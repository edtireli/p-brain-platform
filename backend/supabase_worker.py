
from __future__ import annotations

import os
import socket
import subprocess
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


def _utc_now_iso() -> str:
	return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _env(name: str, default: Optional[str] = None) -> str:
	val = os.getenv(name)
	if val is None or val.strip() == "":
		if default is None:
			raise RuntimeError(f"Missing required env var: {name}")
		return default
	return val


def _env_bool(name: str, default: bool = False) -> bool:
	raw = os.getenv(name)
	if raw is None:
		return default
	v = raw.strip().lower()
	if v in {"1", "true", "yes", "y", "on"}:
		return True
	if v in {"0", "false", "no", "n", "off"}:
		return False
	return default


def _as_subject_id_and_root(source_path: str) -> tuple[str, str]:
	p = Path(source_path).expanduser().resolve()
	if p.is_dir():
		return p.name, str(p.parent)
	# If a file was stored accidentally, treat its parent as subject folder.
	if p.exists():
		return p.parent.name, str(p.parent.parent)
	# If the path doesn't exist on this worker, still attempt basename logic.
	return Path(source_path).name, str(Path(source_path).parent)


@dataclass(frozen=True)
class WorkerConfig:
	supabase_url: str
	supabase_service_role_key: str
	poll_interval_s: float
	worker_id: str

	pbrain_main_py: str
	pbrain_python: str
	pbrain_run_diffusion: bool
	pbrain_turbo: bool
	pbrain_ai_dir: Optional[str]


def load_config() -> WorkerConfig:
	supabase_url = _env("SUPABASE_URL")
	supabase_service_role_key = _env("SUPABASE_SERVICE_ROLE_KEY")

	poll_interval_s = float(os.getenv("PBRAIN_WORKER_POLL_INTERVAL", "2.5"))
	worker_id = os.getenv("PBRAIN_WORKER_ID") or socket.gethostname()

	pbrain_main_py = _env("PBRAIN_MAIN_PY")
	pbrain_python = os.getenv("PBRAIN_PYTHON") or os.sys.executable
	pbrain_run_diffusion = _env_bool("PBRAIN_RUN_DIFFUSION", default=True)
	pbrain_turbo = _env_bool("PBRAIN_TURBO", default=True)

	# Optional: point p-brain's AI_MODEL_PATHS env overrides at a local folder.
	# If unset, we default to a sibling `backend/AI` directory when present.
	default_ai = str(Path(__file__).with_name("AI"))
	pbrain_ai_dir = os.getenv("PBRAIN_AI_DIR") or (default_ai if Path(default_ai).is_dir() else None)

	return WorkerConfig(
		supabase_url=supabase_url,
		supabase_service_role_key=supabase_service_role_key,
		poll_interval_s=poll_interval_s,
		worker_id=worker_id,
		pbrain_main_py=pbrain_main_py,
		pbrain_python=pbrain_python,
		pbrain_run_diffusion=pbrain_run_diffusion,
		pbrain_turbo=pbrain_turbo,
		pbrain_ai_dir=pbrain_ai_dir,
	)


def _import_supabase():
	try:
		from supabase import create_client  # type: ignore

		return create_client
	except Exception as exc:
		raise RuntimeError(
			"Missing Python dependency 'supabase'. Install backend requirements."
		) from exc


def _sb(cfg: WorkerConfig):
	create_client = _import_supabase()
	return create_client(cfg.supabase_url, cfg.supabase_service_role_key)


def _stage_statuses_running() -> Dict[str, str]:
	return {
		"import": "running",
		"t1_fit": "running",
		"input_functions": "running",
		"time_shift": "running",
		"segmentation": "running",
		"tissue_ctc": "running",
		"modelling": "running",
		"diffusion": "running",
		"montage_qc": "running",
	}


def _stage_statuses_done() -> Dict[str, str]:
	return {
		"import": "done",
		"t1_fit": "done",
		"input_functions": "done",
		"time_shift": "done",
		"segmentation": "done",
		"tissue_ctc": "done",
		"modelling": "done",
		"diffusion": "done",
		"montage_qc": "done",
	}


def _stage_statuses_failed(failed_stage: str) -> Dict[str, str]:
	statuses = _stage_statuses_running()
	statuses[failed_stage] = "failed"
	return statuses


def _update_subject_stage_statuses(sb: Any, subject_id: str, statuses: Dict[str, str]) -> None:
	sb.table("subjects").update({"stage_statuses": statuses, "updated_at": _utc_now_iso()}).eq(
		"id", subject_id
	).execute()


def _update_job(sb: Any, job_id: str, patch: Dict[str, Any]) -> None:
	patch = dict(patch)
	patch.setdefault("updated_at", _utc_now_iso())
	sb.table("jobs").update(patch).eq("id", job_id).execute()


def _bulk_update_jobs_for_subject(sb: Any, subject_id: str, statuses: Iterable[str], patch: Dict[str, Any]) -> None:
	patch = dict(patch)
	patch.setdefault("updated_at", _utc_now_iso())
	q = sb.table("jobs").update(patch).eq("subject_id", subject_id)
	q = q.in_("status", list(statuses))
	q.execute()


def _claim_one_job(sb: Any, worker_id: str) -> Optional[Dict[str, Any]]:
	# Fetch a small batch of queued jobs (oldest-first if created_at exists).
	query = sb.table("jobs").select("*").eq("status", "queued").limit(10)
	try:
		query = query.order("created_at", desc=False)
	except Exception:
		# Some schemas might not have created_at.
		pass

	resp = query.execute()
	rows: List[Dict[str, Any]] = resp.data or []
	for row in rows:
		job_id = row.get("id")
		if not job_id:
			continue
		# Atomic claim: only flip queued -> running if still queued.
		patch = {
			"status": "running",
			"start_time": _utc_now_iso(),
			"progress": 0,
			"current_step": f"Claimed by {worker_id}",
			"error": None,
		}
		upd = (
			sb.table("jobs")
			.update(patch)
			.eq("id", job_id)
			.eq("status", "queued")
			.execute()
		)
		claimed = upd.data or []
		if claimed:
			return claimed[0]
	return None


def _run_pbrain_full_pipeline(cfg: WorkerConfig, *, source_path: str, log_path: Path) -> None:
	subject_id, data_root = _as_subject_id_and_root(source_path)
	cmd = [
		cfg.pbrain_python,
		cfg.pbrain_main_py,
		"--id",
		subject_id,
		"--mode",
		"auto",
		"--data-dir",
		data_root,
	]
	if cfg.pbrain_run_diffusion:
		cmd.append("--diffusion")

	env = os.environ.copy()
	if cfg.pbrain_turbo:
		env["PBRAIN_TURBO"] = "1"

	if cfg.pbrain_ai_dir:
		ai_dir = str(Path(cfg.pbrain_ai_dir).expanduser().resolve())
		env.setdefault("SLICE_CLASSIFIER_RICA_MODEL", str(Path(ai_dir) / "slice_classifier_model_rica.keras"))
		env.setdefault("RICA_ROI_MODEL", str(Path(ai_dir) / "rica_roi_model.keras"))
		env.setdefault("SLICE_CLASSIFIER_SS_MODEL", str(Path(ai_dir) / "ss_slice_classifier.keras"))
		env.setdefault("SS_ROI_MODEL", str(Path(ai_dir) / "ss_roi_model.keras"))

	# Ensure relative AI model paths resolve (AI/...).
	pbrain_dir = str(Path(cfg.pbrain_main_py).expanduser().resolve().parent)

	log_path.parent.mkdir(parents=True, exist_ok=True)
	with log_path.open("wb") as f:
		proc = subprocess.Popen(
			cmd,
			cwd=pbrain_dir,
			env=env,
			stdout=f,
			stderr=subprocess.STDOUT,
		)
		rc = proc.wait()

	if rc != 0:
		raise RuntimeError(f"p-brain exited with code {rc}. See log: {log_path}")


def main() -> None:
	cfg = load_config()
	sb = _sb(cfg)

	logs_dir = Path(os.getenv("PBRAIN_WORKER_LOG_DIR", str(Path(__file__).with_name("logs"))))

	print(f"[supabase-worker] worker_id={cfg.worker_id}")
	print(f"[supabase-worker] poll_interval_s={cfg.poll_interval_s}")
	print(f"[supabase-worker] pbrain_main_py={cfg.pbrain_main_py}")
	print(f"[supabase-worker] pbrain_python={cfg.pbrain_python}")

	while True:
		job = _claim_one_job(sb, cfg.worker_id)
		if not job:
			time.sleep(cfg.poll_interval_s)
			continue

		job_id = str(job.get("id"))
		subject_id = str(job.get("subject_id"))

		subj_resp = (
			sb.table("subjects")
			.select("id,source_path")
			.eq("id", subject_id)
			.limit(1)
			.execute()
		)
		subjects = subj_resp.data or []
		if not subjects:
			_update_job(
				sb,
				job_id,
				{
					"status": "failed",
					"end_time": _utc_now_iso(),
					"error": f"Subject not found: {subject_id}",
				},
			)
			continue

		source_path = (subjects[0].get("source_path") or "").strip()
		if not source_path:
			_update_job(
				sb,
				job_id,
				{
					"status": "failed",
					"end_time": _utc_now_iso(),
					"error": "Subject has empty source_path; worker cannot locate data on disk.",
				},
			)
			continue

		run_id = f"{subject_id}-{job_id}"[:80]
		log_path = logs_dir / f"{run_id}.log"

		# Mark all queued jobs for this subject as running. We execute the full
		# p-brain automatic pipeline once per subject run.
		_bulk_update_jobs_for_subject(
			sb,
			subject_id,
			statuses=["queued"],
			patch={
				"status": "running",
				"start_time": _utc_now_iso(),
				"progress": 0,
				"current_step": f"Running full p-brain pipeline on worker {cfg.worker_id}",
				"error": None,
				"log_path": str(log_path),
			},
		)

		try:
			_update_subject_stage_statuses(sb, subject_id, _stage_statuses_running())
			_bulk_update_jobs_for_subject(
				sb,
				subject_id,
				statuses=["running"],
				patch={"progress": 5, "current_step": "Starting p-brain automatic pipeline"},
			)

			_run_pbrain_full_pipeline(cfg, source_path=source_path, log_path=log_path)

			_update_subject_stage_statuses(sb, subject_id, _stage_statuses_done())
			_bulk_update_jobs_for_subject(
				sb,
				subject_id,
				statuses=["running"],
				patch={
					"status": "completed",
					"progress": 100,
					"current_step": "Completed",
					"end_time": _utc_now_iso(),
				},
			)
		except Exception as exc:
			err = str(exc)
			_update_subject_stage_statuses(sb, subject_id, _stage_statuses_failed("modelling"))
			_bulk_update_jobs_for_subject(
				sb,
				subject_id,
				statuses=["running"],
				patch={
					"status": "failed",
					"end_time": _utc_now_iso(),
					"error": err[:2000],
					"current_step": "Failed",
				},
			)


if __name__ == "__main__":
	main()

