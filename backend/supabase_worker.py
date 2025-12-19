
from __future__ import annotations

import os
import socket
import subprocess
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional


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


@dataclass(frozen=True)
class WorkerConfig:
	supabase_url: str
	supabase_service_role_key: str
	bucket: str
	storage_root: Path
	poll_interval_s: float
	worker_id: str

	pbrain_main_py: str
	pbrain_python: str
	pbrain_run_diffusion: bool
	pbrain_turbo: bool
	pbrain_ai_dir: Optional[str]
	logs_dir: Path


def load_config() -> WorkerConfig:
	supabase_url = _env("SUPABASE_URL")
	supabase_service_role_key = _env("SUPABASE_SERVICE_ROLE_KEY")
	bucket = _env("PBRAIN_STORAGE_BUCKET", "pbrain")
	storage_root = Path(_env("PBRAIN_STORAGE_ROOT")).expanduser().resolve()
	poll_interval_s = float(os.getenv("PBRAIN_WORKER_POLL_INTERVAL", "2.5"))
	worker_id = os.getenv("PBRAIN_WORKER_ID") or socket.gethostname()

	pbrain_main_py = _env("PBRAIN_MAIN_PY")
	pbrain_python = os.getenv("PBRAIN_PYTHON") or os.sys.executable
	pbrain_run_diffusion = _env_bool("PBRAIN_RUN_DIFFUSION", default=True)
	pbrain_turbo = _env_bool("PBRAIN_TURBO", default=True)
	pbrain_ai_dir = os.getenv("PBRAIN_AI_DIR") or None
	logs_dir = Path(os.getenv("PBRAIN_WORKER_LOG_DIR", str(Path(__file__).with_name("logs")))).expanduser().resolve()

	return WorkerConfig(
		supabase_url=supabase_url,
		supabase_service_role_key=supabase_service_role_key,
		bucket=bucket,
		storage_root=storage_root,
		poll_interval_s=poll_interval_s,
		worker_id=worker_id,
		pbrain_main_py=pbrain_main_py,
		pbrain_python=pbrain_python,
		pbrain_run_diffusion=pbrain_run_diffusion,
		pbrain_turbo=pbrain_turbo,
		pbrain_ai_dir=pbrain_ai_dir,
		logs_dir=logs_dir,
	)


def _import_supabase():
	try:
		from supabase import create_client  # type: ignore

		return create_client
	except Exception as exc:
		raise RuntimeError("Missing Python dependency 'supabase'. Install backend requirements.") from exc


def _sb(cfg: WorkerConfig):
	create_client = _import_supabase()
	return create_client(cfg.supabase_url, cfg.supabase_service_role_key)


def _claim_job(sb: Any, worker_id: str) -> Optional[Dict[str, Any]]:
	resp = sb.rpc("claim_job", {"p_worker_id": worker_id}).execute()
	data = resp.data
	if not data:
		return None
	if isinstance(data, list):
		return data[0] if data else None
	if isinstance(data, dict):
		return data
	return None


def _update_job(sb: Any, job_id: str, patch: Dict[str, Any]) -> None:
	payload = {**patch, "updated_at": _utc_now_iso()}
	sb.table("jobs").update(payload).eq("id", job_id).execute()


def _log_event(sb: Any, job_id: str, level: str, message: str) -> None:
	try:
		sb.table("job_events").insert({"job_id": job_id, "level": level, "message": message}).execute()
	except Exception:
		# Keep runner alive even if logging fails.
		pass


def _insert_output(sb: Any, job_id: str, kind: str, storage_path: str, meta: Dict[str, Any]) -> None:
	try:
		sb.table("job_outputs").insert({
			"job_id": job_id,
			"kind": kind,
			"storage_path": storage_path,
			"meta": meta,
		}).execute()
	except Exception:
		pass


def _upload_log(sb: Any, bucket: str, storage_path: str, log_path: Path) -> None:
	try:
		with log_path.open("rb") as f:
			sb.storage.from_(bucket).upload(storage_path, f, {"upsert": True})
	except Exception:
		pass


def _resolve_subject(payload: Dict[str, Any], storage_root: Path) -> tuple[str, Path]:
	rel_path = (payload.get("relative_path") or payload.get("path") or payload.get("source_path") or "").strip()
	if not rel_path:
		raise RuntimeError("Job payload missing relative_path/path/source_path")
	full = (storage_root / rel_path).expanduser().resolve()
	if not full.exists():
		raise RuntimeError(f"Local path not found on runner: {full}")
	subject_id = str(payload.get("subject_id") or full.name)
	return subject_id, full


def _run_pbrain(cfg: WorkerConfig, subject_id: str, subject_path: Path, log_path: Path) -> None:
	cmd = [
		cfg.pbrain_python,
		cfg.pbrain_main_py,
		"--id",
		subject_id,
		"--mode",
		"auto",
		"--data-dir",
		str(subject_path.parent if subject_path.is_dir() else subject_path.parent),
	]
	if cfg.pbrain_run_diffusion:
		cmd.append("--diffusion")

	env = os.environ.copy()
	if cfg.pbrain_turbo:
		env["PBRAIN_TURBO"] = "1"

	ai_dir_raw = cfg.pbrain_ai_dir
	if ai_dir_raw:
		ai_dir = Path(ai_dir_raw).expanduser().resolve()
		env.setdefault("SLICE_CLASSIFIER_RICA_MODEL", str(ai_dir / "slice_classifier_model_rica.keras"))
		env.setdefault("RICA_ROI_MODEL", str(ai_dir / "rica_roi_model.keras"))
		env.setdefault("SLICE_CLASSIFIER_SS_MODEL", str(ai_dir / "ss_slice_classifier.keras"))
		env.setdefault("SS_ROI_MODEL", str(ai_dir / "ss_roi_model.keras"))

	log_path.parent.mkdir(parents=True, exist_ok=True)
	with log_path.open("wb") as f:
		proc = subprocess.Popen(
			cmd,
			cwd=str(Path(cfg.pbrain_main_py).expanduser().resolve().parent),
			env=env,
			stdout=f,
			stderr=subprocess.STDOUT,
		)
		rc = proc.wait()
	if rc != 0:
		raise RuntimeError(f"p-brain exited with code {rc}; see log")


def _process_job(cfg: WorkerConfig, sb: Any, job: Dict[str, Any]) -> None:
	job_id = str(job.get("id"))
	payload: Dict[str, Any] = job.get("payload") or {}
	_log_event(sb, job_id, "info", f"claiming on {cfg.worker_id}")

	try:
		subject_id, subject_path = _resolve_subject(payload, cfg.storage_root)
	except Exception as exc:
		err = str(exc)
		_update_job(sb, job_id, {"status": "failed", "error": err, "finished_at": _utc_now_iso(), "end_time": _utc_now_iso()})
		_log_event(sb, job_id, "error", err)
		return

	log_path = cfg.logs_dir / f"{job_id}.log"
	_update_job(
		sb,
		job_id,
		{
			"status": "running",
			"runner_id": cfg.worker_id,
			"claimed_at": job.get("claimed_at") or _utc_now_iso(),
			"start_time": job.get("start_time") or _utc_now_iso(),
			"current_step": "running pipeline",
			"progress": 0,
			"log_path": str(log_path),
			"error": None,
		},
	)
	_log_event(sb, job_id, "info", f"started {subject_id} at {subject_path}")

	try:
		_run_pbrain(cfg, subject_id, subject_path, log_path)
		_update_job(
			sb,
			job_id,
			{
				"status": "completed",
				"progress": 1,
				"current_step": "done",
				"finished_at": _utc_now_iso(),
				"end_time": _utc_now_iso(),
			},
		)
		_log_event(sb, job_id, "info", "completed")
	except Exception as exc:
		err = str(exc)
		_update_job(
			sb,
			job_id,
			{
				"status": "failed",
				"error": err,
				"finished_at": _utc_now_iso(),
				"end_time": _utc_now_iso(),
				"current_step": "failed",
			},
		)
		_log_event(sb, job_id, "error", err)
		return

	# Upload log to Storage + register as output (best-effort)
	storage_path = f"jobs/{job_id}/logs/runner.log"
	_upload_log(sb, cfg.bucket, storage_path, log_path)
	_insert_output(
		sb,
		job_id,
		kind="log",
		storage_path=storage_path,
		meta={"runner_id": cfg.worker_id},
	)


def main() -> None:
	cfg = load_config()
	sb = _sb(cfg)

	print(f"[supabase-worker] worker_id={cfg.worker_id}")
	print(f"[supabase-worker] poll_interval_s={cfg.poll_interval_s}")
	print(f"[supabase-worker] storage_root={cfg.storage_root}")
	print(f"[supabase-worker] bucket={cfg.bucket}")

	while True:
		try:
			job = _claim_job(sb, cfg.worker_id)
			if not job:
				time.sleep(cfg.poll_interval_s)
				continue
			_process_job(cfg, sb, job)
		except KeyboardInterrupt:
			print("[supabase-worker] stopping")
			break
		except Exception as exc:
			print(f"[supabase-worker] loop error: {exc}")
			time.sleep(cfg.poll_interval_s)


if __name__ == "__main__":
	main()

