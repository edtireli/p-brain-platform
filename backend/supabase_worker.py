
from __future__ import annotations

import os
import socket
import subprocess
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

import requests


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
	pbrain_python = os.getenv("PBRAIN_PYTHON")
	if not pbrain_python:
		try:
			pbrain_dir = Path(pbrain_main_py).expanduser().resolve().parent
			candidates = [
				pbrain_dir / ".venv" / "bin" / "python",
				pbrain_dir / "venv" / "bin" / "python",
				pbrain_dir / ".venv" / "Scripts" / "python.exe",
				pbrain_dir / "venv" / "Scripts" / "python.exe",
			]
			for c in candidates:
				if c.exists():
					pbrain_python = str(c)
					break
		except Exception:
			pbrain_python = None
	if not pbrain_python:
		pbrain_python = os.sys.executable
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
	raise RuntimeError("supabase SDK is no longer used; worker uses HTTP via requests")


class SupabaseHttp:
	def __init__(self, url: str, service_role_key: str):
		self.url = url.rstrip("/")
		self.key = service_role_key
		self.s = requests.Session()
		self.s.headers.update(
			{
				"apikey": self.key,
				"authorization": f"Bearer {self.key}",
				"content-type": "application/json",
			}
		)

	def rpc(self, fn: str, payload: Dict[str, Any]) -> Any:
		r = self.s.post(f"{self.url}/rest/v1/rpc/{fn}", json=payload, timeout=30)
		if r.status_code >= 400:
			raise RuntimeError(f"rpc {fn} failed: {r.status_code} {r.text[:300]}")
		# Can be object, list, or null
		return r.json() if r.text.strip() else None

	def select_one(self, table: str, columns: str, eq: Dict[str, str]) -> Optional[Dict[str, Any]]:
		params = {"select": columns, **{k: f"eq.{v}" for k, v in eq.items()}}
		r = self.s.get(f"{self.url}/rest/v1/{table}", params=params, timeout=30)
		if r.status_code >= 400:
			raise RuntimeError(f"select {table} failed: {r.status_code} {r.text[:300]}")
		data = r.json()
		if isinstance(data, list):
			return data[0] if data else None
		return data if isinstance(data, dict) else None

	def update(self, table: str, eq: Dict[str, str], patch: Dict[str, Any]) -> None:
		headers = {"prefer": "return=minimal"}
		params = {k: f"eq.{v}" for k, v in eq.items()}
		r = self.s.patch(f"{self.url}/rest/v1/{table}", params=params, json=patch, headers=headers, timeout=30)
		if r.status_code >= 400:
			raise RuntimeError(f"update {table} failed: {r.status_code} {r.text[:300]}")

	def insert(self, table: str, row: Dict[str, Any]) -> None:
		headers = {"prefer": "return=minimal"}
		r = self.s.post(f"{self.url}/rest/v1/{table}", json=row, headers=headers, timeout=30)
		if r.status_code >= 400:
			raise RuntimeError(f"insert {table} failed: {r.status_code} {r.text[:300]}")

	def upsert(self, table: str, row: Dict[str, Any], on_conflict: str) -> None:
		headers = {"prefer": "resolution=merge-duplicates,return=minimal"}
		r = self.s.post(
			f"{self.url}/rest/v1/{table}",
			params={"on_conflict": on_conflict},
			json=row,
			headers=headers,
			timeout=30,
		)
		if r.status_code >= 400:
			raise RuntimeError(f"upsert {table} failed: {r.status_code} {r.text[:300]}")

	def upload(self, bucket: str, path: str, content: bytes, upsert: bool = True) -> None:
		# Storage API accepts raw bytes. Keep best-effort; ignore failures.
		headers = {
			"apikey": self.key,
			"authorization": f"Bearer {self.key}",
			"content-type": "application/octet-stream",
			"x-upsert": "true" if upsert else "false",
		}
		r = requests.post(f"{self.url}/storage/v1/object/{bucket}/{path}", data=content, headers=headers, timeout=60)
		if r.status_code >= 400:
			raise RuntimeError(f"upload failed: {r.status_code} {r.text[:200]}")


def _sb(cfg: WorkerConfig) -> SupabaseHttp:
	return SupabaseHttp(cfg.supabase_url, cfg.supabase_service_role_key)


def _claim_job(sb: SupabaseHttp, worker_id: str) -> Optional[Dict[str, Any]]:
	data = sb.rpc("claim_job", {"p_worker_id": worker_id})
	if not data:
		return None
	if isinstance(data, list):
		j = data[0] if data else None
		return j if isinstance(j, dict) and j.get("id") else None
	if isinstance(data, dict):
		return data if data.get("id") else None
	return None


def _update_job(sb: SupabaseHttp, job_id: str, patch: Dict[str, Any]) -> None:
	payload = {**patch, "updated_at": _utc_now_iso()}
	sb.update("jobs", {"id": job_id}, payload)



def _log_event(sb: SupabaseHttp, job_id: str, level: str, message: str) -> None:
	try:
		sb.insert("job_events", {"job_id": job_id, "level": level, "message": message})
	except Exception:
		# Keep runner alive even if logging fails.
		pass


def _insert_output(sb: SupabaseHttp, job_id: str, kind: str, storage_path: str, meta: Dict[str, Any]) -> None:
	try:
		sb.insert(
			"job_outputs",
			{
				"job_id": job_id,
				"kind": kind,
				"storage_path": storage_path,
				"meta": meta,
			},
		)
	except Exception:
		pass


def _upload_log(sb: SupabaseHttp, bucket: str, storage_path: str, log_path: Path) -> None:
	try:
		content = log_path.read_bytes()
		sb.upload(bucket, storage_path, content, upsert=True)
	except Exception:
		pass


def _heartbeat(sb: SupabaseHttp, cfg: WorkerConfig) -> None:
	try:
		sb.upsert(
			"worker_heartbeats",
			{
				"worker_id": cfg.worker_id,
				"last_seen": _utc_now_iso(),
				"hostname": socket.gethostname(),
				"meta": {
					"storage_root": str(cfg.storage_root),
					"bucket": cfg.bucket,
					"poll_interval_s": cfg.poll_interval_s,
					"pbrain_main_py": cfg.pbrain_main_py,
				},
			},
			on_conflict="worker_id",
		)
	except Exception:
		# Never crash the worker due to observability.
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


def _process_job(cfg: WorkerConfig, sb: SupabaseHttp, job: Dict[str, Any]) -> None:
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
	storage_path = f"jobs/{job_id}/logs/runner.log"
	final_status: str = "failed"
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
		final_status = "completed"
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
		final_status = "failed"
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

	# Upload log to Storage + register as output (best-effort), even on failures.
	_upload_log(sb, cfg.bucket, storage_path, log_path)
	_insert_output(sb, job_id, kind="log", storage_path=storage_path, meta={"runner_id": cfg.worker_id, "status": final_status})


def main() -> None:
	cfg = load_config()
	sb = _sb(cfg)

	print(f"[supabase-worker] worker_id={cfg.worker_id}")
	print(f"[supabase-worker] poll_interval_s={cfg.poll_interval_s}")
	print(f"[supabase-worker] storage_root={cfg.storage_root}")
	print(f"[supabase-worker] bucket={cfg.bucket}")

	while True:
		try:
			_heartbeat(sb, cfg)
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

