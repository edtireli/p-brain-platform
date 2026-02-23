
from __future__ import annotations

import os
import socket
import subprocess
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from fnmatch import fnmatch
from pathlib import Path
from typing import Any, Dict, Optional

import requests


# Keep stage ids in sync with the web UI (src/lib/supabase-engine.ts).
_STAGES = (
	"import",
	"t1_fit",
	"input_functions",
	"time_shift",
	"segmentation",
	"tissue_ctc",
	"modelling",
	"diffusion",
	"montage_qc",
)


def _default_stage_statuses() -> Dict[str, str]:
	# Keep in sync with UI types: not_run | running | done | failed
	return {stage: "not_run" for stage in _STAGES}


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

	def select_many(
		self,
		table: str,
		columns: str,
		*,
		eq: Optional[Dict[str, str]] = None,
		order: Optional[str] = None,
		limit: Optional[int] = None,
	) -> list[Dict[str, Any]]:
		params: Dict[str, Any] = {"select": columns}
		if eq:
			params.update({k: f"eq.{v}" for k, v in eq.items()})
		if order:
			params["order"] = order
		if limit is not None:
			params["limit"] = str(int(limit))
		r = self.s.get(f"{self.url}/rest/v1/{table}", params=params, timeout=30)
		if r.status_code >= 400:
			raise RuntimeError(f"select {table} failed: {r.status_code} {r.text[:300]}")
		data = r.json()
		return data if isinstance(data, list) else ([] if not data else [data])

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


def _stage_index(stage_id: str) -> int:
	try:
		return list(_STAGES).index(stage_id)
	except ValueError:
		return 999


def _is_stage_done(status: Any) -> bool:
	return str(status or "").strip().lower() == "done"


def _is_stage_failed(status: Any) -> bool:
	return str(status or "").strip().lower() == "failed"


def _is_stage_running(status: Any) -> bool:
	return str(status or "").strip().lower() == "running"


def _job_ready(stage_id: str, stage_statuses: Dict[str, Any]) -> bool:
	# Enforce a simple linear dependency chain based on _STAGES ordering.
	idx = _stage_index(stage_id)
	if idx <= 0:
		return True
	# If any prior stage failed, don't run downstream stages.
	for s in _STAGES[:idx]:
		if _is_stage_failed(stage_statuses.get(s)):
			return False
	# Require all prior stages done.
	return all(_is_stage_done(stage_statuses.get(s)) for s in _STAGES[:idx])


def _claim_next_ready_job(sb: SupabaseHttp, worker_id: str) -> Optional[Dict[str, Any]]:
	# We don't use the claim_job RPC because it cannot filter for stage readiness.
	# This repo typically runs a single local worker; best-effort claiming is sufficient.
	queued = sb.select_many(
		"jobs",
		"id,project_id,subject_id,stage_id,status,payload,created_at,start_time,claimed_at",
		eq={"status": "queued"},
		order="created_at.asc",
		limit=120,
	)
	if not queued:
		return None

	# Preload stage_statuses for subjects referenced by queued jobs (best-effort).
	subject_ids = sorted({str(j.get("subject_id") or "") for j in queued if j.get("subject_id")})
	stage_by_subject: Dict[str, Dict[str, Any]] = {}
	for sid in subject_ids[:50]:
		try:
			row = sb.select_one("subjects", "id,stage_statuses:stage_statuses", {"id": sid})
			st = (row or {}).get("stage_statuses")
			stage_by_subject[sid] = st if isinstance(st, dict) else {}
		except Exception:
			stage_by_subject[sid] = {}

	def sort_key(j: Dict[str, Any]) -> tuple:
		created = str(j.get("created_at") or "")
		payload = j.get("payload") or {}
		idx = payload.get("stage_index")
		try:
			idx_i = int(idx)
		except Exception:
			idx_i = _stage_index(str(j.get("stage_id") or ""))
		return (created, idx_i, str(j.get("id") or ""))

	queued_sorted = sorted(queued, key=sort_key)
	for j in queued_sorted:
		sid = str(j.get("subject_id") or "")
		stage_id = str(j.get("stage_id") or "")
		statuses = stage_by_subject.get(sid) or {}
		# Allow claiming jobs whose stage is already done/failed so we can mark them terminal.
		if _is_stage_done(statuses.get(stage_id)) or _is_stage_failed(statuses.get(stage_id)):
			return j
		if not _job_ready(stage_id, statuses):
			continue
		return j
	return None


def _update_job(sb: SupabaseHttp, job_id: str, patch: Dict[str, Any]) -> None:
	payload = {**patch, "updated_at": _utc_now_iso()}
	sb.update("jobs", {"id": job_id}, payload)


def _update_subject(sb: SupabaseHttp, subject_id: str, patch: Dict[str, Any]) -> None:
	payload = {**patch, "updated_at": _utc_now_iso()}
	sb.update("subjects", {"id": subject_id}, payload)


def _set_subject_stage(sb: SupabaseHttp, subject_id: str, stage_id: str, status: str) -> None:
	try:
		row = sb.select_one("subjects", "id,stage_statuses:stage_statuses", {"id": subject_id})
		current = (row or {}).get("stage_statuses") or {}
		if not isinstance(current, dict):
			current = {}
		next_statuses: Dict[str, Any] = {**_default_stage_statuses(), **current, stage_id: status}
		_update_subject(sb, subject_id, {"stage_statuses": next_statuses})
	except Exception:
		# Never crash the worker due to status propagation.
		pass


def _fail_running_stages(sb: SupabaseHttp, subject_id: str) -> None:
	try:
		row = sb.select_one("subjects", "id,stage_statuses:stage_statuses", {"id": subject_id})
		current = (row or {}).get("stage_statuses") or {}
		if not isinstance(current, dict):
			current = {}
		next_statuses: Dict[str, Any] = {**_default_stage_statuses(), **current}
		changed = False
		for k, v in list(next_statuses.items()):
			if v == "running":
				next_statuses[k] = "failed"
				changed = True
		if not changed:
			# If nothing was marked running, still surface failure on the entry stage.
			next_statuses["import"] = "failed"
		_update_subject(sb, subject_id, {"stage_statuses": next_statuses})
	except Exception:
		pass


def _update_subject_flags(sb: SupabaseHttp, subject_id: str, *, has_t1: bool, has_dce: bool, has_diffusion: bool) -> None:
	try:
		_update_subject(
			sb,
			subject_id,
			{
				"has_t1": bool(has_t1),
				"has_dce": bool(has_dce),
				"has_diffusion": bool(has_diffusion),
			},
		)
	except Exception:
		pass


def _scan_subject_inputs(subject_path: Path) -> tuple[bool, bool, bool]:
	"""Fast input presence scan for the Import & Index stage.

	This is intentionally lightweight: it should take milliseconds to seconds.
	"""
	nifti_dir = subject_path / "NIfTI"
	if not nifti_dir.exists() or not nifti_dir.is_dir():
		return False, False, False

	try:
		names = [p.name.lower() for p in nifti_dir.iterdir() if p.is_file()]
	except Exception:
		return False, False, False

	# Backstop heuristics aligned with observed datasets.
	has_t1 = any("t1" in n and n.endswith((".nii", ".nii.gz")) for n in names)
	# DCE / perfusion acquisitions are large and often include these tokens.
	has_dce = any(("perf" in n or "dce" in n) and n.endswith((".nii", ".nii.gz")) for n in names)
	# Diffusion often includes DWI/ADC tokens or specific known prefixes.
	has_diff = any(
		("dwi" in n or "adc" in n or "reg-dwi" in n or "isodwi" in n) and n.endswith((".nii", ".nii.gz"))
		for n in names
	)
	return has_t1, has_dce, has_diff


def _scan_subject_inputs_with_patterns(subject_path: Path, folder_structure: Dict[str, Any] | None) -> tuple[bool, bool, bool]:
	"""Scan for inputs using the same wildcard patterns as sync-artifacts + the UI config."""
	if not folder_structure or not isinstance(folder_structure, dict):
		return _scan_subject_inputs(subject_path)

	nifti_dir = subject_path / "NIfTI"
	if not nifti_dir.exists() or not nifti_dir.is_dir():
		return _scan_subject_inputs(subject_path)

	try:
		files = [p.name for p in nifti_dir.iterdir() if p.is_file()]
	except Exception:
		return _scan_subject_inputs(subject_path)

	def pats(key: str, fallback: str) -> list[str]:
		raw = str(folder_structure.get(key) or "").strip()
		if not raw:
			raw = fallback
		return [p.strip() for p in raw.split(",") if p.strip()]

	def match_any(patterns: list[str]) -> bool:
		for pat in patterns:
			lp = pat.lower()
			for f in files:
				lf = f.lower()
				if fnmatch(lf, lp) and (lf.endswith(".nii") or lf.endswith(".nii.gz")):
					return True
		return False

	# Keep these defaults aligned with scripts/sync-artifacts.mjs.
	has_t1 = match_any(pats("t1Pattern", "WIPcs_T1W_3D_TFE_32channel.nii*,*T1*.nii*"))
	has_t2 = match_any(
		pats(
			"t2Pattern",
			"WIPcs_3D_Brain_VIEW_T2_32chSHC.nii*,ax*WIPcs_3D_Brain_VIEW_T2_32chSHC.nii*,WIPAxT2TSEmatrix.nii*,*T2*.nii*",
		)
	)
	has_flair = match_any(
		pats(
			"flairPattern",
			"WIPcs_3D_Brain_VIEW_FLAIR_SHC.nii*,ax*WIPcs_3D_Brain_VIEW_FLAIR_SHC.nii*,*FLAIR*.nii*",
		)
	)
	has_dce = match_any(pats("dcePattern", "WIPDelRec-hperf120long.nii*,WIPhperf120long.nii*,*DCE*.nii*"))
	has_diff = match_any(
		pats(
			"diffusionPattern",
			"Reg-DWInySENSE.nii*,Reg-DWInySENSE_ADC.nii*,isoDWIb-1000*.nii*,WIPDTI_RSI_*.nii*,WIPDWI_RSI_*.nii*,*DTI*.nii*",
		)
	)
	# Treat T2/FLAIR presence as non-fatal for flags; primary flags are T1, DCE, diffusion.
	return bool(has_t1), bool(has_dce), bool(has_diff)



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




def _normalize_subject_path(raw: str, storage_root: Path) -> Path:
	"""Resolve a subject path from a payload value.

	The UI typically stores a *relative* path (e.g. "20250218x3") relative to
	`storage_root`. Some users may provide paths like "data/20250218x3" when
	`storage_root` itself is the "data" folder; normalize that to avoid
	".../data/data/...".
	"""
	val = (raw or "").strip().replace("\\", "/")
	if not val:
		raise RuntimeError("Job payload missing relative_path/path/source_path")

	root = storage_root.expanduser().resolve()

	# 1) If an absolute path is provided and exists, accept it.
	try_abs = Path(val).expanduser()
	if try_abs.is_absolute():
		abs_resolved = try_abs.resolve()
		if abs_resolved.exists():
			return abs_resolved
		# If it's absolute but doesn't exist (e.g. different machine), fall back to
		# treating it as relative to storage_root.
		val = val.lstrip("/")

	# 2) If payload accidentally includes the storage_root prefix, make it relative.
	root_str = str(root).rstrip("/")
	if root_str and val.startswith(root_str + "/"):
		val = val[len(root_str) + 1 :]

	# 3) Remove duplicated leading directory (e.g. "data/<x>" when root.name == "data").
	parts = [p for p in val.split("/") if p and p != "."]
	if parts and root.name and parts[0] == root.name:
		parts = parts[1:]

	return (root / Path(*parts)).resolve()


def _resolve_subject(payload: Dict[str, Any], storage_root: Path) -> tuple[str, Path]:
	rel_path = (payload.get("relative_path") or payload.get("path") or payload.get("source_path") or "").strip()
	full = _normalize_subject_path(rel_path, storage_root)
	if not full.exists():
		raise RuntimeError(f"Local path not found on runner: {full}")
	# p-brain expects --id to match the dataset folder name under --data-dir.
	# The Supabase subject_id is typically a UUID and should NOT be used here.
	pbrain_id = str(payload.get("pbrain_id") or full.name)
	return pbrain_id, full


def _run_pbrain(
	cfg: WorkerConfig,
	sb: SupabaseHttp,
	*,
	jobs_by_stage: Dict[str, str],
	subject_db_id: str,
	subject_id: str,
	subject_path: Path,
	log_path: Path,
) -> None:
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
	# Keep AI input-function defaults stable for batch/worker runs.
	# Do not let runner host env accidentally change ROI quality.
	for k in (
		"P_BRAIN_AI_ICA_FRAME_MODE",
		"P_BRAIN_AI_ICA_FRAME_OVERRIDE",
		"P_BRAIN_AI_ROI_POSTPROCESS",
	):
		env.pop(k, None)
	env.setdefault("P_BRAIN_AI_ICA_FRAME_MODE", "intensity")
	env.setdefault("P_BRAIN_AI_ROI_POSTPROCESS", "1")
	if cfg.pbrain_turbo:
		env["PBRAIN_TURBO"] = "1"
		# Ensure headless matplotlib backend for batch runs.
		env.setdefault("MPLBACKEND", "Agg")

	ai_dir_raw = cfg.pbrain_ai_dir
	if ai_dir_raw:
		ai_dir = Path(ai_dir_raw).expanduser().resolve()
		env.setdefault("SLICE_CLASSIFIER_RICA_MODEL", str(ai_dir / "slice_classifier_model_rica.keras"))
		env.setdefault("RICA_ROI_MODEL", str(ai_dir / "rica_roi_model.keras"))
		env.setdefault("SLICE_CLASSIFIER_SS_MODEL", str(ai_dir / "ss_slice_classifier.keras"))
		env.setdefault("SS_ROI_MODEL", str(ai_dir / "ss_roi_model.keras"))

	log_path.parent.mkdir(parents=True, exist_ok=True)

	# Stream output so we can surface progress + stage statuses in the UI.
	last_sent_at: Dict[str, float] = {}
	last_progress_sent: Dict[str, int] = {}

	def send_job_update(job_id: str, *, progress: Optional[int] = None, step: Optional[str] = None, force: bool = False) -> None:
		now = time.monotonic()
		last_t = last_sent_at.get(job_id, 0.0)
		last_p = last_progress_sent.get(job_id, -1)
		p = None if progress is None else int(progress)
		if not force:
			if now - last_t < 0.9:
				return
			if p is not None and abs(p - last_p) < 1:
				p = None
		if p is None and step is None:
			return
		try:
			patch: Dict[str, Any] = {}
			if p is not None:
				patch["progress"] = max(0, min(100, int(p)))
				last_progress_sent[job_id] = int(patch["progress"])
			if step is not None:
				patch["current_step"] = step
			_update_job(sb, job_id, patch)
			last_sent_at[job_id] = now
		except Exception:
			pass

	stage_job_id: Dict[str, str] = {k: v for k, v in jobs_by_stage.items() if v}

	def begin_stage(stage_id: str, label: str) -> None:
		jid = stage_job_id.get(stage_id)
		if subject_db_id:
			_set_subject_stage(sb, subject_db_id, stage_id, "running")
		if jid:
			_update_job(
				sb,
				jid,
				{
					"status": "running",
					"runner_id": cfg.worker_id,
					"claimed_at": _utc_now_iso(),
					"start_time": _utc_now_iso(),
					"error": None,
				},
			)
			send_job_update(jid, progress=0, step=label, force=True)
			_log_event(sb, jid, "info", f"stage started: {label}")

	def end_stage(stage_id: str, label: str) -> None:
		jid = stage_job_id.get(stage_id)
		if subject_db_id:
			_set_subject_stage(sb, subject_db_id, stage_id, "done")
		if jid:
			send_job_update(jid, progress=100, step=label, force=True)
			_update_job(
				sb,
				jid,
				{
					"status": "completed",
					"progress": 100,
					"current_step": label,
					"finished_at": _utc_now_iso(),
					"end_time": _utc_now_iso(),
				},
			)
			_log_event(sb, jid, "info", f"stage completed: {label}")

	active_stage: str = "t1_fit"
	active_job_id: str = stage_job_id.get(active_stage) or ""
	active_stage_progress = 0
	last_event_sent = 0.0

	def bump_stage_progress(delta: int = 1) -> None:
		nonlocal active_stage_progress
		if not active_job_id:
			return
		active_stage_progress = max(0, min(99, active_stage_progress + int(delta)))
		send_job_update(active_job_id, progress=active_stage_progress, step=None)

	def emit_log_for_active(line: str, level: str = "info") -> None:
		nonlocal last_event_sent
		if not active_job_id:
			return
		now = time.monotonic()
		if now - last_event_sent < 0.2:
			return
		last_event_sent = now
		_log_event(sb, active_job_id, level, line)

	with log_path.open("w", encoding="utf-8", errors="replace") as f:
		proc = subprocess.Popen(
			cmd,
			cwd=str(Path(cfg.pbrain_main_py).expanduser().resolve().parent),
			env=env,
			stdout=subprocess.PIPE,
			stderr=subprocess.STDOUT,
			text=True,
			bufsize=1,
		)
		assert proc.stdout is not None
		for raw_line in proc.stdout:
			line = raw_line.rstrip("\n")
			f.write(raw_line)
			# Keep the log reasonably up to date on disk.
			f.flush()

			# Stream selected logs to Supabase for real-time UI.
			if "[AUTO]" in line or "Executing order" in line or "error" in line.lower():
				emit_log_for_active(line)

			# Parse p-brain automatic logs (see p-brain/utils/cli_logging.py).
			# Example: "[AUTO] 12:34:56 | Starting process: T1 fitting"
			if "Starting process:" in line:
				if "T1 fitting" in line:
					active_stage = "t1_fit"
					active_job_id = stage_job_id.get(active_stage) or active_job_id
					active_stage_progress = 0
					begin_stage("t1_fit", "T1 fitting")
				elif "AI input function extraction" in line:
					end_stage("t1_fit", "T1 fitting done")
					active_stage = "input_functions"
					active_job_id = stage_job_id.get(active_stage) or active_job_id
					active_stage_progress = 0
					begin_stage("input_functions", "Input functions")
				elif "Tissue kinetic modelling" in line:
					end_stage("input_functions", "Input functions done")
					active_stage = "modelling"
					active_job_id = stage_job_id.get(active_stage) or active_job_id
					active_stage_progress = 0
					# These are sub-steps inside the modelling stage; mark them active together.
					begin_stage("modelling", "Modelling")
					if subject_db_id:
						_set_subject_stage(sb, subject_db_id, "time_shift", "running")
						_set_subject_stage(sb, subject_db_id, "segmentation", "running")
						_set_subject_stage(sb, subject_db_id, "tissue_ctc", "running")
						if cfg.pbrain_run_diffusion:
							_set_subject_stage(sb, subject_db_id, "diffusion", "running")
				elif "Segmented M0/T1 rendering" in line:
					# Modelling is effectively complete once rendering begins.
					end_stage("modelling", "Modelling done")
					if subject_db_id:
						_set_subject_stage(sb, subject_db_id, "time_shift", "done")
						_set_subject_stage(sb, subject_db_id, "segmentation", "done")
						_set_subject_stage(sb, subject_db_id, "tissue_ctc", "done")
						if cfg.pbrain_run_diffusion:
							_set_subject_stage(sb, subject_db_id, "diffusion", "done")
					active_stage = "montage_qc"
					active_job_id = stage_job_id.get(active_stage) or active_job_id
					active_stage_progress = 0
					begin_stage("montage_qc", "Rendering outputs")

			elif "Completed process:" in line:
				if "T1 fitting" in line:
					end_stage("t1_fit", "T1 fitting done")
				elif "AI input function extraction" in line:
					end_stage("input_functions", "Input functions done")
				elif "Tissue kinetic modelling" in line:
					end_stage("modelling", "Modelling done")
					if subject_db_id:
						_set_subject_stage(sb, subject_db_id, "time_shift", "done")
						_set_subject_stage(sb, subject_db_id, "segmentation", "done")
						_set_subject_stage(sb, subject_db_id, "tissue_ctc", "done")
						if cfg.pbrain_run_diffusion:
							_set_subject_stage(sb, subject_db_id, "diffusion", "done")
				elif "Segmented M0/T1 rendering" in line:
					end_stage("montage_qc", "Rendering done")

			# Opportunistic progress bumps during long runs.
			elif "Generated file:" in line or "Generated image:" in line:
				bump_stage_progress(1)
				# Also surface file generation in the active stage log.
				emit_log_for_active(line)

		rc = proc.wait()

	if rc != 0:
		raise RuntimeError(f"p-brain exited with code {rc}; see log")


def _sync_artifacts(cfg: WorkerConfig, job: Dict[str, Any], subject_path: Path, log_path: Path) -> None:
	"""Upload a minimal set of artifacts to Supabase Storage for the web UI.

	Uses the repo's scripts/sync-artifacts.mjs to keep artifact layout consistent.
	"""
	job_id = str(job.get("id"))
	project_id = str(job.get("project_id") or "")
	if not project_id:
		raise RuntimeError("Job missing project_id; cannot sync artifacts")

	repo_root = Path(__file__).expanduser().resolve().parents[1]
	script = repo_root / "scripts" / "sync-artifacts.mjs"
	if not script.exists():
		raise RuntimeError(f"sync-artifacts script not found: {script}")

	# Favor Node from PATH.
	max_upload_mb = os.getenv("PBRAIN_MAX_UPLOAD_MB") or "45"
	cmd = [
		"node",
		str(script),
		"--project",
		project_id,
		"--subject-dir",
		str(subject_path),
		"--bucket",
		cfg.bucket,
		"--max-upload-mb",
		str(max_upload_mb),
	]

	env = os.environ.copy()
	env.setdefault("SUPABASE_URL", cfg.supabase_url)
	env.setdefault("SUPABASE_SERVICE_ROLE_KEY", cfg.supabase_service_role_key)
	env.setdefault("SUPABASE_BUCKET", cfg.bucket)

	# Align source volume selection with the configured folder structure.
	try:
		payload = job.get("payload") or {}
		fs_cfg = payload.get("folder_structure") if isinstance(payload, dict) else None
		if isinstance(fs_cfg, dict):
			def _set_if(key: str, env_key: str):
				val = str(fs_cfg.get(key) or "").strip()
				if val:
					env[env_key] = val
			_set_if("t1Pattern", "PBRAIN_T1_PATTERN")
			_set_if("t2Pattern", "PBRAIN_T2_PATTERN")
			_set_if("flairPattern", "PBRAIN_FLAIR_PATTERN")
			_set_if("dcePattern", "PBRAIN_DCE_PATTERN")
			_set_if("diffusionPattern", "PBRAIN_DIFFUSION_PATTERN")
	except Exception:
		pass

	# Append output to the same runner log for easy debugging.
	with log_path.open("ab") as f:
		f.write(b"\n[supabase-worker] syncing artifacts to Storage...\n")
		proc = subprocess.Popen(
			cmd,
			cwd=str(repo_root),
			env=env,
			stdout=f,
			stderr=subprocess.STDOUT,
		)
		rc = proc.wait()
		if rc != 0:
			raise RuntimeError(f"sync-artifacts exited with code {rc}; see log")
		f.write(b"[supabase-worker] artifact sync done\n")


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

	subject_db_id = str(job.get("subject_id") or "")
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

	# Fast Import & Index: scan inputs + mark the stage done quickly.
	stage_id = str(job.get("stage_id") or "import")

	# If the stage is already terminal in subjects.stage_statuses, mirror that to the job row.
	try:
		subj = sb.select_one("subjects", "id,stage_statuses:stage_statuses", {"id": subject_db_id}) if subject_db_id else None
		statuses = (subj or {}).get("stage_statuses")
		statuses = statuses if isinstance(statuses, dict) else {}
		if _is_stage_done(statuses.get(stage_id)):
			_update_job(sb, job_id, {"status": "completed", "progress": 100, "current_step": "Already done", "finished_at": _utc_now_iso(), "end_time": _utc_now_iso()})
			final_status = "completed"
			return
		if _is_stage_failed(statuses.get(stage_id)):
			_update_job(sb, job_id, {"status": "failed", "progress": 100, "current_step": "Upstream failed", "finished_at": _utc_now_iso(), "end_time": _utc_now_iso()})
			final_status = "failed"
			return
	except Exception:
		pass

	if stage_id == "import":
		if subject_db_id:
			_set_subject_stage(sb, subject_db_id, "import", "running")
		try:
			_update_job(sb, job_id, {"current_step": "Import & Index", "progress": 0})
			folder_structure = (payload.get("folder_structure") or {}) if isinstance(payload, dict) else {}
			has_t1, has_dce, has_diff = _scan_subject_inputs_with_patterns(subject_path, folder_structure)
			if subject_db_id:
				_update_subject_flags(sb, subject_db_id, has_t1=has_t1, has_dce=has_dce, has_diffusion=has_diff)
				_set_subject_stage(sb, subject_db_id, "import", "done")
			_update_job(sb, job_id, {"status": "completed", "progress": 100, "current_step": "Indexed", "finished_at": _utc_now_iso(), "end_time": _utc_now_iso()})
			_log_event(sb, job_id, "info", f"indexed inputs: t1={has_t1} dce={has_dce} diff={has_diff}")
			final_status = "completed"
			return
		except Exception as exc:
			err = str(exc)
			final_status = "failed"
			_update_job(sb, job_id, {"status": "failed", "error": err, "finished_at": _utc_now_iso(), "end_time": _utc_now_iso(), "current_step": "failed"})
			_log_event(sb, job_id, "error", err)
			if subject_db_id:
				_fail_running_stages(sb, subject_db_id)
			return

	# Compute pipeline: run p-brain once (triggered by the first compute stage),
	# and update all stage jobs/statuses based on [AUTO] log boundaries.
	try:
		# Load all stage jobs for this subject that are still queued/running.
		stage_jobs = sb.select_many(
			"jobs",
			"id,stage_id,status,payload,created_at",
			eq={"subject_id": subject_db_id},
			order="created_at.desc",
			limit=200,
		)
		jobs_by_stage: Dict[str, str] = {}
		for r in stage_jobs:
			st = str(r.get("stage_id") or "")
			if st in _STAGES and str(r.get("status") or "") in {"queued", "running"}:
				# Keep the newest job id for each stage.
				jobs_by_stage.setdefault(st, str(r.get("id")))
		# Ensure the currently claimed job is used for its stage.
		jobs_by_stage[stage_id] = job_id

		# Mark this stage as running.
		if subject_db_id:
			_set_subject_stage(sb, subject_db_id, stage_id, "running")
		_update_job(sb, job_id, {"status": "running", "progress": 0, "current_step": "Starting"})

		# Run p-brain; stages are advanced by parsing [AUTO] logs.
		_run_pbrain(
			cfg,
			sb,
			jobs_by_stage=jobs_by_stage,
			subject_db_id=subject_db_id,
			subject_id=subject_id,
			subject_path=subject_path,
			log_path=log_path,
		)

		# Upload artifacts as the montage_qc stage.
		mq_job = jobs_by_stage.get("montage_qc") or job_id
		if subject_db_id:
			_set_subject_stage(sb, subject_db_id, "montage_qc", "running")
		_update_job(sb, mq_job, {"status": "running", "progress": 0, "current_step": "Uploading artifacts"})
		_log_event(sb, mq_job, "info", "syncing artifacts to Storage")
		_sync_artifacts(cfg, job, subject_path, log_path)
		_update_job(sb, mq_job, {"status": "completed", "progress": 100, "current_step": "Uploaded", "finished_at": _utc_now_iso(), "end_time": _utc_now_iso()})
		if subject_db_id:
			_set_subject_stage(sb, subject_db_id, "montage_qc", "done")
		try:
			_insert_output(
				sb,
				mq_job,
				kind="artifacts_index",
				storage_path=f"projects/{job.get('project_id')}/subjects/{job.get('subject_id')}/artifacts/index.json",
				meta={"runner_id": cfg.worker_id},
			)
		except Exception:
			pass

		# Mark any remaining queued stage jobs for this run as completed.
		for st, jid in list(jobs_by_stage.items()):
			if not jid:
				continue
			try:
				row = sb.select_one("jobs", "id,status", {"id": jid})
				if row and str(row.get("status") or "") in {"queued", "running"}:
					_update_job(sb, jid, {"status": "completed", "progress": 100, "current_step": "Done", "finished_at": _utc_now_iso(), "end_time": _utc_now_iso()})
			except Exception:
				pass
			if subject_db_id and st in _STAGES:
				_set_subject_stage(sb, subject_db_id, st, "done")

		final_status = "completed"
		_log_event(sb, job_id, "info", "completed")
	except Exception as exc:
		err = str(exc)
		final_status = "failed"
		_update_job(sb, job_id, {"status": "failed", "error": err, "finished_at": _utc_now_iso(), "end_time": _utc_now_iso(), "current_step": "failed"})
		_log_event(sb, job_id, "error", err)
		if subject_db_id:
			_fail_running_stages(sb, subject_db_id)

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
			job = _claim_next_ready_job(sb, cfg.worker_id)
			if not job:
				time.sleep(cfg.poll_interval_s)
				continue
			# Claim by flipping queued->running. Best-effort; safe for single-worker setups.
			try:
				_update_job(
					sb,
					str(job.get("id")),
					{
						"status": "running",
						"runner_id": cfg.worker_id,
						"claimed_at": _utc_now_iso(),
						"start_time": job.get("start_time") or _utc_now_iso(),
						"error": None,
					},
				)
			except Exception:
				pass
			_process_job(cfg, sb, job)
		except KeyboardInterrupt:
			print("[supabase-worker] stopping")
			break
		except Exception as exc:
			print(f"[supabase-worker] loop error: {exc}")
			time.sleep(cfg.poll_interval_s)


if __name__ == "__main__":
	main()

