from __future__ import annotations

import asyncio
import re
import fnmatch
import json
import os
import math
import time
import threading
import subprocess
import signal
import sys
import hashlib
import importlib
import colorsys
import pickle
from functools import lru_cache
from dataclasses import asdict, dataclass, field
from datetime import datetime
import shutil
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional
from typing import Tuple

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field


np = None  # type: ignore

# SciPy is optional and heavy; keep it lazy so the backend can start quickly
# (and so packaged builds don't fail at import time).
least_squares = None  # type: ignore
nib = None  # type: ignore


def _ensure_stdio_fds_open() -> None:
    """Ensure file descriptors 0/1/2 exist.

    When the backend is launched from a GUI host on macOS, stdin/stdout/stderr
    can be closed. Some Python subprocesses then crash at startup with:
    "init_sys_streams: can't initialize sys standard streams".

    This is safe to call multiple times.
    """

    if os.name != "posix":
        return

    try:
        devnull_fd = os.open(os.devnull, os.O_RDWR)
    except Exception:
        return

    try:
        for fd in (0, 1, 2):
            try:
                os.fstat(fd)
            except OSError:
                try:
                    os.dup2(devnull_fd, fd)
                except Exception:
                    pass
    finally:
        try:
            os.close(devnull_fd)
        except Exception:
            pass


# Fix stdio for GUI/daemon launches early, before any subprocess calls.
_ensure_stdio_fds_open()


def _load_numpy():
    """Lazy-load numpy so cold starts don't pay the import cost."""

    global np
    if np is not None:
        return np
    try:  # pragma: no cover - optional dependency
        import numpy as _np  # type: ignore

        np = _np  # type: ignore
    except Exception:  # pragma: no cover
        np = None  # type: ignore
    return np


def _get_least_squares():
    """Lazy-load scipy.optimize.least_squares when modelling paths need it."""

    global least_squares
    if least_squares is not None:
        return least_squares
    try:  # pragma: no cover - optional dependency
        mod = importlib.import_module("scipy.optimize")
        least_squares = getattr(mod, "least_squares", None)
        return least_squares
    except Exception:
        return None


def _load_nibabel():
    """Lazy-load nibabel only when volume I/O is needed."""

    global nib
    if nib is not None:
        return nib
    try:  # pragma: no cover - optional dependency
        import nibabel as _nib  # type: ignore

        nib = _nib  # type: ignore
    except Exception:  # pragma: no cover
        nib = None  # type: ignore
    return nib


# Create the FastAPI app early so route decorators below don't fail during import.
# (Some endpoint groups are defined above other initialization helpers.)
app = FastAPI(title="p-brain-web local backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


JobStatus = Literal["queued", "running", "completed", "failed", "cancelled"]
StageId = Literal[
    "import",
    "t1_fit",
    "input_functions",
    "time_shift",
    "segmentation",
    "tissue_ctc",
    "modelling",
    "diffusion",
    "tractography",
    "connectome",
]
StageStatus = Literal["not_run", "running", "done", "failed", "waiting"]


# p-brain uses this special exit code to indicate a non-error stop where
# user interaction (ROI drawing/providing) is required to proceed.
PBRAIN_WAITING_FOR_ROI_EXIT_CODE = 42


def _now_iso() -> str:
    return datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


def _user_data_dir() -> Path:
    # Keep local state out of the .app bundle so updates don't wipe it.
    # Allow override for debugging/tests.
    override = os.environ.get("PBRAIN_WEB_DATA_DIR")
    if override:
        return Path(override).expanduser().resolve()

    app_name = os.environ.get("PBRAIN_WEB_APP_NAME") or "p-brain"

    if sys.platform == "darwin":
        base = Path.home() / "Library" / "Application Support"
    elif sys.platform.startswith("win"):
        base = Path(os.environ.get("APPDATA") or (Path.home() / "AppData" / "Roaming"))
    else:
        base = Path(os.environ.get("XDG_CONFIG_HOME") or (Path.home() / ".config"))

    return (base / app_name).expanduser().resolve()


def _legacy_db_path() -> Path:
    # Historical/dev default (beside this file).
    return Path(__file__).with_name("db.json")


def _db_path() -> Path:
    # Preferred persistent location.
    override = os.environ.get("PBRAIN_WEB_DB_PATH")
    if override:
        return Path(override).expanduser().resolve()

    data_dir = _user_data_dir()
    p = data_dir / "db.json"

    # One-time migration from legacy path if it exists.
    try:
        legacy = _legacy_db_path()
        if legacy.exists() and not p.exists():
            p.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(str(legacy), str(p))
    except Exception:
        # Never block startup on migration.
        pass

    return p


@dataclass
class Project:
    id: str
    name: str
    storagePath: str
    createdAt: str
    updatedAt: str
    copyDataIntoProject: bool
    config: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Subject:
    id: str
    projectId: str
    name: str
    sourcePath: str
    createdAt: str
    updatedAt: str
    hasT1: bool
    hasDCE: bool
    hasDiffusion: bool
    stageStatuses: Dict[str, StageStatus]


@dataclass
class Job:
    id: str
    projectId: str
    subjectId: str
    stageId: StageId
    status: JobStatus
    progress: int
    currentStep: str
    startTime: Optional[str] = None
    endTime: Optional[str] = None
    estimatedTimeRemaining: Optional[int] = None
    error: Optional[str] = None
    logPath: Optional[str] = None


class CreateProjectRequest(BaseModel):
    name: str
    storagePath: str
    copyDataIntoProject: bool = False


class UpdateProjectRequest(BaseModel):
    name: Optional[str] = None
    storagePath: Optional[str] = None
    copyDataIntoProject: Optional[bool] = None


class UpdateProjectConfigRequest(BaseModel):
    configUpdate: Dict[str, Any] = Field(default_factory=dict)


class FolderStructurePreviewRequest(BaseModel):
    folderStructure: Dict[str, Any] = Field(default_factory=dict)


def _split_fallback_patterns(raw: Any) -> List[str]:
    s = str(raw or "").strip()
    if not s:
        return []
    parts = [p.strip() for p in s.split(",")]
    return [p for p in parts if p]


def _extract_subject_id_from_pattern(pattern: str, folder_name: str) -> str:
    if "{subject_id}" not in pattern:
        return folder_name
    parts = pattern.split("{subject_id}")
    if len(parts) != 2:
        return folder_name
    prefix, suffix = parts
    if prefix and not folder_name.startswith(prefix):
        return folder_name
    if suffix and not folder_name.endswith(suffix):
        return folder_name
    start = len(prefix)
    end = len(folder_name) - len(suffix) if suffix else len(folder_name)
    if end < start:
        return folder_name
    return folder_name[start:end]


def _iter_nifti_relpaths(base_dir: Path, max_files: int = 5000) -> List[str]:
    out: List[str] = []
    try:
        for root, _dirs, files in os.walk(str(base_dir)):
            for fn in files:
                lf = fn.lower()
                if not (lf.endswith(".nii") or lf.endswith(".nii.gz")):
                    continue
                p = Path(root) / fn
                try:
                    rel = p.relative_to(base_dir)
                except Exception:
                    rel = p.name
                out.append(rel.as_posix())
                if len(out) >= max_files:
                    return sorted(out)
    except Exception:
        return []
    return sorted(out)


def _first_match(files: List[str], pattern_list: List[str]) -> Optional[str]:
    if not files or not pattern_list:
        return None
    for pat in pattern_list:
        cleaned = str(pat).strip().lstrip("/")
        if not cleaned:
            continue
        for f in files:
            try:
                if fnmatch.fnmatchcase(f, cleaned):
                    return f
            except Exception:
                continue
    return None


class ImportSubjectsRequest(BaseModel):
    subjects: List[Dict[str, str]]  # {name, sourcePath}


class RunFullPipelineRequest(BaseModel):
    subjectIds: List[str]
    # Optional subset of stages to run (comma order not important).
    # When provided, the backend expands required dependencies and runs a minimal chain.
    stageIds: Optional[List[StageId]] = None


class RunStageRequest(BaseModel):
    stageId: StageId
    runDependencies: bool = True
    # Optional per-run env overrides for the p-brain subprocess.
    # Intended for narrowly-scoped toggles (e.g. forcing ROI_METHOD=file after user ROI upload).
    envOverrides: Optional[Dict[str, str]] = None


class AnalysisPearsonRequest(BaseModel):
    x: List[float]
    y: List[float]


class AnalysisGroupCompareRequest(BaseModel):
    a: List[float]
    b: List[float]


class AnalysisOlsRequest(BaseModel):
    y: List[float]
    X: List[List[float]]
    columns: List[str]


STAGE_DEPENDENCIES: Dict[StageId, List[StageId]] = {
    "import": [],
    "t1_fit": ["import"],
    # AIF/VIF extraction should be runnable without requiring T1/M0 fitting.
    # (T1 fitting is needed later for tissue CTC/modelling, but not for selecting ROIs.)
    "input_functions": ["import"],
    "time_shift": ["input_functions"],
    "segmentation": ["time_shift"],
    "tissue_ctc": ["segmentation"],
    "modelling": ["tissue_ctc"],
    "diffusion": ["modelling"],
    "tractography": ["diffusion"],
    "connectome": ["tractography"],
}


class EnsureArtifactsResponse(BaseModel):
    started: bool
    jobs: List[Dict[str, Any]] = Field(default_factory=list)
    reason: str = ""


class ResolveDefaultVolumeResponse(BaseModel):
    path: str


class VolumeInfoRequest(BaseModel):
    path: str
    kind: Optional[str] = None


class VolumeSliceRequest(BaseModel):
    path: str
    z: int
    t: int = 0


class MapVolumeResponse(BaseModel):
    id: str
    name: str
    unit: str
    path: str
    group: str


class MontageImageResponse(BaseModel):
    id: str
    name: str
    path: str


class RoiMaskVolumeResponse(BaseModel):
    id: str
    name: str
    path: str
    roiType: str
    roiSubType: str


class AppSettings(BaseModel):
    firstName: str = ""
    onboardingCompleted: bool = False
    pbrainMainPy: str = ""  # absolute path to p-brain/main.py
    fastsurferDir: str = ""  # absolute path to FastSurfer repo root
    freesurferHome: str = ""  # absolute path to FREESURFER_HOME
    # Optional p-brain configuration overrides (forwarded as env vars).
    pbrainT1Fit: str = "ir"  # auto|ir|vfa|none
    pbrainVfaGlob: str = ""  # comma-separated glob(s), e.g. "*VFA*.nii*,*flip*.nii*"


class UpdateAppSettingsRequest(BaseModel):
    firstName: Optional[str] = None
    onboardingCompleted: Optional[bool] = None
    pbrainMainPy: Optional[str] = None
    fastsurferDir: Optional[str] = None
    freesurferHome: Optional[str] = None
    pbrainT1Fit: Optional[str] = None
    pbrainVfaGlob: Optional[str] = None


class SystemDepsResponse(BaseModel):
    pbrainMainPy: Dict[str, Any]
    freesurfer: Dict[str, Any]
    fastsurfer: Dict[str, Any]


class InstallFastSurferRequest(BaseModel):
    installDir: str


class ScanSystemDepsRequest(BaseModel):
    # If true, write discovered paths into settings.
    apply: bool = True


class InstallPBrainRequest(BaseModel):
    installDir: str


class InstallPBrainRequirementsRequest(BaseModel):
    # Optional. If omitted, uses the repo inferred from settings/pbrainMainPy.
    pbrainDir: Optional[str] = None


class SaveRoiVoxelsRequest(BaseModel):
    roiType: str
    roiSubType: str
    sliceIndex: int
    frameIndex: int = 0
    # Raw nibabel slice coordinates as [row, col] pairs.
    # These will be converted to p-brain's rotated in-plane ROI coordinate frame on disk.
    voxels: List[List[int]] = Field(default_factory=list)


class DB:
    def __init__(self) -> None:
        self.projects: List[Project] = []
        self.subjects: List[Subject] = []
        self.jobs: List[Job] = []
        self.settings: Dict[str, Any] = {}
        self._load()

        # Fast lookup + caching for high-frequency polling endpoints.
        self._job_by_id: Dict[str, Job] = {j.id: j for j in self.jobs}
        self._jobs_version: int = 0
        # (projectId, subjectId, status) -> (version, serialized jobs list)
        self._jobs_cache: Dict[tuple[str, str, str], tuple[int, List[Dict[str, Any]]]] = {}

        # job_id -> asyncio.Task
        self._job_tasks: Dict[str, asyncio.Task] = {}
        # job_id -> process
        self._job_processes: Dict[str, asyncio.subprocess.Process] = {}
        # subject_id -> job_ids (the stage jobs created for the current run)
        self._subject_job_ids: Dict[str, List[str]] = {}
        # subject_id -> current stage index
        self._subject_stage_index: Dict[str, int] = {}

    def _touch_jobs(self) -> None:
        self._jobs_version += 1
        self._jobs_cache.clear()

    def _reindex_jobs(self) -> None:
        self._job_by_id = {j.id: j for j in self.jobs}
        self._touch_jobs()

    def _load(self) -> None:
        path = _db_path()
        if not path.exists():
            return
        try:
            raw = json.loads(path.read_text())
            self.projects = [Project(**p) for p in raw.get("projects", [])]
            self.subjects = [Subject(**s) for s in raw.get("subjects", [])]
            self.jobs = [Job(**j) for j in raw.get("jobs", [])]
            self.settings = raw.get("settings", {}) or {}
        except Exception:
            # If corrupted, start fresh (still local-only).
            self.projects = []
            self.subjects = []
            self.jobs = []
            self.settings = {}

    def save(self) -> None:
        path = _db_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "projects": [asdict(p) for p in self.projects],
            "subjects": [asdict(s) for s in self.subjects],
            "jobs": [asdict(j) for j in self.jobs],
            "settings": self.settings or {},
        }
        path.write_text(json.dumps(payload, indent=2))


db = DB()

# Background warm/preflight state so we can amortize heavy imports after startup.
_warm_lock = asyncio.Lock()
_warm_started = False
_warm_finished = False
_warm_error = ""
_warm_steps: Dict[str, float] = {}


def _max_concurrent_runs() -> int:
    try:
        v = int(os.environ.get("PBRAIN_WEB_MAX_CONCURRENT_RUNS", "1").strip())
        return max(1, v)
    except Exception:
        return 1


# Global throttle for p-brain runs so a "run full" doesn't spawn 100+ concurrent jobs.
_RUN_SEMAPHORE = asyncio.Semaphore(_max_concurrent_runs())


STAGES: List[StageId] = [
    "import",
    "t1_fit",
    "input_functions",
    "time_shift",
    "segmentation",
    "tissue_ctc",
    "modelling",
    "diffusion",
    "tractography",
    "connectome",
]


def _default_stage_statuses() -> Dict[str, StageStatus]:
    return {stage: "not_run" for stage in STAGES}


def _find_project(project_id: str) -> Project:
    p = next((p for p in db.projects if p.id == project_id), None)
    if not p:
        raise HTTPException(status_code=404, detail="Project not found")
    return p


def _find_subject(subject_id: str) -> Subject:
    s = next((s for s in db.subjects if s.id == subject_id), None)
    if not s:
        raise HTTPException(status_code=404, detail="Subject not found")
    return s


def _find_job(job_id: str) -> Job:
    j = db._job_by_id.get(job_id)
    if j is None:
        j = next((j for j in db.jobs if j.id == job_id), None)
        if j is not None:
            db._job_by_id[j.id] = j
    if not j:
        raise HTTPException(status_code=404, detail="Job not found")
    return j


def _deep_merge(base: Dict[str, Any], patch: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = dict(base or {})
    for k, v in (patch or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def _python_env_for_pbrain() -> Dict[str, str]:
    env = {**os.environ}
    env.setdefault("PBRAIN_TURBO", "1")
    # Headless execution: prevent matplotlib from selecting GUI backends (e.g. TkAgg).
    # This must be set in the subprocess env *before* matplotlib is imported.
    env["MPLBACKEND"] = "Agg"
    # Defaults for p-brain execution when project config doesn't override.
    # The platform focuses on the TurboFLASH pipeline and IR-based T1 fitting.
    env.setdefault("P_BRAIN_CTC_MODEL", "advanced")
    # Enforce the platform-standard TurboFLASH signal->concentration conversion.
    # This is MATLAB `menu_5` case12 method1 (p-brain: turboflash_advanced).
    env["P_BRAIN_TURBOFLASH_CTC_METHOD"] = "turboflash_advanced"
    # Do not let a user's shell env (or a prior run) silently change the fit mode.
    # Explicit overrides should come from app settings or project config only.
    env.pop("P_BRAIN_T1_FIT", None)
    # Default to deterministic auto-selection (prefers IR when a complete TI series exists,
    # otherwise VFA when available).
    env["P_BRAIN_T1_FIT"] = "auto"

    # Keep defaults aligned with the validated p-brain defaults JSON.
    # (This ensures a fresh project with no saved overrides still matches expected behavior.)
    env.pop("P_BRAIN_TURBO_NPH", None)
    env.pop("P_BRAIN_TURBOFLASH_NPH", None)
    env.setdefault("P_BRAIN_NUMBER_OF_PEAKS", "2")
    env.setdefault("P_BRAIN_CTC_PEAK_RESCALE_THRESHOLD", "4.0")
    env.setdefault("P_BRAIN_VASCULAR_ROI_CURVE_METHOD", "max")
    env.setdefault("P_BRAIN_VASCULAR_ROI_ADAPTIVE_MAX", "1")
    env.setdefault("P_BRAIN_MODELLING_INPUT_FUNCTION", "tscc")
    env.setdefault("P_BRAIN_AIF_USE_SSS", "1")

    # Do not let a user's shell env (or a prior run) silently change AI AIF/VIF
    # behavior. These are controlled via project config only.
    for k in (
        "P_BRAIN_AIF_ARTERY",
        "PBRAIN_AUTO_ARTERY",
        "P_BRAIN_ROI_METHOD",
        "ROI_METHOD",
        "P_BRAIN_AI_ROT90_K",
        "P_BRAIN_AI_SLICE_CONF_THRESHOLDS",
        "P_BRAIN_AI_SLICE_CONF_START",
        "P_BRAIN_AI_SLICE_CONF_MIN",
        "P_BRAIN_AI_SLICE_CONF_STEP",
        "P_BRAIN_AI_ICA_FRAME_STRIDE",
        "P_BRAIN_AI_ICA_FRAME_MODE",
        "P_BRAIN_AI_ICA_FRAME_OVERRIDE",
        "P_BRAIN_AI_ICA_FRAME_SEARCH_MAX",
        "P_BRAIN_AI_ICA_FRAME_SEARCH_STRIDE",
        "P_BRAIN_AI_INPUT_MISSING_FALLBACK",
        "P_BRAIN_AI_ROI_POSTPROCESS",
    ):
        env.pop(k, None)

    # Default to the same behavior as the plain CLI:
    # - If an artery side isn't explicitly configured, extract BOTH ICAs.
    # - Avoid forcing runner-specific heuristics/exports here; let p-brain defaults
    #   (or project config) control ICA frame choice, postprocessing, TSCC selection,
    #   and which intermediate artifacts are written.
    env.setdefault("P_BRAIN_AIF_ARTERY", "BOTH")
    # Keep noninteractive prompts deterministic when p-brain still asks for a choice.
    env.setdefault("PBRAIN_AUTO_ARTERY", "rica")
    # Always run p-brain in non-interactive mode when orchestrated by the app.
    # Some stages prompt via input(), which otherwise EOFs in headless subprocesses.
    env.setdefault("PBRAIN_NONINTERACTIVE", "1")
    # Ensure we can stream logs promptly when stdout is piped.
    env.setdefault("PYTHONUNBUFFERED", "1")
    env.setdefault("PYTHONIOENCODING", "utf-8")
    # Avoid mixing in user-site packages (common source of numpy/nibabel mismatch).
    env.setdefault("PYTHONNOUSERSITE", "1")
    # Improve import reliability when calling p-brain by path.
    main_py = env.get("PBRAIN_MAIN_PY") or _resolve_pbrain_main_py()
    if main_py:
        pbrain_root = str(Path(main_py).expanduser().resolve().parent)
        env["PYTHONPATH"] = pbrain_root + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")

    # Forward optional p-brain tuning knobs from app settings.
    try:
        s = _get_settings()
        t1_fit = str(s.get("pbrainT1Fit") or "").strip().lower()
        if t1_fit in {"auto", "ir", "vfa", "none"}:
            env["P_BRAIN_T1_FIT"] = t1_fit
        vfa_glob = str(s.get("pbrainVfaGlob") or "").strip()
        if vfa_glob:
            env.setdefault("P_BRAIN_VFA_GLOB", vfa_glob)
    except Exception:
        pass
    return env


_AI_MODEL_ENV_TO_FILENAME: Dict[str, str] = {
    "SLICE_CLASSIFIER_RICA_MODEL": "slice_classifier_model_rica.keras",
    "RICA_ROI_MODEL": "rica_roi_model.keras",
    "SLICE_CLASSIFIER_SS_MODEL": "ss_slice_classifier.keras",
    "SS_ROI_MODEL": "ss_roi_model.keras",
}


def _candidate_ai_model_dirs(pbrain_root: Optional[Path]) -> List[Path]:
    """Return ordered directories that may contain AI model binaries."""

    out: List[Path] = []

    def add(p: Optional[Path]) -> None:
        if not p:
            return
        try:
            pp = p.expanduser().resolve()
        except Exception:
            return
        if pp not in out:
            out.append(pp)

    # User override (documented in backend/AI/README.md)
    override = (os.environ.get("PBRAIN_AI_DIR") or "").strip()
    if override:
        try:
            add(Path(override))
        except Exception:
            pass

    # Preferred default: models inside the configured p-brain repo.
    if pbrain_root:
        add(pbrain_root / "AI")

    # Packaged app convenience: also look next to the frozen backend executable.
    # (Users often drop the Zenodo model files there.)
    try:
        exe_dir = Path(sys.executable).expanduser().resolve().parent
        add(exe_dir / "AI")
        # Also consider a sibling of the executable dir.
        add(exe_dir.parent / "AI")
    except Exception:
        pass

    # App-managed persistent location.
    try:
        add(_user_data_dir() / "AI")
    except Exception:
        pass

    # Dev/local backend folder (useful when running backend from source).
    try:
        add(Path(__file__).with_name("AI"))
    except Exception:
        pass

    return out


def _apply_ai_model_overrides(
    env: Dict[str, str],
    *,
    pbrain_root: Optional[Path],
    write_line_sync=None,
) -> Dict[str, str]:
    """Set SLICE_CLASSIFIER_*/ROI_* env vars to absolute paths when available."""

    dirs = _candidate_ai_model_dirs(pbrain_root)
    resolved: Dict[str, str] = {}

    for env_key, fname in _AI_MODEL_ENV_TO_FILENAME.items():
        # Respect explicit per-model overrides.
        if str(env.get(env_key) or "").strip():
            continue
        found: Optional[Path] = None
        for d in dirs:
            try:
                p = d / fname
                if p.exists() and p.is_file():
                    found = p
                    break
            except Exception:
                continue
        if found is not None:
            resolved[env_key] = str(found)

    if resolved:
        env.update(resolved)
        try:
            if write_line_sync is not None:
                used = ", ".join(f"{k}={v}" for k, v in sorted(resolved.items()))
                write_line_sync(f"[{_now_iso()}] AI model overrides: {used}\n")
        except Exception:
            pass
    else:
        try:
            if write_line_sync is not None:
                write_line_sync(f"[{_now_iso()}] AI model overrides: none\n")
        except Exception:
            pass

    return env


def _preflight_ai_models(
    env: Dict[str, str],
    *,
    pbrain_root: Optional[Path],
    write_line_sync=None,
) -> None:
    """Fail loudly when AI ROI is selected but required model files are missing."""

    roi_method = (env.get("P_BRAIN_ROI_METHOD") or env.get("ROI_METHOD") or "ai").strip().lower()
    if roi_method in {"geometry", "deterministic"}:
        return
    if roi_method == "file":
        return

    missing: List[str] = []
    checked: Dict[str, str] = {}
    for env_key, fname in _AI_MODEL_ENV_TO_FILENAME.items():
        raw = (env.get(env_key) or "").strip()
        candidate: Optional[Path] = None
        if raw:
            candidate = Path(raw).expanduser()
        elif pbrain_root:
            candidate = (pbrain_root / "AI" / fname)
        else:
            candidate = None
        if candidate is None:
            missing.append(fname)
            continue
        try:
            cand = candidate.resolve()
        except Exception:
            cand = candidate
        checked[env_key] = str(cand)
        try:
            if not cand.exists() or not cand.is_file():
                missing.append(fname)
        except Exception:
            missing.append(fname)

    try:
        if write_line_sync is not None:
            det = ", ".join(f"{k}={v}" for k, v in sorted(checked.items()))
            write_line_sync(f"[{_now_iso()}] AI model check: {det}\n")
    except Exception:
        pass

    if missing:
        where: List[str] = []
        try:
            if pbrain_root:
                where.append(str(pbrain_root / "AI"))
        except Exception:
            pass
        try:
            where.append(str(_user_data_dir() / "AI"))
        except Exception:
            pass
        where.append("(or set PBRAIN_AI_DIR)")

        raise RuntimeError(
            "Missing AI model file(s): "
            + ", ".join(sorted(set(missing)))
            + ". Download the models from Zenodo and place them in one of: "
            + "; ".join(where)
        )


def _apply_project_pbrain_env_overrides(env: Dict[str, str], project: Project) -> Dict[str, str]:
    cfg = project.config if isinstance(getattr(project, "config", None), dict) else {}
    ctc = cfg.get("ctc") if isinstance(cfg.get("ctc"), dict) else {}

    pbrain = cfg.get("pbrain") if isinstance(cfg.get("pbrain"), dict) else {}
    model_cfg = cfg.get("model") if isinstance(cfg.get("model"), dict) else {}
    input_fn = cfg.get("inputFunction") if isinstance(cfg.get("inputFunction"), dict) else {}
    tissue_cfg = cfg.get("tissue") if isinstance(cfg.get("tissue"), dict) else {}

    # p-brain execution / metadata strictness
    try:
        strict = pbrain.get("strictMetadata")
        if strict is not None:
            env["P_BRAIN_STRICT_METADATA"] = "1" if bool(strict) else "0"
    except Exception:
        pass

    try:
        mp = pbrain.get("multiprocessing")
        if mp is not None:
            env["P_BRAIN_MULTIPROCESSING"] = "1" if bool(mp) else "0"
    except Exception:
        pass

    try:
        cores_raw = pbrain.get("cores")
        cores = str(cores_raw).strip() if cores_raw is not None else ""
        if cores:
            env["P_BRAIN_CORES"] = cores
    except Exception:
        pass

    try:
        fa = pbrain.get("flipAngle")
        if fa is not None:
            # Accept either numeric or 'auto'.
            if isinstance(fa, (int, float)):
                if math.isfinite(float(fa)) and float(fa) > 0:
                    env["P_BRAIN_FLIP_ANGLE"] = str(float(fa))
            else:
                fa_s = str(fa).strip().lower()
                if fa_s:
                    env["P_BRAIN_FLIP_ANGLE"] = fa_s
    except Exception:
        pass

    # T1/M0 fit method selection (IR vs VFA vs auto/none)
    try:
        if isinstance(pbrain, dict) and "t1Fit" in pbrain:
            raw = pbrain.get("t1Fit")
            if raw is None or str(raw).strip() == "":
                env.pop("P_BRAIN_T1_FIT", None)
            else:
                t1_fit = str(raw).strip().lower()
                if t1_fit in {"auto", "ir", "vfa", "none"}:
                    env["P_BRAIN_T1_FIT"] = t1_fit
    except Exception:
        pass

    # Optional series discovery overrides.
    try:
        if isinstance(pbrain, dict):
            vfa_glob = str(pbrain.get("vfaGlob") or "").strip()
            if vfa_glob:
                env["P_BRAIN_VFA_GLOB"] = vfa_glob

            ir_prefixes = str(pbrain.get("irPrefixes") or "").strip()
            if ir_prefixes:
                env["P_BRAIN_IR_PREFIXES"] = ir_prefixes

            ir_ti = str(pbrain.get("irTi") or "").strip()
            if ir_ti:
                env["P_BRAIN_IR_TI"] = ir_ti
    except Exception:
        pass

    model = str(ctc.get("model") or "").strip().lower()
    if model in {"saturation", "turboflash", "advanced"}:
        env["P_BRAIN_CTC_MODEL"] = model
        if model in {"turboflash", "advanced"}:
            # Enforce platform standard: MATLAB menu_5 case12 method1.
            env["P_BRAIN_TURBOFLASH_CTC_METHOD"] = "turboflash_advanced"

    # Input-function AI behavior (AIF/VIF detection thresholds and fallback policy)
    try:
        ai_cfg = input_fn.get("ai") if isinstance(input_fn.get("ai"), dict) else {}
        start = ai_cfg.get("sliceConfStart")
        min_v = ai_cfg.get("sliceConfMin")
        step = ai_cfg.get("sliceConfStep")

        def _clamp01(v: Any) -> Optional[float]:
            try:
                f = float(v)
                if not math.isfinite(f):
                    return None
                return max(0.01, min(1.0, f))
            except Exception:
                return None

        s0 = _clamp01(start)
        smin = _clamp01(min_v)
        sstep = _clamp01(step)
        if s0 is not None:
            env["P_BRAIN_AI_SLICE_CONF_START"] = str(s0)
        if smin is not None:
            env["P_BRAIN_AI_SLICE_CONF_MIN"] = str(smin)
        if sstep is not None:
            # Step should be reasonable; clamp more tightly.
            env["P_BRAIN_AI_SLICE_CONF_STEP"] = str(max(0.01, min(0.5, sstep)))

        fallback = str(ai_cfg.get("missingFallback") or "").strip().lower()
        if fallback in {"deterministic", "roi"}:
            env["P_BRAIN_AI_INPUT_MISSING_FALLBACK"] = fallback
    except Exception:
        pass

    # Input-function selection
    # - aif: pure arterial curve
    # - vif: pure venous curve (SSS)
    # - adjusted_vif: TSCC (SSS-derived time-shifted/rescaled curve)
    try:
        src = str(input_fn.get("source") or "").strip().lower()
        if src == "aif":
            env["P_BRAIN_MODELLING_INPUT_FUNCTION"] = "aif"
            env["P_BRAIN_AIF_USE_SSS"] = "0"
        elif src == "vif":
            env["P_BRAIN_MODELLING_INPUT_FUNCTION"] = "vif"
            env["P_BRAIN_AIF_USE_SSS"] = "0"
        elif src in {"adjusted_vif", "tscc", "sss"}:
            # Legacy names treated as TSCC.
            env["P_BRAIN_MODELLING_INPUT_FUNCTION"] = "tscc"
            env["P_BRAIN_AIF_USE_SSS"] = "1"
    except Exception:
        pass

    # Input-function artery preference (RICA vs LICA).
    # p-brain uses this to select which carotid ROI to extract and (for LICA)
    # whether to mirror slices for inference.
    try:
        artery = str(
            input_fn.get("artery")
            or input_fn.get("aifArtery")
            or input_fn.get("preferredArtery")
            or ""
        ).strip().lower()
        if artery in {"rica", "lica"}:
            env["P_BRAIN_AIF_ARTERY"] = artery.upper()
            # Keep noninteractive prompts aligned with the selected artery.
            env["PBRAIN_AUTO_ARTERY"] = artery
        elif artery in {"both", "bilateral", "rica+lica", "lica+rica"}:
            env["P_BRAIN_AIF_ARTERY"] = "BOTH"
            env.setdefault("PBRAIN_AUTO_ARTERY", "rica")
    except Exception:
        pass

    # Vascular ROI curve extraction settings
    try:
        method = str(input_fn.get("vascularRoiCurveMethod") or "").strip().lower()
        if method in {"max", "mean", "median"}:
            env["P_BRAIN_VASCULAR_ROI_CURVE_METHOD"] = method

        if isinstance(input_fn, dict) and "vascularRoiAdaptiveMax" in input_fn:
            env["P_BRAIN_VASCULAR_ROI_ADAPTIVE_MAX"] = "1" if bool(input_fn.get("vascularRoiAdaptiveMax")) else "0"
    except Exception:
        pass

    # Tissue/atlas ROI aggregation settings
    try:
        agg = str(tissue_cfg.get("roiAggregation") or "").strip().lower()
        if agg in {"mean", "median"}:
            env["P_BRAIN_TISSUE_ROI_AGGREGATION"] = agg
    except Exception:
        pass

    # Kinetic model selector (validated set only)
    try:
        pk_model = model_cfg.get("pkModel")
        if pk_model is None:
            # Default to running both validated models.
            env["P_BRAIN_MODEL"] = "both"
        else:
            pk = str(pk_model).strip().lower()
            # Keep accepting older config values, but always emit canonical validated keys.
            if pk in {
                "both",
                "all",
                "patlak_tikhonov",
                "patlak_tikhonov_fast",
                "patlak-then-tikhonov",
                "patlak-then-tikhonov-fast",
                "patlak_then_tikhonov",
                "patlak_then_tikhonov_fast",
            }:
                env["P_BRAIN_MODEL"] = "both"
            elif pk == "patlak":
                env["P_BRAIN_MODEL"] = "patlak"
            elif pk in {
                "tikhonov",
                "tikhonov_only",
                "tikhonov-only",
                "tikhonov_fast",
                "tikhonov-only-fast",
                "tikhonov_only_fast",
                "tik_fast",
                "tik-fast",
                "tikfast",
                "tikh-fast",
                "two_compartment",
                "2comp",
                "two-comp",
                "two-compartment",
            }:
                env["P_BRAIN_MODEL"] = "tikhonov"
            else:
                env["P_BRAIN_MODEL"] = "both"
    except Exception:
        pass

    # Deconvolution / residue settings (Tikhonov-based metrics)
    try:
        penalty = model_cfg.get("tikhonovPenalty")
        if penalty is not None:
            penalty_s = str(penalty).strip().lower()
            if penalty_s in {"identity", "derivative"}:
                env["P_BRAIN_TIKHONOV_PENALTY"] = penalty_s
    except Exception:
        pass

    try:
        nonneg = model_cfg.get("residueEnforceNonneg")
        if nonneg is not None:
            env["P_BRAIN_RESIDUE_ENFORCE_NONNEG"] = "1" if bool(nonneg) else "0"
    except Exception:
        pass

    try:
        mono = model_cfg.get("residueEnforceMonotone")
        if mono is not None:
            env["P_BRAIN_RESIDUE_ENFORCE_MONOTONE"] = "1" if bool(mono) else "0"
    except Exception:
        pass

    # TurboFLASH nph override.
    # - If turboNph is explicitly null/"auto" in project config, clear any override env.
    # - If turboNph is a positive integer, set override env.
    try:
        if isinstance(ctc, dict) and "turboNph" in ctc:
            nph_raw = ctc.get("turboNph")
            if nph_raw is None or str(nph_raw).strip().lower() in {"", "auto"}:
                env.pop("P_BRAIN_TURBO_NPH", None)
                env.pop("P_BRAIN_TURBOFLASH_NPH", None)
            else:
                nph = int(float(nph_raw))
                if nph >= 1:
                    env["P_BRAIN_TURBO_NPH"] = str(nph)
    except Exception:
        pass

    peaks_raw = ctc.get("numberOfPeaks")
    try:
        peaks = int(peaks_raw)
        if peaks >= 1:
            env["P_BRAIN_NUMBER_OF_PEAKS"] = str(peaks)
    except Exception:
        pass

    peak_thresh_raw = ctc.get("peakRescaleThreshold")
    try:
        peak_thresh = float(peak_thresh_raw)
        if math.isfinite(peak_thresh) and peak_thresh >= 0:
            env["P_BRAIN_CTC_PEAK_RESCALE_THRESHOLD"] = str(peak_thresh)
    except Exception:
        pass

    # TSCC Max selection forked-peak skipping
    try:
        if isinstance(ctc, dict) and "skipForkedMaxCtcPeaks" in ctc:
            env["P_BRAIN_TSCC_SKIP_FORKED_PEAKS"] = "1" if bool(ctc.get("skipForkedMaxCtcPeaks")) else "0"
    except Exception:
        pass

    # CTC exports (maps + full 4D)
    try:
        if isinstance(ctc, dict) and "writeCtcMaps" in ctc:
            env["P_BRAIN_CTC_MAPS"] = "1" if bool(ctc.get("writeCtcMaps")) else "0"
    except Exception:
        pass

    try:
        if isinstance(ctc, dict) and "writeCtc4d" in ctc:
            env["P_BRAIN_CTC_4D"] = "1" if bool(ctc.get("writeCtc4d")) else "0"
    except Exception:
        pass

    try:
        if isinstance(ctc, dict) and "ctcMapSlice" in ctc:
            raw_slice = ctc.get("ctcMapSlice")
            slice_i = int(float(raw_slice))
            if slice_i >= 1:
                env["P_BRAIN_CTC_MAP_SLICE"] = str(slice_i)
    except Exception:
        pass

    return env


def _apply_pbrain_env_overrides(env: Dict[str, str], overrides: Optional[Dict[str, str]]) -> Dict[str, str]:
    if not overrides:
        return env
    allowed = {"P_BRAIN_ROI_METHOD", "ROI_METHOD"}
    bad = [k for k in overrides.keys() if k not in allowed]
    if bad:
        raise HTTPException(status_code=400, detail=f"Unsupported env override(s): {', '.join(sorted(bad))}")

    for k, v in overrides.items():
        vv = "" if v is None else str(v)
        if vv.strip() == "":
            env.pop(k, None)
        else:
            env[k] = vv
    return env


def _require_numpy() -> None:
    if _load_numpy() is None:
        raise HTTPException(status_code=500, detail="Backend missing dependency: numpy")


def _require_nibabel() -> None:
    if _load_nibabel() is None:
        raise HTTPException(status_code=500, detail="Backend missing dependency: nibabel")


def _require_scipy() -> None:
    if _get_least_squares() is None:
        raise HTTPException(status_code=500, detail="Backend missing dependency: scipy")


def _analysis_dir_for_subject(subject: Subject) -> Path:
    return Path(subject.sourcePath).expanduser().resolve() / "Analysis"


@lru_cache(maxsize=1)
def _freesurfer_lut() -> Dict[int, str]:
    """Parse FreeSurferColorLUT.txt into {id: name}.

    Uses FREESURFER_HOME or configured settings.
    """

    s = _get_settings()
    fs_home = (os.environ.get("FREESURFER_HOME") or s.get("freesurferHome") or "").strip()

    lut_path: Optional[Path] = None
    if fs_home:
        candidate = Path(fs_home).expanduser().resolve() / "FreeSurferColorLUT.txt"
        if candidate.exists():
            lut_path = candidate

    # Fallbacks for common macOS installs (useful in packaged apps where shell env vars may not propagate).
    if lut_path is None:
        candidates: List[Path] = [
            Path("/Applications/freesurfer/7.4.1"),
            Path("/Applications/freesurfer"),
            Path("/usr/local/freesurfer"),
        ]
        for base in candidates:
            try:
                base = base.expanduser().resolve()
            except Exception:
                continue
            direct = base / "FreeSurferColorLUT.txt"
            if direct.exists():
                lut_path = direct
                break
            if base.exists() and base.is_dir():
                # Look one level down for versioned installs.
                try:
                    for child in sorted(base.iterdir(), reverse=True):
                        candidate = child / "FreeSurferColorLUT.txt"
                        if candidate.exists():
                            lut_path = candidate
                            break
                except Exception:
                    pass
            if lut_path is not None:
                break

    if lut_path is None or not lut_path.exists():
        return {}

    out: Dict[int, str] = {}
    try:
        for raw in lut_path.read_text(errors="ignore").splitlines():
            line = (raw or "").strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 2:
                continue
            try:
                idx = int(parts[0])
            except Exception:
                continue
            name = parts[1]
            if name:
                out[idx] = name
    except Exception:
        return {}

    return out


def _fs_region_name(region: Any) -> str:
    """Map a numeric atlas label to a FreeSurfer region name when possible."""

    if region is None:
        return ""
    s = str(region).strip()
    if not s:
        return ""
    # Only remap purely-numeric keys.
    if not re.fullmatch(r"\d+", s):
        return s
    try:
        idx = int(s)
    except Exception:
        return s

    lut = _freesurfer_lut()
    name = lut.get(idx)
    return name or s


def _nifti_dir_for_subject(project: Project, subject: Subject) -> Path:
    cfg = project.config or {}
    fs = cfg.get("folderStructure") if isinstance(cfg.get("folderStructure"), dict) else {}
    use_nested = bool(fs.get("useNestedStructure", True))
    nifti_subfolder = str(fs.get("niftiSubfolder", "nifti"))
    root = Path(subject.sourcePath).expanduser().resolve()
    if use_nested:
        # Try configured folder name first.
        candidate = root / nifti_subfolder
        if candidate.exists():
            return candidate

        # Heuristic: p-brain datasets often use capitalised folder names.
        for alt in ("NIfTI", "nifti", "NIFTI"):
            candidate = root / alt
            if candidate.exists():
                return candidate

    return root


def _safe_resolve_path(project: Project, subject: Optional[Subject], path_str: str) -> Path:
    # Allow reading only within project storagePath OR within subject sourcePath.
    p = Path(path_str).expanduser().resolve()
    allowed_roots = [Path(project.storagePath).expanduser().resolve()]
    if subject is not None:
        allowed_roots.append(Path(subject.sourcePath).expanduser().resolve())
    for root in allowed_roots:
        try:
            p.relative_to(root)
            return p
        except Exception:
            continue
    raise HTTPException(status_code=403, detail="Path is outside allowed roots")


def _glob_first(base: Path, pattern: str) -> Optional[Path]:
    # fnmatch against all files recursively.
    # Pattern can target a filename ("*.nii.gz") or a relative path ("anat/*T1w*.nii.gz").
    pattern = (pattern or "").strip()
    if not pattern:
        return None
    for f in base.rglob("*"):
        if not f.is_file():
            continue
        try:
            rel = f.relative_to(base).as_posix()
        except Exception:
            rel = f.name
        if fnmatch.fnmatch(rel, pattern) or fnmatch.fnmatch(f.name, pattern):
            return f
    return None


def _scan_nifti_presence(
    base: Path,
    *,
    t1_patterns: List[str],
    dce_patterns: List[str],
    diff_patterns: List[str],
) -> Dict[str, bool]:
    """Single-pass scan used during subject import.

    Avoids repeated expensive `rglob` traversals on large datasets.
    """

    base = Path(base)
    has_nifti = False
    t1_by_pattern = False
    dce_by_pattern = False
    diff_by_pattern = False
    t1_by_name = False
    dce_by_name = False
    diff_by_name = False

    # Normalize patterns once.
    t1p = [p.strip() for p in (t1_patterns or []) if str(p).strip()]
    dcep = [p.strip() for p in (dce_patterns or []) if str(p).strip()]
    diffp = [p.strip() for p in (diff_patterns or []) if str(p).strip()]

    # We only need to know if any matching files exist; stop early when everything is found.
    want_t1 = True
    want_dce = True
    want_diff = True

    def done() -> bool:
        if not has_nifti:
            return False
        if want_t1 and not (t1_by_pattern or t1_by_name):
            return False
        if want_dce and not (dce_by_pattern or dce_by_name):
            return False
        if want_diff and not (diff_by_pattern or diff_by_name):
            return False
        return True

    if not base.exists():
        return {
            "has_nifti": False,
            "t1_by_pattern": False,
            "dce_by_pattern": False,
            "diff_by_pattern": False,
            "t1_by_name": False,
            "dce_by_name": False,
            "diff_by_name": False,
        }

    for root, dirs, files in os.walk(base, topdown=True, followlinks=False):
        # Prune hidden dirs and Analysis outputs (common large trees).
        dirs[:] = [
            d
            for d in dirs
            if not d.startswith(".") and d.lower() not in {"analysis"}
        ]

        rel_root = os.path.relpath(root, str(base))
        rel_root = "" if rel_root == "." else rel_root

        for name in files:
            # Cheap extension filter first.
            lower = name.lower()
            if not (lower.endswith(".nii") or lower.endswith(".nii.gz")):
                continue

            has_nifti = True

            rel = f"{rel_root}/{name}" if rel_root else name
            rel = rel.replace(os.sep, "/")

            if not t1_by_pattern and t1p and any(fnmatch.fnmatch(rel, p) or fnmatch.fnmatch(name, p) for p in t1p):
                t1_by_pattern = True
            if not dce_by_pattern and dcep and any(fnmatch.fnmatch(rel, p) or fnmatch.fnmatch(name, p) for p in dcep):
                dce_by_pattern = True
            if not diff_by_pattern and diffp and any(fnmatch.fnmatch(rel, p) or fnmatch.fnmatch(name, p) for p in diffp):
                diff_by_pattern = True

            if not t1_by_name and "t1" in lower:
                t1_by_name = True
            if not dce_by_name and "dce" in lower:
                dce_by_name = True
            if not diff_by_name and ("dwi" in lower or "dti" in lower):
                diff_by_name = True

            if done():
                return {
                    "has_nifti": has_nifti,
                    "t1_by_pattern": t1_by_pattern,
                    "dce_by_pattern": dce_by_pattern,
                    "diff_by_pattern": diff_by_pattern,
                    "t1_by_name": t1_by_name,
                    "dce_by_name": dce_by_name,
                    "diff_by_name": diff_by_name,
                }

    return {
        "has_nifti": has_nifti,
        "t1_by_pattern": t1_by_pattern,
        "dce_by_pattern": dce_by_pattern,
        "diff_by_pattern": diff_by_pattern,
        "t1_by_name": t1_by_name,
        "dce_by_name": dce_by_name,
        "diff_by_name": diff_by_name,
    }


def _split_patterns(raw: Any) -> List[str]:
    if raw is None:
        return []
    if isinstance(raw, list):
        parts = [str(p).strip() for p in raw]
        return [p for p in parts if p]
    s = str(raw)
    # Allow comma-separated patterns for fallbacks.
    parts = [p.strip() for p in s.split(",")]
    return [p for p in parts if p]


def _nifti_dir_for_path(project: Project, root: Path) -> Path:
    cfg = project.config or {}
    fs = cfg.get("folderStructure") if isinstance(cfg.get("folderStructure"), dict) else {}
    use_nested = bool(fs.get("useNestedStructure", True))
    nifti_subfolder = str(fs.get("niftiSubfolder", "nifti"))

    if use_nested:
        candidate = root / nifti_subfolder if nifti_subfolder else root
        if candidate.exists():
            return candidate
        for alt in ("NIfTI", "nifti", "NIFTI"):
            candidate = root / alt
            if candidate.exists():
                return candidate
    return root


def _resolve_default_volume_path(project: Project, subject: Subject, kind: str) -> Path:
    cfg = project.config or {}
    fs = cfg.get("folderStructure") if isinstance(cfg.get("folderStructure"), dict) else {}
    nifti_dir = _nifti_dir_for_subject(project, subject)

    if kind == "dce":
        patterns = _split_patterns(fs.get("dcePattern", "*DCE*.nii.gz"))
    elif kind == "t1":
        patterns = _split_patterns(fs.get("t1Pattern", "*T1*.nii.gz"))
    elif kind == "t2":
        patterns = _split_patterns(fs.get("t2Pattern", "*T2*.nii.gz"))
    elif kind == "flair":
        patterns = _split_patterns(fs.get("flairPattern", "*FLAIR*.nii.gz"))
    elif kind == "diffusion":
        patterns = _split_patterns(fs.get("diffusionPattern", "*DTI*.nii.gz"))
    else:
        raise HTTPException(status_code=400, detail="Unknown kind")

    f: Optional[Path] = None
    for pattern in patterns:
        f = _glob_first(nifti_dir, pattern)
        if f:
            break
    if not f:
        # fallback: first nifti
        for ext in ("*.nii.gz", "*.nii"):
            f = _glob_first(nifti_dir, ext)
            if f:
                break
    if not f:
        raise HTTPException(status_code=404, detail=f"No volume found for kind={kind}")
    return f


@lru_cache(maxsize=8)
def _load_nifti(path_str: str):
    _require_nibabel()
    assert nib is not None
    return nib.load(path_str)


def _volume_info_from_img(img, *, kind: Optional[str] = None, path: Optional[str] = None) -> Dict[str, Any]:
    _require_numpy()
    assert np is not None
    shape = tuple(int(x) for x in img.shape)
    zooms = img.header.get_zooms()
    voxel_size = [float(zooms[0]), float(zooms[1]), float(zooms[2])] if len(zooms) >= 3 else [1.0, 1.0, 1.0]
    dims: List[int] = [shape[0], shape[1], shape[2] if len(shape) >= 3 else 1]
    if len(shape) >= 4:
        dims.append(shape[3])

    def _is_map_kind(k: Optional[str], p: Optional[str]) -> bool:
        if (k or "").strip().lower() == "map":
            return True
        # Fallback heuristic: most map calls come from MapsView which always passes kind=map.
        if not p:
            return False
        name = os.path.basename(p).lower()
        return any(tok in name for tok in ("ki_", "vp_", "cbf_", "mtt_", "cth_", "fa_", "md_", "ad_", "rd_", "mo_"))

    def _estimate_intensity_range(vals):
        finite = vals[np.isfinite(vals)]
        if finite.size == 0:
            return None, None
        vmin, vmax = np.percentile(finite, (2.0, 98.0))
        if not np.isfinite(vmin) or not np.isfinite(vmax):
            return None, None
        if np.isclose(vmin, vmax):
            delta = float(abs(vmin) if vmin else 1.0)
            vmax = vmin + delta
        return float(vmin), float(vmax)

    # Default: cheap min/max from a central slice/timeframe.
    z = dims[2] // 2
    t = 0
    if len(shape) >= 4:
        arr = np.asanyarray(img.dataobj[:, :, z, t])
    else:
        arr = np.asanyarray(img.dataobj[:, :, z])
    vmin = float(np.nanmin(arr)) if arr.size else 0.0
    vmax = float(np.nanmax(arr)) if arr.size else 0.0

    info: Dict[str, Any] = {
        "dimensions": dims,
        "voxelSize": voxel_size,
        "dataType": str(img.get_data_dtype()),
        "min": vmin,
        "max": vmax,
    }

    # For parameter maps, match montage defaults:
    # - slider bounds: min/max of finite nonzero voxels
    # - default window: 2–98% of finite nonzero voxels (or fixed overrides like FA 0–1)
    if _is_map_kind(kind, path):
        try:
            # Maps are typically 3D; if 4D, use the first volume.
            vol = np.asanyarray(img.dataobj)
            if vol.ndim >= 4:
                vol = vol[:, :, :, 0]
            finite = vol[np.isfinite(vol)]
            finite = finite[np.abs(finite) > 1e-8]

            if finite.size:
                info["min"] = float(np.min(finite))
                info["max"] = float(np.max(finite))

                disp_min, disp_max = _estimate_intensity_range(finite)
                # Fixed montage overrides for FA.
                base = (os.path.basename(path or "") or "").lower()
                base = base[:-7] if base.endswith(".nii.gz") else (base[:-4] if base.endswith(".nii") else base)
                if base.startswith("fa_map") or base.startswith("fa_") or base == "fa":
                    disp_min, disp_max = 0.0, 1.0

                if disp_min is not None and disp_max is not None:
                    info["displayMin"] = float(disp_min)
                    info["displayMax"] = float(disp_max)
        except Exception:
            # Keep cheap info if anything goes wrong.
            pass

    return info


def _analysis_metrics_table(subject: Subject, view: str = "atlas") -> Dict[str, Any]:
    analysis_dir = _analysis_dir_for_subject(subject)

    v = (view or "atlas").strip().lower()

    if v == "atlas":
        candidates = [
            analysis_dir / "Ki_values_atlas_patlak.json",
            analysis_dir / "Ki_values_atlas_tikhonov.json",
        ]
        p = next((c for c in candidates if c.exists()), None)
        if not p:
            return {"rows": []}
        raw = json.loads(p.read_text())

        def _is_pk_ventricular_region(region_raw: str, region_mapped: str) -> bool:
            # Filter ventricular/CSF parcels from pharmacokinetic atlas tables.
            # Applies to both pre-mapped keys and mapped FreeSurfer region names.
            for candidate in (region_mapped, region_raw):
                s = (candidate or "").strip()
                if not s:
                    continue
                if s.isdigit() and int(s) in {4, 5, 14, 15, 24, 43, 44, 72}:
                    return True
                lower = s.lower()
                if (
                    "ventricle" in lower
                    or lower.endswith("vent")
                    or "csf" in lower
                ):
                    return True
            return False

        rows: List[Dict[str, Any]] = []
        if isinstance(raw, dict):
            for region, v0 in raw.items():
                if not isinstance(v0, dict):
                    continue
                mapped = _fs_region_name(region)
                if _is_pk_ventricular_region(str(region), mapped):
                    continue
                row: Dict[str, Any] = {
                    "region": mapped,
                    "Ki": v0.get("Ki"),
                    "vp": v0.get("vp"),
                    "Ktrans": v0.get("Ktrans"),
                    "ve": v0.get("ve"),
                    "CBF": v0.get("CBF_tikhonov") or v0.get("CBF"),
                    "MTT": v0.get("MTT_tikhonov") or v0.get("MTT"),
                    "CTH": v0.get("CTH_tikhonov") or v0.get("CTH"),
                }
                rows.append(row)

        rows.sort(key=lambda r: r.get("region") or "")
        return {"rows": rows}

    # Newer pipelines write model-specific files (tikhonov/patlak).
    # Prefer tikhonov since it contains the CBF/MTT/CTH fields used by the UI.
    candidates = [
        analysis_dir / "AI_values_median_total_tikhonov.json",
        analysis_dir / "AI_values_median_total.json",
        analysis_dir / "AI_values_median_total_patlak.json",
    ]
    p = next((c for c in candidates if c.exists()), None)
    if not p:
        return {"rows": []}
    raw = json.loads(p.read_text())

    rows: List[Dict[str, Any]] = []
    for k, v in raw.items():
        if not isinstance(v, dict):
            continue
        if not k.endswith("_median_total"):
            continue
        region = k.replace("_median_total", "")
        row: Dict[str, Any] = {
            "region": region,
            "Ki": v.get("Ki"),
            "CBF": v.get("CBF_tikhonov"),
            "MTT": v.get("MTT_tikhonov"),
            "CTH": v.get("CTH_tikhonov"),
        }
        rows.append(row)

    rows.sort(key=lambda r: r.get("region") or "")
    return {"rows": rows}


def _analysis_metrics_table_from_dir(subject_dir: Path, view: str = "atlas") -> Dict[str, Any]:
    analysis_dir = subject_dir.expanduser().resolve() / "Analysis"

    v = (view or "atlas").strip().lower()

    if v == "atlas":
        candidates = [
            analysis_dir / "Ki_values_atlas_patlak.json",
            analysis_dir / "Ki_values_atlas_tikhonov.json",
        ]
        p = next((c for c in candidates if c.exists()), None)
        if not p:
            return {"rows": []}
        raw = json.loads(p.read_text())

        def _is_pk_ventricular_region(region_raw: str, region_mapped: str) -> bool:
            for candidate in (region_mapped, region_raw):
                s = (candidate or "").strip()
                if not s:
                    continue
                if s.isdigit() and int(s) in {4, 5, 14, 15, 24, 43, 44, 72}:
                    return True
                lower = s.lower()
                if (
                    "ventricle" in lower
                    or lower.endswith("vent")
                    or "csf" in lower
                ):
                    return True
            return False

        rows: List[Dict[str, Any]] = []
        if isinstance(raw, dict):
            for region, v0 in raw.items():
                if not isinstance(v0, dict):
                    continue
                mapped = _fs_region_name(region)
                if _is_pk_ventricular_region(str(region), mapped):
                    continue
                row: Dict[str, Any] = {
                    "region": mapped,
                    "Ki": v0.get("Ki"),
                    "vp": v0.get("vp"),
                    "Ktrans": v0.get("Ktrans"),
                    "ve": v0.get("ve"),
                    "CBF": v0.get("CBF_tikhonov") or v0.get("CBF"),
                    "MTT": v0.get("MTT_tikhonov") or v0.get("MTT"),
                    "CTH": v0.get("CTH_tikhonov") or v0.get("CTH"),
                }
                rows.append(row)

        rows.sort(key=lambda r: r.get("region") or "")
        return {"rows": rows}

    candidates = [
        analysis_dir / "AI_values_median_total_tikhonov.json",
        analysis_dir / "AI_values_median_total.json",
        analysis_dir / "AI_values_median_total_patlak.json",
    ]
    p = next((c for c in candidates if c.exists()), None)
    if not p:
        return {"rows": []}
    raw = json.loads(p.read_text())

    rows: List[Dict[str, Any]] = []
    for k, v0 in raw.items():
        if not isinstance(v0, dict):
            continue
        if not str(k).endswith("_median_total"):
            continue
        region = str(k).replace("_median_total", "")
        row: Dict[str, Any] = {
            "region": region,
            "Ki": v0.get("Ki"),
            "CBF": v0.get("CBF_tikhonov"),
            "MTT": v0.get("MTT_tikhonov"),
            "CTH": v0.get("CTH_tikhonov"),
        }
        rows.append(row)

    rows.sort(key=lambda r: r.get("region") or "")
    return {"rows": rows}


def _analysis_tractography_path(subject: Subject, explicit: Optional[str] = None) -> Optional[Path]:
    if explicit:
        try:
            p = Path(explicit).expanduser().resolve()
            if p.exists() and p.is_file():
                return p
        except Exception:
            return None

    analysis_dir = _analysis_dir_for_subject(subject)
    candidates = [
        analysis_dir / "diffusion" / "tractography.trk",
        analysis_dir / "tractography.trk",
    ]
    for c in candidates:
        if c.exists() and c.is_file():
            return c

    try:
        for p in analysis_dir.rglob("tractography*.trk"):
            if not p.is_file() or p.name.startswith("._"):
                continue
            return p
    except Exception:
        return None
    return None


def _analysis_tractography_streamlines(
    subject: Subject,
    *,
    max_streamlines: int = 0,
    max_points_per_streamline: int = 120,
    path_override: Optional[str] = None,
) -> Dict[str, Any]:
    _require_nibabel()
    _require_numpy()
    assert nib is not None
    assert np is not None

    tract_path = _analysis_tractography_path(subject, explicit=path_override)
    if not tract_path:
        return {"path": path_override or "", "streamlines": [], "error": "Tractography file not found"}

    try:
        from nibabel import streamlines as nib_streamlines  # type: ignore
    except Exception:
        raise HTTPException(status_code=500, detail="Backend missing dependency: nibabel.streamlines")

    load_error = ""
    try:
        tractogram = nib_streamlines.load(str(tract_path)).tractogram
        sls = tractogram.streamlines
    except Exception as exc:
        load_error = str(exc)
        return {"path": str(tract_path), "streamlines": [], "error": load_error}

    # Downsample streamlines to keep payload light, but default to all streamlines (max_streamlines=0 means no cap).
    selected: List[list] = []
    colours: List[list] = []
    try:
        total = len(sls)
    except Exception:
        total = 0

    if total and max_streamlines > 0 and total > max_streamlines:
        step = max(1, int(total / max_streamlines))
        indices = range(0, total, step)
    else:
        indices = range(0, total) if total else range(0)

    for i in indices:
        if max_streamlines > 0 and len(selected) >= max_streamlines:
            break
        try:
            pts = np.asarray(sls[i], dtype=float)
        except Exception:
            continue
        if pts.ndim != 2 or pts.shape[1] < 3:
            continue
        # Subsample points.
        n = int(pts.shape[0])
        if max_points_per_streamline > 0 and n > max_points_per_streamline:
            stride = max(1, int(n / max_points_per_streamline))
            pts = pts[::stride]
        pts = pts[:, :3]
        selected.append(pts.tolist())

        # Colour by dominant orientation (replicates original p-brain colouring).
        try:
            segments = np.diff(pts, axis=0)
            norms = np.linalg.norm(segments, axis=1, keepdims=True)
            valid = norms[:, 0] > 0
            if valid.any():
                directions = segments[valid] / norms[valid]
                mean_dir = directions.mean(axis=0)
            else:
                mean_dir = np.array([0.0, 0.0, 1.0], dtype=float)
            angle = float(np.arctan2(mean_dir[1], mean_dir[0]))
            hue = (angle + np.pi) / (2 * np.pi)
            r, g, b = colorsys.hsv_to_rgb(hue, 0.85, 1.0)
            colours.append([float(r), float(g), float(b)])
        except Exception:
            colours.append([0.7, 0.7, 0.7])

    return {
        "path": str(tract_path),
        "streamlines": selected,
        "colors": colours,
        "totalStreamlines": int(total or 0),
        "returned": len(selected),
        "error": load_error,
    }


def _analysis_connectome_paths(subject: Subject) -> Dict[str, Optional[Path]]:
    """Return expected connectome artifact paths under the subject Analysis directory."""

    analysis_dir = _analysis_dir_for_subject(subject)
    diffusion_dir = analysis_dir / "diffusion"
    candidates = {
        "matrix": diffusion_dir / "connectome_matrix.csv",
        "labels": diffusion_dir / "connectome_labels.csv",
        "metrics": diffusion_dir / "connectome_metrics.json",
        "image": diffusion_dir / "connectome_circular.png",
    }

    out: Dict[str, Optional[Path]] = {}
    for k, p in candidates.items():
        out[k] = p if p.exists() and p.is_file() else None
    return out


@app.get("/subjects/{subject_id}/connectome")
def get_subject_connectome(subject_id: str) -> Dict[str, Any]:
    subject = _find_subject(subject_id)
    paths = _analysis_connectome_paths(subject)
    metrics_path = paths.get("metrics")

    payload: Dict[str, Any] = {
        "available": bool(metrics_path),
        "files": {
            "matrix": str(paths["matrix"]) if paths.get("matrix") else None,
            "labels": str(paths["labels"]) if paths.get("labels") else None,
            "metrics": str(metrics_path) if metrics_path else None,
            "image": str(paths["image"]) if paths.get("image") else None,
        },
        "metrics": None,
    }

    if metrics_path and metrics_path.exists():
        try:
            payload["metrics"] = json.loads(metrics_path.read_text(encoding="utf-8"))
        except Exception as exc:
            payload["available"] = False
            payload["error"] = f"Failed to read connectome metrics: {exc}"

    return payload


@app.get("/subjects/{subject_id}/connectome/file")
def get_subject_connectome_file(subject_id: str, kind: str) -> FileResponse:
    """Download connectome artifacts (matrix/labels/metrics) for a subject."""

    subject = _find_subject(subject_id)
    project = _find_project(subject.projectId)

    paths = _analysis_connectome_paths(subject)
    kind_norm = str(kind or "").strip().lower()
    if kind_norm not in {"matrix", "labels", "metrics", "image"}:
        raise HTTPException(status_code=400, detail="kind must be one of: matrix, labels, metrics, image")

    p = paths.get(kind_norm)
    if not p or not p.exists() or not p.is_file():
        raise HTTPException(status_code=404, detail="Connectome file not found")

    # Enforce subject/project sandbox.
    safe = _safe_resolve_path(project, subject, str(p))

    # Ensure connectome labels use the same FreeSurfer naming scheme as atlas tables.
    # Some connectome runs may contain numeric names; rewrite them on-the-fly using FreeSurferColorLUT.
    if kind_norm == "labels" and safe.suffix.lower() == ".csv":
        try:
            import csv
            import io

            lut = _freesurfer_lut()
            if lut:
                original = safe.read_text(encoding="utf-8", errors="replace")
                reader = csv.reader(io.StringIO(original))
                rows = list(reader)
                if rows:
                    header = rows[0]
                    label_col = 0
                    name_col: Optional[int] = None
                    for i, col in enumerate(header):
                        c = str(col).strip().lower()
                        if c == "label":
                            label_col = i
                        elif c == "name":
                            name_col = i

                    if name_col is not None:
                        changed = False
                        for r in range(1, len(rows)):
                            row = rows[r]
                            if not row or len(row) <= max(label_col, name_col):
                                continue
                            raw_label = str(row[label_col] or "").strip()
                            if not raw_label:
                                continue
                            mapped = _fs_region_name(raw_label)
                            if not mapped or mapped == raw_label:
                                continue

                            current = str(row[name_col] or "").strip()
                            # Rewrite when missing or numeric-only (e.g., "1017").
                            if current == "" or current.isdigit() or current == raw_label:
                                row[name_col] = mapped
                                changed = True

                        if changed:
                            out = io.StringIO()
                            writer = csv.writer(out)
                            writer.writerows(rows)
                            return Response(
                                content=out.getvalue(),
                                media_type="text/csv",
                                headers={
                                    "Content-Disposition": f"attachment; filename=\"{safe.name}\"",
                                },
                            )
        except Exception:
            # Fall back to serving the file as-is.
            pass

    mt = "application/octet-stream"
    lower = safe.name.lower()
    if lower.endswith(".csv"):
        mt = "text/csv"
    elif lower.endswith(".json"):
        mt = "application/json"
    elif lower.endswith(".png"):
        mt = "image/png"
    return FileResponse(str(safe), media_type=mt)


def _analysis_curves(subject: Subject) -> List[Dict[str, Any]]:
    _require_numpy()
    assert np is not None
    analysis_dir = _analysis_dir_for_subject(subject)
    tp = _load_time_points_best_effort(subject)

    curves: List[Dict[str, Any]] = []

    def _default_time_points(n: int) -> List[float]:
        return np.arange(int(n), dtype=float).astype(float).tolist()

    def _time_points_for_len(n: int) -> List[float]:
        if tp is None:
            return _default_time_points(n)
        nn = int(min(int(tp.size), int(n)))
        return tp[:nn].astype(float).tolist()

    def _aligned_time_and_values(values: Any) -> Optional[tuple[List[float], List[float]]]:
        try:
            vv = np.asarray(values, dtype=float).reshape(-1)
        except Exception:
            return None
        if vv.size < 3:
            return None
        if tp is None:
            return _default_time_points(int(vv.size)), vv.astype(float).tolist()
        if tp.size < 3:
            return None
        n = int(min(int(tp.size), int(vv.size)))
        if n < 3:
            return None
        return tp[:n].astype(float).tolist(), vv[:n].astype(float).tolist()

    def _read_max_artery_type() -> Optional[str]:
        info = analysis_dir / "max_info.json"
        if not info.exists() or not info.is_file():
            return None
        try:
            raw = json.loads(info.read_text())
        except Exception:
            return None
        if not isinstance(raw, list):
            return None
        for entry in raw:
            if isinstance(entry, str) and entry.startswith("Max artery type:"):
                return entry.split(":", 1)[1].strip() or None
        return None

    def _load_tscc_max_curve() -> Optional[tuple[str, Any]]:
        tscc_dir = analysis_dir / "TSCC Data" / "Max"
        if not tscc_dir.exists() or not tscc_dir.is_dir():
            return None
        files = sorted(
            [p for p in tscc_dir.glob("*.npy") if p.is_file() and not p.name.startswith(".")]
        )
        if not files:
            return None
        arr = np.asarray(np.load(str(files[0])), dtype=float).reshape(-1)
        if arr.size < 3:
            return None
        label = _read_max_artery_type() or "Max"
        return label, arr

    # Prefer shifted curves if present.
    def pick_curve(glob_path: Path) -> Optional[Path]:
        matches = sorted(glob_path.parent.glob(glob_path.name))
        return matches[0] if matches else None

    # AIF: p-brain uses the time-shifted & rescaled VIF as the effective AIF.
    tscc = _load_tscc_max_curve()
    if tscc is not None:
        label, arr = tscc
        n = int(arr.size) if tp is None else min(int(tp.size), int(arr.size))
        curves.append(
            {
                "id": "aif_tscc_max",
                "name": f"AIF (Time-shifted VIF: {label})",
                "timePoints": _time_points_for_len(n),
                "values": arr[:n].astype(float).tolist(),
                "unit": "mM",
            }
        )
    else:
        # Fallback: first available arterial curve.
        artery_dir = analysis_dir / "CTC Data" / "Artery"
        if artery_dir.exists():
            for subtype_dir in sorted([d for d in artery_dir.iterdir() if d.is_dir()]):
                pth = pick_curve(subtype_dir / "CTC_shifted_slice_*.npy") or pick_curve(subtype_dir / "CTC_slice_*.npy")
                if pth and pth.exists():
                    aligned = _aligned_time_and_values(np.load(str(pth)))
                    if not aligned:
                        continue
                    tt, vv = aligned
                    curves.append({"id": f"aif_{subtype_dir.name}", "name": f"AIF ({subtype_dir.name})", "timePoints": tt, "values": vv, "unit": "mM"})
                    break

    # VIF (vein)
    vein_dir = analysis_dir / "CTC Data" / "Vein"
    if vein_dir.exists():
        for subtype_dir in sorted([d for d in vein_dir.iterdir() if d.is_dir()]):
            pth = pick_curve(subtype_dir / "CTC_shifted_slice_*.npy") or pick_curve(subtype_dir / "CTC_slice_*.npy")
            if pth and pth.exists():
                aligned = _aligned_time_and_values(np.load(str(pth)))
                if not aligned:
                    continue
                tt, vv = aligned
                curves.append({"id": f"vif_{subtype_dir.name}", "name": f"VIF ({subtype_dir.name})", "timePoints": tt, "values": vv, "unit": "mM"})
                break

    # Tissue curves: include all available tissue subtypes.
    tissue_dir = analysis_dir / "CTC Data" / "Tissue"
    if tissue_dir.exists():
        for subtype_dir in sorted([d for d in tissue_dir.iterdir() if d.is_dir()]):
            pth = pick_curve(subtype_dir / "CTC_shifted_slice_*.npy") or pick_curve(subtype_dir / "CTC_slice_*.npy")
            if pth and pth.exists():
                aligned = _aligned_time_and_values(np.load(str(pth)))
                if not aligned:
                    continue
                tt, vv = aligned
                curves.append({
                    "id": f"tissue_{subtype_dir.name}",
                    "name": f"Tissue ({subtype_dir.name})",
                    "timePoints": tt,
                    "values": vv,
                    "unit": "mM",
                })

        # Newer pipeline: Analysis/CTC Data/Tissue/AI/{region}_AI_Tissue_slice_*_segmented_median.npy
        ai_dir = tissue_dir / "AI"
        if ai_dir.exists():
            grouped: Dict[str, List[Path]] = {}
            for p in sorted(ai_dir.glob("*_AI_Tissue_slice_*_segmented_median.npy")):
                if not p.is_file() or p.name.startswith("._"):
                    continue
                region = p.name.split("_AI_Tissue_slice_")[0]
                if not region:
                    continue
                grouped.setdefault(region, []).append(p)

            for region, files in sorted(grouped.items()):
                stacks: List[Any] = []
                for p in files:
                    arr = np.asarray(np.load(str(p)), dtype=float).reshape(-1)
                    if arr.size >= 3 and np.all(np.isfinite(arr) | np.isnan(arr)):
                        stacks.append(arr)
                if not stacks:
                    continue
                min_len = min(a.size for a in stacks)
                stack = np.stack([a[:min_len] for a in stacks], axis=0)
                vals = np.nanmedian(stack, axis=0).astype(float).tolist()
                curves.append(
                    {
                        "id": f"tissue_ai_{region}",
                        "name": f"Tissue ({region})",
                        "timePoints": _time_points_for_len(min_len),
                        "values": vals,
                        "unit": "mM",
                    }
                )

    return curves


def _analysis_roi_overlays(subject: Subject) -> List[Dict[str, Any]]:
    """Return saved ROI voxel overlays (AIF/VIF selection) as lightweight bboxes.

    p-brain writes ROI voxel coordinates under:
      Analysis/ROI Data/<type>/<subtype>/ROI_voxels_slice_<N>.npy
    and the chosen frame index under:
      Analysis/Frame Data/<type>/<subtype>/frame_index_slice_<N>.npy

    NOTE: p-brain's ROI selection tooling rotates the in-plane axes
    (np.rot90(..., k=-1, axes=(0, 1))) before saving ROI voxel coordinates.
    That means the saved coordinates are effectively in a rotated (x, y)
    frame, not the raw nibabel slice frame.

    To render these ROIs on slices served by `/volumes/slice` (which returns
    the raw nibabel in-plane axes), we convert the saved coordinates back into
    the raw slice frame using the DCE volume shape.
    """

    _require_numpy()
    assert np is not None

    analysis_dir = _analysis_dir_for_subject(subject)
    roi_root = analysis_dir / "ROI Data"
    frame_root = analysis_dir / "Frame Data"
    if not roi_root.exists() or not roi_root.is_dir():
        return []

    # Best-effort: infer in-plane shape from the subject's DCE volume.
    # This lets us transform rotated ROI coordinates back into the raw slice frame.
    plane_shape: Optional[Tuple[int, int]] = None
    try:
        root = Path(subject.sourcePath).expanduser().resolve()
        nifti_dir = root
        for alt in ("NIfTI", "nifti", "NIFTI"):
            cand = root / alt
            if cand.exists() and cand.is_dir():
                nifti_dir = cand
                break

        p = (
            _glob_first(nifti_dir, "*DCE*.nii.gz")
            or _glob_first(nifti_dir, "*DCE*.nii")
            or _glob_first(nifti_dir, "*.nii.gz")
            or _glob_first(nifti_dir, "*.nii")
        )
        if p:
            img = _load_nifti(str(p))
            sh = tuple(int(x) for x in getattr(img, "shape", ()) or ())
            if len(sh) >= 2 and sh[0] > 0 and sh[1] > 0:
                plane_shape = (sh[0], sh[1])
    except Exception:
        plane_shape = None

    overlays: List[Dict[str, Any]] = []

    for roi_path in roi_root.rglob("ROI_voxels_slice_*.npy"):
        try:
            m = re.search(r"ROI_voxels_slice_(\d+)\.npy$", roi_path.name)
            if not m:
                continue
            slice_num_1 = int(m.group(1))
            slice_index = slice_num_1 - 1
            rel = roi_path.relative_to(roi_root)
            parts = list(rel.parts)
            if len(parts) < 3:
                continue
            roi_type = str(parts[0])
            roi_subtype = str(parts[1])

            # ROI voxels are saved as 2D integer pairs.
            vox = np.load(str(roi_path))
            arr = np.asarray(vox)
            if arr.ndim != 2 or arr.shape[1] < 2:
                continue

            a0 = np.asarray(arr[:, 0], dtype=int)
            a1 = np.asarray(arr[:, 1], dtype=int)

            # If we can infer volume dimensions, map from p-brain's rotated
            # in-plane frame (x, y) back to raw nibabel slice indices (row, col).
            # For np.rot90(k=-1): B[i, j] = A[X-1-j, i]  =>  A[row, col] = B[col, X-1-row]
            # Thus: row = X-1-y, col = x.
            if plane_shape is not None:
                x_dim, y_dim = plane_shape  # raw slice shape from nibabel: (X, Y)
                # Only apply the transform if coordinates appear to fall within the rotated bounds.
                # This is a best-effort heuristic; if it fails we fall back to identity.
                if (
                    a0.size > 0
                    and a1.size > 0
                    and int(np.min(a0)) >= 0
                    and int(np.min(a1)) >= 0
                    and int(np.max(a0)) < y_dim
                    and int(np.max(a1)) < x_dim
                ):
                    rows = (x_dim - 1 - a1).astype(int)
                    cols = a0.astype(int)
                else:
                    rows = a0
                    cols = a1
            else:
                rows = a0
                cols = a1
            if rows.size == 0 or cols.size == 0:
                continue

            row0 = int(np.min(rows))
            row1 = int(np.max(rows))
            col0 = int(np.min(cols))
            col1 = int(np.max(cols))

            frame_index: Optional[int] = None
            frame_path = frame_root / roi_type / roi_subtype / f"frame_index_slice_{slice_num_1}.npy"
            if frame_path.exists() and frame_path.is_file():
                try:
                    fi = np.load(str(frame_path))
                    fi_arr = np.asarray(fi).reshape(-1)
                    if fi_arr.size:
                        frame_index = int(fi_arr[0])
                except Exception:
                    frame_index = None

            overlays.append(
                {
                    "id": f"{roi_type}/{roi_subtype}/slice_{slice_index}",
                    "roiType": roi_type,
                    "roiSubType": roi_subtype,
                    "sliceIndex": int(slice_index),
                    "frameIndex": frame_index,
                    "row0": row0,
                    "row1": row1,
                    "col0": col0,
                    "col1": col1,
                }
            )
        except Exception:
            # Best-effort: skip any malformed ROI artifacts.
            continue

    overlays.sort(key=lambda o: (str(o.get("roiType", "")), str(o.get("roiSubType", "")), int(o.get("sliceIndex", 0))))
    return overlays


def _analysis_roi_mask_volumes(project: Project, subject: Subject) -> List[Dict[str, Any]]:
    """Generate and return ROI mask NIfTI volumes aligned to the subject's DCE volume.

    This avoids fragile canvas bbox rendering by producing true voxel masks
    that can be overlaid using the same slice extraction pipeline.

    Output:
      Analysis/ROI NIfTI/{roiType}__{roiSubType}__mask.nii.gz
    """

    _require_numpy()
    _require_nibabel()
    assert np is not None
    assert nib is not None

    analysis_dir = _analysis_dir_for_subject(subject)
    roi_root = analysis_dir / "ROI Data"
    if not roi_root.exists() or not roi_root.is_dir():
        return []

    # Reference: DCE volume determines shape + affine.
    try:
        ref_path = _resolve_default_volume_path(project, subject, "dce")
    except Exception:
        # Fallback to any NIfTI if patterns don't match.
        root = Path(subject.sourcePath).expanduser().resolve()
        nifti_dir = _nifti_dir_for_path(project, root)
        ref_path = (
            _glob_first(nifti_dir, "*.nii.gz")
            or _glob_first(nifti_dir, "*.nii")
            or None
        )
        if not ref_path:
            return []

    ref_img = _load_nifti(str(ref_path))
    ref_shape = tuple(int(x) for x in getattr(ref_img, "shape", ()) or ())
    if len(ref_shape) < 3:
        return []

    x_dim, y_dim, z_dim = int(ref_shape[0]), int(ref_shape[1]), int(ref_shape[2])
    if x_dim <= 0 or y_dim <= 0 or z_dim <= 0:
        return []

    out_dir = analysis_dir / "ROI NIfTI"
    try:
        out_dir.mkdir(parents=True, exist_ok=True)
    except Exception:
        return []

    def safe_part(s: str) -> str:
        return re.sub(r"[^A-Za-z0-9._-]+", "_", (s or "").strip()) or "roi"

    results: List[Dict[str, Any]] = []

    # Group ROI voxel files by (roiType, roiSubType).
    grouped: Dict[Tuple[str, str], List[Path]] = {}
    for roi_path in roi_root.rglob("ROI_voxels_slice_*.npy"):
        if not roi_path.is_file() or roi_path.name.startswith("._"):
            continue
        try:
            rel = roi_path.relative_to(roi_root)
        except Exception:
            continue
        parts = list(rel.parts)
        if len(parts) < 3:
            continue
        roi_type = str(parts[0])
        roi_subtype = str(parts[1])
        grouped.setdefault((roi_type, roi_subtype), []).append(roi_path)

    for (roi_type, roi_subtype), files in sorted(grouped.items(), key=lambda kv: (kv[0][0], kv[0][1])):
        mask = np.zeros((x_dim, y_dim, z_dim), dtype=np.uint8)

        for roi_path in sorted(files):
            try:
                m = re.search(r"ROI_voxels_slice_(\d+)\.npy$", roi_path.name)
                if not m:
                    continue
                slice_num_1 = int(m.group(1))
                z = int(slice_num_1 - 1)
                if z < 0 or z >= z_dim:
                    continue

                vox = np.load(str(roi_path))
                arr = np.asarray(vox)
                if arr.ndim != 2 or arr.shape[1] < 2:
                    continue

                a0 = np.asarray(arr[:, 0], dtype=int)
                a1 = np.asarray(arr[:, 1], dtype=int)
                if a0.size == 0 or a1.size == 0:
                    continue

                # Map from p-brain's rotated in-plane frame back to raw nibabel slice indices.
                # For np.rot90(k=-1): A[row, col] = B[col, X-1-row]
                # Thus: row = X-1-y, col = x.
                if (
                    int(np.min(a0)) >= 0
                    and int(np.min(a1)) >= 0
                    and int(np.max(a0)) < y_dim
                    and int(np.max(a1)) < x_dim
                ):
                    rows = (x_dim - 1 - a1).astype(int)
                    cols = a0.astype(int)
                else:
                    rows = a0
                    cols = a1

                # Clamp to bounds.
                rows = np.clip(rows, 0, x_dim - 1)
                cols = np.clip(cols, 0, y_dim - 1)

                mask[rows, cols, z] = 1
            except Exception:
                continue

        if int(np.sum(mask)) <= 0:
            continue

        out_name = f"{safe_part(roi_type)}__{safe_part(roi_subtype)}__mask.nii.gz"
        out_path = out_dir / out_name
        try:
            img = nib.Nifti1Image(mask, affine=ref_img.affine)
            img.header.set_data_dtype(np.uint8)
            nib.save(img, str(out_path))
        except Exception:
            continue

        results.append(
            {
                "id": f"roi_mask::{roi_type}/{roi_subtype}",
                "name": f"ROI Mask ({roi_type}/{roi_subtype})",
                "path": str(out_path),
                "roiType": roi_type,
                "roiSubType": roi_subtype,
            }
        )

    return results

@app.post("/subjects/{subject_id}/roi-voxels")
def save_subject_roi_voxels(subject_id: str, req: SaveRoiVoxelsRequest) -> Dict[str, Any]:
    """Persist a user-drawn ROI voxel list for a single slice.

    Writes p-brain compatible artifacts:
      Analysis/ROI Data/<roiType>/<roiSubType>/ROI_voxels_slice_<N>.npy
      Analysis/Frame Data/<roiType>/<roiSubType>/frame_index_slice_<N>.npy

    Input voxels are provided in the raw nibabel slice frame as [row, col].
    p-brain stores ROI coordinates in a rotated in-plane frame (np.rot90(k=-1)).
    We convert using the DCE volume shape so downstream p-brain stages can consume them.
    """

    _require_numpy()
    _require_nibabel()
    assert np is not None
    assert nib is not None

    subject = _find_subject(subject_id)
    project = _find_project(subject.projectId)

    roi_type = str(req.roiType or "").strip()
    roi_subtype = str(req.roiSubType or "").strip()
    if not roi_type or not roi_subtype:
        raise HTTPException(status_code=400, detail="roiType and roiSubType are required")

    if (
        ".." in roi_type
        or ".." in roi_subtype
        or "/" in roi_type
        or "/" in roi_subtype
        or "\\" in roi_type
        or "\\" in roi_subtype
    ):
        raise HTTPException(status_code=400, detail="Invalid roiType/roiSubType")

    slice_index = int(req.sliceIndex)
    if slice_index < 0:
        raise HTTPException(status_code=400, detail="sliceIndex must be >= 0")

    frame_index = int(req.frameIndex)
    if frame_index < 0:
        frame_index = 0

    analysis_dir = _analysis_dir_for_subject(subject)
    roi_root = analysis_dir / "ROI Data" / roi_type / roi_subtype
    frame_root = analysis_dir / "Frame Data" / roi_type / roi_subtype
    roi_root.mkdir(parents=True, exist_ok=True)
    frame_root.mkdir(parents=True, exist_ok=True)

    # Use DCE volume to validate slice bounds and perform raw->rotated coordinate transform.
    try:
        ref_path = _resolve_default_volume_path(project, subject, "dce")
        ref_img = _load_nifti(str(ref_path))
        sh = tuple(int(x) for x in getattr(ref_img, "shape", ()) or ())
        if len(sh) < 3:
            raise RuntimeError("DCE volume has unexpected shape")
        x_dim, y_dim, z_dim = int(sh[0]), int(sh[1]), int(sh[2])
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to resolve DCE volume shape: {e}")

    if slice_index >= z_dim:
        raise HTTPException(status_code=400, detail=f"sliceIndex out of bounds (0..{max(0, z_dim-1)})")

    # Convert raw [row, col] -> p-brain saved [x, y] in rotated in-plane frame.
    # Mapping used elsewhere:
    #   raw row = X-1-y, raw col = x
    # Inverse:
    #   x = raw col, y = X-1-raw row
    raw_pairs: List[Tuple[int, int]] = []
    for item in (req.voxels or []):
        try:
            if not isinstance(item, (list, tuple)) or len(item) < 2:
                continue
            r = int(item[0])
            c = int(item[1])
        except Exception:
            continue
        if r < 0 or c < 0 or r >= x_dim or c >= y_dim:
            continue
        raw_pairs.append((r, c))

    out_vox_path = roi_root / f"ROI_voxels_slice_{slice_index + 1}.npy"
    out_frame_path = frame_root / f"frame_index_slice_{slice_index + 1}.npy"

    if len(raw_pairs) == 0:
        # Treat empty payload as delete for this slice.
        try:
            if out_vox_path.exists():
                out_vox_path.unlink()
        except Exception:
            pass
        try:
            if out_frame_path.exists():
                out_frame_path.unlink()
        except Exception:
            pass
    else:
        arr = np.asarray(raw_pairs, dtype=np.int64)
        rows = arr[:, 0]
        cols = arr[:, 1]
        xs = cols
        ys = (x_dim - 1 - rows)
        saved = np.stack([xs, ys], axis=1).astype(np.int16, copy=False)
        np.save(str(out_vox_path), saved)
        np.save(str(out_frame_path), np.asarray(int(frame_index), dtype=np.int16))

    # Best-effort: remove stale cached ROI mask so next GET regenerates.
    try:
        mask_dir = analysis_dir / "ROI NIfTI"
        if mask_dir.is_dir():
            for p in mask_dir.glob(f"{roi_type}__{roi_subtype}__mask.nii*"):
                try:
                    p.unlink()
                except Exception:
                    pass
    except Exception:
        pass

    return {
        "ok": True,
        "roiType": roi_type,
        "roiSubType": roi_subtype,
        "sliceIndex": slice_index,
        "frameIndex": int(frame_index),
        "savedVoxelCount": int(len(raw_pairs)),
    }


def _analysis_curves_from_dir(subject_dir: Path) -> List[Dict[str, Any]]:
    _require_numpy()
    assert np is not None
    analysis_dir = subject_dir.expanduser().resolve() / "Analysis"
    time_path = analysis_dir / "Fitting" / "time_points_s.npy"
    time_points: Optional[List[float]] = None
    if time_path.exists():
        try:
            tp = np.asarray(np.load(str(time_path)), dtype=float).reshape(-1)
            if tp.size >= 3 and np.all(np.isfinite(tp)):
                time_points = tp.astype(float).tolist()
        except Exception:
            time_points = None

    def _tp_for(n: int) -> List[float]:
        nn = int(n)
        if time_points is None:
            return np.arange(nn, dtype=float).astype(float).tolist()
        n2 = int(min(int(len(time_points)), nn))
        return [float(x) for x in time_points[:n2]]

    curves: List[Dict[str, Any]] = []

    def _read_max_artery_type() -> Optional[str]:
        info = analysis_dir / "max_info.json"
        if not info.exists() or not info.is_file():
            return None
        try:
            raw = json.loads(info.read_text())
        except Exception:
            return None
        if not isinstance(raw, list):
            return None
        for entry in raw:
            if isinstance(entry, str) and entry.startswith("Max artery type:"):
                return entry.split(":", 1)[1].strip() or None
        return None

    def _load_tscc_max_curve() -> Optional[tuple[str, Any]]:
        tscc_dir = analysis_dir / "TSCC Data" / "Max"
        if not tscc_dir.exists() or not tscc_dir.is_dir():
            return None
        files = sorted(
            [p for p in tscc_dir.glob("*.npy") if p.is_file() and not p.name.startswith(".")]
        )
        if not files:
            return None
        arr = np.asarray(np.load(str(files[0])), dtype=float).reshape(-1)
        if arr.size < 3:
            return None
        label = _read_max_artery_type() or "Max"
        return label, arr

    def pick_curve(glob_path: Path) -> Optional[Path]:
        matches = sorted(glob_path.parent.glob(glob_path.name))
        return matches[0] if matches else None

    tscc = _load_tscc_max_curve()
    if tscc is not None:
        label, arr = tscc
        n = int(arr.size) if time_points is None else min(len(time_points), int(arr.size))
        curves.append(
            {
                "id": "aif_tscc_max",
                "name": f"AIF (Time-shifted VIF: {label})",
                "timePoints": _tp_for(n),
                "values": arr[:n].astype(float).tolist(),
                "unit": "mM",
            }
        )
    else:
        artery_dir = analysis_dir / "CTC Data" / "Artery"
        if artery_dir.exists():
            for subtype_dir in sorted([d for d in artery_dir.iterdir() if d.is_dir()]):
                pth = pick_curve(subtype_dir / "CTC_shifted_slice_*.npy") or pick_curve(subtype_dir / "CTC_slice_*.npy")
                if pth and pth.exists():
                    vv = np.asarray(np.load(str(pth)), dtype=float).reshape(-1)
                    n = int(vv.size) if time_points is None else min(int(len(time_points)), int(vv.size))
                    curves.append({
                        "id": f"aif_{subtype_dir.name}",
                        "name": f"AIF ({subtype_dir.name})",
                        "timePoints": _tp_for(n),
                        "values": vv[:n].astype(float).tolist(),
                        "unit": "mM",
                    })
                    break

    vein_dir = analysis_dir / "CTC Data" / "Vein"
    if vein_dir.exists():
        for subtype_dir in sorted([d for d in vein_dir.iterdir() if d.is_dir()]):
            pth = pick_curve(subtype_dir / "CTC_shifted_slice_*.npy") or pick_curve(subtype_dir / "CTC_slice_*.npy")
            if pth and pth.exists():
                vv = np.asarray(np.load(str(pth)), dtype=float).reshape(-1)
                n = int(vv.size) if time_points is None else min(int(len(time_points)), int(vv.size))
                curves.append({
                    "id": f"vif_{subtype_dir.name}",
                    "name": f"VIF ({subtype_dir.name})",
                    "timePoints": _tp_for(n),
                    "values": vv[:n].astype(float).tolist(),
                    "unit": "mM",
                })
                break

    tissue_dir = analysis_dir / "CTC Data" / "Tissue"
    if tissue_dir.exists():
        for subtype_dir in sorted([d for d in tissue_dir.iterdir() if d.is_dir()]):
            pth = pick_curve(subtype_dir / "CTC_shifted_slice_*.npy") or pick_curve(subtype_dir / "CTC_slice_*.npy")
            if pth and pth.exists():
                vv = np.asarray(np.load(str(pth)), dtype=float).reshape(-1)
                n = int(vv.size) if time_points is None else min(int(len(time_points)), int(vv.size))
                curves.append({
                    "id": f"tissue_{subtype_dir.name}",
                    "name": f"Tissue ({subtype_dir.name})",
                    "timePoints": _tp_for(n),
                    "values": vv[:n].astype(float).tolist(),
                    "unit": "mM",
                })

        ai_dir = tissue_dir / "AI"
        if ai_dir.exists():
            grouped: Dict[str, List[Path]] = {}
            for p in sorted(ai_dir.glob("*_AI_Tissue_slice_*_segmented_median.npy")):
                if not p.is_file() or p.name.startswith("._"):
                    continue
                region = p.name.split("_AI_Tissue_slice_")[0]
                if not region:
                    continue
                grouped.setdefault(region, []).append(p)

            for region, files in sorted(grouped.items()):
                stacks: List[Any] = []
                for p in files:
                    arr = np.asarray(np.load(str(p)), dtype=float).reshape(-1)
                    if arr.size >= 3 and np.all(np.isfinite(arr) | np.isnan(arr)):
                        stacks.append(arr)
                if not stacks:
                    continue
                min_len = min(a.size for a in stacks)
                stack = np.stack([a[:min_len] for a in stacks], axis=0)
                vals = np.nanmedian(stack, axis=0).astype(float).tolist()
                curves.append(
                    {
                        "id": f"tissue_ai_{region}",
                        "name": f"Tissue ({region})",
                        "timePoints": _tp_for(min_len),
                        "values": vals,
                        "unit": "mM",
                    }
                )

    return curves


def _get_config_value(project: Project, keys: List[str], default: Any) -> Any:
    cur: Any = project.config or {}
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


def _region_to_ai_key(region: str) -> str:
    r = (region or "").strip().lower()
    mapping = {
        "gm": "cortical_gm",
        "cortical_gm": "cortical_gm",
        "subcortical_gm": "subcortical_gm",
        "brainstem": "gm_brainstem",
        "gm_brainstem": "gm_brainstem",
        "cerebellum_gm": "gm_cerebellum",
        "gm_cerebellum": "gm_cerebellum",
        "wm": "wm",
        "wm_cerebellum": "wm_cerebellum",
        "wm_cc": "wm_cc",
        "boundary": "boundary",
    }
    return mapping.get(r, "cortical_gm")


def _load_time_points(subject: Subject) -> Any:
    _require_numpy()
    assert np is not None
    analysis_dir = _analysis_dir_for_subject(subject)
    time_path = analysis_dir / "Fitting" / "time_points_s.npy"
    if time_path.exists():
        t = np.asarray(np.load(str(time_path)), dtype=float).reshape(-1)
        if t.size >= 3 and np.all(np.isfinite(t)):
            return t

    # Fallback: derive time points from the configured DCE volume NIfTI header.
    t2 = _load_time_points_best_effort(subject)
    if t2 is None or int(t2.size) < 3 or not np.all(np.isfinite(t2)):
        raise HTTPException(status_code=404, detail="Missing time points (time_points_s.npy and DCE header fallback unavailable)")
    return t2


def _load_time_points_best_effort(subject: Subject) -> Optional[Any]:
    """Best-effort time axis for plotting/modelling.

    Prefers Analysis/Fitting/time_points_s.npy, otherwise derives from the DCE NIfTI
    header (4th zoom) and 4D length.
    """
    _require_numpy()
    assert np is not None
    analysis_dir = _analysis_dir_for_subject(subject)
    time_path = analysis_dir / "Fitting" / "time_points_s.npy"
    if time_path.exists():
        try:
            t = np.asarray(np.load(str(time_path)), dtype=float).reshape(-1)
            if t.size >= 3 and np.all(np.isfinite(t)):
                return t
        except Exception:
            pass

    try:
        project = _find_project(subject.projectId)
    except Exception:
        return None

    try:
        dce_path = _resolve_default_volume_path(project, subject, "dce")
    except Exception:
        return None

    try:
        _require_nibabel()
        img = _load_nifti(str(dce_path))
        shape = img.shape
        if len(shape) < 4:
            return None
        n = int(shape[3])
        if n < 3:
            return None
        zooms = img.header.get_zooms()
        dt = float(zooms[3]) if len(zooms) >= 4 else 1.0
        if not np.isfinite(dt) or dt <= 0:
            dt = 1.0
        return (np.arange(n, dtype=float) * dt).astype(float)
    except Exception:
        return None




def _analysis_map_volumes(subject: Subject) -> List[Dict[str, Any]]:
    # Use the same robust on-disk scan as local-dir mode so we pick up newer
    # pipeline outputs (e.g. *_patlak / *_tikhonov).
    return _analysis_map_volumes_from_dir(Path(subject.sourcePath))


def _analysis_map_volumes_from_dir(subject_dir: Path) -> List[Dict[str, Any]]:
    analysis_dir = subject_dir.expanduser().resolve() / "Analysis"
    if not analysis_dir.exists():
        return []

    def base_no_ext(name: str) -> str:
        n = name
        if n.lower().endswith(".nii.gz"):
            return n[:-7]
        if n.lower().endswith(".nii"):
            return n[:-4]
        return n

    diffusion_keys = {"fa", "md", "ad", "rd", "mo"}

    def infer_group(p: Path) -> str:
        parts = {x.lower() for x in p.parts}
        if "diffusion" in parts:
            return "diffusion"
        b = base_no_ext(p.name).lower()
        if b in diffusion_keys or b.startswith("tensor_") or b.endswith("_tensor") or b.startswith("fa_"):
            return "diffusion"
        return "modelling"

    def infer_unit(base: str) -> str:
        b = base.lower()
        if b.startswith("ki"):
            return "ml/100g/min"
        if b.startswith("cbf"):
            return "ml/100g/min"
        if b.startswith("mtt"):
            return "s"
        if b.startswith("cth"):
            return "s"
        if b.startswith("vp"):
            return "fraction"
        if b.startswith("fa"):
            return "fraction"
        # md/ad/rd units vary; keep empty to avoid lying
        return ""

    # Collect any NIfTI volumes under Analysis (including diffusion/).
    files: List[Path] = []
    for p in analysis_dir.rglob("*.nii"):
        files.append(p)
    for p in analysis_dir.rglob("*.nii.gz"):
        files.append(p)

    out: List[Dict[str, Any]] = []
    seen: set[str] = set()
    for p in sorted(set(files)):
        if not p.is_file():
            continue
        if p.name.startswith("._") or p.name == ".DS_Store":
            continue
        # Skip noisy debug outputs.
        if "native_debug" in p.name.lower():
            continue

        base = base_no_ext(p.name)
        base_l = base.lower()

        # Only expose validated model outputs.
        if (
            "two_compartment" in base_l
            or "patlak_tikhonov" in base_l
            or "patlak_then_tikhonov" in base_l
            or "tikhonov_fast" in base_l
            or "tik_fast" in base_l
            or "tikfast" in base_l
        ):
            continue
        key = str(p)
        if key in seen:
            continue
        seen.add(key)

        group = infer_group(p)
        unit = infer_unit(base)
        out.append(
            {
                "id": base,
                "name": base,
                "unit": unit,
                "path": str(p),
                "group": group,
            }
        )

    # Stable ordering: modelling first, then diffusion; then alphabetical.
    out.sort(key=lambda m: (0 if m.get("group") == "modelling" else 1, str(m.get("name") or "")))
    return out


def _montage_images_from_dir(subject_dir: Path) -> List[Dict[str, Any]]:
    montage_dir = subject_dir.expanduser().resolve() / "Images" / "AI" / "Montages"
    if not montage_dir.exists():
        return []
    images: List[Dict[str, Any]] = []
    for p in sorted(montage_dir.glob("*.png")):
        if not p.is_file():
            continue
        if p.name.startswith("._") or p.name == ".DS_Store":
            continue
        images.append({"id": p.stem, "name": p.name, "path": str(p)})
    return images


def _montage_dir_for_subject(subject: Subject) -> Path:
    return Path(subject.sourcePath).expanduser().resolve() / "Images" / "AI" / "Montages"


def _subject_montage_images(subject: Subject) -> List[Dict[str, Any]]:
    montage_dir = _montage_dir_for_subject(subject)
    if not montage_dir.exists():
        return []
    images: List[Dict[str, Any]] = []
    for p in sorted(montage_dir.glob("*.png")):
        if not p.is_file():
            continue
        images.append(
            {
                "id": p.stem,
                "name": p.name,
                "path": str(p),
            }
        )
    return images


def _load_aif_curve(subject: Subject) -> Any:
    _require_numpy()
    assert np is not None
    analysis_dir = _analysis_dir_for_subject(subject)

    # Prefer p-brain's time-shifted & rescaled VIF (TSCC Data/Max) when available.
    tscc_dir = analysis_dir / "TSCC Data" / "Max"
    if tscc_dir.exists() and tscc_dir.is_dir():
        files = sorted(
            [p for p in tscc_dir.glob("*.npy") if p.is_file() and not p.name.startswith(".")]
        )
        if files:
            c = np.asarray(np.load(str(files[0])), dtype=float).reshape(-1)
            if c.size >= 3 and np.all(np.isfinite(c) | np.isnan(c)):
                return np.nan_to_num(c, nan=0.0, posinf=0.0, neginf=0.0)

    artery_dir = analysis_dir / "CTC Data" / "Artery"
    if not artery_dir.exists():
        raise HTTPException(status_code=404, detail="Missing input function outputs")

    def pick_curve(dir_path: Path) -> Optional[Path]:
        shifted = sorted(dir_path.glob("CTC_shifted_slice_*.npy"))
        if shifted:
            return shifted[0]
        raw = sorted(dir_path.glob("CTC_slice_*.npy"))
        return raw[0] if raw else None

    for subtype_dir in sorted([d for d in artery_dir.iterdir() if d.is_dir()]):
        pth = pick_curve(subtype_dir)
        if pth and pth.exists():
            c = np.asarray(np.load(str(pth)), dtype=float).reshape(-1)
            if c.size >= 3 and np.all(np.isfinite(c)):
                return c
    raise HTTPException(status_code=404, detail="No AIF curve found")


def _load_ai_tissue_curve(subject: Subject, region_key: str) -> Any:
    _require_numpy()
    assert np is not None
    analysis_dir = _analysis_dir_for_subject(subject)
    ai_dir = analysis_dir / "CTC Data" / "Tissue" / "AI"
    if ai_dir.exists():
        patt = f"{region_key}_AI_Tissue_slice_*_segmented_median.npy"
        files = sorted(ai_dir.glob(patt))
        if files:
            curves: List[Any] = []
            for p in files:
                arr = np.asarray(np.load(str(p)), dtype=float).reshape(-1)
                if arr.size >= 3 and np.all(np.isfinite(arr) | np.isnan(arr)):
                    curves.append(arr)
            if curves:
                min_len = min(c.size for c in curves)
                stack = np.stack([c[:min_len] for c in curves], axis=0)
                return np.nanmedian(stack, axis=0)

    # Fallback: first available tissue curve.
    tissue_dir = analysis_dir / "CTC Data" / "Tissue"
    if tissue_dir.exists():
        for subtype_dir in sorted([d for d in tissue_dir.iterdir() if d.is_dir()]):
            pth = sorted(subtype_dir.glob("CTC_slice_*.npy"))
            if pth:
                arr = np.asarray(np.load(str(pth[0])), dtype=float).reshape(-1)
                if arr.size >= 3:
                    return arr
    raise HTTPException(status_code=404, detail="No tissue curve found")


def _align_curves(t: Any, aif: Any, tissue: Any) -> tuple[Any, Any, Any, float]:
    _require_numpy()
    assert np is not None
    n = int(min(t.size, aif.size, tissue.size))
    t = t[:n]
    aif = aif[:n]
    tissue = tissue[:n]
    if n < 3:
        raise HTTPException(status_code=400, detail="Insufficient samples")
    # Use first positive delta as dt.
    deltas = np.diff(t)
    deltas = deltas[np.isfinite(deltas) & (deltas > 0)]
    if deltas.size == 0:
        raise HTTPException(status_code=400, detail="Invalid time axis")
    dt = float(deltas[0])
    return t, aif, tissue, dt


def _patlak_from_curves(t: Any, aif: Any, tissue: Any, *, window_start_fraction: float) -> Dict[str, Any]:
    _require_numpy()
    assert np is not None

    # Patlak coordinates
    dt = np.diff(t)
    x = np.concatenate(([0.0], np.cumsum(aif[:-1] * dt)))
    with np.errstate(divide="ignore", invalid="ignore"):
        x = x / aif
        y = tissue / aif

    good = np.isfinite(x) & np.isfinite(y) & (aif != 0)
    if good.sum() < 3:
        raise HTTPException(status_code=400, detail="Not enough valid Patlak points")

    x_max = float(np.nanmax(x[good]))
    ws = float(window_start_fraction)
    if not (0.0 < ws < 1.0):
        ws = 1.0 / 3.0
    window = (x >= ws * x_max) & (x <= x_max)
    fit_mask = good & window
    if fit_mask.sum() < 3:
        fit_mask = good

    xv = x[fit_mask]
    yv = y[fit_mask]
    xm = float(np.mean(xv))
    ym = float(np.mean(yv))
    denom = float(np.sum((xv - xm) ** 2))
    if denom <= 0:
        raise HTTPException(status_code=400, detail="Degenerate Patlak fit")
    slope = float(np.sum((xv - xm) * (yv - ym)) / denom)
    intercept = float(ym - slope * xm)

    yhat = intercept + slope * xv
    ss_res = float(np.sum((yv - yhat) ** 2))
    ss_tot = float(np.sum((yv - float(np.mean(yv))) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")

    x_line = [float(np.nanmin(xv)), float(np.nanmax(xv))]
    y_line = [intercept + slope * x_line[0], intercept + slope * x_line[1]]

    return {
        "x": x.astype(float).tolist(),
        "y": y.astype(float).tolist(),
        "Ki": float(slope * 6000.0),
        "vp": float(intercept),
        "r2": float(r2),
        "fitLineX": x_line,
        "fitLineY": [float(v) for v in y_line],
        "windowStart": float(ws),
    }


def _deconvolution_from_curves(
    t: Any,
    aif: Any,
    tissue: Any,
    *,
    dt: float,
    lambd: float,
    hematocrit: float,
    tissue_density: float,
) -> Dict[str, Any]:
    _require_numpy()
    assert np is not None

    n = int(min(t.size, aif.size, tissue.size))
    t = t[:n]
    aif = aif[:n]
    tissue = tissue[:n]

    # Build convolution matrix A (Toeplitz, trapezoidal)
    A = np.zeros((n, n), dtype=float)
    for i in range(n):
        A[i, : i + 1] = aif[i::-1] * dt
        if i == 0:
            A[i, 0] = 0.0
        else:
            A[i, 0] *= 0.5
            A[i, i] *= 0.5

    lam = float(max(lambd, 0.0))
    ata = A.T @ A
    regularised = ata + (lam**2) * np.eye(n, dtype=float)
    rhs = A.T @ tissue
    try:
        impulse = np.linalg.solve(regularised, rhs)
    except np.linalg.LinAlgError:
        impulse = np.linalg.lstsq(regularised, rhs, rcond=None)[0]

    g0 = float(impulse[0]) if impulse.size else float("nan")
    if not np.isfinite(g0) or g0 <= 0.0:
        residue = np.full_like(impulse, np.nan)
        cbf = float("nan")
    else:
        residue = impulse / g0
        # AIF is plasma; convert to whole-blood flow (1 - Hct)
        scale = 1.0 - float(np.clip(hematocrit, 0.0, 0.9))
        cbf = 6000.0 * g0 * scale / float(max(tissue_density, 1e-6))

    # MTT from residue
    mtt = float(np.trapz(np.clip(residue, 0.0, None), dx=dt)) if np.all(np.isfinite(residue)) else float("nan")

    # CTH from h(t) = -d/dt residue
    if np.all(np.isfinite(residue)) and residue.size >= 5:
        h = np.maximum(0.0, -np.gradient(np.clip(residue, 0.0, None), dt, edge_order=2))
        area = float(np.trapz(h, dx=dt))
        if area > 0:
            h = h / area
            mu = float(np.trapz(t * h, dx=dt))
            var = float(np.trapz(((t - mu) ** 2) * h, dx=dt))
            cth = float(np.sqrt(max(var, 0.0)))
        else:
            h = np.full_like(residue, np.nan)
            cth = float("nan")
    else:
        h = np.full_like(residue, np.nan)
        cth = float("nan")

    return {
        "timePoints": t.astype(float).tolist(),
        "residue": residue.astype(float).tolist(),
        "h_t": h.astype(float).tolist(),
        "CBF": float(cbf),
        "MTT": float(mtt),
        "CTH": float(cth),
    }


def _stage_jobs_for_subject(subject_id: str) -> Dict[StageId, Job]:
    # Return most-recent stage jobs for the subject from the current run (kept in-memory).
    job_ids = db._subject_job_ids.get(subject_id, [])
    jobs: Dict[StageId, Job] = {}
    for jid in job_ids:
        job = db._job_by_id.get(jid)
        if job:
            jobs[job.stageId] = job
    return jobs


def _set_stage_status(subject: Subject, stage: StageId, status: StageStatus) -> None:
    subject.stageStatuses[str(stage)] = status
    subject.updatedAt = _now_iso()


def _set_job(job: Job, *, status: JobStatus, progress: int, step: str, error: Optional[str] = None) -> None:
    job.status = status
    job.progress = int(progress)
    job.currentStep = step
    if error:
        job.error = error
    db._touch_jobs()


def _require_stage_dependencies(subject: Subject, stage: StageId) -> None:
    deps = STAGE_DEPENDENCIES.get(stage, [])
    for dep in deps:
        if subject.stageStatuses.get(str(dep)) != "done":
            raise HTTPException(status_code=409, detail=f"Stage '{stage}' requires '{dep}' to be done")


def _pbrain_stage_cli_args(*, data_root: Path, subject: Subject, stage: StageId) -> List[str]:
    main_py = _resolve_pbrain_main_py()
    if not main_py:
        raise RuntimeError("p-Brain path is not set. Set PBRAIN_MAIN_PY or configure it in Settings.")
    main_py_path = Path(main_py).expanduser().resolve()
    if not main_py_path.exists():
        raise RuntimeError(f"p-Brain main.py not found: {main_py_path}")

    runner_path = _ensure_stage_runner_script(data_root, force=True)
    args: List[str] = [
        _pbrain_python_executable(),
        "-u",
        str(runner_path),
        "--stage",
        str(stage),
        "--id",
        str(subject.name),
        "--data-dir",
        str(data_root),
    ]
    if stage == "diffusion":
        args.append("--diffusion")
    return args


def _pbrain_stage_cli_args_with_python(*, python_exe: str, data_root: Path, subject: Subject, stage: StageId) -> List[str]:
    main_py = _resolve_pbrain_main_py()
    if not main_py:
        raise RuntimeError("p-Brain path is not set. Set PBRAIN_MAIN_PY or configure it in Settings.")
    main_py_path = Path(main_py).expanduser().resolve()
    if not main_py_path.exists():
        raise RuntimeError(f"p-Brain main.py not found: {main_py_path}")

    runner_path = _ensure_stage_runner_script(data_root, force=True)
    args: List[str] = [
        python_exe,
        "-u",
        str(runner_path),
        "--stage",
        str(stage),
        "--id",
        str(subject.name),
        "--data-dir",
        str(data_root),
    ]
    if stage == "diffusion":
        args.append("--diffusion")
    return args


def _project_t1m0_force_enabled(project: Project) -> bool:
    try:
        cfg = project.config if isinstance(project.config, dict) else {}
        pb = cfg.get("pbrain") if isinstance(cfg.get("pbrain"), dict) else {}
        return bool(pb.get("t1m0Force", True))
    except Exception:
        return True


async def _run_pbrain_single_stage(*, project: Project, subject: Subject, job: Job, env_overrides: Optional[Dict[str, str]] = None) -> None:
    async with _RUN_SEMAPHORE:
        data_root = Path(project.storagePath).expanduser().resolve()
        log_fh = None
        proc: asyncio.subprocess.Process | None = None

        try:
            if not data_root.exists():
                raise RuntimeError(f"Project storagePath/data root does not exist: {data_root}")

            # Log file for this stage run.
            # Prefer storing logs under the subject folder so artifacts are self-contained.
            logs_dir: Path
            try:
                logs_dir = Path(subject.sourcePath).expanduser().resolve() / ".pbrain-web" / "logs"
                logs_dir.mkdir(parents=True, exist_ok=True)
            except Exception:
                logs_dir = data_root / ".pbrain-web" / "logs"
                logs_dir.mkdir(parents=True, exist_ok=True)
            log_path = logs_dir / f"stage_{job.stageId}_{subject.id}_{int(datetime.utcnow().timestamp()*1000)}.log"
            job.logPath = str(log_path)
            if not job.startTime:
                job.startTime = _now_iso()
            db._touch_jobs()

            log_fh = log_path.open("a", encoding="utf-8", errors="replace")

            def write_line_sync(line: str) -> None:
                if not log_fh:
                    return
                log_fh.write(line)
                log_fh.flush()

            write_line_sync(f"[{_now_iso()}] Stage run created\n")
            write_line_sync(f"[{_now_iso()}] Subject: {subject.name} ({subject.id})\n")
            write_line_sync(f"[{_now_iso()}] Stage: {job.stageId}\n")

            # Mark stage/job running.
            _set_job(job, status="running", progress=5, step="Starting")
            _set_stage_status(subject, job.stageId, "running")
            db.save()

            python_exe = await _select_pbrain_python(write_line_sync=write_line_sync)
            args = _pbrain_stage_cli_args_with_python(
                python_exe=python_exe,
                data_root=data_root,
                subject=subject,
                stage=job.stageId,
            )

            # Respect project-level T1/M0 fit mode selection even if global app
            # settings/environment differ.
            try:
                cfg = project.config if isinstance(project.config, dict) else {}
                pb = cfg.get("pbrain") if isinstance(cfg.get("pbrain"), dict) else {}
                t1_fit = str(pb.get("t1Fit") or "").strip().lower()
                if t1_fit in {"auto", "ir", "vfa", "none"}:
                    args += ["--t1-fit", t1_fit]
                # Project-level default: always do a fresh T1/M0 fit unless user disables it.
                if job.stageId == "t1_fit" and bool(pb.get("t1m0Force", True)):
                    args.append("--t1m0-force")
            except Exception:
                pass

            # Keep stage-runner behavior consistent with the full p-brain CLI.
            # In particular, voxelwise maps can be extremely expensive; only enable when configured.
            if job.stageId == "modelling":
                compute_ki, compute_cbf = _stage_runner_voxelwise_flags(project)
                if compute_ki:
                    args.append("--voxelwise")
                if compute_cbf:
                    args.append("--cbf")
            write_line_sync(f"[{_now_iso()}] ARGS: {args!r}\n")

            main_py = _resolve_pbrain_main_py()
            pbrain_cwd = str(Path(main_py).expanduser().resolve().parent) if main_py else None
            if pbrain_cwd:
                write_line_sync(f"[{_now_iso()}] CWD: {pbrain_cwd}\n")

            env = _apply_project_pbrain_env_overrides(_python_env_for_pbrain(), project)
            env = _apply_pbrain_env_overrides(env, env_overrides)

            # Ensure AI model paths are resolved consistently for p-brain runs.
            # This matters in the packaged app where the p-brain repo clone may not
            # include the large Zenodo-provided model binaries.
            try:
                pbrain_root = Path(pbrain_cwd).expanduser().resolve() if pbrain_cwd else None
            except Exception:
                pbrain_root = None
            env = _apply_ai_model_overrides(env, pbrain_root=pbrain_root, write_line_sync=write_line_sync)
            if job.stageId == "input_functions":
                _preflight_ai_models(env, pbrain_root=pbrain_root, write_line_sync=write_line_sync)
                await _preflight_pbrain_ai_deps(
                    python_exe,
                    env=env,
                    cwd=pbrain_cwd,
                    write_line_sync=write_line_sync,
                )
            if job.stageId in {"diffusion", "tractography", "connectome"}:
                await _preflight_pbrain_diffusion_deps(
                    python_exe,
                    env=env,
                    cwd=pbrain_cwd,
                    write_line_sync=write_line_sync,
                )
            try:
                keys = [
                    "P_BRAIN_CTC_MODEL",
                    "P_BRAIN_TURBOFLASH_CTC_METHOD",
                    "P_BRAIN_T1_FIT",
                    "P_BRAIN_TURBO_NPH",
                    "P_BRAIN_NUMBER_OF_PEAKS",
                    "P_BRAIN_CTC_PEAK_RESCALE_THRESHOLD",
                    "P_BRAIN_MODELLING_INPUT_FUNCTION",
                    "P_BRAIN_AIF_USE_SSS",
                    "P_BRAIN_AIF_ARTERY",
                    "PBRAIN_AUTO_ARTERY",
                    "P_BRAIN_VASCULAR_ROI_CURVE_METHOD",
                    "P_BRAIN_VASCULAR_ROI_ADAPTIVE_MAX",
                    "P_BRAIN_AI_ICA_FRAME_MODE",
                    "P_BRAIN_AI_ICA_FRAME_OVERRIDE",
                    "P_BRAIN_AI_ROI_POSTPROCESS",
                    "P_BRAIN_TSCC_SKIP_FORKED_PEAKS",
                    "P_BRAIN_CTC_MAPS",
                    "P_BRAIN_CTC_4D",
                    "P_BRAIN_CTC_MAP_SLICE",
                    "P_BRAIN_STRICT_METADATA",
                    "P_BRAIN_MULTIPROCESSING",
                    "P_BRAIN_CORES",
                ]
                summary = " ".join(f"{k}={env.get(k)!r}" for k in keys if k in env)
                write_line_sync(f"[{_now_iso()}] ENV: {summary}\n")
            except Exception:
                pass

            proc = await asyncio.create_subprocess_exec(
                *args,
                # In daemonized contexts stdin can be an invalid FD (especially when
                # launched by a GUI host). Make stdin explicit to keep Python stable.
                stdin=asyncio.subprocess.DEVNULL,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
                cwd=pbrain_cwd,
                env=env,
            )
            db._job_processes[job.id] = proc

            assert proc.stdout is not None
            last_output = time.monotonic()
            while True:
                try:
                    line_b = await asyncio.wait_for(proc.stdout.readline(), timeout=60.0)
                except asyncio.TimeoutError:
                    quiet_s = max(0, int(time.monotonic() - last_output))
                    pid = getattr(proc, "pid", None)
                    pid_s = f" pid={pid}" if pid else ""
                    write_line_sync(f"[{_now_iso()}] (still running; no output for {quiet_s}s){pid_s}\n")
                    if proc.returncode is not None:
                        break
                    continue
                if not line_b:
                    break
                line = line_b.decode(errors="replace")
                write_line_sync(line)
                last_output = time.monotonic()

            rc = await proc.wait()
            if job.status == "cancelled":
                return
            if rc == PBRAIN_WAITING_FOR_ROI_EXIT_CODE:
                _set_job(job, status="completed", progress=100, step="Waiting for user ROI")
                job.endTime = _now_iso()
                _set_stage_status(subject, job.stageId, "waiting")
                db.save()
                return
            if rc != 0:
                raise RuntimeError(f"Stage '{job.stageId}' exited with code {rc}")

            _set_job(job, status="completed", progress=100, step="Completed")
            job.endTime = _now_iso()
            _set_stage_status(subject, job.stageId, "done")
            db.save()

        except Exception as exc:
            err = str(exc)
            if job.status not in {"cancelled", "completed"}:
                _set_job(job, status="failed", progress=int(job.progress or 0), step="Failed", error=err)
                job.endTime = _now_iso()
                _set_stage_status(subject, job.stageId, "failed")
                db.save()
            raise

        finally:
            try:
                if log_fh:
                    log_fh.close()
            except Exception:
                pass
            db._job_processes.pop(job.id, None)
            db._job_tasks.pop(job.id, None)


def _pbrain_python_executable() -> str:
    """Pick the interpreter used to run p-brain.

    In PyInstaller builds, `sys.executable` is the frozen backend binary, not a
    Python interpreter. Use a real python (prefer `python3`) in that case.
    """

    override = (os.environ.get("PBRAIN_PYTHON") or "").strip()
    if override:
        # Allow either an absolute path or a command discoverable on PATH.
        if os.path.sep in override or override.startswith("/"):
            if Path(override).exists():
                return override
            raise RuntimeError(f"PBRAIN_PYTHON not found: {override}")
        resolved = shutil.which(override)
        if resolved:
            return resolved
        raise RuntimeError(f"PBRAIN_PYTHON not found on PATH: {override}")

    if bool(getattr(sys, "frozen", False)):
        # Prefer Homebrew python over Apple's /usr/bin/python3 to avoid
        # accidentally picking up an older system python with incompatible
        # user-site packages.
        for candidate in [
            "/opt/homebrew/bin/python3",
            "/usr/local/bin/python3",
        ]:
            if Path(candidate).exists():
                return candidate

        resolved = shutil.which("python3")
        if resolved:
            return resolved

        if Path("/usr/bin/python3").exists():
            return "/usr/bin/python3"
        raise RuntimeError(
            "Python 3 is required to run p-brain from the packaged app. "
            "Install python3 (or set PBRAIN_PYTHON to a valid interpreter)."
        )

    return sys.executable


async def _select_pbrain_python(*, write_line_sync=None) -> str:
    """Pick a Python interpreter that can run p-brain.

    Tries PBRAIN_PYTHON (if set), otherwise probes a small set of likely
    interpreters and returns the first one that passes dependency preflight.
    """

    override = (os.environ.get("PBRAIN_PYTHON") or "").strip()
    candidates: List[str] = []

    def add_candidate(p: Optional[str]) -> None:
        if not p:
            return
        if p not in candidates:
            candidates.append(p)

    managed_fail: Optional[str] = None

    if override:
        if os.path.sep in override or override.startswith("/"):
            add_candidate(override)
        else:
            add_candidate(shutil.which(override))
    else:
        # If p-brain is configured to point at a repo checkout that already has a
        # virtualenv (common for dev/power-users), prefer that interpreter so the
        # app uses the exact same deps and behavior as running p-brain from CLI.
        try:
            main_py = _resolve_pbrain_main_py()
            if main_py:
                root = Path(main_py).expanduser().resolve().parent
                for candidate in (
                    root / ".venv" / "bin" / "python3",
                    root / ".venv" / "bin" / "python",
                    root / "venv" / "bin" / "python3",
                    root / "venv" / "bin" / "python",
                ):
                    try:
                        if candidate.exists() and candidate.is_file():
                            await _preflight_pbrain_python(str(candidate))
                            return str(candidate)
                    except Exception:
                        continue
        except Exception:
            pass

        # Next choice: app-managed venv (auto-installs deps if missing).
        try:
            if write_line_sync is None:
                def write_line_sync(_line: str) -> None:
                    return
            venv_python = await _ensure_managed_pbrain_venv(write_line_sync=write_line_sync)
            await _preflight_pbrain_python(venv_python)
            return venv_python
        except Exception as exc:
            managed_fail = str(exc)
            # In packaged builds, we don't want to silently fall back to a system
            # python that likely lacks numpy/nibabel. Surface the real problem.
            try:
                if write_line_sync is not None:
                    write_line_sync(f"[{_now_iso()}] Managed venv setup failed: {exc}\n")
            except Exception:
                pass
            if bool(getattr(sys, "frozen", False)):
                raise RuntimeError(
                    "Managed Python environment setup failed. "
                    "This app creates and uses its own venv for p-brain dependencies; "
                    "see the job log above for the pip/install error details."
                ) from exc
            # Dev mode: fall back to probing system interpreters.

        def _conda_python_candidates() -> List[str]:
            out: List[str] = []

            # Common conda roots on macOS (especially when launched from a GUI
            # host where PATH won't include conda shims).
            roots = [
                Path("/opt/homebrew/Caskroom/miniconda/base"),
                Path("/opt/homebrew/Caskroom/miniforge/base"),
                Path("/opt/homebrew/anaconda3"),
                Path.home() / "miniconda3",
                Path.home() / "miniforge3",
                Path.home() / "anaconda3",
            ]

            preferred_envs = [
                "tf_macos",
                "pbrain",
                "p-brain",
            ]

            for root in roots:
                try:
                    # Base python (if present)
                    for p in [root / "bin" / "python3", root / "bin" / "python"]:
                        if p.exists():
                            out.append(str(p))

                    envs_dir = root / "envs"
                    if envs_dir.exists() and envs_dir.is_dir():
                        # Preferred envs first.
                        for name in preferred_envs:
                            for p in [envs_dir / name / "bin" / "python3", envs_dir / name / "bin" / "python"]:
                                if p.exists():
                                    out.append(str(p))
                        # Then any other env.
                        for p in envs_dir.glob("*/bin/python3"):
                            if p.exists():
                                out.append(str(p))
                        for p in envs_dir.glob("*/bin/python"):
                            if p.exists():
                                out.append(str(p))
                except Exception:
                    continue

            # De-dup while preserving order.
            deduped: List[str] = []
            for p in out:
                if p not in deduped:
                    deduped.append(p)
            return deduped


        if bool(getattr(sys, "frozen", False)):
            add_candidate("/opt/homebrew/bin/python3")
            add_candidate("/usr/local/bin/python3")
            add_candidate(shutil.which("python3"))
            add_candidate("/usr/bin/python3")
        else:
            add_candidate(sys.executable)
            add_candidate(shutil.which("python3"))
            for p in _conda_python_candidates():
                add_candidate(p)

    # Filter to existing executables.
    filtered: List[str] = []
    for c in candidates:
        if c.startswith("/"):
            if Path(c).exists():
                filtered.append(c)
        else:
            resolved = shutil.which(c)
            if resolved:
                filtered.append(resolved)
    if not filtered:
        raise RuntimeError(
            "No usable python interpreter found for p-brain. "
            "Install python3 or set PBRAIN_PYTHON to a valid interpreter."
        )

    failures: List[str] = []
    for python_exe in filtered:
        try:
            await _preflight_pbrain_python(python_exe)
            return python_exe
        except Exception as exc:
            failures.append(f"- {python_exe}: {exc}")

    if managed_fail:
        failures.append(f"- managed venv: {managed_fail}")

    raise RuntimeError(
        "No available Python interpreter passed dependency preflight.\n"
        + "\n".join(failures)
        + "\n\nFix: install numpy+nibabel into one of the above interpreters (or create a venv) and point PBRAIN_PYTHON at it."
    )


_PYTHON_PREFLIGHT: Dict[str, Tuple[bool, str, float]] = {}


async def _preflight_pbrain_python(python_exe: str) -> None:
    """Validate the selected interpreter can import core deps.

    This catches common mixed-site failures (e.g., numpy>=2 with old nibabel)
    and provides a concise actionable message.
    """

    now = time.time()
    cached = _PYTHON_PREFLIGHT.get(python_exe)
    if cached and (now - cached[2]) < 300:
        ok, msg, _ts = cached
        if ok:
            return
        raise RuntimeError(msg)

    probe = (
        "import sys; "
        "import numpy as np; "
        "import nibabel as nib; "
        "import modules; "
        "print(sys.version.split()[0]); "
        "print(np.__version__); "
        "print(nib.__version__)"
    )

    try:
        proc = await asyncio.create_subprocess_exec(
            python_exe,
            "-c",
            probe,
            # In packaged/daemonized contexts stdin can be an invalid FD, which can
            # crash Python at startup (init_sys_streams). Make stdin explicit.
            stdin=asyncio.subprocess.DEVNULL,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=_python_env_for_pbrain(),
        )
        out_b, err_b = await proc.communicate()
        out = (out_b or b"").decode(errors="replace").strip()
        err = (err_b or b"").decode(errors="replace").strip()
        if proc.returncode != 0:
            msg = (
                "Selected Python for p-brain failed dependency preflight.\n"
                f"Python: {python_exe}\n"
                f"Error: {err or out}\n\n"
                "Fix: install compatible deps into that interpreter, or set PBRAIN_PYTHON to an interpreter that already has them.\n"
                "If you see `np.sctypes`/NumPy 2.0 errors, upgrade nibabel or pin numpy<2 in that env."
            )
            _PYTHON_PREFLIGHT[python_exe] = (False, msg, now)
            raise RuntimeError(msg)

        _PYTHON_PREFLIGHT[python_exe] = (True, out, now)
    except Exception as exc:
        msg = f"Python preflight failed for {python_exe}: {exc}"
        _PYTHON_PREFLIGHT[python_exe] = (False, msg, now)
        raise


async def _preflight_pbrain_ai_deps(python_exe: str, *, env: Dict[str, str], cwd: Optional[str], write_line_sync=None) -> None:
    """Validate AI ROI dependencies (tensorflow + cv2) for input-function extraction.

    If running from the app-managed venv, missing deps are installed on-demand.
    """

    # If the user explicitly chose deterministic or file-based ROIs, skip.
    roi_method = (env.get("P_BRAIN_ROI_METHOD") or env.get("ROI_METHOD") or "ai").strip().lower()
    if roi_method in {"deterministic", "geometry", "file"}:
        return

    probe = (
        "import sys; "
        "import tensorflow as tf; "
        "import cv2; "
        "print(sys.version.split()[0]); "
        "print(getattr(tf, '__version__', 'unknown')); "
        "print(getattr(cv2, '__version__', 'unknown'))"
    )

    async def _probe_once() -> str:
        proc = await asyncio.create_subprocess_exec(
            python_exe,
            "-c",
            probe,
            # In packaged/daemonized contexts stdin can be an invalid FD, which can
            # crash Python at startup (init_sys_streams). Make stdin explicit.
            stdin=asyncio.subprocess.DEVNULL,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=cwd,
            env=env,
        )
        out_b, err_b = await proc.communicate()
        out = (out_b or b"").decode(errors="replace").strip()
        err = (err_b or b"").decode(errors="replace").strip()
        if proc.returncode != 0:
            raise RuntimeError(err or out or "unknown error")
        return out

    try:
        out = await _probe_once()
        try:
            if write_line_sync is not None:
                write_line_sync(f"[{_now_iso()}] AI deps OK: {out.replace(chr(10), ' | ')}\n")
        except Exception:
            pass
        return
    except Exception as exc:
        # On-demand install for the managed venv.
        try:
            if _is_managed_pbrain_venv_python(python_exe):
                python_exe2 = await _ensure_managed_pbrain_optional_deps(
                    kind="ai",
                    write_line_sync=write_line_sync,
                )
                # If the managed venv was recreated with a different Python version,
                # re-run the probe with the updated interpreter.
                python_exe = python_exe2
                out = await _probe_once()
                try:
                    if write_line_sync is not None:
                        write_line_sync(
                            f"[{_now_iso()}] AI deps OK (after install): {out.replace(chr(10), ' | ')}\n"
                        )
                except Exception:
                    pass
                return
        except Exception as install_exc:
            raise RuntimeError(
                "AI input-function extraction requires TensorFlow and OpenCV. "
                f"Automatic install into managed venv failed: {install_exc}"
            ) from install_exc

        raise RuntimeError(
            "AI input-function extraction requires TensorFlow and OpenCV in the selected Python environment. "
            f"Python: {python_exe}. Error: {exc}. "
            "Fix: run the app's 'Install p-brain requirements' (managed venv) or point PBRAIN_PYTHON at an env with tensorflow-macos/tensorflow-metal + opencv-python installed."
        )


async def _preflight_pbrain_diffusion_deps(python_exe: str, *, cwd: Optional[str], env: Dict[str, str], write_line_sync=None) -> None:
    """Validate diffusion/tractography deps (dipy) and install on-demand for managed venv."""

    probe = "import dipy; print(getattr(dipy, '__version__', 'unknown'))"

    async def _probe_once() -> str:
        proc = await asyncio.create_subprocess_exec(
            python_exe,
            "-c",
            probe,
            stdin=asyncio.subprocess.DEVNULL,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=cwd,
            env=env,
        )
        out_b, err_b = await proc.communicate()
        out = (out_b or b"").decode(errors="replace").strip()
        err = (err_b or b"").decode(errors="replace").strip()
        if proc.returncode != 0:
            raise RuntimeError(err or out or "unknown error")
        return out

    try:
        ver = await _probe_once()
        try:
            if write_line_sync is not None:
                write_line_sync(f"[{_now_iso()}] Diffusion deps OK: dipy {ver}\n")
        except Exception:
            pass
        return
    except Exception as exc:
        try:
            if _is_managed_pbrain_venv_python(python_exe):
                await _ensure_managed_pbrain_optional_deps(kind="diffusion", write_line_sync=write_line_sync)
                ver = await _probe_once()
                try:
                    if write_line_sync is not None:
                        write_line_sync(f"[{_now_iso()}] Diffusion deps OK (after install): dipy {ver}\n")
                except Exception:
                    pass
                return
        except Exception as install_exc:
            raise RuntimeError(
                "Diffusion/tractography requires dipy. "
                f"Automatic install into managed venv failed: {install_exc}"
            ) from install_exc
        raise RuntimeError(
            "Diffusion/tractography requires dipy in the selected Python environment. "
            f"Python: {python_exe}. Error: {exc}."
        )


def _pbrain_root_dir() -> Path:
    main_py = os.environ.get("PBRAIN_MAIN_PY") or _resolve_pbrain_main_py()
    if not main_py:
        raise RuntimeError("p-Brain path is not set. Set PBRAIN_MAIN_PY or configure it in Settings.")
    p = Path(main_py).expanduser().resolve()
    if not p.exists():
        raise RuntimeError(f"p-Brain main.py not found: {p}")
    return p.parent


def _managed_pbrain_venv_dir() -> Path:
    return _user_data_dir() / "pbrain-venv"


def _requirements_fingerprint(req_path: Path) -> str:
    try:
        data = req_path.read_bytes()
    except Exception:
        return "missing"
    return hashlib.sha256(data).hexdigest()


def _managed_venv_python_path() -> Path:
    return _managed_pbrain_venv_dir() / "bin" / "python3"


def _is_managed_pbrain_venv_python(python_exe: str) -> bool:
    try:
        p = Path(python_exe).expanduser().resolve()
        return p == _managed_venv_python_path().expanduser().resolve()
    except Exception:
        return False


async def _run_cmd_logged(
    args: List[str],
    *,
    cwd: Optional[str],
    env: Dict[str, str],
    write_line_sync,
) -> None:
    write_line_sync(f"[{_now_iso()}] $ {' '.join(map(str, args))}\n")
    proc = await asyncio.create_subprocess_exec(
        *args,
        # In packaged/daemonized contexts stdin can be an invalid FD, which can
        # crash Python at startup (e.g. when creating a venv). Make stdin explicit.
        stdin=asyncio.subprocess.DEVNULL,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
        cwd=cwd,
        env=env,
    )
    assert proc.stdout is not None
    while True:
        line_b = await proc.stdout.readline()
        if not line_b:
            break
        write_line_sync(line_b.decode(errors="replace"))
    rc = await proc.wait()
    if rc != 0:
        raise RuntimeError(f"Command failed (exit {rc}): {args!r}")


async def _ensure_managed_pbrain_venv(*, write_line_sync, prefer_ai_compatible: bool = False) -> str:
    """Create/update an app-managed venv with p-brain dependencies.

    This makes the packaged app self-contained even if the user has not
    installed python deps globally.
    """

    venv_dir = _managed_pbrain_venv_dir()
    venv_dir.parent.mkdir(parents=True, exist_ok=True)

    # Pick a bootstrap python that exists. Prefer versions compatible with
    # scientific wheels (and tensorflow-macos), avoiding bleeding-edge Python.
    bootstrap = None
    if prefer_ai_compatible:
        candidates = [
            "/opt/homebrew/bin/python3.11",
            "/opt/homebrew/opt/python@3.11/bin/python3.11",
            shutil.which("python3.11"),
            "/opt/homebrew/bin/python3.10",
            "/opt/homebrew/opt/python@3.10/bin/python3.10",
            shutil.which("python3.10"),
            "/opt/homebrew/bin/python3.12",
            "/opt/homebrew/opt/python@3.12/bin/python3.12",
        ]
    else:
        candidates = [
            # Homebrew Python versioned binaries (most reliable)
            "/opt/homebrew/bin/python3.12",
            "/opt/homebrew/opt/python@3.12/bin/python3.12",
            "/opt/homebrew/bin/python3.11",
            "/opt/homebrew/opt/python@3.11/bin/python3.11",
            shutil.which("python3.11"),
            "/opt/homebrew/bin/python3.10",
            "/opt/homebrew/opt/python@3.10/bin/python3.10",
            shutil.which("python3.10"),
        ]

    candidates += [
        # Homebrew Miniconda (GUI apps often lack the user's shell PATH).
        "/opt/homebrew/Caskroom/miniconda/base/bin/python3",
        "/opt/homebrew/Caskroom/miniconda/base/bin/python",
        "/opt/homebrew/anaconda3/bin/python3",
        "/opt/homebrew/anaconda3/bin/python",
        "/usr/local/bin/python3.11",
        "/usr/local/bin/python3.10",
        "/opt/homebrew/bin/python3",
        shutil.which("python3"),
        "/usr/bin/python3",
    ]

    for candidate in candidates:
        if not candidate:
            continue
        if candidate.startswith("/"):
            if Path(candidate).exists():
                bootstrap = candidate
                break
        else:
            resolved = shutil.which(candidate)
            if resolved:
                bootstrap = resolved
                break
    if not bootstrap:
        raise RuntimeError("python3 is required to create the p-brain environment")

    venv_python = venv_dir / "bin" / "python3"

    async def _read_python_version(python_exe: str) -> str:
        proc = await asyncio.create_subprocess_exec(
            python_exe,
            "-c",
            "import sys; print(f'{sys.version_info[0]}.{sys.version_info[1]}')",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=_python_env_for_pbrain(),
        )
        out_b, _err_b = await proc.communicate()
        if proc.returncode != 0:
            return "unknown"
        return (out_b or b"").decode(errors="replace").strip() or "unknown"

    bootstrap_ver = await _read_python_version(bootstrap)
    bootstrap_marker = venv_dir / ".bootstrap.txt"
    existing_bootstrap = bootstrap_marker.read_text(encoding="utf-8", errors="ignore").strip() if bootstrap_marker.exists() else ""

    # If venv exists but was created with a different bootstrap (or unknown), recreate.
    if venv_dir.exists() and existing_bootstrap and existing_bootstrap != f"{bootstrap}|{bootstrap_ver}":
        write_line_sync(f"[{_now_iso()}] Recreating managed venv (bootstrap changed)\n")
        shutil.rmtree(venv_dir, ignore_errors=True)

    if venv_dir.exists() and venv_python.exists():
        venv_ver = await _read_python_version(str(venv_python))
        # Tensorflow/science wheels generally lag; treat 3.13+ as unsupported for managed env.
        try:
            major_s, minor_s = venv_ver.split(".", 1)
            if int(major_s) >= 3 and int(minor_s) >= 13:
                write_line_sync(f"[{_now_iso()}] Recreating managed venv (Python {venv_ver} unsupported)\n")
                shutil.rmtree(venv_dir, ignore_errors=True)
        except Exception:
            pass

    if not venv_python.exists():
        await _run_cmd_logged(
            [bootstrap, "-m", "venv", str(venv_dir)],
            cwd=None,
            env=_python_env_for_pbrain(),
            write_line_sync=write_line_sync,
        )
        bootstrap_marker.write_text(f"{bootstrap}|{bootstrap_ver}", encoding="utf-8")

    # Ensure pip tooling.
    await _run_cmd_logged(
        [str(venv_python), "-m", "pip", "install", "-U", "pip", "setuptools", "wheel"],
        cwd=None,
        env=_python_env_for_pbrain(),
        write_line_sync=write_line_sync,
    )

    pbrain_root = _pbrain_root_dir()
    req_path = pbrain_root / "requirements.txt"
    marker = venv_dir / ".pbrain_requirements.sha256"
    # Version the marker so we can change the install strategy without leaving
    # users with a stale/broken managed venv.
    want_fp = f"core-v1:{_requirements_fingerprint(req_path) if req_path.exists() else 'missing'}"
    have_fp = marker.read_text(encoding="utf-8", errors="ignore").strip() if marker.exists() else ""

    def _split_core_vs_optional_requirements(lines: List[str]) -> Tuple[List[str], List[str], List[str]]:
        """Return (core_lines, ai_lines, diffusion_lines) given raw requirements.txt lines.

        Optional packages are installed on-demand for the managed venv.
        """
        core: List[str] = []
        ai: List[str] = []
        diffusion: List[str] = []

        ai_pkgs = {"tensorflow", "tensorflow-macos", "tensorflow-metal", "opencv-python"}
        diffusion_pkgs = {"dipy"}

        for raw in lines:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            pkg = line.split("==", 1)[0].split(">=", 1)[0].split("<=", 1)[0].split("<", 1)[0].split(">", 1)[0]
            pkg = pkg.strip().lower()
            if pkg in diffusion_pkgs:
                diffusion.append(line)
            elif pkg in ai_pkgs:
                ai.append(line)
            else:
                core.append(line)

        # Handle the legacy 'tensorflow' name on macOS by translating it into
        # tensorflow-macos + tensorflow-metal (but keep it AI-scoped).
        if sys.platform == "darwin":
            normalized_ai: List[str] = []
            saw_tensorflow = any(
                (l.strip().lower().startswith("tensorflow") and not l.strip().lower().startswith("tensorflow-") )
                for l in ai
            )
            for l in ai:
                ll = l.strip().lower()
                if ll == "tensorflow":
                    continue
                normalized_ai.append(l)
            if saw_tensorflow:
                normalized_ai.append("tensorflow-macos")
                normalized_ai.append("tensorflow-metal")
            ai = normalized_ai

        return core, ai, diffusion


    def _write_optional_requirement_files() -> None:
        if not req_path.exists():
            return
        try:
            lines = req_path.read_text(encoding="utf-8", errors="ignore").splitlines()
            core_lines, ai_lines, diffusion_lines = _split_core_vs_optional_requirements(lines)

            core_req = venv_dir / "requirements.core.txt"
            core_req.write_text("\n".join(core_lines) + "\n", encoding="utf-8")

            ai_req = venv_dir / "requirements.ai.txt"
            ai_req.write_text("\n".join(ai_lines) + "\n", encoding="utf-8")

            diff_req = venv_dir / "requirements.diffusion.txt"
            diff_req.write_text("\n".join(diffusion_lines) + "\n", encoding="utf-8")
        except Exception:
            # Best-effort; missing optional files will be regenerated during reinstall.
            return


    async def _install_deps() -> None:
        if req_path.exists():
            # Install a CORE subset first (numpy/nibabel/scipy/etc) so early
            # stages can run even if AI packages are not available for the
            # user's Python version. AI deps are installed on-demand.
            lines = req_path.read_text(encoding="utf-8", errors="ignore").splitlines()
            core_lines, ai_lines, diffusion_lines = _split_core_vs_optional_requirements(lines)

            core_req = venv_dir / "requirements.core.txt"
            core_req.write_text("\n".join(core_lines) + "\n", encoding="utf-8")

            ai_req = venv_dir / "requirements.ai.txt"
            ai_req.write_text("\n".join(ai_lines) + "\n", encoding="utf-8")

            diff_req = venv_dir / "requirements.diffusion.txt"
            diff_req.write_text("\n".join(diffusion_lines) + "\n", encoding="utf-8")

            # On macOS, pip can backtrack into very old scikit-image versions
            # that require building from source with build tooling incompatible
            # with Python 3.12+. Force a known wheel-available version.
            pip_args: List[str] = [str(venv_python), "-m", "pip", "install", "--prefer-binary", "-r", str(core_req)]
            if sys.platform == "darwin":
                constraints = venv_dir / "constraints.darwin.txt"
                constraints.write_text(
                    "\n".join(
                        [
                            "numpy<2",
                            "scikit-image==0.25.2",
                        ]
                    )
                    + "\n",
                    encoding="utf-8",
                )
                pip_args += ["-c", str(constraints)]

            await _run_cmd_logged(
                pip_args,
                cwd=str(pbrain_root),
                env=_python_env_for_pbrain(),
                write_line_sync=write_line_sync,
            )
        else:
            # Fallback minimum deps needed to start p-brain.
            await _run_cmd_logged(
                [str(venv_python), "-m", "pip", "install", "numpy", "nibabel"],
                cwd=str(pbrain_root),
                env=_python_env_for_pbrain(),
                write_line_sync=write_line_sync,
            )
        marker.write_text(want_fp, encoding="utf-8")

    if want_fp != have_fp:
        await _install_deps()

        # We just changed the environment; clear any cached negative preflight.
        try:
            _PYTHON_PREFLIGHT.pop(str(venv_python), None)
        except Exception:
            pass

    # Keep the optional requirement files present even if core deps are already up-to-date.
    _write_optional_requirement_files()

    # Validate that the managed env actually imports what we need.
    # If the venv was partially installed or corrupted, force one reinstall.
    try:
        try:
            _PYTHON_PREFLIGHT.pop(str(venv_python), None)
        except Exception:
            pass
        await _preflight_pbrain_python(str(venv_python))
    except Exception as exc:
        write_line_sync(f"[{_now_iso()}] Managed venv preflight failed; reinstalling deps once: {exc}\n")
        try:
            if marker.exists():
                marker.unlink()
        except Exception:
            pass
        await _install_deps()
        try:
            _PYTHON_PREFLIGHT.pop(str(venv_python), None)
        except Exception:
            pass
        await _preflight_pbrain_python(str(venv_python))

    return str(venv_python)


async def _ensure_managed_pbrain_optional_deps(*, kind: Literal["ai", "diffusion"], write_line_sync=None) -> str:
    """Install optional managed-venv deps on-demand and return the venv python.

    For AI deps we prefer creating the venv with a TensorFlow-compatible Python.
    """

    if write_line_sync is None:
        def write_line_sync(_line: str) -> None:
            return

    prefer_ai = bool(kind == "ai")
    venv_python = await _ensure_managed_pbrain_venv(write_line_sync=write_line_sync, prefer_ai_compatible=prefer_ai)
    venv_dir = _managed_pbrain_venv_dir()
    pbrain_root = _pbrain_root_dir()

    if kind == "ai":
        req_file = venv_dir / "requirements.ai.txt"
        marker = venv_dir / ".pbrain_requirements.ai.sha256"
        label = "AI"
    else:
        req_file = venv_dir / "requirements.diffusion.txt"
        marker = venv_dir / ".pbrain_requirements.diffusion.sha256"
        label = "diffusion"

    # If we don't have the split files (older venv), regenerate by forcing a core ensure.
    if not req_file.exists():
        await _ensure_managed_pbrain_venv(write_line_sync=write_line_sync)

    # Nothing to do if there are no optional deps.
    try:
        lines = req_file.read_text(encoding="utf-8", errors="ignore").splitlines()
        has_any = any(l.strip() and not l.strip().startswith("#") for l in lines)
    except Exception:
        has_any = False
    if not has_any:
        return venv_python

    try:
        fp = hashlib.sha256(req_file.read_bytes()).hexdigest()
    except Exception:
        fp = "missing"
    want = f"{kind}-v1:{fp}"
    have = marker.read_text(encoding="utf-8", errors="ignore").strip() if marker.exists() else ""
    if want == have:
        return venv_python

    pip_args: List[str] = [venv_python, "-m", "pip", "install", "--prefer-binary", "-r", str(req_file)]
    if sys.platform == "darwin":
        constraints = venv_dir / "constraints.darwin.txt"
        if constraints.exists():
            pip_args += ["-c", str(constraints)]

    try:
        write_line_sync(f"[{_now_iso()}] Installing optional {label} deps into managed venv...\n")
    except Exception:
        pass
    await _run_cmd_logged(
        pip_args,
        cwd=str(pbrain_root),
        env=_python_env_for_pbrain(),
        write_line_sync=write_line_sync,
    )
    try:
        # Optional installs can change import availability; clear cached preflight.
        _PYTHON_PREFLIGHT.pop(str(_managed_venv_python_path()), None)
    except Exception:
        pass
    marker.write_text(want, encoding="utf-8")
    return venv_python


def _pbrain_cli_args(project: Project, subject: Subject) -> List[str]:
    main_py = _resolve_pbrain_main_py()
    if not main_py:
        raise RuntimeError("p-Brain path is not set. Set PBRAIN_MAIN_PY or configure it in Settings.")
    main_py_path = Path(main_py).expanduser().resolve()
    if not main_py_path.exists():
        raise RuntimeError(f"p-Brain main.py not found: {main_py_path}")

    data_root = Path(project.storagePath).expanduser().resolve()
    args: List[str] = [
        _pbrain_python_executable(),
        "-u",
        str(main_py_path),
        "--mode",
        "auto",
        "--id",
        subject.name,
        "--data-dir",
        str(data_root),
    ]

    cfg = project.config or {}
    model_cfg = cfg.get("model", {}) if isinstance(cfg.get("model", {}), dict) else {}
    voxel_cfg = cfg.get("voxelwise", {}) if isinstance(cfg.get("voxelwise", {}), dict) else {}

    # Project-level T1/M0 fit selection.
    try:
        pb = cfg.get("pbrain") if isinstance(cfg.get("pbrain"), dict) else {}
        t1_fit = str(pb.get("t1Fit") or "").strip().lower()
        if t1_fit in {"auto", "ir", "vfa", "none"}:
            args += ["--t1-fit", t1_fit]
    except Exception:
        pass

    # Keep behaviour aligned with p-brain CLI flags.
    lambd = model_cfg.get("lambdaTikhonov")
    if isinstance(lambd, (int, float)):
        args += ["--lambda", str(float(lambd))]
    else:
        # Default to L-curve auto lambda when user has not pinned a value.
        args.append("--enable-lcurve")
    if model_cfg.get("autoLambda") is True:
        args.append("--enable-lcurve")

    # Write optional maps
    if "writeMTT" in voxel_cfg:
        args += ["--write-mtt", str(bool(voxel_cfg.get("writeMTT"))).lower()]
    if "writeCTH" in voxel_cfg:
        args += ["--write-cth", str(bool(voxel_cfg.get("writeCTH"))).lower()]

    # Diffusion: only request when the subject has diffusion.
    if subject.hasDiffusion:
        args.append("--diffusion")

    return args


def _pbrain_cli_args_with_python(python_exe: str, project: Project, subject: Subject) -> List[str]:
    main_py = _resolve_pbrain_main_py()
    if not main_py:
        raise RuntimeError("p-Brain path is not set. Set PBRAIN_MAIN_PY or configure it in Settings.")
    main_py_path = Path(main_py).expanduser().resolve()
    if not main_py_path.exists():
        raise RuntimeError(f"p-Brain main.py not found: {main_py_path}")

    data_root = Path(project.storagePath).expanduser().resolve()
    args: List[str] = [
        python_exe,
        "-u",
        str(main_py_path),
        "--mode",
        "auto",
        "--id",
        subject.name,
        "--data-dir",
        str(data_root),
    ]

    cfg = project.config or {}
    model_cfg = cfg.get("model", {}) if isinstance(cfg.get("model", {}), dict) else {}
    voxel_cfg = cfg.get("voxelwise", {}) if isinstance(cfg.get("voxelwise", {}), dict) else {}

    # Project-level T1/M0 fit selection.
    try:
        pb = cfg.get("pbrain") if isinstance(cfg.get("pbrain"), dict) else {}
        t1_fit = str(pb.get("t1Fit") or "").strip().lower()
        if t1_fit in {"auto", "ir", "vfa", "none"}:
            args += ["--t1-fit", t1_fit]
    except Exception:
        pass

    lambd = model_cfg.get("lambdaTikhonov")
    if isinstance(lambd, (int, float)):
        args += ["--lambda", str(float(lambd))]
    else:
        args.append("--enable-lcurve")
    if model_cfg.get("autoLambda") is True:
        args.append("--enable-lcurve")

    if "writeMTT" in voxel_cfg:
        args += ["--write-mtt", str(bool(voxel_cfg.get("writeMTT"))).lower()]
    if "writeCTH" in voxel_cfg:
        args += ["--write-cth", str(bool(voxel_cfg.get("writeCTH"))).lower()]

    if subject.hasDiffusion:
        args.append("--diffusion")

    return args


_STAGE_RUNNER_VERSION = "32"


_STAGE_RUNNER_VERSION_RE = re.compile(r"stage runner \(version:\s*([0-9]+)\)")


def _read_stage_runner_version(text: str | None) -> str | None:
    if not text:
        return None
    try:
        m = _STAGE_RUNNER_VERSION_RE.search(text)
        if not m:
            return None
        v = (m.group(1) or "").strip()
        return v or None
    except Exception:
        return None


def _stage_runner_voxelwise_flags(project: Project) -> tuple[bool, bool]:
    """Translate project voxelwise config into stage-runner flags.

    p-brain's main CLI does not expose --voxelwise/--cbf, but the generated
    stage runner does. Historically, p-brain-web stored voxelwise settings as:
      voxelwise.enabled, voxelwise.writeMTT, voxelwise.writeCTH

    Backwards compatibility: also accept legacy keys computeKi/computeCBF.
    """

    cfg = project.config if isinstance(getattr(project, "config", None), dict) else {}
    voxel = cfg.get("voxelwise") if isinstance(cfg.get("voxelwise"), dict) else {}

    # Default to parity with p-brain auto runs: produce voxelwise outputs unless
    # explicitly disabled per-project.
    enabled_raw = voxel.get("enabled")
    enabled = True if enabled_raw is None else bool(enabled_raw)

    compute_ki = bool(voxel.get("computeKi")) if "computeKi" in voxel else bool(enabled)

    if "computeCBF" in voxel:
        compute_cbf = bool(voxel.get("computeCBF"))
    else:
        # Only compute residue-based perfusion maps when voxelwise is enabled.
        write_mtt_raw = voxel.get("writeMTT")
        write_cth_raw = voxel.get("writeCTH")
        write_mtt = True if write_mtt_raw is None else bool(write_mtt_raw)
        write_cth = True if write_cth_raw is None else bool(write_cth_raw)
        compute_cbf = bool(enabled) and (write_mtt or write_cth)

    # If we request voxelwise Ki, default to also computing CBF/MTT/CTH unless
    # computeCBF was explicitly provided.
    if compute_ki and "computeCBF" not in voxel:
        compute_cbf = True

    return bool(compute_ki), bool(compute_cbf)


def _stage_runner_path(data_root: Path) -> Path:
    return data_root / ".pbrain-web" / "runner" / "pbrain_stage_runner.py"


def _ensure_stage_runner_script(data_root: Path, *, force: bool = False) -> Path:
    runner_path = _stage_runner_path(data_root)
    # Be robust in packaged builds: avoid referencing exception variables outside
    # their except-block scope (can raise UnboundLocalError in some Python versions).
    mkdir_error: Exception | None = None
    try:
        runner_path.parent.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        mkdir_error = e

    if mkdir_error is not None:
        # We'll still try to proceed; the write below will surface a clearer error.
        pass

    content = """#!/usr/bin/env python3
# pbrain-web stage runner (version: __STAGE_RUNNER_VERSION__)

import argparse
import os
import shutil
import sys
import subprocess
import builtins
import time
import threading
import resource
import functools

# When running in batch/headless mode, ensure matplotlib uses a non-GUI backend.
if os.environ.get('PBRAIN_TURBO') == '1':
    os.environ.setdefault('MPLBACKEND', 'Agg')

try:
    # Helpful for debugging stale runner issues in the field.
    print('[runner] pbrain-web stage runner version: __STAGE_RUNNER_VERSION__', flush=True)
except Exception:
    pass

# Ensure p-brain repo imports resolve even though this runner lives outside it.
# p-brain-web typically sets cwd to the repo root when invoking this script.
try:
    _cwd = os.path.abspath(os.getcwd())
    if _cwd and _cwd not in sys.path:
        sys.path.insert(0, _cwd)
except Exception:
    pass

from utils.settings import setup_directories
from utils.parameters import global_filenames, global_parameters, refresh_nifti_directory
from modules.start import parrec2nifti
from modules.opt07_axials import check_axial
from modules.opt01_T1_fit import T1_fit
import modules.opt01_T1_fit as opt01_T1_fit
import modules.opt03_time_shifting as opt03_time_shifting
from modules import input_function_AI
import numpy as np
import nibabel as nib

import utils.settings as settings
from utils.loading import discover_ir_series, discover_vfa_series
import modules.AI_tissue_functions as AIT
from utils.compare_matlab import compare_t1m0_to_matlab
from utils.cli_logging import (
    install_auto_logging_hooks,
    uninstall_auto_logging_hooks,
    auto_logging_suppressed,
    log_process_start,
    log_process_end,
)


def _install_numpy_mkdir_wrappers() -> None:
    '''Ensure numpy save calls don't fail if parent dirs are missing.

    p-brain sometimes writes into nested analysis folders (e.g. 'Analysis/TSCC Data/<subtype>/...')
    without creating the directory first.

    We wrap numpy's save APIs to create the parent directory when given a path.
    '''

    try:
        import os

        def _wrap(fn):
            if getattr(fn, "__pbrainweb_mkdir_wrapped__", False):
                return fn

            def _wrapped(file, *args, **kwargs):
                try:
                    if isinstance(file, (str, os.PathLike)):
                        parent = os.path.dirname(os.fspath(file))
                        if parent:
                            os.makedirs(parent, exist_ok=True)
                except Exception:
                    pass
                return fn(file, *args, **kwargs)

            setattr(_wrapped, "__pbrainweb_mkdir_wrapped__", True)
            return _wrapped

        np.save = _wrap(np.save)
        np.savez = _wrap(np.savez)
        np.savez_compressed = _wrap(np.savez_compressed)
    except Exception:
        pass


def _install_noninteractive_input(default_choice: str = "rica") -> None:
    '''Make p-brain input() safe in headless runs.

    Some upstream stages prompt for artery/ROI selection and y/n confirmations.
    In PBRAIN_NONINTERACTIVE mode we auto-answer with sensible defaults.
    '''

    force = os.environ.get("PBRAIN_NONINTERACTIVE") == "1"
    try:
        is_tty = bool(getattr(sys.stdin, "isatty", lambda: False)())
    except Exception:
        is_tty = False
    if is_tty and not force:
        return

    choice = (os.environ.get("PBRAIN_AUTO_INPUT") or "").strip() or default_choice
    auto_artery = (os.environ.get("PBRAIN_AUTO_ARTERY") or "").strip() or default_choice

    def _auto_input(prompt: str = "") -> str:
        try:
            if prompt:
                # Keep prompts visible in logs.
                print(prompt, end="", flush=True)
        except Exception:
            pass

        p = (prompt or "").strip().lower()

        # Time shifting: artery choice prompt.
        if "enter the number corresponding to your choice" in p or "corresponding to your choice" in p:
            return auto_artery

        # Common y/n prompt.
        if "(y/n" in p or "(y/n):" in p or "(y/n)" in p:
            return "y"

        return choice

    builtins.input = _auto_input


def _set_turbo_mode(enabled: bool) -> None:
    # Mirror main.py behaviour for modules that use a module-level turbo flag.
    try:
        import utils.plotting as plotting
        import modules.opt01_T1_fit as _opt01
        import modules.AI_input_functions as _aif
        import modules.AI_tissue_functions as _tissue
        import modules.opt03_time_shifting as _ts
        import modules.opt02_input_functions as _if
        import modules.opt04_tissue_function as _tf
        import modules.opt05_BBB_parameters as _bbb
        import modules.opt00_images as _img
        for m in (plotting, _opt01, _aif, _tissue, _ts, _if, _tf, _bbb, _img):
            setattr(m, 'turbo_mode', enabled)
    except Exception:
        pass


def _format_rss_mb(ru_maxrss: float) -> float:
    # On macOS ru_maxrss is bytes; on Linux it's kilobytes.
    try:
        v = float(ru_maxrss)
    except Exception:
        return 0.0
    if v <= 0:
        return 0.0
    # Heuristic: if it's huge, treat as bytes.
    if v > 1024 * 1024:
        return v / (1024.0 * 1024.0)
    return v / 1024.0


def _newest_file_under(paths, *, max_files: int = 8000, max_depth: int = 6):
    newest_path = None
    newest_mtime = -1.0

    def walk(p: str, depth: int):
        nonlocal newest_path, newest_mtime, max_files
        if max_files <= 0:
            return
        if depth > max_depth:
            return
        try:
            with os.scandir(p) as it:
                for entry in it:
                    if max_files <= 0:
                        return
                    try:
                        if entry.is_dir(follow_symlinks=False):
                            walk(entry.path, depth + 1)
                        else:
                            st = entry.stat(follow_symlinks=False)
                            max_files -= 1
                            mt = float(getattr(st, "st_mtime", 0.0) or 0.0)
                            if mt > newest_mtime:
                                newest_mtime = mt
                                newest_path = entry.path
                    except Exception:
                        continue
        except Exception:
            return

    for p in paths or []:
        if not p:
            continue
        try:
            if os.path.isdir(p):
                walk(p, 0)
        except Exception:
            continue

    return newest_path, newest_mtime


def _run_with_heartbeat(label: str, fn, *, heartbeat_s: float = 30.0, watch_paths=None):
    start = time.time()
    stop = threading.Event()

    def beat():
        while not stop.wait(heartbeat_s):
            try:
                ru = resource.getrusage(resource.RUSAGE_SELF)
                cpu = float(getattr(ru, "ru_utime", 0.0) or 0.0) + float(getattr(ru, "ru_stime", 0.0) or 0.0)
                rss_mb = _format_rss_mb(getattr(ru, "ru_maxrss", 0.0) or 0.0)
            except Exception:
                cpu = 0.0
                rss_mb = 0.0

            newest_path, newest_mtime = _newest_file_under(watch_paths or [])
            newest_age = None
            try:
                if newest_mtime and newest_mtime > 0:
                    newest_age = max(0.0, time.time() - float(newest_mtime))
            except Exception:
                newest_age = None

            elapsed = max(0.0, time.time() - start)
            if newest_path and newest_age is not None:
                print(
                    "[heartbeat] %s: elapsed=%.0fs cpu=%.0fs rss=%.0fMB newest='%s' (%.0fs ago)"
                    % (label, elapsed, cpu, rss_mb, newest_path, newest_age),
                    flush=True,
                )
            else:
                print("[heartbeat] %s: elapsed=%.0fs cpu=%.0fs rss=%.0fMB" % (label, elapsed, cpu, rss_mb), flush=True)

    t = threading.Thread(target=beat, daemon=True)
    t.start()
    print("[start] %s" % (label,), flush=True)
    try:
        return fn()
    finally:
        stop.set()
        try:
            t.join(timeout=1.0)
        except Exception:
            pass
        elapsed = max(0.0, time.time() - start)
        print("[done] %s (elapsed=%.0fs)" % (label, elapsed), flush=True)


def _preamble(subject_id: str, data_root: str):
    data_directory, analysis_directory, nifti_directory, image_directory = setup_directories(subject_id, data_root)
    filenames = global_filenames(nifti_directory)
    parameters = global_parameters()

    # This matches main.py ordering and is expected to be idempotent.
    parrec2nifti(data_directory, nifti_directory)
    filenames = global_filenames(nifti_directory)
    parameters = global_parameters()
    refresh_nifti_directory(nifti_directory)
    check_axial(nifti_directory, filenames)
    return data_directory, analysis_directory, nifti_directory, image_directory, filenames, parameters


def _maybe_force_delete_t1m0_outputs(analysis_directory: str) -> None:
    fitting_dir = os.path.join(analysis_directory, 'Fitting')
    if not os.path.isdir(fitting_dir):
        return
    names = [
        'voxel_matrix.pkl',
        'voxel_T1_matrix.pkl',
        'voxel_M0_matrix.pkl',
        't1_map.nii.gz',
        'm0_map.nii.gz',
        't1_map_in_dce.nii.gz',
        'm0_map_in_dce.nii.gz',
    ]
    for name in names:
        for candidate in (name, f'._{name}'):
            path = os.path.join(fitting_dir, candidate)
            try:
                if os.path.exists(path):
                    os.remove(path)
            except Exception:
                pass


def main() -> int:
    p = argparse.ArgumentParser(description='Run a single p-brain stage (invoked by p-brain-web)')
    p.add_argument('--stage', required=True)
    p.add_argument('--id', required=True)
    p.add_argument('--data-dir', required=True)
    p.add_argument('--t1-fit', dest='t1_fit', type=str, default=None)
    p.add_argument('--t1m0-force', dest='t1m0_force', action='store_true')
    p.add_argument('--compare-matlab', dest='compare_matlab', action='store_true')
    p.add_argument('--compare-matlab-path', dest='compare_matlab_path', type=str, default=None)
    p.add_argument('--diffusion', action='store_true')
    p.add_argument('--voxelwise', action='store_true')
    p.add_argument('--cbf', action='store_true')
    args = p.parse_args()

    stage = str(args.stage).strip().lower()
    subject_id = str(args.id)
    data_root = str(args.data_dir)

    # Emit a concise config summary for reproducibility.
    try:
        keys = [
            'P_BRAIN_CTC_MODEL',
            'P_BRAIN_TURBOFLASH_CTC_METHOD',
            'P_BRAIN_T1_FIT',
            'P_BRAIN_TURBO_NPH',
            'P_BRAIN_NUMBER_OF_PEAKS',
            'P_BRAIN_CTC_PEAK_RESCALE_THRESHOLD',
            'P_BRAIN_MODELLING_INPUT_FUNCTION',
            'P_BRAIN_AIF_USE_SSS',
            'P_BRAIN_VASCULAR_ROI_CURVE_METHOD',
            'P_BRAIN_VASCULAR_ROI_ADAPTIVE_MAX',
            'P_BRAIN_TSCC_SKIP_FORKED_PEAKS',
            'P_BRAIN_CTC_MAPS',
            'P_BRAIN_CTC_4D',
            'P_BRAIN_CTC_MAP_SLICE',
        ]
        summary = ' '.join([f"{k}={os.environ.get(k)!r}" for k in keys if os.environ.get(k) is not None])
        print(f"[runner] stage={stage} id={subject_id} data_dir={data_root} {summary}", flush=True)
    except Exception:
        pass

    # Project-level override: make sure T1/M0 fit selection is applied even when
    # the runner was launched with inherited env vars.
    try:
        if args.t1_fit is not None:
            v = str(args.t1_fit).strip().lower()
            if v in ('auto', 'ir', 'vfa', 'none'):
                settings.T1_FIT_MODE = v
                os.environ['P_BRAIN_T1_FIT'] = v
    except Exception:
        pass

    # Mirror p-brain main.py behaviour: when running TurboFLASH CTC, default the
    # T1 recovery model to the matching TurboFLASH TI-series fit unless explicitly
    # overridden.
    try:
        ctc_model = (os.environ.get('P_BRAIN_CTC_MODEL') or getattr(settings, 'CTC_MODEL', '') or '').strip().lower()
        if ctc_model in {'turboflash', 'advanced'} and (os.environ.get('P_BRAIN_T1_RECOVERY_MODEL') or '').strip() == '':
            try:
                settings.T1_RECOVERY_MODEL = 'turboflash'
            except Exception:
                pass
            os.environ['P_BRAIN_T1_RECOVERY_MODEL'] = 'turboflash'
    except Exception:
        pass

    turbo_env = os.environ.get('PBRAIN_TURBO') == '1'
    _set_turbo_mode(turbo_env)

    _install_noninteractive_input()

    hooks_installed = False
    try:
        data_directory, analysis_directory, nifti_directory, image_directory, filenames, parameters = _preamble(subject_id, data_root)

        # Import/index only.
        if stage == 'import':
            print('[import] Prepared inputs and refreshed indices')
            return 0

        # Make sure downstream numpy saves won't fail due to missing directories.
        # Must be installed *before* auto-logging hooks, since those may wrap/replace
        # numpy save functions.
        _install_numpy_mkdir_wrappers()

        install_auto_logging_hooks()
        hooks_installed = True
        with auto_logging_suppressed():
            pass

        if stage == 't1_fit':
            # User requirement: T1/M0 fitting must follow p-brain's CLI semantics.
            # Delegate to `main.py --t1m0-only` and forward relevant flags.
            if getattr(args, 't1m0_force', False):
                _maybe_force_delete_t1m0_outputs(analysis_directory)
            cmd = [sys.executable, '-u', 'main.py', '--id', subject_id, '--data-dir', data_root, '--t1m0-only']
            if getattr(args, 't1_fit', None):
                cmd += ['--t1-fit', str(args.t1_fit)]
            if getattr(args, 'compare_matlab', False):
                cmd += ['--compare-matlab']
                if getattr(args, 'compare_matlab_path', None):
                    cmd += ['--compare-matlab-path', str(args.compare_matlab_path)]

            env = dict(os.environ)
            # Ensure headless behaviour for spawned p-brain process.
            env.setdefault('PBRAIN_TURBO', '1')
            env.setdefault('PBRAIN_NONINTERACTIVE', '1')

            print(f"[t1_fit] Delegating to p-brain: {cmd}", flush=True)
            rc = subprocess.call(cmd, env=env)
            if int(rc) != 0:
                print(f"[t1_fit] p-brain exited with code {rc}", flush=True)
                return int(rc)
            return 0

        if stage == 'input_functions':
            log_process_start('Input function extraction')

            def _debug_dce_sanity():
                try:
                    dce_filename = None
                    try:
                        dce_filename = filenames[8]
                    except Exception:
                        dce_filename = None
                    dce_path = os.path.join(nifti_directory, dce_filename) if dce_filename else None
                    if not dce_path or not os.path.exists(dce_path):
                        print(f"[input_functions] DCE missing or not found: {dce_path}", flush=True)
                        return
                    try:
                        img = nib.load(dce_path)
                        sh = tuple(int(x) for x in (getattr(img, 'shape', None) or ()))
                        print(f"[input_functions] DCE: {dce_filename} shape={sh}", flush=True)
                    except Exception as e:
                        print(f"[input_functions] DCE load failed: {dce_path} err={e}", flush=True)
                except Exception:
                    pass

            def _maybe_repair_deterministic_tmp(missing_path: str) -> bool:
                '''If deterministic fallback writes into a tmp analysis dir, ensure it has T1/M0-in-DCE maps.

                p-brain's deterministic ROI path can call plotting which looks for:
                    <tmp>/Fitting/t1_map_in_dce.nii.gz and m0_map_in_dce.nii.gz
                If those are only present in the main Analysis/Fitting, copy them in.
                '''

                try:
                    if not missing_path:
                        return False
                    mp = str(missing_path)
                    if '.pbrain_tmp_deterministic_roi' not in mp:
                        return False
                    if not mp.replace('\\\\', '/').endswith('/Fitting/voxel_T1_matrix.pkl'):
                        return False

                    tmp_fit = os.path.dirname(mp)
                    tmp_analysis = os.path.dirname(tmp_fit)
                    main_fit = os.path.join(analysis_directory, 'Fitting')
                    if not os.path.isdir(main_fit):
                        return False

                    os.makedirs(tmp_fit, exist_ok=True)
                    copied = 0
                    for name in ('t1_map_in_dce.nii.gz', 'm0_map_in_dce.nii.gz', 't1_map_in_dce.nii', 'm0_map_in_dce.nii'):
                        src = os.path.join(main_fit, name)
                        dst = os.path.join(tmp_fit, name)
                        try:
                            if os.path.exists(src) and not os.path.exists(dst):
                                shutil.copy2(src, dst)
                                copied += 1
                        except Exception:
                            pass
                    if copied > 0:
                        print(f"[input_functions] Repaired deterministic tmp fitting dir: {tmp_analysis} (copied {copied} map(s))", flush=True)
                        return True
                except Exception:
                    return False
                return False

            try:
                from modules.input_function_dispatch import (
                    PBRAIN_WAITING_FOR_ROI_EXIT_CODE,
                    InputFunctionUserInteractionRequired,
                    run_input_function,
                )

                _debug_dce_sanity()

                # One retry to handle deterministic tmp-dir missing T1/M0 maps.
                for attempt in range(2):
                    try:
                        run_input_function(analysis_directory, nifti_directory, image_directory, filenames, parameters)
                        break
                    except InputFunctionUserInteractionRequired as e:
                        raise
                    except FileNotFoundError as e:
                        msg = str(e)
                        missing_path = getattr(e, 'filename', None) or ''

                        # ROI-file mode: treat missing ROI root as "waiting".
                        if 'Missing ROI Data directory' in msg or msg.replace('\\\\', '/').endswith('/Analysis/ROI Data'):
                            print('PBRAIN_WAITING_STAGE=input_functions missing=roi_data')
                            log_process_end('Input function extraction')
                            return int(PBRAIN_WAITING_FOR_ROI_EXIT_CODE)

                        # Deterministic fallback can fail during plotting if tmp dir lacks maps.
                        if 'voxel_T1_matrix.pkl' in msg and _maybe_repair_deterministic_tmp(missing_path or msg):
                            if attempt == 0:
                                continue

                        raise

            except InputFunctionUserInteractionRequired as e:
                try:
                    missing = getattr(e, 'missing', None) or []
                    print(f"PBRAIN_WAITING_STAGE=input_functions missing={','.join(missing)}")
                except Exception:
                    pass
                log_process_end('Input function extraction')
                return int(PBRAIN_WAITING_FOR_ROI_EXIT_CODE)

            log_process_end('Input function extraction')
            return 0

        if stage == 'time_shift':
            log_process_start('Time shifting')
            # Upstream bug: utils.mapping.choice2subtype returns an undefined `type` and may
            # also leave `subtype` unset for unexpected choices. opt03_time_shifting expects
            # a string subtype, so patch the module-global reference.
            try:
                def _choice2subtype_patched(c):
                    s = str(c).strip().lower()
                    lut = {{
                        'b': 'Basilar',
                        '4': 'Basilar',
                        'lica': 'Left Interior Carotid',
                        '2': 'Left Interior Carotid',
                        'rica': 'Right Interior Carotid',
                        '3': 'Right Interior Carotid',
                        'lmca': 'Left Middle Cerebral',
                        '5': 'Left Middle Cerebral',
                        'rmca': 'Right Middle Cerebral',
                        '6': 'Right Middle Cerebral',
                    }}
                    return lut.get(s, 'Right Interior Carotid')

                opt03_time_shifting.choice2subtype = _choice2subtype_patched
            except Exception:
                pass

            opt03_time_shifting.time_shifting(analysis_directory, nifti_directory, image_directory)
            log_process_end('Time shifting')
            return 0

        # --- AI tissue pipeline split into true sub-stages --------------------
        (
            t1_3D_filename,
            axial_t1_3D_filename,
            t2_3D_filename,
            axial_t2_3D_filename,
            flair_3D_filename,
            axial_flair_3D_filename,
            axial_t2_2D_filename,
            diffusion_filename,
            dce_filename,
        ) = filenames

        IsVFA, IsIR, apple_metal, boundary, RERUN_SEGMENTATION, SEGMENTATION_METHOD, _ = parameters

        t1_path = os.path.join(nifti_directory, t1_3D_filename) if t1_3D_filename else None
        t2_path = os.path.join(nifti_directory, axial_t2_2D_filename) if axial_t2_2D_filename else None
        dce_path = os.path.join(nifti_directory, dce_filename) if dce_filename else None

        flip_angle_deg = None
        try:
            if dce_path:
                flip_angle_deg = AIT.resolve_flip_angle_deg(dce_path, default=None)
        except Exception:
            flip_angle_deg = None

        seg_dir = os.path.join(nifti_directory, 'segmentation')
        sid = 'segmentation'
        seg_mgz_path = os.path.join(seg_dir, sid, 'mri', 'aparc.DKTatlas+aseg.deep.mgz')

        def _fastsurfer_run_sh() -> str:
            direct = (os.environ.get('FASTSURFER_RUN_SH') or '').strip()
            if direct:
                return direct
            root = (os.environ.get('FASTSURFER_DIR') or '').strip()
            if root:
                return os.path.join(root, 'run_fastsurfer.sh')
            # Legacy/dev fallback.
            return '/Users/edt/FastSurfer/run_fastsurfer.sh'

        def _load_t1m0():
            t1_pkl = os.path.join(analysis_directory, 'Fitting', 'voxel_T1_matrix.pkl')
            m0_pkl = os.path.join(analysis_directory, 'Fitting', 'voxel_M0_matrix.pkl')
            if not os.path.exists(t1_pkl) or not os.path.exists(m0_pkl):
                raise RuntimeError('Missing T1/M0 fit outputs. Run T1/M0 fitting first.')
            return AIT.load_from_pickle(t1_pkl), AIT.load_from_pickle(m0_pkl)

        def _load_timepoints(data_4d: np.ndarray, ref_img) -> np.ndarray:
            TR = ref_img.header.get_zooms()[-1]
            num_volumes = data_4d.shape[-1]
            total_scan_duration = TR * num_volumes
            return np.linspace(0, total_scan_duration, num_volumes)

        def _masks_and_images():
            if not dce_path or not os.path.exists(dce_path):
                raise RuntimeError('Missing DCE NIfTI. Ensure DCE is present and imported.')
            if not os.path.exists(seg_mgz_path):
                raise RuntimeError('Missing segmentation output. Run Segmentation first.')

            (
                wm_mask_t2, wm_mask_dce,
                cortical_gm_mask_t2, cortical_gm_mask_dce,
                subcortical_gm_mask_t2, subcortical_gm_mask_dce,
                gm_brainstem_mask_t2, gm_brainstem_mask_dce,
                gm_cerebellum_mask_t2, gm_cerebellum_mask_dce,
                wm_cerebellum_mask_t2, wm_cerebellum_mask_dce,
                wm_cc_mask_t2, wm_cc_mask_dce,
            ) = AIT.coregistration(seg_mgz_path=seg_mgz_path, dce_path=dce_path, t2_path=t2_path)

            t2_img = nib.load(t2_path).get_fdata()
            ref_img = nib.load(dce_path)
            data_4d = np.array(ref_img.get_fdata())
            return (
                t2_img, ref_img, data_4d,
                wm_mask_t2, wm_mask_dce,
                cortical_gm_mask_t2, cortical_gm_mask_dce,
                subcortical_gm_mask_t2, subcortical_gm_mask_dce,
                gm_brainstem_mask_t2, gm_brainstem_mask_dce,
                gm_cerebellum_mask_t2, gm_cerebellum_mask_dce,
                wm_cerebellum_mask_t2, wm_cerebellum_mask_dce,
                wm_cc_mask_t2, wm_cc_mask_dce,
            )

        if stage == 'segmentation':
            if not t1_path or not os.path.exists(t1_path):
                print('[segmentation] Missing T1 structural input; skipping segmentation')
                return 0
            if not t2_path or not os.path.exists(t2_path):
                print('[segmentation] Missing T2 structural input; skipping segmentation')
                return 0
            if not dce_path or not os.path.exists(dce_path):
                print('[segmentation] Missing DCE NIfTI; skipping segmentation')
                return 0
            fastsurfer_path = _fastsurfer_run_sh()
            os.makedirs(seg_dir, exist_ok=True)
            print('[segmentation] Running FastSurfer + mask generation')
            AIT.segmentation(
                fastsurfer_path,
                seg_mgz_path,
                t1_path,
                seg_dir,
                sid,
                apple_metal,
                RERUN_SEGMENTATION,
                SEGMENTATION_METHOD,
            )
            print('[segmentation] Coregistering masks into T2/DCE space')
            AIT.coregistration(seg_mgz_path=seg_mgz_path, dce_path=dce_path, t2_path=t2_path)
            return 0

        if stage == 'tissue_ctc':
            if not os.path.exists(seg_mgz_path):
                print('[tissue_ctc] Missing segmentation output; skipping tissue curves')
                return 0
            if not t2_path or not os.path.exists(t2_path):
                print('[tissue_ctc] Missing T2 structural input; skipping tissue curves')
                return 0
            print('[tissue_ctc] Computing tissue curves (no modelling)')
            (
                t2_img, ref_img, data_4d,
                wm_mask_t2, wm_mask_dce,
                cortical_gm_mask_t2, cortical_gm_mask_dce,
                subcortical_gm_mask_t2, subcortical_gm_mask_dce,
                gm_brainstem_mask_t2, gm_brainstem_mask_dce,
                gm_cerebellum_mask_t2, gm_cerebellum_mask_dce,
                wm_cerebellum_mask_t2, wm_cerebellum_mask_dce,
                wm_cc_mask_t2, wm_cc_mask_dce,
            ) = _masks_and_images()
            T1_matrix, M0_matrix = _load_t1m0()
            time_points_s = _load_timepoints(data_4d, ref_img)

            AIT.compute_and_plot_ctcs_median(
                data_4d,
                t2_img,
                wm_mask_t2,
                cortical_gm_mask_t2,
                subcortical_gm_mask_t2,
                wm_mask_dce,
                cortical_gm_mask_dce,
                subcortical_gm_mask_dce,
                T1_matrix,
                M0_matrix,
                analysis_directory,
                time_points_s,
                image_directory,
                dce_path=dce_path,
                ref_affine=ref_img.affine,
                ref_header=ref_img.header.copy(),
                boundary=boundary,
                compute_per_voxel_Ki=False,
                compute_per_voxel_CBF=False,
                gm_brainstem_mask_t2=gm_brainstem_mask_t2,
                gm_brainstem_mask_dce=gm_brainstem_mask_dce,
                gm_cerebellum_mask_t2=gm_cerebellum_mask_t2,
                gm_cerebellum_mask_dce=gm_cerebellum_mask_dce,
                wm_cerebellum_mask_t2=wm_cerebellum_mask_t2,
                wm_cerebellum_mask_dce=wm_cerebellum_mask_dce,
                wm_cc_mask_t2=wm_cc_mask_t2,
                wm_cc_mask_dce=wm_cc_mask_dce,
                flip_angle_deg=flip_angle_deg,
            )
            return 0

        if stage == 'modelling':
            model_setting = settings.KINETIC_MODEL.lower()
            models = ['patlak', 'tikhonov'] if model_setting == 'both' else [model_setting]

            compute_ki = bool(getattr(args, 'voxelwise', False))
            compute_cbf = bool(getattr(args, 'cbf', False))

            def _is_missing_input_curves_error(exc: FileNotFoundError) -> bool:
                try:
                    msg = str(exc).lower()
                except Exception:
                    msg = ''
                return (
                    ('ctc data/artery' in msg)
                    or ('tscc data/max' in msg)
                    or ('arterial concentration' in msg)
                    or ('no .npy files found' in msg)
                )

            # If segmentation isn't available, still allow voxelwise-only modelling.
            if not os.path.exists(seg_mgz_path) or not t2_path or not os.path.exists(t2_path):
                if not dce_path or not os.path.exists(dce_path):
                    raise RuntimeError('Missing DCE NIfTI. Ensure DCE is present and imported.')
                if not compute_ki and not compute_cbf:
                    print('[modelling] Missing segmentation and voxelwise outputs not requested; skipping modelling')
                    return 0

                print('[modelling] Segmentation missing; running voxelwise-only modelling')
                ref_img = nib.load(dce_path)
                data_4d = np.array(ref_img.get_fdata())
                T1_matrix, M0_matrix = _load_t1m0()
                time_points_s = _load_timepoints(data_4d, ref_img)

                watch = [analysis_directory, image_directory]
                hb = float(os.environ.get('PBRAIN_HEARTBEAT_S') or 30.0)
                for m in models:
                    settings.KINETIC_MODEL = m
                    print(f'[modelling] Running {{m}} model (voxelwise-only)')

                    try:
                        _run_with_heartbeat(
                            "modelling/%s: compute_and_plot_ctcs_median (voxelwise-only ki=%s cbf=%s)" % (m, str(compute_ki).lower(), str(compute_cbf).lower()),
                            lambda: AIT.compute_and_plot_ctcs_median(
                                data_4d,
                                None,
                                None,
                                None,
                                None,
                                None,
                                None,
                                None,
                                T1_matrix,
                                M0_matrix,
                                analysis_directory,
                                time_points_s,
                                image_directory,
                                dce_path=dce_path,
                                ref_affine=ref_img.affine,
                                ref_header=ref_img.header.copy(),
                                boundary=False,
                                compute_per_voxel_Ki=compute_ki,
                                compute_per_voxel_CBF=compute_cbf,
                                flip_angle_deg=flip_angle_deg,
                                voxelwise_only=True,
                            ),
                            heartbeat_s=hb,
                            watch_paths=watch,
                        )
                    except FileNotFoundError as exc:
                        if _is_missing_input_curves_error(exc):
                            print(f"[modelling] Missing arterial/input-function curves; skipping modelling ({{exc}})")
                            return 0
                        raise

                    try:
                        suffix = '_patlak' if m == 'patlak' else '_tikhonov'
                        AIT._rename_model_outputs(analysis_directory, image_directory, suffix, boundary=False)
                    except Exception:
                        pass

                return 0

            ai_base = os.path.join(image_directory, 'AI')
            screenshot_name = 'AI_input_function_ROIs.png'
            screenshot_backup = os.path.join(image_directory, screenshot_name)
            screenshot_src = os.path.join(ai_base, screenshot_name)
            if os.path.exists(screenshot_src):
                shutil.copy2(screenshot_src, screenshot_backup)

            (
                t2_img, ref_img, data_4d,
                wm_mask_t2, wm_mask_dce,
                cortical_gm_mask_t2, cortical_gm_mask_dce,
                subcortical_gm_mask_t2, subcortical_gm_mask_dce,
                gm_brainstem_mask_t2, gm_brainstem_mask_dce,
                gm_cerebellum_mask_t2, gm_cerebellum_mask_dce,
                wm_cerebellum_mask_t2, wm_cerebellum_mask_dce,
                wm_cc_mask_t2, wm_cc_mask_dce,
            ) = _masks_and_images()
            T1_matrix, M0_matrix = _load_t1m0()
            time_points_s = _load_timepoints(data_4d, ref_img)

            atlas_path = os.path.join(
                nifti_directory,
                'segmentation',
                'segmentation',
                'mri',
                'aparc.DKTatlas+aseg.deep_in_DCE.nii.gz',
            )
            if not os.path.exists(atlas_path):
                raise RuntimeError('Missing atlas segmentation in DCE space. Run Segmentation first.')

            try:
                C_a_full, _ = AIT.get_input_function_curve(analysis_directory)
            except FileNotFoundError as exc:
                if _is_missing_input_curves_error(exc):
                    print(f"[modelling] Missing arterial/input-function curves; skipping modelling ({{exc}})")
                    return 0
                raise
            if C_a_full is None or len(C_a_full) == 0:
                raise RuntimeError('Missing input function curve. Run AIF/VIF extraction first.')

            try:
                print(
                    "[modelling] Inputs: data_4d.shape=%s timepoints=%d atlas=%s"
                    % (str(getattr(data_4d, 'shape', None)), int(len(time_points_s)), str(atlas_path)),
                    flush=True,
                )
            except Exception:
                pass

            watch = [analysis_directory, image_directory]
            hb = float(os.environ.get('PBRAIN_HEARTBEAT_S') or 30.0)
            try:
                print(
                    "[modelling] Options: voxelwise=%s cbf=%s" % (str(compute_ki).lower(), str(compute_cbf).lower()),
                    flush=True,
                )
            except Exception:
                pass

            for m in models:
                settings.KINETIC_MODEL = m
                print(f'[modelling] Running {{m}} model')

                if os.path.exists(ai_base):
                    shutil.rmtree(ai_base, ignore_errors=True)
                os.makedirs(ai_base, exist_ok=True)
                if os.path.exists(screenshot_backup):
                    shutil.copy2(screenshot_backup, os.path.join(ai_base, screenshot_name))

                try:
                    _run_with_heartbeat(
                        "modelling/%s: compute_and_plot_ctcs_median (voxelwise=%s cbf=%s)" % (m, str(compute_ki).lower(), str(compute_cbf).lower()),
                        lambda: AIT.compute_and_plot_ctcs_median(
                            data_4d,
                            t2_img,
                            wm_mask_t2,
                            cortical_gm_mask_t2,
                            subcortical_gm_mask_t2,
                            wm_mask_dce,
                            cortical_gm_mask_dce,
                            subcortical_gm_mask_dce,
                            T1_matrix,
                            M0_matrix,
                            analysis_directory,
                            time_points_s,
                            image_directory,
                            dce_path=dce_path,
                            ref_affine=ref_img.affine,
                            ref_header=ref_img.header.copy(),
                            boundary=boundary,
                            compute_per_voxel_Ki=compute_ki,
                            compute_per_voxel_CBF=compute_cbf,
                            gm_brainstem_mask_t2=gm_brainstem_mask_t2,
                            gm_brainstem_mask_dce=gm_brainstem_mask_dce,
                            gm_cerebellum_mask_t2=gm_cerebellum_mask_t2,
                            gm_cerebellum_mask_dce=gm_cerebellum_mask_dce,
                            wm_cerebellum_mask_t2=wm_cerebellum_mask_t2,
                            wm_cerebellum_mask_dce=wm_cerebellum_mask_dce,
                            wm_cc_mask_t2=wm_cc_mask_t2,
                            wm_cc_mask_dce=wm_cc_mask_dce,
                            flip_angle_deg=flip_angle_deg,
                        ),
                        heartbeat_s=hb,
                        watch_paths=watch,
                    )
                except FileNotFoundError as exc:
                    if _is_missing_input_curves_error(exc):
                        print(f"[modelling] Missing arterial/input-function curves; skipping modelling ({{exc}})")
                        return 0
                    raise

                compute_CTC_meta = (
                    functools.partial(AIT.compute_CTC, flip_angle_deg=flip_angle_deg)
                    if flip_angle_deg is not None
                    else AIT.compute_CTC
                )

                _run_with_heartbeat(
                    "modelling/%s: compute_Ki_from_atlas" % (m,),
                    lambda: AIT.compute_Ki_from_atlas(
                        atlas_path=atlas_path,
                        data_4d=data_4d,
                        T1_matrix=T1_matrix,
                        M0_matrix=M0_matrix,
                        time_points_s=time_points_s,
                        C_a_full=C_a_full,
                        affine=ref_img.affine,
                        output_directory=analysis_directory,
                        compute_CTC=compute_CTC_meta,
                        find_baseline_point_advanced=AIT.find_baseline_point_advanced,
                        custom_shifter=AIT.custom_shifter,
                        patlak_analysis_plotting=AIT.patlak_analysis_plotting,
                    ),
                    heartbeat_s=hb,
                    watch_paths=watch,
                )

                suffix = '_patlak' if m == 'patlak' else '_tikhonov'
                AIT._rename_model_outputs(analysis_directory, image_directory, suffix, boundary)

            if os.path.exists(screenshot_backup):
                os.remove(screenshot_backup)
            return 0

        if stage == 'diffusion':
            diffusion_filename = filenames[-2] if filenames else None
            if not diffusion_filename:
                print('[diffusion] No diffusion filename available; nothing to do')
                return 0
            from modules import opt08_fa

            dce_filename = filenames[-1] if filenames else None
            dce_path = os.path.join(nifti_directory, dce_filename) if dce_filename else None
            print('[diffusion] Computing diffusion metrics')
            opt08_fa.compute_fa(
                nifti_directory,
                analysis_directory,
                image_directory,
                diffusion_filename=diffusion_filename,
                dce_path=dce_path,
            )
            return 0

        if stage == 'tractography':
            diffusion_filename = filenames[-2] if filenames else None
            if not diffusion_filename:
                print('[tractography] No diffusion filename available; nothing to do')
                return 0
            from modules import tractography

            log_process_start('Tractography')
            # Default to best-effort CSD (+ ACT + PFT). If the acquisition cannot support CSD,
            # allow fallback rather than failing the whole stage.
            os.environ.setdefault('P_BRAIN_TRACK_REQUIRE_REQUESTED_MODEL', '0')
            # Note: generate_tractography writes outputs under:
            # - analysis_directory/diffusion (tractography.tck, density NIfTI, debug JSON)
            # - image_directory/tractography (render PNG, optional montage/animation)
            tractography.generate_tractography(
                nifti_directory=nifti_directory,
                analysis_directory=analysis_directory,
                image_directory=image_directory,
                diffusion_filename=diffusion_filename,
                diffusion_model='CSD',
                enable_act=True,
                enable_pft=True,
                create_montage=True,
            )

            # Best-effort connectome export (matrix + topology metrics). Non-fatal on failure.
            try:
                from modules import connectome

                connectome.compute_connectome(
                    nifti_directory=nifti_directory,
                    analysis_directory=analysis_directory,
                    diffusion_filename=diffusion_filename,
                )
                print('[connectome] Saved connectome outputs under Analysis/diffusion')
            except Exception as exc:
                print(f'[connectome] Skipped (unable to compute): {{exc}}')

            log_process_end('Tractography')
            return 0

        if stage == 'connectome':
            diffusion_filename = filenames[-2] if filenames else None
            if not diffusion_filename:
                print('[connectome] No diffusion filename available; nothing to do')
                return 0

            from modules import connectome

            log_process_start('Connectome')
            connectome.compute_connectome(
                nifti_directory=nifti_directory,
                analysis_directory=analysis_directory,
                diffusion_filename=diffusion_filename,
            )
            print('[connectome] Saved connectome outputs under Analysis/diffusion')
            log_process_end('Connectome')
            return 0

        raise SystemExit(f'Unknown stage: {{stage}}')

    finally:
        if hooks_installed:
            try:
                uninstall_auto_logging_hooks()
            except Exception:
                pass


if __name__ == '__main__':
    raise SystemExit(main())
"""

    content = content.replace("__STAGE_RUNNER_VERSION__", _STAGE_RUNNER_VERSION)

    try:
        if force:
            try:
                runner_path.unlink(missing_ok=True)
            except Exception:
                pass
        else:
            existing = None
            if runner_path.exists():
                existing = runner_path.read_text(encoding="utf-8", errors="ignore")
            existing_version = _read_stage_runner_version(existing)
            # Only skip rewriting when the file is both:
            # - the expected version, and
            # - contains key features (guards against partial/manual edits or older templates).
            if (
                existing
                and existing_version == _STAGE_RUNNER_VERSION
                and "[t1_fit] Delegating to p-brain" in existing
            ):
                return runner_path
            # If we detect a stale runner version, try to remove it first so we
            # never accidentally execute an old script due to partial writes.
            if existing_version and existing_version != _STAGE_RUNNER_VERSION:
                try:
                    runner_path.unlink(missing_ok=True)
                except Exception:
                    pass
    except Exception:
        pass

    write_error: Exception | None = None
    try:
        runner_path.write_text(content, encoding="utf-8")
        try:
            runner_path.chmod(0o755)
        except Exception:
            pass
    except Exception as e:
        write_error = e

    if write_error is not None:
        raise RuntimeError(f"Failed to write stage runner script: {runner_path} ({write_error})") from write_error

    return runner_path


async def _run_pbrain_auto(*, project: Project, subject: Subject) -> None:
    # Serialize runs globally so stages/logs progress sequentially across the whole app.
    async with _RUN_SEMAPHORE:
        data_root = Path(project.storagePath).expanduser().resolve()
        jobs = _stage_jobs_for_subject(subject.id)
        stage_order: List[StageId] = STAGES
        log_fh = None
        proc: asyncio.subprocess.Process | None = None

        try:
            if not data_root.exists():
                raise RuntimeError(f"Project storagePath/data root does not exist: {data_root}")
            if not jobs:
                raise RuntimeError("No stage jobs registered for subject run")

            # Shared log file per run.
            # Prefer storing logs under the subject folder so artifacts are self-contained.
            logs_dir: Path
            try:
                logs_dir = Path(subject.sourcePath).expanduser().resolve() / ".pbrain-web" / "logs"
                logs_dir.mkdir(parents=True, exist_ok=True)
            except Exception:
                logs_dir = data_root / ".pbrain-web" / "logs"
                logs_dir.mkdir(parents=True, exist_ok=True)
            log_path = logs_dir / f"run_{subject.id}_{int(datetime.utcnow().timestamp()*1000)}.log"

            for job in jobs.values():
                job.logPath = str(log_path)
                job.startTime = _now_iso()

            db._touch_jobs()

            # Open log file early so it is never empty, even if we fail before spawning p-brain.
            log_fh = log_path.open("a", encoding="utf-8", errors="replace")

            def write_line_sync(line: str) -> None:
                if not log_fh:
                    return
                log_fh.write(line)
                log_fh.flush()

            write_line_sync(f"[{_now_iso()}] Run created\n")
            write_line_sync(f"[{_now_iso()}] Subject: {subject.name} ({subject.id})\n")
            write_line_sync(f"[{_now_iso()}] Project: {project.name} ({project.id})\n")

            if not subject.hasDiffusion:
                jobs["diffusion"].status = "completed"
                jobs["diffusion"].progress = 100
                jobs["diffusion"].currentStep = "Skipped (no diffusion)"
                jobs["diffusion"].endTime = _now_iso()
                # Keep the subject stage status as not_run.
                subject.stageStatuses["diffusion"] = "not_run"

                jobs["tractography"].status = "completed"
                jobs["tractography"].progress = 100
                jobs["tractography"].currentStep = "Skipped (no diffusion)"
                jobs["tractography"].endTime = _now_iso()
                subject.stageStatuses["tractography"] = "not_run"

                jobs["connectome"].status = "completed"
                jobs["connectome"].progress = 100
                jobs["connectome"].currentStep = "Skipped (no diffusion)"
                jobs["connectome"].endTime = _now_iso()
                subject.stageStatuses["connectome"] = "not_run"

            # Stage: import starts when the run actually begins.
            _set_job(jobs["import"], status="running", progress=5, step="Preparing inputs")
            _set_stage_status(subject, "import", "running")
            db.save()

            python_exe = await _select_pbrain_python(write_line_sync=write_line_sync)
            args = _pbrain_cli_args_with_python(python_exe, project, subject)

            # Make sure we run from inside the p-brain folder for consistent relative paths.
            main_py = _resolve_pbrain_main_py()
            if not main_py:
                raise RuntimeError("p-Brain path is not set. Set PBRAIN_MAIN_PY or configure it in Settings.")
            main_py_path = Path(main_py).expanduser().resolve()
            if not main_py_path.exists():
                raise RuntimeError(f"p-Brain main.py not found: {main_py_path}")
            pbrain_cwd = str(main_py_path.parent)

            proc = await asyncio.create_subprocess_exec(
                *args,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
                cwd=pbrain_cwd,
                env=_python_env_for_pbrain(),
            )

            # Track process under every stage job so cancelling any of them works.
            for job in jobs.values():
                db._job_processes[job.id] = proc

            write_line_sync(f"[{_now_iso()}] Starting p-brain\n")
            write_line_sync(f"[{_now_iso()}] CWD: {pbrain_cwd}\n")
            write_line_sync(f"[{_now_iso()}] ARGS: {args!r}\n")

            # Helper to advance stage jobs in order.
            db._subject_stage_index[subject.id] = 0

            def begin_stage(stage: StageId, step: str, progress: int) -> None:
                if jobs[stage].status in {"completed", "failed", "cancelled"}:
                    return
                _set_job(jobs[stage], status="running", progress=progress, step=step)
                _set_stage_status(subject, stage, "running")

            def finish_stage(stage: StageId) -> None:
                if jobs[stage].status in {"failed", "cancelled"}:
                    return
                _set_job(jobs[stage], status="completed", progress=100, step="Completed")
                jobs[stage].endTime = _now_iso()
                _set_stage_status(subject, stage, "done")

            assert proc.stdout is not None
            while True:
                try:
                    # If p-brain is quiet/buffered, keep the log alive so the UI doesn't look stuck.
                    line_b = await asyncio.wait_for(proc.stdout.readline(), timeout=60.0)
                except asyncio.TimeoutError:
                    write_line_sync(f"[{_now_iso()}] (still running; no output yet)\n")
                    if proc.returncode is not None:
                        break
                    continue
                if not line_b:
                    break
                line = line_b.decode(errors="replace")
                write_line_sync(line)

                low = line.lower()

                # Map p-brain auto logs to the web app stage model.
                if "starting process: t1 fitting" in low:
                    finish_stage("import")
                    begin_stage("t1_fit", "T1/M0 fitting", 10)
                elif "completed process: t1 fitting" in low:
                    finish_stage("t1_fit")

                elif "starting process: ai input function extraction" in low:
                    begin_stage("input_functions", "AI AIF/VIF extraction", 10)
                elif "starting process: time shifting" in low:
                    begin_stage("time_shift", "Time shifting", 10)
                elif "completed process: time shifting" in low:
                    finish_stage("time_shift")
                elif "completed process: ai input function extraction" in low:
                    finish_stage("input_functions")

                elif "starting process: tissue kinetic modelling" in low:
                    # This umbrella stage covers segmentation + tissue curves + modelling, and optionally diffusion.
                    begin_stage("segmentation", "Segmentation", 10)
                    begin_stage("tissue_ctc", "Tissue curves", 10)
                    begin_stage("modelling", "Model fitting", 10)
                    if subject.hasDiffusion:
                        begin_stage("diffusion", "Diffusion processing", 10)
                elif "completed process: tissue kinetic modelling" in low:
                    finish_stage("segmentation")
                    finish_stage("tissue_ctc")
                    finish_stage("modelling")
                    if subject.hasDiffusion:
                        finish_stage("diffusion")

                # Note: tractography is run by p-brain-web after p-brain finishes.

            rc = await proc.wait()
            if any(j.status == "cancelled" for j in jobs.values()):
                return
            if rc == PBRAIN_WAITING_FOR_ROI_EXIT_CODE:
                # Stop without error: input-functions require a user-provided ROI.
                wait_stage: StageId = "input_functions"
                try:
                    _set_job(jobs[wait_stage], status="completed", progress=100, step="Waiting for user ROI")
                    jobs[wait_stage].endTime = _now_iso()
                    _set_stage_status(subject, wait_stage, "waiting")
                except Exception:
                    pass

                # Everything after input-functions remains not_run.
                try:
                    idx = stage_order.index(wait_stage)
                except Exception:
                    idx = -1
                if idx >= 0:
                    for st in stage_order[idx + 1 :]:
                        if st in jobs and jobs[st].status in {"queued", "running"}:
                            _set_job(jobs[st], status="completed", progress=0, step="Not run (waiting for ROI)")
                            jobs[st].endTime = _now_iso()
                            _set_stage_status(subject, st, "not_run")
                db.save()
                return

            if rc != 0:
                raise RuntimeError(f"p-brain exited with code {rc}")

            # Run tractography as a separate post-step when diffusion is present.
            # (p-brain auto mode does not currently emit tractography markers.)
            if subject.hasDiffusion and jobs.get("tractography"):
                tj = jobs["tractography"]
                if tj.status not in {"completed", "failed", "cancelled"}:
                    begin_stage("tractography", "Tractography", 10)
                    db.save()
                    try:
                        runner_path = _ensure_stage_runner_script(data_root, force=True)
                        stage_args = [
                            python_exe,
                            "-u",
                            str(runner_path),
                            "--stage",
                            "tractography",
                            "--id",
                            str(subject.name),
                            "--data-dir",
                            str(data_root),
                        ]
                        await _run_cmd_logged(
                            stage_args,
                            cwd=pbrain_cwd,
                            env=_python_env_for_pbrain(),
                            write_line_sync=write_line_sync,
                        )
                        finish_stage("tractography")
                    except Exception as exc:
                        err = str(exc)
                        _set_job(tj, status="failed", progress=tj.progress, step="Failed", error=err)
                        tj.endTime = _now_iso()
                        _set_stage_status(subject, "tractography", "failed")
                        db.save()
                        raise

            # Run connectome as a separate post-step when tractography is present.
            if subject.hasDiffusion and jobs.get("connectome"):
                cj = jobs["connectome"]
                if cj.status not in {"completed", "failed", "cancelled"}:
                    begin_stage("connectome", "Connectome", 10)
                    db.save()
                    try:
                        runner_path = _ensure_stage_runner_script(data_root, force=True)
                        stage_args = [
                            python_exe,
                            "-u",
                            str(runner_path),
                            "--stage",
                            "connectome",
                            "--id",
                            str(subject.name),
                            "--data-dir",
                            str(data_root),
                        ]
                        await _run_cmd_logged(
                            stage_args,
                            cwd=pbrain_cwd,
                            env=_python_env_for_pbrain(),
                            write_line_sync=write_line_sync,
                        )
                        finish_stage("connectome")
                    except Exception as exc:
                        err = str(exc)
                        _set_job(cj, status="failed", progress=cj.progress, step="Failed", error=err)
                        cj.endTime = _now_iso()
                        _set_stage_status(subject, "connectome", "failed")
                        db.save()
                        raise

            # If p-brain exits cleanly but some stages never emitted markers, complete them.
            for stage in stage_order:
                if jobs[stage].status in {"queued", "running"}:
                    _set_job(jobs[stage], status="completed", progress=100, step="Completed")
                    jobs[stage].endTime = _now_iso()
                    _set_stage_status(subject, stage, "done")
            db.save()

        except Exception as exc:
            # Catch any early failures (permissions, missing PBRAIN_MAIN_PY, etc.) so the UI doesn't
            # get stuck in "Queued" forever.
            err = str(exc)
            for stage in stage_order:
                if stage in jobs and jobs[stage].status in {"queued", "running"}:
                    _set_job(jobs[stage], status="failed", progress=jobs[stage].progress, step="Failed", error=err)
                    jobs[stage].endTime = _now_iso()
                    _set_stage_status(subject, stage, "failed")
            db.save()
            raise

        finally:
            try:
                if log_fh:
                    log_fh.close()
            except Exception:
                pass
            for job in jobs.values():
                db._job_processes.pop(job.id, None)
                db._job_tasks.pop(job.id, None)
            db._subject_job_ids.pop(subject.id, None)
            db._subject_stage_index.pop(subject.id, None)
async def _warm_backend_once() -> None:
    """Kick off heavyweight imports and basic filesystem prep in the background."""

    global _warm_started, _warm_finished, _warm_error, _warm_steps
    async with _warm_lock:
        if _warm_finished:
            return
        if _warm_started:
            return
        _warm_started = True

    steps: Dict[str, float] = {}
    try:
        t0 = time.perf_counter()
        await asyncio.to_thread(_user_data_dir().mkdir, parents=True, exist_ok=True)
        steps["data_dir"] = round(time.perf_counter() - t0, 3)

        t1 = time.perf_counter()
        await asyncio.to_thread(_load_numpy)
        steps["numpy"] = round(time.perf_counter() - t1, 3)

        t2 = time.perf_counter()
        await asyncio.to_thread(_load_nibabel)
        steps["nibabel"] = round(time.perf_counter() - t2, 3)

        t3 = time.perf_counter()
        await asyncio.to_thread(_get_least_squares)
        steps["scipy_least_squares"] = round(time.perf_counter() - t3, 3)

        _warm_steps = steps
        _warm_finished = True
        _warm_error = ""
    except Exception as exc:
        _warm_error = str(exc)
    finally:
        async with _warm_lock:
            _warm_started = False


def _schedule_warmup() -> bool:
    if _warm_finished or _warm_started:
        return False
    try:
        loop = asyncio.get_event_loop()
        loop.create_task(_warm_backend_once())
        return True
    except Exception:
        return False


@app.on_event("startup")
async def _on_startup_warm() -> None:  # pragma: no cover - best-effort background task
    _schedule_warmup()


def _mount_frontend_if_present() -> None:
    """Serve the built React app from this backend if `../dist` exists.

    This is the recommended "neuroscientist" path: one local URL over HTTP,
    no GitHub Pages, no HTTPS, and no certificate prompts.
    """

    try:
        # Allow packaged launchers to point at a UI directory explicitly.
        override = os.environ.get("PBRAIN_WEB_DIST")
        if override:
            dist_dir = Path(override).expanduser().resolve()
        else:
            dist_dir = Path(__file__).resolve().parent.parent / "dist"
        index = dist_dir / "index.html"
        if not index.exists():
            return
        app.mount("/", StaticFiles(directory=str(dist_dir), html=True), name="ui")
    except Exception:
        # If anything goes wrong, keep API running.
        return


def _get_settings() -> Dict[str, Any]:
    # Keep a small, stable schema and tolerate older db.json.
    s = db.settings or {}
    return {
        "firstName": str(s.get("firstName") or ""),
        "onboardingCompleted": bool(s.get("onboardingCompleted") or False),
        "pbrainMainPy": str(s.get("pbrainMainPy") or ""),
        "fastsurferDir": str(s.get("fastsurferDir") or ""),
        "freesurferHome": str(s.get("freesurferHome") or ""),
        "pbrainT1Fit": str(s.get("pbrainT1Fit") or ""),
        "pbrainVfaGlob": str(s.get("pbrainVfaGlob") or ""),
    }


def _set_settings(patch: Dict[str, Any]) -> Dict[str, Any]:
    current = _get_settings()
    for k, v in (patch or {}).items():
        if v is None:
            continue
        if k in current:
            current[k] = v
    # normalize strings
    for k in [
        "firstName",
        "pbrainMainPy",
        "fastsurferDir",
        "freesurferHome",
        "pbrainT1Fit",
        "pbrainVfaGlob",
    ]:
        current[k] = str(current.get(k) or "").strip()
    # normalize enumerations
    if current.get("pbrainT1Fit"):
        v = str(current.get("pbrainT1Fit") or "").strip().lower()
        if v not in {"auto", "ir", "vfa", "none"}:
            v = ""
        current["pbrainT1Fit"] = v
    current["onboardingCompleted"] = bool(current.get("onboardingCompleted") or False)
    db.settings = current
    db.save()
    return current


def _resolve_pbrain_main_py() -> str:
    env = os.environ.get("PBRAIN_MAIN_PY")
    if env and env.strip():
        return env.strip()
    s = _get_settings().get("pbrainMainPy")
    configured = str(s or "").strip()
    if configured:
        return configured

    # Fresh installs commonly have neither PBRAIN_MAIN_PY nor settings configured.
    # Try a one-time auto-discovery using the existing deps scanner and persist it.
    global _AUTO_SCANNED_PBRAIN_MAIN
    try:
        if not _AUTO_SCANNED_PBRAIN_MAIN:
            _AUTO_SCANNED_PBRAIN_MAIN = True
            res = _scan_system_deps(apply=True)
            found = (res.get("found") or {}).get("pbrainMainPy")
            found_s = str(found or "").strip()
            if found_s:
                return found_s
    except Exception:
        pass

    return ""


_AUTO_SCANNED_PBRAIN_MAIN = False


def _system_deps() -> Dict[str, Any]:
    s = _get_settings()

    pbrain_main = _resolve_pbrain_main_py()
    pbrain_ok = bool(pbrain_main) and Path(pbrain_main).expanduser().exists()

    # FreeSurfer
    recon_all = shutil.which("recon-all")
    fs_home = (os.environ.get("FREESURFER_HOME") or s.get("freesurferHome") or "").strip()
    fs_home_ok = bool(fs_home) and Path(fs_home).expanduser().exists()
    freesurfer_ok = bool(recon_all) or fs_home_ok

    # FastSurfer
    fastsurfer_dir = (s.get("fastsurferDir") or "").strip()
    run_sh = ""
    if fastsurfer_dir:
        candidate = Path(fastsurfer_dir).expanduser() / "run_fastsurfer.sh"
        if candidate.exists():
            run_sh = str(candidate)
    fastsurfer_ok = bool(run_sh)

    return {
        "pbrainMainPy": {
            "configured": pbrain_main,
            "exists": pbrain_ok,
        },
        "freesurfer": {
            "reconAll": recon_all or "",
            "freesurferHome": fs_home,
            "ok": freesurfer_ok,
        },
        "fastsurfer": {
            "fastsurferDir": fastsurfer_dir,
            "runScript": run_sh,
            "ok": fastsurfer_ok,
        },
    }


def _looks_like_pbrain_repo_dir(p: Path) -> bool:
    try:
        return p.is_dir() and (p / "main.py").exists()
    except Exception:
        return False


def _looks_like_fastsurfer_repo_dir(p: Path) -> bool:
    try:
        return p.is_dir() and (p / "run_fastsurfer.sh").exists()
    except Exception:
        return False


def _find_first_existing(paths: List[Path]) -> Optional[Path]:
    for p in paths:
        try:
            pp = p.expanduser().resolve()
        except Exception:
            continue
        if pp.exists():
            return pp
    return None


def _walk_find_repo(
    roots: List[Path],
    repo_dir_name: str,
    predicate,
    *,
    max_depth: int = 4,
    max_dirs: int = 6000,
) -> Optional[Path]:
    """Best-effort shallow search for a repo-like directory.

    Avoids scanning huge trees by skipping hidden folders and common build caches.
    """

    skip_names = {
        ".git",
        "node_modules",
        "dist",
        "build",
        "__pycache__",
        ".venv",
        "venv",
        "Library",
        ".cache",
    }

    scanned = 0
    for root in roots:
        try:
            r = root.expanduser().resolve()
        except Exception:
            continue
        if not r.exists() or not r.is_dir():
            continue

        # Quick direct hit: root/<repo_dir_name>
        direct = r / repo_dir_name
        if predicate(direct):
            return direct

        for dirpath, dirnames, _filenames in os.walk(r):
            scanned += 1
            if scanned > max_dirs:
                return None

            try:
                d = Path(dirpath)
                depth = 0
                try:
                    depth = len(d.relative_to(r).parts)
                except Exception:
                    depth = 0
                if depth >= max_depth:
                    dirnames[:] = []
                    continue
            except Exception:
                continue

            dirnames[:] = [
                n
                for n in dirnames
                if n not in skip_names and not n.startswith(".") and not n.startswith("_")
            ]

            if repo_dir_name in dirnames:
                candidate = Path(dirpath) / repo_dir_name
                if predicate(candidate):
                    return candidate

    return None


def _scan_system_deps(apply: bool) -> Dict[str, Any]:
    s = _get_settings()

    home = Path.home()
    common_roots = [
        home,
        home / "Desktop",
        home / "Documents",
        home / "Downloads",
        home / "src",
        home / "code",
        home / "repos",
        home / "git",
    ]

    # p-Brain
    pbrain_main_current = str((s.get("pbrainMainPy") or "").strip())
    pbrain_main_candidate: Optional[str] = None
    if pbrain_main_current:
        try:
            pp = Path(pbrain_main_current).expanduser()
            if pp.exists():
                pbrain_main_candidate = str(pp.resolve())
        except Exception:
            pass

    if not pbrain_main_candidate:
        # First check a few obvious candidates
        obvious = [
            home / "p-brain" / "main.py",
            home / "Desktop" / "p-brain" / "main.py",
            home / "Documents" / "p-brain" / "main.py",
            home / "src" / "p-brain" / "main.py",
            home / "code" / "p-brain" / "main.py",
            home / "repos" / "p-brain" / "main.py",
        ]
        hit = _find_first_existing(obvious)
        if hit:
            pbrain_main_candidate = str(hit)
        else:
            repo = _walk_find_repo(common_roots, "p-brain", _looks_like_pbrain_repo_dir)
            if repo:
                pbrain_main_candidate = str((repo / "main.py").resolve())

    # FastSurfer
    fastsurfer_dir_current = str((s.get("fastsurferDir") or "").strip())
    fastsurfer_dir_candidate: Optional[str] = None
    if fastsurfer_dir_current:
        try:
            fp = Path(fastsurfer_dir_current).expanduser()
            if _looks_like_fastsurfer_repo_dir(fp):
                fastsurfer_dir_candidate = str(fp.resolve())
        except Exception:
            pass
    if not fastsurfer_dir_candidate:
        obvious_fs = [
            home / "FastSurfer",
            home / "Desktop" / "FastSurfer",
            home / "Documents" / "FastSurfer",
            home / "src" / "FastSurfer",
            home / "code" / "FastSurfer",
            home / "repos" / "FastSurfer",
        ]
        repo = _find_first_existing(obvious_fs)
        if repo and _looks_like_fastsurfer_repo_dir(repo):
            fastsurfer_dir_candidate = str(repo)
        else:
            repo = _walk_find_repo(common_roots, "FastSurfer", _looks_like_fastsurfer_repo_dir)
            if repo:
                fastsurfer_dir_candidate = str(repo.resolve())

    # FreeSurfer (we only set freesurferHome when it looks like a real folder)
    fs_home_current = str((s.get("freesurferHome") or "").strip())
    freesurfer_home_candidate: Optional[str] = None
    if fs_home_current:
        try:
            fh = Path(fs_home_current).expanduser()
            if fh.exists():
                freesurfer_home_candidate = str(fh.resolve())
        except Exception:
            pass
    if not freesurfer_home_candidate:
        default_homes = [
            Path("/Applications/freesurfer"),
            Path("/usr/local/freesurfer"),
            home / "freesurfer",
        ]
        hit = _find_first_existing(default_homes)
        if hit:
            freesurfer_home_candidate = str(hit)

    patch: Dict[str, Any] = {}
    if apply:
        if pbrain_main_candidate:
            patch["pbrainMainPy"] = pbrain_main_candidate
        if fastsurfer_dir_candidate:
            patch["fastsurferDir"] = fastsurfer_dir_candidate
        if freesurfer_home_candidate:
            patch["freesurferHome"] = freesurfer_home_candidate
        if patch:
            _set_settings(patch)

    return {
        "ok": True,
        "applied": bool(apply and patch),
        "settingsPatch": patch,
        "found": {
            "pbrainMainPy": pbrain_main_candidate or "",
            "fastsurferDir": fastsurfer_dir_candidate or "",
            "freesurferHome": freesurfer_home_candidate or "",
        },
        "deps": _system_deps(),
    }


@app.get("/settings")
def get_settings() -> Dict[str, Any]:
    return _get_settings()


@app.patch("/settings")
def patch_settings(req: UpdateAppSettingsRequest) -> Dict[str, Any]:
    patch = req.model_dump(exclude_unset=True)
    return _set_settings(patch)


@app.get("/system/deps")
def get_system_deps() -> Dict[str, Any]:
    return _system_deps()


@app.post("/system/deps/scan")
def scan_system_deps(req: ScanSystemDepsRequest) -> Dict[str, Any]:
    return _scan_system_deps(apply=bool(req.apply))


@app.post("/system/deps/pbrain/install")
def install_pbrain(req: InstallPBrainRequest) -> Dict[str, Any]:
    install_dir = Path(req.installDir).expanduser().resolve()
    install_dir.mkdir(parents=True, exist_ok=True)

    if shutil.which("git") is None:
        raise HTTPException(status_code=500, detail="git is not installed")

    target = install_dir / "p-brain"
    main_py = target / "main.py"
    if target.exists() and main_py.exists():
        _set_settings({"pbrainMainPy": str(main_py)})
        return {"ok": True, "pbrainDir": str(target), "pbrainMainPy": str(main_py)}

    if target.exists() and not main_py.exists():
        raise HTTPException(status_code=409, detail=f"{target} exists but does not look like p-brain")

    try:
        subprocess.run(
            ["git", "clone", "--depth", "1", "https://github.com/edtireli/p-brain.git", str(target)],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
    except subprocess.CalledProcessError as e:
        out = (e.stdout or "").strip()
        raise HTTPException(status_code=500, detail=f"p-brain clone failed: {out[-800:]}")

    if not main_py.exists():
        raise HTTPException(status_code=500, detail="p-brain clone completed but main.py not found")

    _set_settings({"pbrainMainPy": str(main_py)})
    return {"ok": True, "pbrainDir": str(target), "pbrainMainPy": str(main_py)}


@app.post("/system/deps/pbrain/requirements/install")
def install_pbrain_requirements(req: InstallPBrainRequirementsRequest) -> Dict[str, Any]:
    pbrain_dir: Optional[Path] = None
    if req.pbrainDir:
        try:
            pbrain_dir = Path(req.pbrainDir).expanduser().resolve()
        except Exception:
            raise HTTPException(status_code=400, detail="Invalid pbrainDir")
    else:
        configured = _resolve_pbrain_main_py()
        if configured:
            try:
                pbrain_dir = Path(configured).expanduser().resolve().parent
            except Exception:
                pbrain_dir = None

    if not pbrain_dir or not pbrain_dir.exists() or not pbrain_dir.is_dir():
        raise HTTPException(status_code=400, detail="p-brain directory not found; set pbrainMainPy first")

    req_file = pbrain_dir / "requirements.txt"
    if not req_file.exists():
        raise HTTPException(status_code=404, detail="requirements.txt not found in p-brain directory")

    # Prefer using the backend runtime; if pip is not available (packaged builds), fall back to python3/python.
    candidates: List[List[str]] = []
    if sys.executable:
        candidates.append([sys.executable, "-m", "pip", "install", "-r", str(req_file)])
    for exe in ["python3", "python"]:
        p = shutil.which(exe)
        if p:
            candidates.append([p, "-m", "pip", "install", "-r", str(req_file)])

    last_out = ""
    for cmd in candidates:
        try:
            r = subprocess.run(
                cmd,
                check=True,
                cwd=str(pbrain_dir),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
            )
            out = (r.stdout or "").strip()
            return {
                "ok": True,
                "command": " ".join(cmd),
                "pbrainDir": str(pbrain_dir),
                "outputTail": out[-2000:],
            }
        except subprocess.CalledProcessError as e:
            last_out = (e.stdout or "").strip()
            continue

    raise HTTPException(status_code=500, detail=f"pip install failed: {last_out[-800:]}")


@app.post("/system/deps/fastsurfer/install")
def install_fastsurfer(req: InstallFastSurferRequest) -> Dict[str, Any]:
    install_dir = Path(req.installDir).expanduser().resolve()
    install_dir.mkdir(parents=True, exist_ok=True)

    if shutil.which("git") is None:
        raise HTTPException(status_code=500, detail="git is not installed")

    target = install_dir / "FastSurfer"
    if target.exists() and (target / "run_fastsurfer.sh").exists():
        _set_settings({"fastsurferDir": str(target)})
        return {"ok": True, "fastsurferDir": str(target)}

    if target.exists() and not (target / "run_fastsurfer.sh").exists():
        raise HTTPException(status_code=409, detail=f"{target} exists but does not look like FastSurfer")

    try:
        subprocess.run(
            ["git", "clone", "--depth", "1", "https://github.com/Deep-MI/FastSurfer.git", str(target)],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
    except subprocess.CalledProcessError as e:
        out = (e.stdout or "").strip()
        raise HTTPException(status_code=500, detail=f"FastSurfer clone failed: {out[-800:]}")

    if not (target / "run_fastsurfer.sh").exists():
        raise HTTPException(status_code=500, detail="FastSurfer clone completed but run_fastsurfer.sh not found")

    _set_settings({"fastsurferDir": str(target)})
    return {"ok": True, "fastsurferDir": str(target)}


def _frontend_dist_dir() -> Optional[Path]:
    """Return the directory containing the built frontend, if available."""

    try:
        override = os.environ.get("PBRAIN_WEB_DIST")
        if override:
            dist_dir = Path(override).expanduser().resolve()
        else:
            dist_dir = Path(__file__).resolve().parent.parent / "dist"
        index = dist_dir / "index.html"
        if not index.exists():
            return None
        return dist_dir
    except Exception:
        return None


def _is_local_origin(req: Request) -> bool:
    """Basic CSRF guard for endpoints that change allowed filesystem roots.

    We only accept requests coming from the local app (or no Origin header).
    """

    origin = (req.headers.get("origin") or "").strip()
    if not origin or origin.lower() == "null":
        return True
    try:
        from urllib.parse import urlparse

        u = urlparse(origin)
        host = (u.hostname or "").lower()
        return host in {"127.0.0.1", "localhost"}
    except Exception:
        return False


@app.get("/", include_in_schema=False)
def _frontend_index() -> FileResponse:
    """Serve the SPA entrypoint when running with a bundled frontend."""

    dist_dir = _frontend_dist_dir()
    if not dist_dir:
        raise HTTPException(status_code=404, detail="Not Found")
    return FileResponse(str(dist_dir / "index.html"))


class AllowRootRequest(BaseModel):
    path: str


@app.post("/local/allow-root")
def allow_root(body: AllowRootRequest, request: Request) -> Dict[str, Any]:
    if not _is_local_origin(request):
        raise HTTPException(status_code=403, detail="Forbidden")

    raw = (body.path or "").strip()
    if not raw:
        raise HTTPException(status_code=400, detail="Missing path")
    try:
        p = Path(raw).expanduser().resolve()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid path")
    if not p.is_absolute() or not p.exists() or not p.is_dir():
        raise HTTPException(status_code=400, detail="Path must be an existing directory")

    _set_dynamic_allowed_root(str(p))
    return {"ok": True, "path": str(p)}




@app.get("/_spark/loaded")
def spark_loaded() -> Dict[str, Any]:
    return {"ok": True}


@app.get("/health")
def health() -> Dict[str, Any]:
    return {"ok": True, "time": _now_iso()}


@app.post("/system/backend/restart")
def system_backend_restart(request: Request) -> Dict[str, Any]:
    """Request the backend process to exit so the launcher can restart it.

    Note: in dev (uvicorn launched manually), this will stop the server but will
    not automatically restart it unless an external supervisor is running.
    """

    if not _is_local_origin(request):
        raise HTTPException(status_code=403, detail="Forbidden")

    delay_ms = 250

    def _exit_soon() -> None:
        try:
            time.sleep(delay_ms / 1000.0)
        except Exception:
            pass
        os._exit(0)

    try:
        t = threading.Thread(target=_exit_soon, daemon=True)
        t.start()
    except Exception:
        # Worst-case: still return OK; user can restart manually.
        pass

    return {"ok": True, "willExit": True, "delayMs": delay_ms}


@app.post("/system/backend/refresh-runners")
def system_backend_refresh_runners(request: Request) -> Dict[str, Any]:
    """Rewrite stage-runner scripts for all known projects."""

    if not _is_local_origin(request):
        raise HTTPException(status_code=403, detail="Forbidden")

    refreshed: List[str] = []
    skipped: List[str] = []

    for p in list(db.projects or []):
        try:
            root = Path(str(p.storagePath)).expanduser().resolve()
            if not root.exists():
                skipped.append(str(root))
                continue
            runner = _ensure_stage_runner_script(root, force=True)
            refreshed.append(str(runner))
        except Exception:
            try:
                skipped.append(str(getattr(p, "storagePath", "")) or "")
            except Exception:
                skipped.append("")

    return {"ok": True, "refreshed": refreshed, "skipped": skipped}


@app.post("/system/warm")
async def warm_backend() -> Dict[str, Any]:
    started = _schedule_warmup()
    # If a warmup is already running, wait briefly so the caller sees progress.
    if not started and _warm_started:
        try:
            await asyncio.wait_for(_warm_lock.acquire(), timeout=0.05)
            _warm_lock.release()
        except Exception:
            pass
    return {
        "started": started,
        "done": _warm_finished,
        "error": _warm_error,
        "steps": dict(_warm_steps or {}),
    }


def _allowed_roots() -> List[Path]:
    # Runtime-augmented roots (e.g., chosen via native folder picker).
    global _DYNAMIC_ALLOWED_ROOTS
    raw = os.environ.get("PBRAIN_ALLOWED_ROOTS")
    if raw is None:
        raw = os.environ.get("PBRAIN_STORAGE_ROOT")
    roots = []
    for part in str(raw or "").split(","):
        p = part.strip()
        if not p:
            continue
        try:
            roots.append(Path(p).expanduser().resolve())
        except Exception:
            continue

    # Add dynamic roots last (take precedence for UI flows).
    try:
        for r in list(_DYNAMIC_ALLOWED_ROOTS or []):
            try:
                rr = Path(r).expanduser().resolve()
                if rr not in roots:
                    roots.append(rr)
            except Exception:
                continue
    except Exception:
        pass
    return roots


_DYNAMIC_ALLOWED_ROOTS: List[Path] = []


def _set_dynamic_allowed_root(root: Path) -> None:
    global _DYNAMIC_ALLOWED_ROOTS
    try:
        rr = root.expanduser().resolve()
    except Exception:
        return
    _DYNAMIC_ALLOWED_ROOTS = [rr]


def _require_allowed_roots() -> List[Path]:
    roots = _allowed_roots()
    if not roots:
        raise HTTPException(
            status_code=400,
            detail="Local file access disabled. Set PBRAIN_STORAGE_ROOT (or PBRAIN_ALLOWED_ROOTS) on the backend.",
        )
    return roots


def _assert_path_allowed(p: Path) -> Path:
    roots = _require_allowed_roots()
    try:
        rp = p.expanduser().resolve()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid path")
    for r in roots:
        try:
            if rp.is_relative_to(r):
                return rp
        except Exception:
            # Python <3.9 compatibility not needed here, but keep safe.
            rp_str = str(rp)
            r_str = str(r)
            if rp_str == r_str or rp_str.startswith(r_str.rstrip(os.sep) + os.sep):
                return rp
    raise HTTPException(status_code=403, detail="Path not allowed")


@app.get("/local/list")
def local_list(dir: str, glob: str = "*", recursive: bool = True, limit: int = 500) -> Dict[str, Any]:
    """List files under a local directory (guarded by PBRAIN_STORAGE_ROOT).

    Intended for local-only UI use. Returns absolute paths.
    """

    root = _assert_path_allowed(Path(dir))
    if not root.exists() or not root.is_dir():
        return {"files": []}

    max_n = int(limit) if isinstance(limit, (int, str)) else 500
    max_n = max(1, min(max_n, 5000))

    patt = str(glob or "*")
    out: List[Dict[str, Any]] = []

    # Use fnmatch against relative paths for flexibility.
    def iter_paths() -> Any:
        if recursive:
            yield from root.rglob("*")
        else:
            yield from root.glob("*")

    for p in iter_paths():
        if len(out) >= max_n:
            break
        if not p.is_file():
            continue
        if p.name == ".DS_Store" or p.name.startswith("._"):
            continue
        rel = p.relative_to(root).as_posix()
        if not fnmatch.fnmatch(rel, patt) and not fnmatch.fnmatch(p.name, patt):
            continue
        out.append({"name": p.name, "path": str(p)})

    return {"files": out}


@app.get("/local/file")
def local_file(path: str) -> FileResponse:
    """Serve a local file by absolute path (guarded by PBRAIN_STORAGE_ROOT)."""

    p = _assert_path_allowed(Path(path))
    if not p.exists() or not p.is_file():
        raise HTTPException(status_code=404, detail="File not found")

    lower = p.name.lower()
    if lower.endswith(".nii") or lower.endswith(".nii.gz"):
        mt = "application/octet-stream"
    elif lower.endswith(".json"):
        mt = "application/json"
    elif lower.endswith(".png"):
        mt = "image/png"
    else:
        mt = "application/octet-stream"
    return FileResponse(str(p), media_type=mt)


@app.get("/local/analysis/curves")
def local_analysis_curves(subjectDir: str) -> Dict[str, Any]:
    d = _assert_path_allowed(Path(subjectDir))
    return {"curves": _analysis_curves_from_dir(d)}


@app.get("/local/analysis/maps")
def local_analysis_maps(subjectDir: str) -> Dict[str, Any]:
    d = _assert_path_allowed(Path(subjectDir))
    return {"maps": _analysis_map_volumes_from_dir(d)}


@app.get("/local/analysis/metrics")
def local_analysis_metrics(subjectDir: str, view: str = "atlas") -> Dict[str, Any]:
    d = _assert_path_allowed(Path(subjectDir))
    return _analysis_metrics_table_from_dir(d, view=view)


@app.get("/local/montages")
def local_montages(subjectDir: str) -> Dict[str, Any]:
    d = _assert_path_allowed(Path(subjectDir))
    return {"montages": _montage_images_from_dir(d)}


@app.post("/local/pick-folder")
def local_pick_folder() -> Dict[str, Any]:
    """Open a native folder picker on the machine running this backend.

    Returns the absolute selected path and whitelists it for subsequent /local/* access.
    """

    try:
        import tkinter as tk
        from tkinter import filedialog
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Folder picker unavailable (tkinter not installed): {e}")

    try:
        root = tk.Tk()
        root.withdraw()
        root.attributes('-topmost', True)
        path = filedialog.askdirectory(title="Select patient data folder")
        try:
            root.destroy()
        except Exception:
            pass
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Folder picker failed: {e}")

    if not path:
        return {"cancelled": True}

    p = Path(path).expanduser().resolve()
    _set_dynamic_allowed_root(p)
    return {"path": str(p)}


@app.get("/local/resolve-storage-path")
def local_resolve_storage_path(folderName: str, sampleSubject: Optional[str] = None, limit: int = 5) -> Dict[str, Any]:
    """Resolve a dropped folder name (e.g. 'data') into an absolute storage path.

    Browsers often cannot provide absolute paths for drag-and-drop folders.
    This endpoint searches within allowed roots (PBRAIN_ALLOWED_ROOTS / PBRAIN_STORAGE_ROOT)
    and common mount points for a directory whose basename matches folderName and that
    contains plausible subject directories.
    """

    folder = (folderName or "").strip()
    if not folder or "/" in folder or "\\" in folder:
        raise HTTPException(status_code=400, detail="Invalid folderName")

    sample = (sampleSubject or "").strip()
    subj_re = re.compile(r"^\d{8}x\d+$", re.IGNORECASE)
    if sample and not subj_re.match(sample):
        # Keep sample optional; ignore invalid sample values.
        sample = ""

    # Candidate roots: configured allowed roots, plus common macOS mounts.
    roots: List[Path] = []
    try:
        roots.extend(_require_allowed_roots())
    except Exception:
        # If local file access isn't configured, we can't do anything meaningful.
        raise

    extra = [Path("/Volumes")]  # macOS external disks
    for p in extra:
        try:
            rp = p.expanduser().resolve()
            if rp.exists() and rp.is_dir() and rp not in roots:
                roots.append(rp)
        except Exception:
            continue

    max_limit = int(limit) if isinstance(limit, (int, str)) else 5
    max_limit = max(1, min(max_limit, 20))

    def score_candidate(cand: Path) -> int:
        score = 0
        try:
            if sample and (cand / sample).exists():
                score += 100
        except Exception:
            pass
        # Count a few subject-like subdirectories.
        try:
            n = 0
            for entry in os.scandir(cand):
                if not entry.is_dir():
                    continue
                name = entry.name
                if name.startswith("."):
                    continue
                if subj_re.match(name):
                    n += 1
                    if n >= 10:
                        break
            score += min(n, 10)
        except Exception:
            pass
        return score

    found: List[Tuple[int, Path]] = []

    def try_add_dir(p: Path) -> None:
        try:
            rp = p.expanduser().resolve()
            # Ensure candidate is within allowed roots to keep access guard consistent.
            _assert_path_allowed(rp)
            if not rp.exists() or not rp.is_dir():
                return
            found.append((score_candidate(rp), rp))
        except Exception:
            return

    # 1) Check direct children of each root.
    for r in roots:
        try_add_dir(r / folder)

    # 2) For /Volumes, check /Volumes/*/<folder>
    for r in roots:
        if str(r).rstrip(os.sep) == "/Volumes":
            try:
                for entry in os.scandir(r):
                    if not entry.is_dir():
                        continue
                    try_add_dir(Path(entry.path) / folder)
            except Exception:
                pass

    # Pick best scored candidates.
    found.sort(key=lambda x: (-x[0], str(x[1]).lower()))
    out: List[Dict[str, Any]] = []
    seen: set = set()
    for sc, p in found:
        s = str(p)
        if s in seen:
            continue
        seen.add(s)
        out.append({"path": s, "score": sc})
        if len(out) >= max_limit:
            break

    return {"candidates": out}


@app.post("/_spark/loaded")
def spark_loaded_post() -> Dict[str, Any]:
    return {"ok": True}


@app.get("/projects")
def get_projects() -> List[Dict[str, Any]]:
    return [asdict(p) for p in db.projects]


@app.get("/projects/{project_id}")
def get_project(project_id: str) -> Dict[str, Any]:
    p = _find_project(project_id)
    return asdict(p)


@app.get("/projects/{project_id}/scan-subject-folders")
def scan_subject_folders(project_id: str) -> Dict[str, Any]:
    p = _find_project(project_id)
    root = Path(p.storagePath).expanduser().resolve()
    if not root.exists() or not root.is_dir():
        raise HTTPException(status_code=400, detail="Project storage path is not a directory")

    subjects: List[Dict[str, str]] = []
    subj_name_re = re.compile(r"^\d{8}x\d+$", re.IGNORECASE)
    for entry in os.scandir(root):
        if not entry.is_dir():
            continue
        name = entry.name
        if name.startswith("."):
            continue

        sp = Path(entry.path)
        looks_like_subject = bool(subj_name_re.match(name))
        if not looks_like_subject:
            nifti_dir = sp / "NIfTI"
            looks_like_subject = nifti_dir.exists() and nifti_dir.is_dir()
        if not looks_like_subject:
            continue

        subjects.append({"name": name, "sourcePath": str(Path(entry.path).resolve())})

    subjects.sort(key=lambda x: x["name"].lower())
    return {"subjects": subjects}


@app.post("/projects/{project_id}/folder-structure/preview")
def preview_folder_structure(project_id: str, req: FolderStructurePreviewRequest) -> Dict[str, Any]:
    p = _find_project(project_id)
    root = Path(p.storagePath).expanduser().resolve()
    if not root.exists() or not root.is_dir():
        raise HTTPException(status_code=400, detail="Project storage path is not a directory")

    cfg = req.folderStructure or {}
    subject_folder_pattern = str(cfg.get("subjectFolderPattern") or "{subject_id}").strip() or "{subject_id}"
    subject_glob = subject_folder_pattern.replace("{subject_id}", "*")
    use_nested = bool(cfg.get("useNestedStructure", True))
    nifti_subfolder = str(cfg.get("niftiSubfolder") or "").strip()

    keys_and_patterns: List[Tuple[str, List[str]]] = [
        ("t1", _split_fallback_patterns(cfg.get("t1Pattern"))),
        ("t2", _split_fallback_patterns(cfg.get("t2Pattern"))),
        ("flair", _split_fallback_patterns(cfg.get("flairPattern"))),
        ("dce", _split_fallback_patterns(cfg.get("dcePattern"))),
        ("diffusion", _split_fallback_patterns(cfg.get("diffusionPattern"))),
    ]

    subjects_out: List[Dict[str, Any]] = []
    errors: List[str] = []

    try:
        for entry in os.scandir(root):
            if not entry.is_dir():
                continue
            name = entry.name
            if name.startswith("."):
                continue
            try:
                if not fnmatch.fnmatchcase(name, subject_glob):
                    continue
            except Exception:
                continue

            subject_path = Path(entry.path).resolve()
            base_dir = subject_path
            if use_nested and nifti_subfolder:
                candidate = subject_path / nifti_subfolder
                if candidate.exists() and candidate.is_dir():
                    base_dir = candidate

            files = _iter_nifti_relpaths(base_dir)
            matches: Dict[str, Optional[str]] = {}
            for key, pats in keys_and_patterns:
                matches[key] = _first_match(files, pats)

            subjects_out.append(
                {
                    "name": name,
                    "subjectId": _extract_subject_id_from_pattern(subject_folder_pattern, name),
                    "base": str(base_dir),
                    "matches": matches,
                    "fileCount": len(files),
                }
            )
    except Exception as e:
        errors.append(str(e))

    subjects_out.sort(key=lambda x: str(x.get("name", "")).lower())
    return {"ok": True, "subjects": subjects_out, "errors": errors}


@app.delete("/projects/{project_id}")
def delete_project(project_id: str) -> Dict[str, Any]:
    _find_project(project_id)

    subject_ids = {s.id for s in db.subjects if s.projectId == project_id}

    db.projects = [p for p in db.projects if p.id != project_id]
    db.subjects = [s for s in db.subjects if s.projectId != project_id]
    db.jobs = [j for j in db.jobs if j.projectId != project_id and j.subjectId not in subject_ids]
    db._reindex_jobs()

    # best-effort cleanup of in-memory runners
    for subject_id in subject_ids:
        db._subject_job_ids.pop(subject_id, None)
        db._subject_stage_index.pop(subject_id, None)

    db.save()
    return {"ok": True}


@app.post("/projects")
def create_project(req: CreateProjectRequest) -> Dict[str, Any]:
    project = Project(
        id=f"proj_{int(datetime.utcnow().timestamp()*1000)}",
        name=req.name,
        storagePath=req.storagePath,
        createdAt=_now_iso(),
        updatedAt=_now_iso(),
        copyDataIntoProject=req.copyDataIntoProject,
        config={
            "ctc": {
                "model": "advanced",
                "turboNph": None,
            },
            "pbrain": {
                "t1Fit": "ir",
                "t1m0Force": True,
                "flipAngle": "auto",
            },
        },
    )
    db.projects.append(project)
    db.save()
    return asdict(project)


@app.patch("/projects/{project_id}")
def update_project(project_id: str, req: UpdateProjectRequest) -> Dict[str, Any]:
    p = _find_project(project_id)
    if req.name is not None:
        p.name = str(req.name)
    if req.storagePath is not None:
        p.storagePath = str(req.storagePath)
    if req.copyDataIntoProject is not None:
        p.copyDataIntoProject = bool(req.copyDataIntoProject)
    p.updatedAt = _now_iso()
    db.save()
    return asdict(p)


@app.patch("/projects/{project_id}/config")
def update_project_config(project_id: str, req: UpdateProjectConfigRequest) -> Dict[str, Any]:
    p = _find_project(project_id)
    p.config = _deep_merge(p.config or {}, req.configUpdate or {})
    p.updatedAt = _now_iso()
    db.save()
    return asdict(p)


@app.get("/projects/{project_id}/subjects")
def get_subjects(project_id: str) -> List[Dict[str, Any]]:
    _find_project(project_id)
    return [asdict(s) for s in db.subjects if s.projectId == project_id]


def _read_float_file(p: Path) -> Optional[float]:
    try:
        raw = p.read_text(errors="ignore").strip().splitlines()[0].strip()
        if not raw:
            return None
        v = float(raw)
        if not math.isfinite(v):
            return None
        return v
    except Exception:
        return None


def _project_analysis_dataset(project: Project, view: str) -> Dict[str, Any]:
    v = (view or "total").strip().lower()
    if v not in ("total", "atlas"):
        v = "total"

    subjects = [s for s in db.subjects if s.projectId == project.id]
    rows: List[Dict[str, Any]] = []

    for s in subjects:
        table = _analysis_metrics_table(s, view=v)
        for r in (table.get("rows") or []):
            if not isinstance(r, dict):
                continue
            out = {
                "subjectId": s.id,
                "subjectName": s.name,
                "region": str(r.get("region") or ""),
            }
            for k, val in r.items():
                if k == "region":
                    continue
                out[k] = val
            rows.append(out)

        # Best-effort diffusion summary values (if present).
        try:
            analysis_dir = _analysis_dir_for_subject(s)
            diff_dir = analysis_dir / "diffusion"
            fa_all = _read_float_file(diff_dir / "fa_mean.txt")
            fa_wm = _read_float_file(diff_dir / "fa_mean_wm.txt")
            fa_gm = _read_float_file(diff_dir / "fa_mean_gm.txt")
            if fa_all is not None:
                rows.append({"subjectId": s.id, "subjectName": s.name, "region": "FA_total", "FA": fa_all})
            if fa_wm is not None:
                rows.append({"subjectId": s.id, "subjectName": s.name, "region": "FA_wm", "FA": fa_wm})
            if fa_gm is not None:
                rows.append({"subjectId": s.id, "subjectName": s.name, "region": "FA_gm", "FA": fa_gm})
        except Exception:
            pass

    regions = sorted({str(r.get("region") or "") for r in rows if str(r.get("region") or "")})
    metrics: set = set()
    for r in rows:
        for k in r.keys():
            if k in ("subjectId", "subjectName", "region"):
                continue
            metrics.add(k)

    return {"view": v, "rows": rows, "regions": regions, "metrics": sorted(metrics)}


@app.get("/projects/{project_id}/analysis/dataset")
def get_project_analysis_dataset(project_id: str, view: str = "total") -> Dict[str, Any]:
    p = _find_project(project_id)
    return _project_analysis_dataset(p, view=view)


@app.post("/analysis/stats/pearson")
def analysis_stats_pearson(req: AnalysisPearsonRequest) -> Dict[str, Any]:
    _require_scipy()
    _require_numpy()
    assert np is not None
    from scipy import stats  # type: ignore

    x = np.asarray(req.x, dtype=float)
    y = np.asarray(req.y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    if x.size < 3:
        raise HTTPException(status_code=400, detail="Need at least 3 paired samples")

    r, p = stats.pearsonr(x, y)
    return {"n": int(x.size), "r": float(r), "p": float(p)}


@app.post("/analysis/stats/group-compare")
def analysis_stats_group_compare(req: AnalysisGroupCompareRequest) -> Dict[str, Any]:
    _require_scipy()
    _require_numpy()
    assert np is not None
    from scipy import stats  # type: ignore

    a = np.asarray(req.a, dtype=float)
    b = np.asarray(req.b, dtype=float)
    a = a[np.isfinite(a)]
    b = b[np.isfinite(b)]
    if a.size < 2 or b.size < 2:
        raise HTTPException(status_code=400, detail="Need at least 2 samples per group")

    mean_a = float(np.mean(a))
    mean_b = float(np.mean(b))

    # Welch's t-test
    t, t_p = stats.ttest_ind(a, b, equal_var=False)

    # Mann–Whitney U (robust fallback)
    try:
        mw_u, mw_p = stats.mannwhitneyu(a, b, alternative="two-sided")
    except Exception:
        mw_u, mw_p = float("nan"), float("nan")

    # Cohen's d (pooled SD)
    try:
        sa = float(np.std(a, ddof=1))
        sb = float(np.std(b, ddof=1))
        pooled = math.sqrt(((a.size - 1) * sa * sa + (b.size - 1) * sb * sb) / (a.size + b.size - 2))
        cohen_d = float((mean_a - mean_b) / pooled) if pooled > 0 else 0.0
    except Exception:
        cohen_d = 0.0

    sh_a: Optional[float] = None
    sh_b: Optional[float] = None
    try:
        if 3 <= a.size <= 5000:
            _, sh_a = stats.shapiro(a)
            sh_a = float(sh_a)
    except Exception:
        sh_a = None
    try:
        if 3 <= b.size <= 5000:
            _, sh_b = stats.shapiro(b)
            sh_b = float(sh_b)
    except Exception:
        sh_b = None

    return {
        "na": int(a.size),
        "nb": int(b.size),
        "meanA": mean_a,
        "meanB": mean_b,
        "t": float(t),
        "t_p": float(t_p),
        "mw_u": float(mw_u),
        "mw_p": float(mw_p),
        "cohen_d": float(cohen_d),
        "shapiroA_p": sh_a,
        "shapiroB_p": sh_b,
    }


@app.post("/analysis/stats/ols")
def analysis_stats_ols(req: AnalysisOlsRequest) -> Dict[str, Any]:
    _require_scipy()
    _require_numpy()
    assert np is not None
    from scipy import stats  # type: ignore

    y = np.asarray(req.y, dtype=float)
    X = np.asarray(req.X, dtype=float)
    if y.ndim != 1:
        raise HTTPException(status_code=400, detail="y must be 1D")
    if X.ndim != 2:
        raise HTTPException(status_code=400, detail="X must be 2D")
    if X.shape[0] != y.shape[0]:
        raise HTTPException(status_code=400, detail="X and y must have same number of rows")
    if len(req.columns) != X.shape[1]:
        raise HTTPException(status_code=400, detail="columns length must match X columns")
    if y.size < X.shape[1] + 2:
        raise HTTPException(status_code=400, detail="Not enough samples")

    # Remove non-finite rows
    mask = np.isfinite(y)
    mask = mask & np.all(np.isfinite(X), axis=1)
    y = y[mask]
    X = X[mask, :]
    n = int(y.size)
    p = int(X.shape[1])
    if n < p + 2:
        raise HTTPException(status_code=400, detail="Not enough finite samples")

    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    y_hat = X @ beta
    resid = y - y_hat

    sse = float(np.sum(resid ** 2))
    sst = float(np.sum((y - float(np.mean(y))) ** 2))
    r2 = 1.0 - (sse / sst) if sst > 0 else 0.0

    df_resid = n - p
    sigma2 = sse / df_resid if df_resid > 0 else float("nan")

    # Covariance of beta
    try:
        xtx_inv = np.linalg.inv(X.T @ X)
        cov = sigma2 * xtx_inv
        se = np.sqrt(np.diag(cov))
    except Exception:
        se = np.full((p,), float("nan"), dtype=float)

    coeffs: List[Dict[str, Any]] = []
    for i, name in enumerate(req.columns):
        b = float(beta[i])
        s = float(se[i])
        t = float(b / s) if s and math.isfinite(s) and s != 0 else float("nan")
        pval = float(2.0 * (1.0 - stats.t.cdf(abs(t), df=df_resid))) if math.isfinite(t) and df_resid > 0 else float("nan")
        coeffs.append({"name": str(name), "beta": b, "se": s, "t": t, "p": pval})

    sh_p: Optional[float] = None
    try:
        if 3 <= resid.size <= 5000:
            _, sh_p = stats.shapiro(resid)
            sh_p = float(sh_p)
    except Exception:
        sh_p = None

    return {
        "n": n,
        "df_resid": int(df_resid),
        "r2": float(r2),
        "residual_shapiro_p": sh_p,
        "coefficients": coeffs,
    }


@app.get("/subjects/{subject_id}")
def get_subject(subject_id: str) -> Dict[str, Any]:
    s = _find_subject(subject_id)
    return asdict(s)


@app.get("/subjects/{subject_id}/default-volume")
def get_default_volume(subject_id: str, kind: str = "dce") -> Dict[str, Any]:
    subject = _find_subject(subject_id)
    project = _find_project(subject.projectId)
    p = _resolve_default_volume_path(project, subject, kind)
    return {"path": str(p)}


@app.post("/volumes/info")
def get_volume_info(req: VolumeInfoRequest, subjectId: Optional[str] = None, projectId: Optional[str] = None) -> Dict[str, Any]:
    _require_nibabel()
    if not projectId and subjectId:
        subject = _find_subject(subjectId)
        project = _find_project(subject.projectId)
    elif projectId:
        project = _find_project(projectId)
        subject = _find_subject(subjectId) if subjectId else None
    else:
        raise HTTPException(status_code=400, detail="projectId or subjectId is required")

    p = _safe_resolve_path(project, subject, req.path)
    img = _load_nifti(str(p))
    info = _volume_info_from_img(img, kind=req.kind, path=str(p))
    info["path"] = str(p)
    return info


@app.post("/volumes/slice")
def get_volume_slice(req: VolumeSliceRequest, subjectId: Optional[str] = None, projectId: Optional[str] = None) -> Dict[str, Any]:
    _require_nibabel()
    _require_numpy()
    assert np is not None

    if not projectId and subjectId:
        subject = _find_subject(subjectId)
        project = _find_project(subject.projectId)
    elif projectId:
        project = _find_project(projectId)
        subject = _find_subject(subjectId) if subjectId else None
    else:
        raise HTTPException(status_code=400, detail="projectId or subjectId is required")

    p = _safe_resolve_path(project, subject, req.path)
    img = _load_nifti(str(p))
    shape = img.shape
    if len(shape) < 3:
        raise HTTPException(status_code=400, detail="Volume must be 3D or 4D")

    z = int(req.z)
    if z < 0 or z >= int(shape[2]):
        raise HTTPException(status_code=400, detail="z out of range")

    if len(shape) >= 4:
        t = int(req.t)
        if t < 0 or t >= int(shape[3]):
            raise HTTPException(status_code=400, detail="t out of range")
        arr = np.asanyarray(img.dataobj[:, :, z, t]).astype(float)
    else:
        arr = np.asanyarray(img.dataobj[:, :, z]).astype(float)

    vmin = float(np.nanmin(arr)) if arr.size else 0.0
    vmax = float(np.nanmax(arr)) if arr.size else 0.0
    data = arr.tolist()
    return {"data": data, "min": vmin, "max": vmax}


@app.get("/subjects/{subject_id}/curves")
def get_subject_curves(subject_id: str) -> Dict[str, Any]:
    subject = _find_subject(subject_id)
    curves = _analysis_curves(subject)
    return {"curves": curves}


@app.get("/subjects/{subject_id}/roi-overlays")
def get_subject_roi_overlays(subject_id: str) -> Dict[str, Any]:
    subject = _find_subject(subject_id)
    overlays = _analysis_roi_overlays(subject)
    return {"overlays": overlays}


@app.get("/subjects/{subject_id}/roi-masks")
def get_subject_roi_masks(subject_id: str) -> Dict[str, Any]:
    subject = _find_subject(subject_id)
    project = _find_project(subject.projectId)
    masks = _analysis_roi_mask_volumes(project, subject)
    return {"masks": masks}


class ForcedRoiRef(BaseModel):
    roiType: str
    roiSubType: str
    sliceIndex: int


class InputFunctionForcesRequest(BaseModel):
    forcedAif: Optional[ForcedRoiRef] = None
    forcedVif: Optional[ForcedRoiRef] = None


class IgnoreFromAnalysisRequest(BaseModel):
    ignore: bool


@app.get("/subjects/{subject_id}/input-function-forces")
def get_subject_input_function_forces(subject_id: str) -> Dict[str, Any]:
    subject = _find_subject(subject_id)
    analysis_dir = _analysis_dir_for_subject(subject)
    p = analysis_dir / "input_function_forces.json"
    if not p.exists() or not p.is_file():
        return {"forcedAif": None, "forcedVif": None}
    try:
        with open(p, "r", encoding="utf-8") as f:
            raw = json.load(f)
        if not isinstance(raw, dict):
            return {"forcedAif": None, "forcedVif": None}
        forced_aif = raw.get("forcedAif")
        forced_vif = raw.get("forcedVif")
        return {"forcedAif": forced_aif if isinstance(forced_aif, dict) else None, "forcedVif": forced_vif if isinstance(forced_vif, dict) else None}
    except Exception:
        return {"forcedAif": None, "forcedVif": None}


@app.put("/subjects/{subject_id}/input-function-forces")
def set_subject_input_function_forces(subject_id: str, req: InputFunctionForcesRequest) -> Dict[str, Any]:
    subject = _find_subject(subject_id)
    analysis_dir = _analysis_dir_for_subject(subject)
    analysis_dir.mkdir(parents=True, exist_ok=True)

    def _valid_part(s: str) -> bool:
        ss = str(s or "").strip()
        if not ss:
            return False
        if ".." in ss or "/" in ss or "\\" in ss:
            return False
        return True

    forced_aif = req.forcedAif.dict() if req.forcedAif is not None else None
    forced_vif = req.forcedVif.dict() if req.forcedVif is not None else None

    if forced_aif is not None:
        if str(forced_aif.get("roiType") or "") != "Artery":
            raise HTTPException(status_code=400, detail="forcedAif.roiType must be 'Artery'")
        if not _valid_part(str(forced_aif.get("roiSubType") or "")):
            raise HTTPException(status_code=400, detail="forcedAif.roiSubType is required")
        try:
            forced_aif["sliceIndex"] = int(forced_aif.get("sliceIndex"))
        except Exception:
            raise HTTPException(status_code=400, detail="forcedAif.sliceIndex must be an integer")
        if int(forced_aif["sliceIndex"]) < 0:
            raise HTTPException(status_code=400, detail="forcedAif.sliceIndex must be >= 0")

    if forced_vif is not None:
        if str(forced_vif.get("roiType") or "") != "Vein":
            raise HTTPException(status_code=400, detail="forcedVif.roiType must be 'Vein'")
        if not _valid_part(str(forced_vif.get("roiSubType") or "")):
            raise HTTPException(status_code=400, detail="forcedVif.roiSubType is required")
        try:
            forced_vif["sliceIndex"] = int(forced_vif.get("sliceIndex"))
        except Exception:
            raise HTTPException(status_code=400, detail="forcedVif.sliceIndex must be an integer")
        if int(forced_vif["sliceIndex"]) < 0:
            raise HTTPException(status_code=400, detail="forcedVif.sliceIndex must be >= 0")

    out = {"forcedAif": forced_aif, "forcedVif": forced_vif}
    p = analysis_dir / "input_function_forces.json"
    tmp = analysis_dir / "input_function_forces.json.tmp"
    try:
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2)
            f.write("\n")
        os.replace(tmp, p)
    finally:
        try:
            if tmp.exists():
                tmp.unlink()
        except Exception:
            pass

    return out


def _ignore_from_analysis_path(subject: Subject) -> Path:
    # Store flag in the subject directory (dataset root).
    return Path(subject.sourcePath).expanduser().resolve() / "ignore_from_analysis.json"


@app.get("/subjects/{subject_id}/ignore-from-analysis")
def get_subject_ignore_from_analysis(subject_id: str) -> Dict[str, Any]:
    subject = _find_subject(subject_id)
    p = _ignore_from_analysis_path(subject)
    if not p.exists() or not p.is_file():
        return {"ignore": False}
    try:
        with open(p, "r", encoding="utf-8") as f:
            raw = json.load(f)
        if isinstance(raw, dict) and isinstance(raw.get("ignore"), bool):
            return {"ignore": bool(raw.get("ignore"))}
        if isinstance(raw, bool):
            return {"ignore": bool(raw)}
        return {"ignore": False}
    except Exception:
        return {"ignore": False}


@app.put("/subjects/{subject_id}/ignore-from-analysis")
def set_subject_ignore_from_analysis(subject_id: str, req: IgnoreFromAnalysisRequest) -> Dict[str, Any]:
    subject = _find_subject(subject_id)
    p = _ignore_from_analysis_path(subject)
    payload = {"ignore": bool(req.ignore)}

    # Atomic write best-effort.
    tmp = p.with_suffix(p.suffix + ".tmp")
    try:
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
            f.write("\n")
        os.replace(tmp, p)
    finally:
        try:
            if tmp.exists():
                tmp.unlink()
        except Exception:
            pass

    return payload


@app.get("/subjects/{subject_id}/tractography")
def get_subject_tractography(subject_id: str, path: Optional[str] = None) -> Dict[str, Any]:
    subject = _find_subject(subject_id)
    return _analysis_tractography_streamlines(subject, path_override=path)


@app.get("/subjects/{subject_id}/metrics")
def get_subject_metrics(subject_id: str, view: str = "atlas") -> Dict[str, Any]:
    subject = _find_subject(subject_id)
    return _analysis_metrics_table(subject, view=view)


@app.get("/subjects/{subject_id}/maps")
def get_subject_maps(subject_id: str) -> Dict[str, Any]:
    subject = _find_subject(subject_id)
    maps = _analysis_map_volumes(subject)
    return {"maps": maps}


@app.get("/subjects/{subject_id}/volumes")
def get_subject_volumes(subject_id: str) -> Dict[str, Any]:
    subject = _find_subject(subject_id)
    project = _find_project(subject.projectId)

    nifti_dir = _nifti_dir_for_subject(project, subject)
    volumes: List[Dict[str, Any]] = []
    seen: set[str] = set()

    # Only surface the configured modality volumes.
    # (Avoid listing every .nii in the folder, which includes unrelated intermediates.)
    for kind in ("dce", "t1", "t2", "flair", "diffusion"):
        try:
            p = _resolve_default_volume_path(project, subject, kind)
            sp = str(p)
            if sp not in seen and p.exists() and p.is_file():
                seen.add(sp)
                volumes.append({"id": sp, "name": p.name, "path": sp, "kind": kind})
        except Exception:
            pass

    volumes.sort(key=lambda v: str(v.get("name", "")))
    return {"volumes": volumes}


@app.get("/subjects/{subject_id}/montages")
def get_subject_montages(subject_id: str) -> Dict[str, Any]:
    subject = _find_subject(subject_id)
    montages = _subject_montage_images(subject)
    return {"montages": montages}


@app.get("/subjects/{subject_id}/montages/image")
def get_subject_montage_image(subject_id: str, path: str) -> FileResponse:
    subject = _find_subject(subject_id)
    project = _find_project(subject.projectId)
    p = _safe_resolve_path(project, subject, path)
    if not p.exists() or not p.is_file():
        raise HTTPException(status_code=404, detail="Montage image not found")
    return FileResponse(str(p), media_type="image/png")


@app.get("/subjects/{subject_id}/patlak")
def get_subject_patlak(subject_id: str, region: str = "gm") -> Dict[str, Any]:
    subject = _find_subject(subject_id)
    project = _find_project(subject.projectId)

    t = _load_time_points(subject)
    aif = _load_aif_curve(subject)
    tissue = _load_ai_tissue_curve(subject, _region_to_ai_key(region))
    t, aif, tissue, _dt = _align_curves(t, aif, tissue)

    window_start = float(_get_config_value(project, ["model", "patlakWindowStartFraction"], 0.4))
    return _patlak_from_curves(t, aif, tissue, window_start_fraction=window_start)


@app.get("/subjects/{subject_id}/deconvolution")
def get_subject_deconvolution(subject_id: str, region: str = "gm") -> Dict[str, Any]:
    subject = _find_subject(subject_id)
    project = _find_project(subject.projectId)

    t = _load_time_points(subject)
    aif = _load_aif_curve(subject)
    tissue = _load_ai_tissue_curve(subject, _region_to_ai_key(region))
    t, aif, tissue, dt = _align_curves(t, aif, tissue)

    lambd = float(_get_config_value(project, ["model", "lambdaTikhonov"], 0.1))
    hematocrit = float(_get_config_value(project, ["physiological", "hematocrit"], 0.42))
    tissue_density = float(_get_config_value(project, ["physiological", "tissueDensity"], 1.04))
    return _deconvolution_from_curves(
        t,
        aif,
        tissue,
        dt=dt,
        lambd=lambd,
        hematocrit=hematocrit,
        tissue_density=tissue_density,
    )


def _discover_ai_region_keys(subject: Subject) -> List[str]:
    """List available AI tissue region keys from on-disk segmented median curves."""
    analysis_dir = _analysis_dir_for_subject(subject)
    ai_dir = analysis_dir / "CTC Data" / "Tissue" / "AI"
    if not ai_dir.exists() or not ai_dir.is_dir():
        return []
    keys: set[str] = set()
    for p in ai_dir.glob("*_AI_Tissue_slice_*_segmented_median.npy"):
        if not p.is_file() or p.name.startswith("."):
            continue
        name = p.name
        if "_AI_Tissue_slice_" not in name:
            continue
        key = name.split("_AI_Tissue_slice_")[0]
        key = (key or "").strip()
        if key:
            keys.add(key)
    return sorted(keys)


def _load_central_voxel_tissue_curve(subject: Subject) -> tuple[Any, tuple[int, int, int]]:
    """Fallback tissue curve from a central voxel in brain_concentration_4d.

    Uses Analysis/CTC Data/Tissue/brain_concentration_4d.nii.gz if present.
    """
    _require_numpy()
    _require_nibabel()
    assert np is not None

    analysis_dir = _analysis_dir_for_subject(subject)
    p = analysis_dir / "CTC Data" / "Tissue" / "brain_concentration_4d.nii.gz"
    if not p.exists() or not p.is_file():
        raise HTTPException(status_code=404, detail="Missing brain_concentration_4d for central-voxel fallback")

    img = _load_nifti(str(p))
    shape = tuple(int(x) for x in (img.shape or ()))
    if len(shape) < 4:
        raise HTTPException(status_code=400, detail="brain_concentration_4d is not 4D")
    nx, ny, nz, nt = shape[0], shape[1], shape[2], shape[3]
    if nt < 3:
        raise HTTPException(status_code=400, detail="brain_concentration_4d has insufficient time frames")

    proxy = img.dataobj
    cx, cy, cz = nx // 2, ny // 2, nz // 2

    def try_voxel(x: int, y: int, z: int) -> Optional[Any]:
        try:
            arr = np.asarray(proxy[x, y, z, :], dtype=float).reshape(-1)
        except Exception:
            return None
        if arr.size < 3:
            return None
        if not np.any(np.isfinite(arr)):
            return None
        arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
        if float(np.nanmax(np.abs(arr))) <= 1e-8:
            return None
        return arr

    # Search outward from center in a small cube.
    max_r = 10
    for r in range(0, max_r + 1):
        for dz in range(-r, r + 1):
            z = cz + dz
            if z < 0 or z >= nz:
                continue
            for dy in range(-r, r + 1):
                y = cy + dy
                if y < 0 or y >= ny:
                    continue
                for dx in range(-r, r + 1):
                    x = cx + dx
                    if x < 0 or x >= nx:
                        continue
                    if dx == 0 and dy == 0 and dz == 0:
                        pass
                    curve = try_voxel(x, y, z)
                    if curve is not None:
                        return curve, (x, y, z)

    # Fallback: sample a sparse grid for any non-zero voxel.
    for z in range(0, nz, max(1, nz // 16)):
        for y in range(0, ny, max(1, ny // 16)):
            for x in range(0, nx, max(1, nx // 16)):
                curve = try_voxel(x, y, z)
                if curve is not None:
                    return curve, (x, y, z)

    raise HTTPException(status_code=404, detail="No valid central voxel curve found in brain_concentration_4d")


def _analysis_tissue_mask_volumes(project: Project, subject: Subject) -> List[Dict[str, Any]]:
    """Discover tissue segmentation mask volumes aligned to the subject's DCE space.

    p-brain datasets can include binary NIfTI masks under the subject's NIfTI folder, e.g.:
      NIfTI/segmentation/**/aparc.DKTatlas+aseg.deep_in_DCE_wm.nii.gz

    We expose these as ROI-mask-like volumes so the UI can highlight a region on the DCE slice.
    """

    root = Path(subject.sourcePath).expanduser().resolve()
    nifti_dir = _nifti_dir_for_subject(project, subject)

    # Heuristic scan: look for FreeSurfer/atlas-derived tissue masks already in DCE space.
    candidates: List[Path] = []
    try:
        for p in nifti_dir.rglob("*_in_DCE_*.nii*"):
            if not p.is_file() or p.name.startswith("._"):
                continue
            name = p.name
            if not (name.endswith(".nii") or name.endswith(".nii.gz")):
                continue
            candidates.append(p)
    except Exception:
        candidates = []

    by_key: Dict[str, Path] = {}
    for p in sorted(candidates, key=lambda q: q.as_posix()):
        base = p.name
        base = re.sub(r"\.(nii\.gz|nii)$", "", base, flags=re.IGNORECASE)
        if "_in_DCE_" not in base:
            continue
        key = base.split("_in_DCE_", 1)[1].strip().lower()
        if not key:
            continue
        # Prefer .nii.gz if both exist; otherwise first seen.
        prev = by_key.get(key)
        if prev is None:
            by_key[key] = p
        else:
            prev_gz = prev.name.lower().endswith(".nii.gz")
            cur_gz = p.name.lower().endswith(".nii.gz")
            if (not prev_gz) and cur_gz:
                by_key[key] = p

    out: List[Dict[str, Any]] = []
    for key in sorted(by_key.keys()):
        p = by_key[key]
        # Ensure returned paths are within allowed roots (subject or project).
        try:
            _safe_resolve_path(project, subject, str(p))
        except Exception:
            continue
        out.append(
            {
                "id": f"tissue_mask_{key}",
                "name": f"Tissue mask ({key})",
                "path": str(p),
                "roiType": "Tissue",
                "roiSubType": key,
                "source": "segmentation",
                "subjectRoot": str(root),
            }
        )
    return out


def _deconvolution_lcurve_from_curves(
    t: Any,
    aif: Any,
    tissue: Any,
    *,
    dt: float,
    lambdas: Any,
) -> Dict[str, Any]:
    _require_numpy()
    assert np is not None

    n = int(min(t.size, aif.size, tissue.size))
    t = t[:n]
    aif = aif[:n]
    tissue = tissue[:n]

    A = np.zeros((n, n), dtype=float)
    for i in range(n):
        A[i, : i + 1] = aif[i::-1] * dt
        if i == 0:
            A[i, 0] = 0.0
        else:
            A[i, 0] *= 0.5
            A[i, i] *= 0.5

    ata = A.T @ A
    rhs = A.T @ tissue
    lams = np.asarray(lambdas, dtype=float).reshape(-1)
    lams = lams[np.isfinite(lams) & (lams > 0)]
    if lams.size == 0:
        lams = np.asarray([0.1], dtype=float)

    residual_norms: List[float] = []
    solution_norms: List[float] = []

    eye = np.eye(n, dtype=float)
    for lam in lams:
        lam2 = float(lam) ** 2
        regularised = ata + lam2 * eye
        try:
            g = np.linalg.solve(regularised, rhs)
        except np.linalg.LinAlgError:
            g = np.linalg.lstsq(regularised, rhs, rcond=None)[0]
        r = (A @ g) - tissue
        residual_norms.append(float(np.linalg.norm(r)))
        solution_norms.append(float(np.linalg.norm(g)))

    return {
        "lambdas": lams.astype(float).tolist(),
        "residualNorms": residual_norms,
        "solutionNorms": solution_norms,
    }


@app.get("/subjects/{subject_id}/deconvolution/regions")
def get_subject_deconvolution_regions(subject_id: str, regions: Optional[str] = None) -> Dict[str, Any]:
    """Compute deconvolution outputs for multiple tissue regions.

    - If `regions` is omitted/empty: uses all on-disk AI tissue region keys.
    - If no region curves exist: falls back to a central voxel curve from brain_concentration_4d.
    """
    subject = _find_subject(subject_id)
    project = _find_project(subject.projectId)

    t = _load_time_points(subject)
    aif = _load_aif_curve(subject)

    lambd = float(_get_config_value(project, ["model", "lambdaTikhonov"], 0.1))
    hematocrit = float(_get_config_value(project, ["physiological", "hematocrit"], 0.42))
    tissue_density = float(_get_config_value(project, ["physiological", "tissueDensity"], 1.04))

    requested: List[str] = []
    if regions is not None:
        requested = [p.strip() for p in str(regions).split(",") if p.strip()]

    keys = requested if requested else _discover_ai_region_keys(subject)

    out_regions: List[Dict[str, Any]] = []
    for k in keys:
        try:
            tissue = _load_ai_tissue_curve(subject, k)
            t2, a2, tissue2, dt = _align_curves(t, aif, tissue)
            data = _deconvolution_from_curves(
                t2,
                a2,
                tissue2,
                dt=dt,
                lambd=lambd,
                hematocrit=hematocrit,
                tissue_density=tissue_density,
            )
            out_regions.append({"key": k, "label": k, "data": data})
        except Exception:
            continue

    if out_regions:
        return {"regions": out_regions, "fallback": None}

    # Fallback: central voxel
    tissue, voxel = _load_central_voxel_tissue_curve(subject)
    t2, a2, tissue2, dt = _align_curves(t, aif, tissue)
    data = _deconvolution_from_curves(
        t2,
        a2,
        tissue2,
        dt=dt,
        lambd=lambd,
        hematocrit=hematocrit,
        tissue_density=tissue_density,
    )
    return {
        "regions": [{"key": "central_voxel", "label": "Central voxel", "data": data}],
        "fallback": {"kind": "central_voxel", "voxel": [int(voxel[0]), int(voxel[1]), int(voxel[2])]},
    }


@app.get("/subjects/{subject_id}/tissue-masks")
def get_subject_tissue_masks(subject_id: str) -> Dict[str, Any]:
    subject = _find_subject(subject_id)
    project = _find_project(subject.projectId)
    masks = _analysis_tissue_mask_volumes(project, subject)
    # Keep only the fields the UI expects.
    cleaned = [
        {
            "id": str(m.get("id")),
            "name": str(m.get("name")),
            "path": str(m.get("path")),
            "roiType": str(m.get("roiType")),
            "roiSubType": str(m.get("roiSubType")),
        }
        for m in masks
    ]
    return {"masks": cleaned}


@app.get("/subjects/{subject_id}/deconvolution/lcurve")
def get_subject_deconvolution_lcurve(subject_id: str, region: str, points: int = 32) -> Dict[str, Any]:
    subject = _find_subject(subject_id)
    project = _find_project(subject.projectId)

    region_key = (region or "").strip()
    if not region_key:
        raise HTTPException(status_code=400, detail="region is required")

    t = _load_time_points(subject)
    aif = _load_aif_curve(subject)
    tissue = _load_ai_tissue_curve(subject, region_key)
    t2, a2, tissue2, dt = _align_curves(t, aif, tissue)

    base_lam = float(_get_config_value(project, ["model", "lambdaTikhonov"], 0.1))
    if not math.isfinite(base_lam) or base_lam <= 0:
        base_lam = 0.1

    try:
        npts = int(points)
    except Exception:
        npts = 32
    npts = max(8, min(80, npts))

    lam_min = max(base_lam / 1000.0, 1e-8)
    lam_max = max(base_lam * 1000.0, lam_min * 10.0)
    lams = np.logspace(math.log10(lam_min), math.log10(lam_max), npts)

    out = _deconvolution_lcurve_from_curves(t2, a2, tissue2, dt=dt, lambdas=lams)
    out.update({"region": region_key, "selectedLambda": float(base_lam)})
    return out


def _load_pickle_array(path: Path) -> Any:
    try:
        with open(path, "rb") as f:
            return pickle.load(f)
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to load {path.name}: {exc}")


def _t1_ir_saturation_model(ti_ms: Any, m0: float, t1_ms: float) -> Any:
    np_mod = _load_numpy()
    if np_mod is None:
        return None
    ti = np_mod.asarray(ti_ms, dtype=float)
    t1 = float(t1_ms)
    if not np_mod.isfinite(t1) or t1 <= 0:
        return np_mod.full_like(ti, np_mod.nan, dtype=float)
    return float(m0) * (1.0 - np_mod.exp(-ti / t1))


@app.get("/subjects/{subject_id}/t1fit/ir")
def get_subject_t1fit_ir(subject_id: str) -> Dict[str, Any]:
    """Return central-voxel IR TI-series points and a fitted curve.

    Reads p-brain fitting artifacts from Analysis/Fitting:
    - voxel_matrix.pkl (TI, X, Y, Z)
    - voxel_T1_matrix.pkl (X, Y, Z) [optional]
    - voxel_M0_matrix.pkl (X, Y, Z) [optional]
    """

    subject = _find_subject(subject_id)
    fit_dir = _analysis_dir_for_subject(subject) / "Fitting"

    voxel_matrix_p = fit_dir / "voxel_matrix.pkl"
    t1_p = fit_dir / "voxel_T1_matrix.pkl"
    m0_p = fit_dir / "voxel_M0_matrix.pkl"

    if not voxel_matrix_p.exists() or not voxel_matrix_p.is_file():
        raise HTTPException(status_code=404, detail="Missing Analysis/Fitting/voxel_matrix.pkl")

    _require_numpy()
    _require_scipy()
    np_mod = _load_numpy()
    lsq = _get_least_squares()
    if np_mod is None or lsq is None:
        raise HTTPException(status_code=500, detail="Backend missing dependency: numpy/scipy")

    voxel_matrix = np_mod.asarray(_load_pickle_array(voxel_matrix_p), dtype=float)
    if voxel_matrix.ndim != 4 or voxel_matrix.shape[0] < 2:
        raise HTTPException(status_code=500, detail=f"Unexpected voxel_matrix shape: {tuple(voxel_matrix.shape)}")

    n_ti, sx, sy, sz = voxel_matrix.shape
    x0, y0, z0 = int(sx // 2), int(sy // 2), int(sz // 2)

    def _extract_at(x: int, y: int, z: int) -> Any:
        return np_mod.asarray(voxel_matrix[:, x, y, z], dtype=float).reshape(-1)

    measured = _extract_at(x0, y0, z0)
    if (not np_mod.isfinite(measured).any()) or float(np_mod.nanmax(np_mod.abs(measured))) <= 0:
        try:
            ref = np_mod.asarray(voxel_matrix[-1, ...], dtype=float)
            if np_mod.isfinite(ref).any():
                flat_idx = int(np_mod.nanargmax(ref))
                x0, y0, z0 = (int(v) for v in np_mod.unravel_index(flat_idx, ref.shape))
                measured = _extract_at(x0, y0, z0)
        except Exception:
            pass

    canonical_ti = [120, 300, 600, 1000, 2000, 4000, 10000]
    ti_ms = canonical_ti if len(canonical_ti) == int(n_ti) else list(range(int(n_ti)))
    ti_ms_arr = np_mod.asarray(ti_ms, dtype=float)

    finite = np_mod.isfinite(ti_ms_arr) & np_mod.isfinite(measured)
    ti_fit = ti_ms_arr[finite]
    y_fit = measured[finite]
    if ti_fit.size < 3:
        raise HTTPException(status_code=404, detail="Not enough IR points for fit")

    y_max = float(np_mod.nanmax(y_fit)) if np_mod.isfinite(y_fit).any() else 0.0
    if not np_mod.isfinite(y_max) or y_max <= 0:
        y_max = 1.0

    def _resid(params: Any) -> Any:
        m0_hat = float(params[0])
        t1_hat = float(params[1])
        return _t1_ir_saturation_model(ti_fit, m0_hat, t1_hat) - y_fit

    result = lsq(
        _resid,
        x0=np_mod.array([y_max, 1000.0], dtype=float),
        bounds=(np_mod.array([1e-6, 50.0]), np_mod.array([np_mod.inf, 20000.0])),
        method="trf",
    )
    m0_hat = float(result.x[0])
    t1_hat = float(result.x[1])

    dense_ti = np_mod.linspace(float(np_mod.nanmin(ti_fit)), float(np_mod.nanmax(ti_fit)), 200)
    dense_y = _t1_ir_saturation_model(dense_ti, m0_hat, t1_hat)

    map_t1 = None
    map_m0 = None
    try:
        if t1_p.exists() and t1_p.is_file():
            t1_map = np_mod.asarray(_load_pickle_array(t1_p), dtype=float)
            if t1_map.ndim == 3 and 0 <= x0 < t1_map.shape[0] and 0 <= y0 < t1_map.shape[1] and 0 <= z0 < t1_map.shape[2]:
                v = float(t1_map[x0, y0, z0])
                map_t1 = v if np_mod.isfinite(v) else None
    except Exception:
        map_t1 = None

    try:
        if m0_p.exists() and m0_p.is_file():
            m0_map = np_mod.asarray(_load_pickle_array(m0_p), dtype=float)
            if m0_map.ndim == 3 and 0 <= x0 < m0_map.shape[0] and 0 <= y0 < m0_map.shape[1] and 0 <= z0 < m0_map.shape[2]:
                v = float(m0_map[x0, y0, z0])
                map_m0 = v if np_mod.isfinite(v) else None
    except Exception:
        map_m0 = None

    return {
        "voxel": [x0, y0, z0],
        "tiMs": [float(x) for x in ti_ms_arr.tolist()],
        "measured": [float(x) if np_mod.isfinite(x) else None for x in measured.tolist()],
        "fit": {"model": "saturation", "m0": m0_hat, "t1Ms": t1_hat},
        "curve": {"tiMs": [float(x) for x in dense_ti.tolist()], "values": [float(x) for x in dense_y.tolist()]},
        "map": {"m0": map_m0, "t1Ms": map_t1},
    }


@app.post("/projects/{project_id}/subjects/import")
def import_subjects(project_id: str, req: ImportSubjectsRequest) -> List[Dict[str, Any]]:
    project = _find_project(project_id)
    cfg = project.config or {}
    fs = cfg.get("folderStructure") if isinstance(cfg.get("folderStructure"), dict) else {}
    t1_patterns = _split_patterns(fs.get("t1Pattern", "*T1*.nii.gz"))
    dce_patterns = _split_patterns(fs.get("dcePattern", "*DCE*.nii.gz"))
    diff_patterns = _split_patterns(fs.get("diffusionPattern", "*DTI*.nii.gz"))

    imported: List[Subject] = []
    for item in req.subjects:
        name = item.get("name")
        source = item.get("sourcePath") or name
        if not name:
            continue

        # Allow the UI to send relative-ish names; resolve against project storagePath.
        source_path = Path(source)
        if not source_path.is_absolute():
            source_path = Path(project.storagePath) / source
        elif str(source).startswith("/") and not source_path.exists():
            # Browsers may send webkit `fullPath` like "/Subject"; treat as relative.
            source_path = Path(project.storagePath) / str(source).lstrip("/")

        sp = source_path.expanduser().resolve()

        nifti_root = _nifti_dir_for_path(project, sp)

        scan = _scan_nifti_presence(
            nifti_root,
            t1_patterns=t1_patterns,
            dce_patterns=dce_patterns,
            diff_patterns=diff_patterns,
        )

        # Basic data presence checks (very conservative; refined later via patterns).
        has_nifti = bool(scan["has_nifti"])

        # Prefer configured patterns.
        has_t1 = bool(scan["t1_by_pattern"])
        has_dce = bool(scan["dce_by_pattern"])
        has_diff = bool(scan["diff_by_pattern"])

        # Fallback heuristics if patterns were empty or didn't match.
        if not has_t1:
            has_t1 = bool(scan["t1_by_name"])
        if not has_dce:
            has_dce = bool(scan["dce_by_name"])
        if not has_diff:
            has_diff = bool(scan["diff_by_name"])

        subject = Subject(
            id=f"subj_{int(datetime.utcnow().timestamp()*1000)}_{name}",
            projectId=project_id,
            name=name,
            sourcePath=str(sp),
            createdAt=_now_iso(),
            updatedAt=_now_iso(),
            hasT1=bool(has_nifti and has_t1),
            hasDCE=bool(has_nifti and has_dce),
            hasDiffusion=bool(has_nifti and has_diff),
            stageStatuses=_default_stage_statuses(),
        )
        db.subjects.append(subject)
        imported.append(subject)

    db.save()
    return [asdict(s) for s in imported]


@app.get("/jobs")
def get_jobs(projectId: Optional[str] = None, subjectId: Optional[str] = None, status: Optional[str] = None) -> List[Dict[str, Any]]:
    cache_key = (projectId or "", subjectId or "", status or "")
    cached = db._jobs_cache.get(cache_key)
    if cached and cached[0] == db._jobs_version:
        return cached[1]

    jobs = db.jobs
    if projectId:
        jobs = [j for j in jobs if j.projectId == projectId]
    if subjectId:
        jobs = [j for j in jobs if j.subjectId == subjectId]
    if status:
        jobs = [j for j in jobs if j.status == status]

    def key(j: Job) -> str:
        return j.startTime or ""

    def to_dict(j: Job) -> Dict[str, Any]:
        return {
            "id": j.id,
            "projectId": j.projectId,
            "subjectId": j.subjectId,
            "stageId": j.stageId,
            "status": j.status,
            "progress": int(j.progress),
            "currentStep": j.currentStep,
            "startTime": j.startTime,
            "endTime": j.endTime,
            "estimatedTimeRemaining": j.estimatedTimeRemaining,
            "error": j.error,
            "logPath": j.logPath,
        }

    result = [to_dict(j) for j in sorted(jobs, key=key, reverse=True)]
    db._jobs_cache[cache_key] = (db._jobs_version, result)
    return result


@app.post("/projects/{project_id}/run-full")
async def run_full(project_id: str, req: RunFullPipelineRequest) -> List[Dict[str, Any]]:
    project = _find_project(project_id)
    created: List[Job] = []

    for subject_id in req.subjectIds:
        subject = _find_subject(subject_id)
        created.extend(await _start_subject_run(project=project, subject=subject, stage_ids=req.stageIds))

    db.save()
    return [asdict(j) for j in created]


@app.post("/subjects/{subject_id}/run-stage")
async def run_stage(subject_id: str, req: RunStageRequest) -> Dict[str, Any]:
    subject = _find_subject(subject_id)
    project = _find_project(subject.projectId)
    stage = req.stageId
    if stage not in STAGES:
        raise HTTPException(status_code=400, detail="Invalid stage")

    job = await _start_subject_stage_run(
        project=project,
        subject=subject,
        stage=stage,
        run_dependencies=req.runDependencies,
        env_overrides=req.envOverrides,
    )
    db.save()
    return asdict(job)


async def _start_subject_run(*, project: Project, subject: Subject, stage_ids: Optional[List[StageId]] = None) -> List[Job]:
    # Disallow starting if there is an active run for this subject.
    if any(j.subjectId == subject.id and j.status in {"queued", "running"} for j in db.jobs):
        raise HTTPException(status_code=409, detail="Subject already has a queued/running job")

    created: List[Job] = []
    job_ids: List[str] = []
    shared_start = _now_iso()
    stages_to_run: List[StageId] = []

    if stage_ids is None:
        # Original behavior: always queue the full chain.
        for stage in STAGES:
            if stage == "diffusion" and not subject.hasDiffusion:
                continue
            stages_to_run.append(stage)
    else:
        # Minimal chain: expand dependencies for the requested subset,
        # always including explicitly requested stages.
        requested: List[StageId] = [s for s in (stage_ids or []) if s in STAGES]
        requested_set = set(requested)

        expanded: set[StageId] = set()
        for s in requested:
            # If a subject doesn't have diffusion data, don't attempt diffusion or its downstream stages.
            if s == "diffusion" and not subject.hasDiffusion:
                continue
            for st in _stage_chain_for_request(subject, s):
                expanded.add(st)

        if expanded:
            t1m0_force = _project_t1m0_force_enabled(project)
            for st in STAGES:
                if st not in expanded:
                    continue
                if st == "diffusion" and not subject.hasDiffusion:
                    continue
                if st in requested_set:
                    stages_to_run.append(st)
                    continue
                if subject.stageStatuses.get(str(st)) != "done" or (t1m0_force and st == "t1_fit"):
                    stages_to_run.append(st)

        # If everything was filtered out (e.g. diffusion requested but absent), do nothing.

    for stage in stages_to_run:
        jid = f"job_{int(datetime.utcnow().timestamp()*1000)}_{subject.id}_{stage}"
        job = Job(
            id=jid,
            projectId=project.id,
            subjectId=subject.id,
            stageId=stage,
            status="queued",
            progress=0,
            currentStep="Queued",
            startTime=shared_start,
        )
        db.jobs.append(job)
        db._job_by_id[job.id] = job
        created.append(job)
        job_ids.append(jid)
        await asyncio.sleep(0)

    db._touch_jobs()

    db._subject_job_ids[subject.id] = job_ids

    # Run stages sequentially using the stage runner for true separation.
    task = asyncio.create_task(_run_pbrain_stage_chain(project=project, subject=subject, jobs=created))
    for jid in job_ids:
        db._job_tasks[jid] = task
    return created


def _stage_chain_for_request(subject: Subject, stage: StageId) -> List[StageId]:
    ordered: List[StageId] = []
    seen: set[StageId] = set()

    def visit(s: StageId) -> None:
        for dep in STAGE_DEPENDENCIES.get(s, []):
            visit(dep)
        if s not in seen:
            seen.add(s)
            ordered.append(s)

    visit(stage)
    return ordered


async def _run_pbrain_stage_chain(*, project: Project, subject: Subject, jobs: List[Job], env_overrides: Optional[Dict[str, str]] = None) -> None:
    try:
        for j in jobs:
            if j.status == "cancelled":
                return
            await _run_pbrain_single_stage(project=project, subject=subject, job=j, env_overrides=env_overrides)
    except asyncio.CancelledError:
        now = _now_iso()
        for j in jobs:
            if j.status in {"queued", "running"}:
                j.status = "cancelled"
                j.currentStep = "Cancelled"
                j.endTime = now
                if subject.stageStatuses.get(str(j.stageId)) == "running":
                    subject.stageStatuses[str(j.stageId)] = "failed"
                    subject.updatedAt = now
        db._touch_jobs()
        db.save()
        raise
    except Exception as exc:
        err = str(exc)
        now = _now_iso()
        for j in jobs:
            if j.status == "queued":
                j.status = "failed"
                j.currentStep = "Skipped (dependency failure)"
                j.error = err
                j.endTime = now
                subject.stageStatuses[str(j.stageId)] = "failed"
                subject.updatedAt = now
        db._touch_jobs()
        db.save()
        raise


async def _start_subject_stage_run(
    *,
    project: Project,
    subject: Subject,
    stage: StageId,
    run_dependencies: bool,
    env_overrides: Optional[Dict[str, str]] = None,
) -> Job:
    # Disallow starting if there is an active run for this subject.
    if any(j.subjectId == subject.id and j.status in {"queued", "running"} for j in db.jobs):
        raise HTTPException(status_code=409, detail="Subject already has a queued/running job")

    if not run_dependencies:
        _require_stage_dependencies(subject, stage)
        chain: List[StageId] = [stage]
    else:
        chain = _stage_chain_for_request(subject, stage)

    # Only auto-run dependencies that aren't already done; always include the requested stage.
    # Exception: when project defaults require a fresh T1/M0 fit, always include t1_fit.
    t1m0_force = _project_t1m0_force_enabled(project)
    stages: List[StageId] = [
        s
        for s in chain
        if s == stage
        or subject.stageStatuses.get(str(s)) != "done"
        or (t1m0_force and s == "t1_fit")
    ]
    if not stages:
        stages = [stage]

    created: List[Job] = []
    job_ids: List[str] = []
    shared_start = _now_iso()
    for st in stages:
        jid = f"job_{int(datetime.utcnow().timestamp()*1000)}_{subject.id}_{st}"
        job = Job(
            id=jid,
            projectId=project.id,
            subjectId=subject.id,
            stageId=st,
            status="queued",
            progress=0,
            currentStep="Queued",
            startTime=shared_start,
        )
        db.jobs.append(job)
        db._job_by_id[job.id] = job
        created.append(job)
        job_ids.append(jid)
        await asyncio.sleep(0)

    db._touch_jobs()

    if len(created) == 1:
        task = asyncio.create_task(
            _run_pbrain_single_stage(project=project, subject=subject, job=created[0], env_overrides=env_overrides)
        )
        db._job_tasks[created[0].id] = task
        await asyncio.sleep(0)
        return created[0]

    task = asyncio.create_task(
        _run_pbrain_stage_chain(project=project, subject=subject, jobs=created, env_overrides=env_overrides)
    )
    for jid in job_ids:
        db._job_tasks[jid] = task
    await asyncio.sleep(0)
    return created[-1]


@app.post("/subjects/{subject_id}/ensure")
async def ensure_subject_artifacts(subject_id: str, kind: str = "all") -> Dict[str, Any]:
    """Ensure key artifacts exist; if missing, trigger a p-brain auto run.

    kind: one of "all", "maps", "curves", "montages", "roi".
    """

    subject = _find_subject(subject_id)
    project = _find_project(subject.projectId)

    want = (kind or "all").strip().lower()
    if want not in {"all", "maps", "curves", "montages", "roi"}:
        raise HTTPException(status_code=400, detail="Invalid kind")

    missing: List[str] = []
    if want in {"all", "roi"}:
        if len(_analysis_roi_overlays(subject)) == 0:
            missing.append("roi-overlays")
    if want in {"all", "curves"}:
        if len(_analysis_curves(subject)) == 0:
            missing.append("curves")
    if want in {"all", "maps"}:
        if len(_analysis_map_volumes(subject)) == 0:
            missing.append("maps")
    if want in {"all", "montages"}:
        if len(_subject_montage_images(subject)) == 0:
            missing.append("montages")

    if not missing:
        return {"started": False, "jobs": [], "reason": "Artifacts already present"}

    # If we're only missing ROI overlays, (re)run up to AIF/VIF extraction.
    if set(missing) == {"roi-overlays"}:
        job = await _start_subject_stage_run(
            project=project,
            subject=subject,
            stage="input_functions",
            run_dependencies=True,
        )
        db.save()
        return {"started": True, "jobs": [asdict(job)], "reason": "Missing: roi-overlays"}

    created = await _start_subject_run(project=project, subject=subject)
    db.save()
    return {"started": True, "jobs": [asdict(j) for j in created], "reason": f"Missing: {', '.join(missing)}"}


def _cancel_jobs_for_subject(subject_id: str) -> tuple[int, int]:
    targets = [j for j in db.jobs if j.subjectId == subject_id and j.status in {"queued", "running"}]
    if not targets:
        return 0, 0

    now = _now_iso()
    subject = next((s for s in db.subjects if s.id == subject_id), None)

    for j in targets:
        j.status = "cancelled"
        j.endTime = now
        j.currentStep = "Cancelled"
        if subject is not None:
            if subject.stageStatuses.get(str(j.stageId)) == "running":
                subject.stageStatuses[str(j.stageId)] = "failed"
                subject.updatedAt = now

    db._touch_jobs()

    terminated = 0
    seen: set[int] = set()
    for j in targets:
        proc = db._job_processes.get(j.id)
        if not proc or proc.returncode is not None:
            continue
        key = getattr(proc, "pid", None) or id(proc)
        if key in seen:
            continue
        seen.add(key)
        try:
            proc.send_signal(signal.SIGTERM)
            terminated += 1
        except ProcessLookupError:
            pass

    tasks: set[asyncio.Task] = set()
    for j in targets:
        task = db._job_tasks.get(j.id)
        if task and not task.done():
            tasks.add(task)
    for task in tasks:
        task.cancel()

    for j in targets:
        db._job_processes.pop(j.id, None)
        db._job_tasks.pop(j.id, None)

    return len(targets), terminated


@app.post("/subjects/{subject_id}/clear-data")
async def clear_subject_data(subject_id: str) -> Dict[str, Any]:
    """Clear a subject's derived outputs so the next run starts fresh.

    Deletes:
      - <subject>/Analysis
      - <subject>/Images
    Keeps:
      - <subject>/NIfTI and all source data

    Also:
      - cancels any queued/running jobs for the subject
      - clears cached stage runner under <project storage>/.pbrain-web/runner
      - resets stage statuses to not_run and removes job history for the subject
    """

    subject = _find_subject(subject_id)
    project = _find_project(subject.projectId)

    cancelled, terminated = _cancel_jobs_for_subject(subject_id)

    subject_path = Path(subject.sourcePath).expanduser().resolve()
    analysis_dir = subject_path / "Analysis"
    images_dir = subject_path / "Images"

    def _rm_tree(p: Path) -> bool:
        try:
            if p.exists():
                shutil.rmtree(p)
                return True
        except Exception:
            return False
        return False

    deleted_analysis = _rm_tree(analysis_dir)
    deleted_images = _rm_tree(images_dir)

    data_root = Path(project.storagePath).expanduser().resolve()
    runner_dir = data_root / ".pbrain-web" / "runner"
    deleted_runner = _rm_tree(runner_dir)

    # Reset stage statuses.
    subject.stageStatuses = _default_stage_statuses()
    subject.updatedAt = _now_iso()

    # Clear job history for this subject so the UI doesn't show stale done/failed stages.
    to_remove = [j for j in db.jobs if j.subjectId == subject_id]
    for j in to_remove:
        db._job_by_id.pop(j.id, None)
        db._job_processes.pop(j.id, None)
        db._job_tasks.pop(j.id, None)
    db.jobs = [j for j in db.jobs if j.subjectId != subject_id]
    db._subject_job_ids.pop(subject_id, None)
    db._touch_jobs()
    db.save()

    return {
        "ok": True,
        "cancelledJobs": cancelled,
        "terminatedProcesses": terminated,
        "deleted": {
            "analysis": deleted_analysis,
            "images": deleted_images,
            "runner": deleted_runner,
        },
        "subject": asdict(subject),
    }


@app.post("/projects/{project_id}/clear-derived-data")
async def clear_project_derived_data(project_id: str) -> Dict[str, Any]:
    """Clear derived outputs for all subjects in a project.

    Deletes:
      - <subject>/Analysis
      - <subject>/Images
    Keeps:
      - <subject>/NIfTI and all source data

    Also cancels queued/running jobs for impacted subjects, resets stage statuses,
    and clears job history so the UI reflects a clean slate.
    """

    project = _find_project(project_id)
    subjects = [s for s in db.subjects if s.projectId == project.id]

    cancelled_total = 0
    terminated_total = 0
    deleted_analysis_dirs = 0
    deleted_images_dirs = 0

    def _rm_tree(p: Path) -> bool:
        try:
            if p.exists():
                shutil.rmtree(p)
                return True
        except Exception:
            return False
        return False

    for subject in subjects:
        cancelled, terminated = _cancel_jobs_for_subject(subject.id)
        cancelled_total += int(cancelled)
        terminated_total += int(terminated)

        subject_path = Path(subject.sourcePath).expanduser().resolve()
        if _rm_tree(subject_path / "Analysis"):
            deleted_analysis_dirs += 1
        if _rm_tree(subject_path / "Images"):
            deleted_images_dirs += 1

        subject.stageStatuses = _default_stage_statuses()
        subject.updatedAt = _now_iso()

    subject_ids = {s.id for s in subjects}
    if subject_ids:
        to_remove = [j for j in db.jobs if j.subjectId in subject_ids]
        for j in to_remove:
            db._job_by_id.pop(j.id, None)
            db._job_processes.pop(j.id, None)
            db._job_tasks.pop(j.id, None)
        db.jobs = [j for j in db.jobs if j.subjectId not in subject_ids]
        for sid in subject_ids:
            db._subject_job_ids.pop(sid, None)
        db._touch_jobs()

    db.save()
    return {
        "ok": True,
        "projectId": project.id,
        "subjectsCleared": len(subjects),
        "cancelledJobs": cancelled_total,
        "terminatedProcesses": terminated_total,
        "deleted": {
            "analysisDirs": deleted_analysis_dirs,
            "imagesDirs": deleted_images_dirs,
        },
    }


@app.post("/jobs/{job_id}/cancel")
async def cancel_job(job_id: str) -> Dict[str, Any]:
    job = _find_job(job_id)
    if job.status not in {"queued", "running"}:
        return asdict(job)

    subject = next((s for s in db.subjects if s.id == job.subjectId), None)

    # Cancel the entire subject run (all stage jobs).
    stage_jobs = [j for j in db.jobs if j.subjectId == job.subjectId and j.startTime == job.startTime]
    if not stage_jobs:
        stage_jobs = [j for j in db.jobs if j.subjectId == job.subjectId and j.status in {"queued", "running"}]

    for j in stage_jobs:
        j.status = "cancelled"
        j.endTime = _now_iso()
        j.currentStep = "Cancelled"

        # StageStatus does not include "cancelled"; mark running stages as failed
        # so the subject UI stops showing "running".
        if subject is not None:
            try:
                if subject.stageStatuses.get(str(j.stageId)) == "running":
                    subject.stageStatuses[str(j.stageId)] = "failed"
                    subject.updatedAt = _now_iso()
            except Exception:
                pass

    db._touch_jobs()

    proc = db._job_processes.get(job.id)
    if proc and proc.returncode is None:
        try:
            proc.send_signal(signal.SIGTERM)
        except ProcessLookupError:
            pass

    # Try cancelling whatever task we stored.
    task = db._job_tasks.get(job.id)
    if task and not task.done():
        task.cancel()

    db.save()
    return asdict(job)


@app.post("/jobs/cancel-all")
async def cancel_all_jobs(projectId: Optional[str] = None) -> Dict[str, Any]:
    """Cancel all queued/running jobs.

    Optional query param:
      - projectId: only cancel jobs for a specific project.
    """

    targets = [
        j
        for j in db.jobs
        if j.status in {"queued", "running"} and (projectId is None or j.projectId == projectId)
    ]
    if not targets:
        return {"cancelled": 0, "terminated": 0}

    now = _now_iso()
    subjects_by_id: Dict[str, Subject] = {s.id: s for s in db.subjects}
    for j in targets:
        j.status = "cancelled"
        j.endTime = now
        j.currentStep = "Cancelled"

        s = subjects_by_id.get(j.subjectId)
        if s is not None:
            # Mark any currently-running stages as failed so the subject UI reflects cancellation.
            if s.stageStatuses.get(str(j.stageId)) == "running":
                s.stageStatuses[str(j.stageId)] = "failed"
                s.updatedAt = now

    db._touch_jobs()

    # Terminate any running subprocesses.
    terminated = 0
    seen: set[int] = set()
    for j in targets:
        proc = db._job_processes.get(j.id)
        if not proc or proc.returncode is not None:
            continue
        key = getattr(proc, "pid", None) or id(proc)
        if key in seen:
            continue
        seen.add(key)
        try:
            proc.send_signal(signal.SIGTERM)
            terminated += 1
        except ProcessLookupError:
            pass

    # Cancel any queued tasks (dedupe: many job ids map to the same task).
    tasks: set[asyncio.Task] = set()
    for j in targets:
        task = db._job_tasks.get(j.id)
        if task and not task.done():
            tasks.add(task)
    for task in tasks:
        task.cancel()

    # Best-effort cleanup of in-memory trackers.
    for j in targets:
        db._job_processes.pop(j.id, None)
        db._job_tasks.pop(j.id, None)

    db.save()
    return {"cancelled": len(targets), "terminated": terminated}


@app.post("/jobs/{job_id}/retry")
async def retry_job(job_id: str) -> Dict[str, Any]:
    old = _find_job(job_id)
    project = _find_project(old.projectId)
    subject = _find_subject(old.subjectId)

    # Disallow retry if there is an active run for this subject.
    if any(j.subjectId == subject.id and j.status in {"queued", "running"} for j in db.jobs):
        raise HTTPException(status_code=409, detail="Subject already has a queued/running job")

    created: List[Job] = []
    job_ids: List[str] = []
    for stage in STAGES:
        jid = f"job_{int(datetime.utcnow().timestamp()*1000)}_{subject.id}_{stage}"
        job = Job(
            id=jid,
            projectId=project.id,
            subjectId=subject.id,
            stageId=stage,
            status="queued",
            progress=0,
            currentStep="Queued",
        )
        db.jobs.append(job)
        db._job_by_id[job.id] = job
        created.append(job)
        job_ids.append(jid)
        await asyncio.sleep(0)

    db._touch_jobs()

    db._subject_job_ids[subject.id] = job_ids
    task = asyncio.create_task(_run_pbrain_auto(project=project, subject=subject))
    for jid in job_ids:
        db._job_tasks[jid] = task
    db.save()

    representative = next((j for j in created if j.stageId == old.stageId), created[0])
    return asdict(representative)


@app.get("/jobs/{job_id}/logs")
def get_job_logs(job_id: str, tail: int = 400) -> Dict[str, Any]:
    job = _find_job(job_id)
    if not job.logPath:
        return {"lines": []}
    p = Path(job.logPath)
    if not p.exists():
        return {"lines": []}

    def _tail_text_file(path: Path, *, max_lines: int, chunk_size: int = 64 * 1024, max_bytes: int = 2 * 1024 * 1024) -> List[str]:
        """Read last N lines efficiently without loading the whole file."""
        max_lines = int(max_lines)
        if max_lines <= 0:
            return []

        try:
            size = path.stat().st_size
        except Exception:
            return []
        if size <= 0:
            return []

        # Read backwards in chunks until we have enough newlines.
        buf = bytearray()
        read_bytes = 0
        nl_count = 0

        with path.open("rb") as f:
            pos = size
            while pos > 0 and nl_count <= max_lines and read_bytes < max_bytes:
                step = chunk_size if pos >= chunk_size else pos
                pos -= step
                f.seek(pos)
                chunk = f.read(step)
                if not chunk:
                    break
                buf[:0] = chunk
                read_bytes += len(chunk)
                nl_count = buf.count(b"\n")

        text = buf.decode("utf-8", errors="replace")
        return text.splitlines()[-max_lines:]

    lines = _tail_text_file(p, max_lines=tail)
    return {"lines": lines}


@app.get("/{full_path:path}", include_in_schema=False)
def _frontend_spa_fallback(full_path: str) -> FileResponse:
    """SPA fallback: serve index.html for non-API, non-file routes."""

    dist_dir = _frontend_dist_dir()
    if not dist_dir:
        raise HTTPException(status_code=404, detail="Not Found")

    # If the request matches an actual file in the dist directory, serve it.
    candidate = (dist_dir / full_path.lstrip("/")).resolve()
    try:
        if str(candidate).startswith(str(dist_dir.resolve())) and candidate.is_file():
            return FileResponse(str(candidate))
    except Exception:
        pass

    return FileResponse(str(dist_dir / "index.html"))


# Optional: serve the built React UI from this backend.
# IMPORTANT: this must be last so it cannot shadow API routes.
_mount_frontend_if_present()
