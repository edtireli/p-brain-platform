from __future__ import annotations

import asyncio
import re
import fnmatch
import json
import os
import math
import time
import subprocess
import signal
import sys
import hashlib
import importlib
import colorsys
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
StageStatus = Literal["not_run", "running", "done", "failed"]


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


class ImportSubjectsRequest(BaseModel):
    subjects: List[Dict[str, str]]  # {name, sourcePath}


class RunFullPipelineRequest(BaseModel):
    subjectIds: List[str]


class RunStageRequest(BaseModel):
    stageId: StageId
    runDependencies: bool = True


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
    "input_functions": ["t1_fit"],
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


class UpdateAppSettingsRequest(BaseModel):
    firstName: Optional[str] = None
    onboardingCompleted: Optional[bool] = None
    pbrainMainPy: Optional[str] = None
    fastsurferDir: Optional[str] = None
    freesurferHome: Optional[str] = None


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
            analysis_dir / "Ki_values_atlas_two_compartment.json",
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

    # Newer pipelines write model-specific files (tikhonov/patlak/two_compartment).
    # Prefer tikhonov since it contains the CBF/MTT/CTH fields used by the UI.
    candidates = [
        analysis_dir / "AI_values_median_total_tikhonov.json",
        analysis_dir / "AI_values_median_total.json",
        analysis_dir / "AI_values_median_total_patlak.json",
        analysis_dir / "AI_values_median_total_two_compartment.json",
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
            analysis_dir / "Ki_values_atlas_two_compartment.json",
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
        analysis_dir / "AI_values_median_total_two_compartment.json",
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
    time_path = analysis_dir / "Fitting" / "time_points_s.npy"
    if not time_path.exists():
        return []
    time_points = np.load(str(time_path)).astype(float).tolist()

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

    # Prefer shifted curves if present.
    def pick_curve(glob_path: Path) -> Optional[Path]:
        matches = sorted(glob_path.parent.glob(glob_path.name))
        return matches[0] if matches else None

    # AIF: p-brain uses the time-shifted & rescaled VIF as the effective AIF.
    tscc = _load_tscc_max_curve()
    if tscc is not None:
        label, arr = tscc
        n = min(len(time_points), int(arr.size))
        curves.append(
            {
                "id": "aif_tscc_max",
                "name": f"AIF (Time-shifted VIF: {label})",
                "timePoints": time_points[:n],
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
                    vals = np.load(str(pth)).astype(float).tolist()
                    curves.append({"id": f"aif_{subtype_dir.name}", "name": f"AIF ({subtype_dir.name})", "timePoints": time_points, "values": vals, "unit": "mM"})
                    break

    # VIF (vein)
    vein_dir = analysis_dir / "CTC Data" / "Vein"
    if vein_dir.exists():
        for subtype_dir in sorted([d for d in vein_dir.iterdir() if d.is_dir()]):
            pth = pick_curve(subtype_dir / "CTC_shifted_slice_*.npy") or pick_curve(subtype_dir / "CTC_slice_*.npy")
            if pth and pth.exists():
                vals = np.load(str(pth)).astype(float).tolist()
                curves.append({"id": f"vif_{subtype_dir.name}", "name": f"VIF ({subtype_dir.name})", "timePoints": time_points, "values": vals, "unit": "mM"})
                break

    # Tissue curves: include all available tissue subtypes.
    tissue_dir = analysis_dir / "CTC Data" / "Tissue"
    if tissue_dir.exists():
        for subtype_dir in sorted([d for d in tissue_dir.iterdir() if d.is_dir()]):
            pth = pick_curve(subtype_dir / "CTC_shifted_slice_*.npy") or pick_curve(subtype_dir / "CTC_slice_*.npy")
            if pth and pth.exists():
                vals = np.load(str(pth)).astype(float).tolist()
                curves.append({
                    "id": f"tissue_{subtype_dir.name}",
                    "name": f"Tissue ({subtype_dir.name})",
                    "timePoints": time_points,
                    "values": vals,
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
                        "timePoints": time_points[:min_len],
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


def _analysis_curves_from_dir(subject_dir: Path) -> List[Dict[str, Any]]:
    _require_numpy()
    assert np is not None
    analysis_dir = subject_dir.expanduser().resolve() / "Analysis"
    time_path = analysis_dir / "Fitting" / "time_points_s.npy"
    if not time_path.exists():
        return []
    time_points = np.load(str(time_path)).astype(float).tolist()

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
        n = min(len(time_points), int(arr.size))
        curves.append(
            {
                "id": "aif_tscc_max",
                "name": f"AIF (Time-shifted VIF: {label})",
                "timePoints": time_points[:n],
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
                    vals = np.load(str(pth)).astype(float).tolist()
                    curves.append({
                        "id": f"aif_{subtype_dir.name}",
                        "name": f"AIF ({subtype_dir.name})",
                        "timePoints": time_points,
                        "values": vals,
                        "unit": "mM",
                    })
                    break

    vein_dir = analysis_dir / "CTC Data" / "Vein"
    if vein_dir.exists():
        for subtype_dir in sorted([d for d in vein_dir.iterdir() if d.is_dir()]):
            pth = pick_curve(subtype_dir / "CTC_shifted_slice_*.npy") or pick_curve(subtype_dir / "CTC_slice_*.npy")
            if pth and pth.exists():
                vals = np.load(str(pth)).astype(float).tolist()
                curves.append({
                    "id": f"vif_{subtype_dir.name}",
                    "name": f"VIF ({subtype_dir.name})",
                    "timePoints": time_points,
                    "values": vals,
                    "unit": "mM",
                })
                break

    tissue_dir = analysis_dir / "CTC Data" / "Tissue"
    if tissue_dir.exists():
        for subtype_dir in sorted([d for d in tissue_dir.iterdir() if d.is_dir()]):
            pth = pick_curve(subtype_dir / "CTC_shifted_slice_*.npy") or pick_curve(subtype_dir / "CTC_slice_*.npy")
            if pth and pth.exists():
                vals = np.load(str(pth)).astype(float).tolist()
                curves.append({
                    "id": f"tissue_{subtype_dir.name}",
                    "name": f"Tissue ({subtype_dir.name})",
                    "timePoints": time_points,
                    "values": vals,
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
                        "timePoints": time_points[:min_len],
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
    if not time_path.exists():
        raise HTTPException(status_code=404, detail="Missing time_points_s.npy")
    t = np.asarray(np.load(str(time_path)), dtype=float).reshape(-1)
    if t.size < 3 or not np.all(np.isfinite(t)):
        raise HTTPException(status_code=400, detail="Invalid time points")
    return t




def _analysis_map_volumes(subject: Subject) -> List[Dict[str, Any]]:
    # Use the same robust on-disk scan as local-dir mode so we pick up newer
    # pipeline outputs (e.g. *_patlak / *_tikhonov / *_two_compartment).
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


def _extended_tofts_model(t: Any, Ktrans: float, ve: float, vp: float, Cp: Any) -> Any:
    _require_numpy()
    assert np is not None
    Ktrans = float(max(Ktrans, 1e-12))
    ve = float(max(ve, 1e-12))
    vp = float(max(vp, 0.0))
    out = np.zeros_like(t, dtype=float)
    for i in range(t.size):
        tau = t[: i + 1]
        cp = Cp[: i + 1]
        integrand = cp * np.exp(-(t[i] - tau) * Ktrans / ve)
        out[i] = Ktrans * float(np.trapz(integrand, x=tau))
    out = out + vp * Cp
    return out


def _tofts_from_curves(t: Any, aif: Any, tissue: Any, *, lambd: float) -> Dict[str, Any]:
    _require_numpy()
    _require_scipy()
    assert np is not None
    ls = _get_least_squares()
    if ls is None:
        raise HTTPException(status_code=500, detail="Backend missing dependency: scipy")

    # Fit Ktrans, ve, vp (all >= 0)
    x0 = np.array([0.001, 0.2, 0.05], dtype=float)

    def residual(theta: Any) -> Any:
        theta = np.clip(theta, 1e-12, None)
        Ktrans, ve, vp = float(theta[0]), float(theta[1]), float(theta[2])
        Ct_pred = _extended_tofts_model(t, Ktrans, ve, vp, aif)
        misfit = (Ct_pred - tissue).astype(float)
        # Tikhonov-style penalty to keep params reasonable
        lam = float(max(lambd, 0.0))
        w = float(np.linalg.norm(misfit) / max(np.linalg.norm(theta), 1e-8))
        penalty = np.sqrt(lam) * w * theta
        return np.concatenate([misfit, penalty.astype(float)])

    sol = ls(residual, x0, bounds=(0.0, np.inf))
    Ktrans, ve, vp = [float(v) for v in sol.x]
    fitted = _extended_tofts_model(t, Ktrans, ve, vp, aif)
    residuals = (tissue - fitted).astype(float)
    return {
        "timePoints": t.astype(float).tolist(),
        "measured": tissue.astype(float).tolist(),
        "fitted": fitted.astype(float).tolist(),
        # UI labels min^-1
        "Ktrans": float(Ktrans * 60.0),
        "ve": float(ve),
        "vp": float(vp),
        "residuals": residuals.tolist(),
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

    runner_path = _ensure_stage_runner_script(data_root)
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

    runner_path = _ensure_stage_runner_script(data_root)
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


async def _run_pbrain_single_stage(*, project: Project, subject: Subject, job: Job) -> None:
    async with _RUN_SEMAPHORE:
        data_root = Path(project.storagePath).expanduser().resolve()
        log_fh = None
        proc: asyncio.subprocess.Process | None = None

        try:
            if not data_root.exists():
                raise RuntimeError(f"Project storagePath/data root does not exist: {data_root}")

            # Log file for this stage run.
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

            # Keep stage-runner behavior consistent with the full p-brain CLI.
            # In particular, voxelwise maps can be extremely expensive; only enable when configured.
            if job.stageId == "modelling":
                cfg = project.config if isinstance(project.config, dict) else {}
                voxelwise_cfg = cfg.get("voxelwise") if isinstance(cfg.get("voxelwise"), dict) else {}
                compute_ki = bool(voxelwise_cfg.get("computeKi"))
                compute_cbf = bool(voxelwise_cfg.get("computeCBF"))
                if compute_ki:
                    args.append("--voxelwise")
                if compute_cbf:
                    args.append("--cbf")
            write_line_sync(f"[{_now_iso()}] ARGS: {args!r}\n")

            main_py = _resolve_pbrain_main_py()
            pbrain_cwd = str(Path(main_py).expanduser().resolve().parent) if main_py else None
            if pbrain_cwd:
                write_line_sync(f"[{_now_iso()}] CWD: {pbrain_cwd}\n")

            proc = await asyncio.create_subprocess_exec(
                *args,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
                cwd=pbrain_cwd,
                env=_python_env_for_pbrain(),
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

    if override:
        if os.path.sep in override or override.startswith("/"):
            add_candidate(override)
        else:
            add_candidate(shutil.which(override))
    else:
        # First choice: app-managed venv (auto-installs deps if missing).
        try:
            if write_line_sync is None:
                def write_line_sync(_line: str) -> None:
                    return
            venv_python = await _ensure_managed_pbrain_venv(write_line_sync=write_line_sync)
            await _preflight_pbrain_python(venv_python)
            return venv_python
        except Exception as exc:
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

        if bool(getattr(sys, "frozen", False)):
            add_candidate("/opt/homebrew/bin/python3")
            add_candidate("/usr/local/bin/python3")
            add_candidate(shutil.which("python3"))
            add_candidate("/usr/bin/python3")
        else:
            add_candidate(sys.executable)
            add_candidate(shutil.which("python3"))

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


async def _ensure_managed_pbrain_venv(*, write_line_sync) -> str:
    """Create/update an app-managed venv with p-brain dependencies.

    This makes the packaged app self-contained even if the user has not
    installed python deps globally.
    """

    venv_dir = _managed_pbrain_venv_dir()
    venv_dir.parent.mkdir(parents=True, exist_ok=True)

    # Pick a bootstrap python that exists. Prefer versions compatible with
    # scientific wheels (and tensorflow-macos), avoiding bleeding-edge Python.
    bootstrap = None
    for candidate in [
        # Homebrew Python versioned binaries (most reliable)
        "/opt/homebrew/bin/python3.12",
        "/opt/homebrew/opt/python@3.12/bin/python3.12",
        "/opt/homebrew/bin/python3.11",
        "/opt/homebrew/opt/python@3.11/bin/python3.11",
        shutil.which("python3.11"),
        "/opt/homebrew/bin/python3.10",
        "/opt/homebrew/opt/python@3.10/bin/python3.10",
        shutil.which("python3.10"),
        "/usr/local/bin/python3.11",
        "/usr/local/bin/python3.10",
        "/opt/homebrew/bin/python3",
        shutil.which("python3"),
        "/usr/bin/python3",
    ]:
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
    want_fp = _requirements_fingerprint(req_path) if req_path.exists() else "missing"
    have_fp = marker.read_text(encoding="utf-8", errors="ignore").strip() if marker.exists() else ""

    async def _install_deps() -> None:
        if req_path.exists():
            # Filter/translate requirements for platform compatibility.
            lines = req_path.read_text(encoding="utf-8", errors="ignore").splitlines()
            out_lines: List[str] = []
            for raw in lines:
                line = raw.strip()
                if not line or line.startswith("#"):
                    continue
                if line.lower() == "tensorflow" and sys.platform == "darwin":
                    out_lines.append("tensorflow-macos")
                    out_lines.append("tensorflow-metal")
                    continue
                out_lines.append(line)

            filtered_req = venv_dir / "requirements.filtered.txt"
            filtered_req.write_text("\n".join(out_lines) + "\n", encoding="utf-8")

            # On macOS, pip can backtrack into very old scikit-image versions
            # that require building from source with build tooling incompatible
            # with Python 3.12+. Force a known wheel-available version.
            pip_args: List[str] = [str(venv_python), "-m", "pip", "install", "--prefer-binary", "-r", str(filtered_req)]
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

    # Validate that the managed env actually imports what we need.
    # If the venv was partially installed or corrupted, force one reinstall.
    try:
        await _preflight_pbrain_python(str(venv_python))
    except Exception as exc:
        write_line_sync(f"[{_now_iso()}] Managed venv preflight failed; reinstalling deps once: {exc}\n")
        try:
            if marker.exists():
                marker.unlink()
        except Exception:
            pass
        await _install_deps()
        await _preflight_pbrain_python(str(venv_python))

    return str(venv_python)


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

    # Keep behaviour aligned with p-brain CLI flags.
    lambd = model_cfg.get("lambdaTikhonov")
    if isinstance(lambd, (int, float)):
        args += ["--lambda", str(float(lambd))]
    if model_cfg.get("autoLambda") is True:
        args.append("--enable-lcurve")

    # Write optional maps
    if "writeMTT" in voxel_cfg:
        args += ["--write-mtt", str(bool(voxel_cfg.get("writeMTT"))).lower()]
    if "writeCTH" in voxel_cfg:
        args += ["--write-cth", str(bool(voxel_cfg.get("writeCTH"))).lower()]

    if voxel_cfg.get("computeKi") is True:
        args.append("--voxelwise")
    if voxel_cfg.get("computeCBF") is True:
        args.append("--cbf")

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

    lambd = model_cfg.get("lambdaTikhonov")
    if isinstance(lambd, (int, float)):
        args += ["--lambda", str(float(lambd))]
    if model_cfg.get("autoLambda") is True:
        args.append("--enable-lcurve")

    if "writeMTT" in voxel_cfg:
        args += ["--write-mtt", str(bool(voxel_cfg.get("writeMTT"))).lower()]
    if "writeCTH" in voxel_cfg:
        args += ["--write-cth", str(bool(voxel_cfg.get("writeCTH"))).lower()]

    if voxel_cfg.get("computeKi") is True:
        args.append("--voxelwise")
    if voxel_cfg.get("computeCBF") is True:
        args.append("--cbf")

    if subject.hasDiffusion:
        args.append("--diffusion")

    return args


_STAGE_RUNNER_VERSION = "16"


def _stage_runner_path(data_root: Path) -> Path:
    return data_root / ".pbrain-web" / "runner" / "pbrain_stage_runner.py"


def _ensure_stage_runner_script(data_root: Path) -> Path:
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

    content = f"""#!/usr/bin/env python3
# pbrain-web stage runner (version: {_STAGE_RUNNER_VERSION})

import argparse
import os
import shutil
import sys
import builtins
import time
import threading
import resource
import functools

# When running in batch/headless mode, ensure matplotlib uses a non-GUI backend.
if os.environ.get('PBRAIN_TURBO') == '1':
    os.environ.setdefault('MPLBACKEND', 'Agg')

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
import modules.AI_tissue_functions as AIT
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


def main() -> int:
    p = argparse.ArgumentParser(description='Run a single p-brain stage (invoked by p-brain-web)')
    p.add_argument('--stage', required=True)
    p.add_argument('--id', required=True)
    p.add_argument('--data-dir', required=True)
    p.add_argument('--diffusion', action='store_true')
    p.add_argument('--voxelwise', action='store_true')
    p.add_argument('--cbf', action='store_true')
    args = p.parse_args()

    stage = str(args.stage).strip().lower()
    subject_id = str(args.id)
    data_root = str(args.data_dir)

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
            log_process_start('T1 fitting')
            T1_fit(data_directory, analysis_directory, nifti_directory, image_directory, filenames, parameters)
            log_process_end('T1 fitting')
            return 0

        if stage == 'input_functions':
            log_process_start('AI input function extraction')
            input_function_AI(analysis_directory, nifti_directory, image_directory, filenames, parameters)
            log_process_end('AI input function extraction')
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

        t1_path = os.path.join(nifti_directory, t1_3D_filename)
        t2_path = os.path.join(nifti_directory, axial_t2_2D_filename)
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
            if not dce_path or not os.path.exists(dce_path):
                raise RuntimeError('Missing DCE NIfTI; cannot coregister masks without DCE.')
            print('[segmentation] Coregistering masks into T2/DCE space')
            AIT.coregistration(seg_mgz_path=seg_mgz_path, dce_path=dce_path, t2_path=t2_path)
            return 0

        if stage == 'tissue_ctc':
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
            models = ['patlak', 'two_compartment'] if model_setting == 'both' else [model_setting]

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

            C_a_full, _ = AIT.get_input_function_curve(analysis_directory)
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
            compute_ki = bool(getattr(args, 'voxelwise', False))
            compute_cbf = bool(getattr(args, 'cbf', False))
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

    try:
        existing = None
        if runner_path.exists():
            existing = runner_path.read_text(encoding="utf-8", errors="ignore")
        if existing and f"version: {_STAGE_RUNNER_VERSION}" in existing:
            return runner_path
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
                        runner_path = _ensure_stage_runner_script(data_root)
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
                        runner_path = _ensure_stage_runner_script(data_root)
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
    }


def _set_settings(patch: Dict[str, Any]) -> Dict[str, Any]:
    current = _get_settings()
    for k, v in (patch or {}).items():
        if v is None:
            continue
        if k in current:
            current[k] = v
    # normalize strings
    for k in ["firstName", "pbrainMainPy", "fastsurferDir", "freesurferHome"]:
        current[k] = str(current.get(k) or "").strip()
    current["onboardingCompleted"] = bool(current.get("onboardingCompleted") or False)
    db.settings = current
    db.save()
    return current


def _resolve_pbrain_main_py() -> str:
    env = os.environ.get("PBRAIN_MAIN_PY")
    if env and env.strip():
        return env.strip()
    s = _get_settings().get("pbrainMainPy")
    return str(s or "").strip()


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
        config={},
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


@app.get("/subjects/{subject_id}/tofts")
def get_subject_tofts(subject_id: str, region: str = "gm") -> Dict[str, Any]:
    subject = _find_subject(subject_id)
    project = _find_project(subject.projectId)

    t = _load_time_points(subject)
    aif = _load_aif_curve(subject)
    tissue = _load_ai_tissue_curve(subject, _region_to_ai_key(region))
    t, aif, tissue, _dt = _align_curves(t, aif, tissue)

    lambd = float(_get_config_value(project, ["model", "lambdaTikhonov"], 0.1))
    return _tofts_from_curves(t, aif, tissue, lambd=lambd)


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
        created.extend(await _start_subject_run(project=project, subject=subject))

    db.save()
    return [asdict(j) for j in created]


@app.post("/subjects/{subject_id}/run-stage")
async def run_stage(subject_id: str, req: RunStageRequest) -> Dict[str, Any]:
    subject = _find_subject(subject_id)
    project = _find_project(subject.projectId)
    stage = req.stageId
    if stage not in STAGES:
        raise HTTPException(status_code=400, detail="Invalid stage")

    job = await _start_subject_stage_run(project=project, subject=subject, stage=stage, run_dependencies=req.runDependencies)
    db.save()
    return asdict(job)


async def _start_subject_run(*, project: Project, subject: Subject) -> List[Job]:
    # Disallow starting if there is an active run for this subject.
    if any(j.subjectId == subject.id and j.status in {"queued", "running"} for j in db.jobs):
        raise HTTPException(status_code=409, detail="Subject already has a queued/running job")

    created: List[Job] = []
    job_ids: List[str] = []
    shared_start = _now_iso()
    stages_to_run: List[StageId] = []
    for stage in STAGES:
        if stage == "diffusion" and not subject.hasDiffusion:
            continue
        stages_to_run.append(stage)

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


async def _run_pbrain_stage_chain(*, project: Project, subject: Subject, jobs: List[Job]) -> None:
    try:
        for j in jobs:
            if j.status == "cancelled":
                return
            await _run_pbrain_single_stage(project=project, subject=subject, job=j)
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
    stages: List[StageId] = [s for s in chain if s == stage or subject.stageStatuses.get(str(s)) != "done"]
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
        task = asyncio.create_task(_run_pbrain_single_stage(project=project, subject=subject, job=created[0]))
        db._job_tasks[created[0].id] = task
        await asyncio.sleep(0)
        return created[0]

    task = asyncio.create_task(_run_pbrain_stage_chain(project=project, subject=subject, jobs=created))
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
