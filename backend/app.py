from __future__ import annotations

import asyncio
import fnmatch
import json
import os
import signal
import sys
from functools import lru_cache
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field


try:
    import numpy as np  # type: ignore
except Exception:  # pragma: no cover
    np = None  # type: ignore

try:
    from scipy.optimize import least_squares  # type: ignore
except Exception:  # pragma: no cover
    least_squares = None  # type: ignore

try:
    import nibabel as nib  # type: ignore
except Exception:  # pragma: no cover
    nib = None  # type: ignore


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
    "montage_qc",
]
StageStatus = Literal["not_run", "running", "done", "failed"]


def _now_iso() -> str:
    return datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


def _db_path() -> Path:
    # Store DB beside this file by default.
    return Path(__file__).with_name("db.json")


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


class UpdateProjectConfigRequest(BaseModel):
    configUpdate: Dict[str, Any] = Field(default_factory=dict)


class ImportSubjectsRequest(BaseModel):
    subjects: List[Dict[str, str]]  # {name, sourcePath}


class RunFullPipelineRequest(BaseModel):
    subjectIds: List[str]


class EnsureArtifactsResponse(BaseModel):
    started: bool
    jobs: List[Dict[str, Any]] = Field(default_factory=list)
    reason: str = ""


class ResolveDefaultVolumeResponse(BaseModel):
    path: str


class VolumeInfoRequest(BaseModel):
    path: str


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


class DB:
    def __init__(self) -> None:
        self.projects: List[Project] = []
        self.subjects: List[Subject] = []
        self.jobs: List[Job] = []
        self._load()

        # job_id -> asyncio.Task
        self._job_tasks: Dict[str, asyncio.Task] = {}
        # job_id -> process
        self._job_processes: Dict[str, asyncio.subprocess.Process] = {}
        # subject_id -> job_ids (the stage jobs created for the current run)
        self._subject_job_ids: Dict[str, List[str]] = {}
        # subject_id -> current stage index
        self._subject_stage_index: Dict[str, int] = {}

    def _load(self) -> None:
        path = _db_path()
        if not path.exists():
            return
        try:
            raw = json.loads(path.read_text())
            self.projects = [Project(**p) for p in raw.get("projects", [])]
            self.subjects = [Subject(**s) for s in raw.get("subjects", [])]
            self.jobs = [Job(**j) for j in raw.get("jobs", [])]
        except Exception:
            # If corrupted, start fresh (still local-only).
            self.projects = []
            self.subjects = []
            self.jobs = []

    def save(self) -> None:
        path = _db_path()
        payload = {
            "projects": [asdict(p) for p in self.projects],
            "subjects": [asdict(s) for s in self.subjects],
            "jobs": [asdict(j) for j in self.jobs],
        }
        path.write_text(json.dumps(payload, indent=2))


db = DB()


STAGES: List[StageId] = [
    "import",
    "t1_fit",
    "input_functions",
    "time_shift",
    "segmentation",
    "tissue_ctc",
    "modelling",
    "diffusion",
    "montage_qc",
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
    j = next((j for j in db.jobs if j.id == job_id), None)
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
    # Improve import reliability when calling p-brain by path.
    main_py = env.get("PBRAIN_MAIN_PY")
    if main_py:
        pbrain_root = str(Path(main_py).expanduser().resolve().parent)
        env["PYTHONPATH"] = pbrain_root + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    return env


def _require_numpy() -> None:
    if np is None:
        raise HTTPException(status_code=500, detail="Backend missing dependency: numpy")


def _require_nibabel() -> None:
    if nib is None:
        raise HTTPException(status_code=500, detail="Backend missing dependency: nibabel")


def _require_scipy() -> None:
    if least_squares is None:
        raise HTTPException(status_code=500, detail="Backend missing dependency: scipy")


def _analysis_dir_for_subject(subject: Subject) -> Path:
    return Path(subject.sourcePath).expanduser().resolve() / "Analysis"


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


def _volume_info_from_img(img) -> Dict[str, Any]:
    _require_numpy()
    assert np is not None
    shape = tuple(int(x) for x in img.shape)
    zooms = img.header.get_zooms()
    voxel_size = [float(zooms[0]), float(zooms[1]), float(zooms[2])] if len(zooms) >= 3 else [1.0, 1.0, 1.0]
    dims: List[int] = [shape[0], shape[1], shape[2] if len(shape) >= 3 else 1]
    if len(shape) >= 4:
        dims.append(shape[3])

    # Cheap min/max from a central slice/timeframe.
    z = dims[2] // 2
    t = 0
    if len(shape) >= 4:
        arr = np.asanyarray(img.dataobj[:, :, z, t])
    else:
        arr = np.asanyarray(img.dataobj[:, :, z])
    vmin = float(np.nanmin(arr)) if arr.size else 0.0
    vmax = float(np.nanmax(arr)) if arr.size else 0.0

    return {
        "dimensions": dims,
        "voxelSize": voxel_size,
        "dataType": str(img.get_data_dtype()),
        "min": vmin,
        "max": vmax,
    }


def _analysis_metrics_table(subject: Subject) -> Dict[str, Any]:
    analysis_dir = _analysis_dir_for_subject(subject)
    p = analysis_dir / "AI_values_median_total.json"
    if not p.exists():
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


def _analysis_curves(subject: Subject) -> List[Dict[str, Any]]:
    _require_numpy()
    assert np is not None
    analysis_dir = _analysis_dir_for_subject(subject)
    time_path = analysis_dir / "Fitting" / "time_points_s.npy"
    if not time_path.exists():
        return []
    time_points = np.load(str(time_path)).astype(float).tolist()

    curves: List[Dict[str, Any]] = []

    # Prefer shifted curves if present.
    def pick_curve(glob_path: Path) -> Optional[Path]:
        matches = sorted(glob_path.parent.glob(glob_path.name))
        return matches[0] if matches else None

    # AIF (artery): first available subtype.
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

    # Tissue curves: include up to 3.
    tissue_dir = analysis_dir / "CTC Data" / "Tissue"
    if tissue_dir.exists():
        count = 0
        for subtype_dir in sorted([d for d in tissue_dir.iterdir() if d.is_dir()]):
            pth = pick_curve(subtype_dir / "CTC_slice_*.npy")
            if pth and pth.exists():
                vals = np.load(str(pth)).astype(float).tolist()
                curves.append({"id": f"tissue_{subtype_dir.name}", "name": f"Tissue ({subtype_dir.name})", "timePoints": time_points, "values": vals, "unit": "mM"})
                count += 1
                if count >= 3:
                    break

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
    """Return a list of on-disk NIfTI map volumes for the subject.

    This is intentionally conservative: we only expose known p-brain outputs
    under the subject's Analysis folder.
    """

    analysis_dir = _analysis_dir_for_subject(subject)
    maps: List[Dict[str, Any]] = []

    # Voxelwise atlas-mapped outputs (written directly to Analysis/).
    atlas_maps = [
        ("ki_atlas", "Ki Map (atlas)", "ml/100g/min", analysis_dir / "Ki_map_atlas.nii.gz"),
        ("sd_ki_atlas", "SD Ki Map (atlas)", "ml/100g/min", analysis_dir / "SD_Ki_map_atlas.nii.gz"),
        ("vp_atlas", "vp Map (atlas)", "fraction", analysis_dir / "vp_map_atlas.nii.gz"),
        (
            "cbf_tikh_atlas",
            "CBF Map (tikhonov, atlas)",
            "ml/100g/min",
            analysis_dir / "CBF_tikhonov_map_atlas.nii.gz",
        ),
        (
            "mtt_tikh_atlas",
            "MTT Map (tikhonov, atlas)",
            "s",
            analysis_dir / "MTT_tikhonov_map_atlas.nii.gz",
        ),
        (
            "cth_tikh_atlas",
            "CTH Map (tikhonov, atlas)",
            "s",
            analysis_dir / "CTH_tikhonov_map_atlas.nii.gz",
        ),
    ]
    for map_id, name, unit, path in atlas_maps:
        if path.exists():
            maps.append({"id": map_id, "name": name, "unit": unit, "path": str(path), "group": "modelling"})

    # Diffusion outputs (Analysis/diffusion/*.nii.gz)
    diffusion_dir = analysis_dir / "diffusion"
    diffusion_maps = [
        ("fa", "FA Map", "fraction", diffusion_dir / "fa_map.nii.gz"),
        ("md", "MD Map", "mm²/s", diffusion_dir / "md_map.nii.gz"),
        ("ad", "AD Map", "mm²/s", diffusion_dir / "ad_map.nii.gz"),
        ("rd", "RD Map", "mm²/s", diffusion_dir / "rd_map.nii.gz"),
    ]
    for map_id, name, unit, path in diffusion_maps:
        if path.exists():
            maps.append({"id": map_id, "name": name, "unit": unit, "path": str(path), "group": "diffusion"})

    # Legacy diffusion output (written to Analysis/ for backwards compatibility).
    legacy_fa = analysis_dir / "FA_map.nii.gz"
    if legacy_fa.exists() and not any(m.get("id") == "fa" for m in maps):
        maps.append({"id": "fa", "name": "FA Map", "unit": "fraction", "path": str(legacy_fa), "group": "diffusion"})

    return maps


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
    artery_dir = analysis_dir / "CTC Data" / "Artery"
    if not artery_dir.exists():
        raise HTTPException(status_code=404, detail="Missing CTC Data/Artery")

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
    assert least_squares is not None

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

    sol = least_squares(residual, x0, bounds=(0.0, np.inf))
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
        job = next((j for j in db.jobs if j.id == jid), None)
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


def _pbrain_cli_args(project: Project, subject: Subject) -> List[str]:
    main_py = os.environ.get("PBRAIN_MAIN_PY")
    if not main_py:
        raise RuntimeError("PBRAIN_MAIN_PY is not set (path to p-brain/main.py)")

    data_root = Path(project.storagePath).expanduser().resolve()
    args: List[str] = [
        sys.executable,
        str(Path(main_py).expanduser().resolve()),
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

    # Diffusion: only request when the subject has diffusion.
    if subject.hasDiffusion:
        args.append("--diffusion")

    return args


async def _run_pbrain_auto(*, project: Project, subject: Subject) -> None:
    data_root = Path(project.storagePath).expanduser().resolve()
    if not data_root.exists():
        raise RuntimeError(f"Project storagePath/data root does not exist: {data_root}")

    jobs = _stage_jobs_for_subject(subject.id)
    if not jobs:
        raise RuntimeError("No stage jobs registered for subject run")

    # Shared log file per run.
    logs_dir = data_root / ".pbrain-web" / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    log_path = logs_dir / f"run_{subject.id}_{int(datetime.utcnow().timestamp()*1000)}.log"

    for job in jobs.values():
        job.logPath = str(log_path)
        job.startTime = _now_iso()

    if not subject.hasDiffusion:
        jobs["diffusion"].status = "completed"
        jobs["diffusion"].progress = 100
        jobs["diffusion"].currentStep = "Skipped (no diffusion)"
        jobs["diffusion"].endTime = _now_iso()
        # Keep the subject stage status as not_run.
        subject.stageStatuses["diffusion"] = "not_run"

    # Stage: import starts immediately.
    _set_job(jobs["import"], status="running", progress=5, step="Preparing inputs")
    _set_stage_status(subject, "import", "running")
    db.save()

    args = _pbrain_cli_args(project, subject)

    # Make sure we run from inside the p-brain folder for consistent relative paths.
    main_py = os.environ.get("PBRAIN_MAIN_PY")
    pbrain_cwd = str(Path(main_py).expanduser().resolve().parent) if main_py else None

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

    # Open log file once and append efficiently.
    log_fh = log_path.open("a", encoding="utf-8", errors="replace")

    def write_line_sync(line: str) -> None:
        log_fh.write(line)
        log_fh.flush()

    # Helper to advance stage jobs in order.
    stage_order: List[StageId] = STAGES
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

    try:
        assert proc.stdout is not None
        while True:
            line_b = await proc.stdout.readline()
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

            elif "starting process: segmented m0/t1 rendering" in low:
                begin_stage("montage_qc", "QC + montage outputs", 10)
            elif "completed process: segmented m0/t1 rendering" in low:
                finish_stage("montage_qc")

            subject.updatedAt = _now_iso()
            db.save()

        rc = await proc.wait()
        if any(j.status == "cancelled" for j in jobs.values()):
            return
        if rc != 0:
            raise RuntimeError(f"p-brain exited with code {rc}")

        # If p-brain exits cleanly but some stages never emitted markers, complete them.
        for stage in stage_order:
            if jobs[stage].status in {"queued", "running"}:
                _set_job(jobs[stage], status="completed", progress=100, step="Completed")
                jobs[stage].endTime = _now_iso()
                _set_stage_status(subject, stage, "done")
        db.save()

    except Exception as exc:
        err = str(exc)
        for stage in stage_order:
            if jobs[stage].status in {"queued", "running"}:
                _set_job(jobs[stage], status="failed", progress=jobs[stage].progress, step="Failed", error=err)
                jobs[stage].endTime = _now_iso()
                _set_stage_status(subject, stage, "failed")
        db.save()
        raise
    finally:
        try:
            log_fh.close()
        except Exception:
            pass
        for job in jobs.values():
            db._job_processes.pop(job.id, None)
            db._job_tasks.pop(job.id, None)
        db._subject_job_ids.pop(subject.id, None)
        db._subject_stage_index.pop(subject.id, None)


app = FastAPI(title="p-brain-web local backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health() -> Dict[str, Any]:
    return {
        "ok": True,
        "time": _now_iso(),
        "pbrain_main_py": os.environ.get("PBRAIN_MAIN_PY"),
    }


@app.get("/_spark/loaded")
def spark_loaded() -> Dict[str, Any]:
    return {"ok": True}


@app.get("/projects")
def get_projects() -> List[Dict[str, Any]]:
    return [asdict(p) for p in db.projects]


@app.get("/projects/{project_id}")
def get_project(project_id: str) -> Dict[str, Any]:
    p = _find_project(project_id)
    return asdict(p)


@app.delete("/projects/{project_id}")
def delete_project(project_id: str) -> Dict[str, Any]:
    _find_project(project_id)

    subject_ids = {s.id for s in db.subjects if s.projectId == project_id}

    db.projects = [p for p in db.projects if p.id != project_id]
    db.subjects = [s for s in db.subjects if s.projectId != project_id]
    db.jobs = [j for j in db.jobs if j.projectId != project_id and j.subjectId not in subject_ids]

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
    info = _volume_info_from_img(img)
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


@app.get("/subjects/{subject_id}/metrics")
def get_subject_metrics(subject_id: str) -> Dict[str, Any]:
    subject = _find_subject(subject_id)
    return _analysis_metrics_table(subject)


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

    # Prefer listing volumes from the NIfTI folder (3D + 4D) for Viewer selection.
    if nifti_dir.exists():
        for p in sorted(nifti_dir.rglob("*.nii*")):
            if not p.is_file():
                continue
            sp = str(p)
            if sp in seen:
                continue
            seen.add(sp)
            volumes.append({"id": sp, "name": p.name, "path": sp})

    # Always include the default volumes if they exist (covers non-nested layouts).
    for kind in ("dce", "t1", "diffusion"):
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
        if not source_path.is_absolute() or str(source).startswith("/") and not Path(source).exists():
            source_path = Path(project.storagePath) / name

        sp = source_path.expanduser().resolve()

        nifti_root = _nifti_dir_for_path(project, sp)

        # Basic data presence checks (very conservative; refined later via patterns).
        has_nifti = any(nifti_root.rglob("*.nii")) or any(nifti_root.rglob("*.nii.gz"))

        # Prefer configured patterns (can be comma-separated fallbacks).
        has_t1 = any(_glob_first(nifti_root, patt) for patt in (t1_patterns or []))
        has_dce = any(_glob_first(nifti_root, patt) for patt in (dce_patterns or []))
        has_diff = any(_glob_first(nifti_root, patt) for patt in (diff_patterns or []))

        # Fallback heuristics if patterns were empty or didn't match.
        if not has_t1:
            has_t1 = any("t1" in p.name.lower() for p in nifti_root.rglob("*.nii*"))
        if not has_dce:
            has_dce = any("dce" in p.name.lower() for p in nifti_root.rglob("*.nii*"))
        if not has_diff:
            has_diff = any("dwi" in p.name.lower() or "dti" in p.name.lower() for p in nifti_root.rglob("*.nii*"))

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
    jobs = db.jobs
    if projectId:
        jobs = [j for j in jobs if j.projectId == projectId]
    if subjectId:
        jobs = [j for j in jobs if j.subjectId == subjectId]
    if status:
        jobs = [j for j in jobs if j.status == status]

    def key(j: Job) -> str:
        return j.startTime or ""

    return [asdict(j) for j in sorted(jobs, key=key, reverse=True)]


@app.post("/projects/{project_id}/run-full")
async def run_full(project_id: str, req: RunFullPipelineRequest) -> List[Dict[str, Any]]:
    project = _find_project(project_id)
    created: List[Job] = []

    for subject_id in req.subjectIds:
        subject = _find_subject(subject_id)
        created.extend(await _start_subject_run(project=project, subject=subject))

    db.save()
    return [asdict(j) for j in created]


async def _start_subject_run(*, project: Project, subject: Subject) -> List[Job]:
    # Disallow starting if there is an active run for this subject.
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
        created.append(job)
        job_ids.append(jid)
        await asyncio.sleep(0)

    db._subject_job_ids[subject.id] = job_ids
    task = asyncio.create_task(_run_pbrain_auto(project=project, subject=subject))
    for jid in job_ids:
        db._job_tasks[jid] = task
    return created


@app.post("/subjects/{subject_id}/ensure")
async def ensure_subject_artifacts(subject_id: str, kind: str = "all") -> Dict[str, Any]:
    """Ensure key artifacts exist; if missing, trigger a p-brain auto run.

    kind: one of "all", "maps", "curves", "montages".
    """

    subject = _find_subject(subject_id)
    project = _find_project(subject.projectId)

    want = (kind or "all").strip().lower()
    if want not in {"all", "maps", "curves", "montages"}:
        raise HTTPException(status_code=400, detail="Invalid kind")

    missing: List[str] = []
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

    created = await _start_subject_run(project=project, subject=subject)
    db.save()
    return {"started": True, "jobs": [asdict(j) for j in created], "reason": f"Missing: {', '.join(missing)}"}


@app.post("/jobs/{job_id}/cancel")
async def cancel_job(job_id: str) -> Dict[str, Any]:
    job = _find_job(job_id)
    if job.status not in {"queued", "running"}:
        return asdict(job)

    # Cancel the entire subject run (all stage jobs).
    stage_jobs = [j for j in db.jobs if j.subjectId == job.subjectId and j.startTime == job.startTime]
    if not stage_jobs:
        stage_jobs = [j for j in db.jobs if j.subjectId == job.subjectId and j.status in {"queued", "running"}]

    for j in stage_jobs:
        j.status = "cancelled"
        j.endTime = _now_iso()
        j.currentStep = "Cancelled"

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
        created.append(job)
        job_ids.append(jid)
        await asyncio.sleep(0)

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
    lines = p.read_text(errors="replace").splitlines()[-tail:]
    return {"lines": lines}
