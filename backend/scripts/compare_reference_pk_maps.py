#!/usr/bin/env python3
"""Compare p-Brain voxelwise PK maps against a MATLAB reference .mat file.

Designed for hemiSure turboflash runs where a reference file like:
  20240618x2_CBF_maps_method4_tik_....mat
contains 3D volumes named e.g. CBF, MTT, CBKi, CBV_p.

Outputs:
  - JSON metrics summary
  - Per-map PNG montages: reference vs p-brain vs difference

Example:
  python3 backend/scripts/compare_reference_pk_maps.py \
    --subject-dir /Volumes/T5_EVO_EDT/hemisure/20240618x2_flot \
    --ref-mat "/Volumes/T5_EVO_EDT/hemisure/20240618x2_flot/20240618x2_CBF_maps_method4_tik_conc_methodT1_map_input_MRsignal_M_fix_slice4_slice4_shifted_PaReLLeL_offset1_.mat"
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np


ABS_PCT_BINS_DEFAULT = [0.0, 2.0, 5.0, 10.0, 25.0, 50.0, 100.0]


def _load_mat(path: Path) -> Dict[str, np.ndarray]:
    from scipy.io import loadmat

    md = loadmat(str(path))
    out: Dict[str, np.ndarray] = {}
    for k, v in md.items():
        if k.startswith("__"):
            continue
        try:
            a = np.asarray(v)
        except Exception:
            continue
        if a.dtype.kind not in {"i", "u", "f"}:
            continue
        a = np.squeeze(a)
        if a.ndim != 3:
            continue
        out[k] = a.astype(float, copy=False)
    return out


def _load_nii(path: Path) -> np.ndarray:
    import nibabel as nib

    img = nib.load(str(path))
    return np.asarray(img.get_fdata(), dtype=float)


def _finite_mask(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.isfinite(a) & np.isfinite(b)


def _sample_xy(x: np.ndarray, y: np.ndarray, mask: np.ndarray, *, max_points: int = 200_000) -> Tuple[np.ndarray, np.ndarray]:
    idx = np.flatnonzero(mask.ravel())
    if idx.size == 0:
        return np.array([], dtype=float), np.array([], dtype=float)
    if idx.size > max_points:
        rng = np.random.default_rng(0)
        idx = rng.choice(idx, size=max_points, replace=False)
    x1 = x.ravel()[idx].astype(float, copy=False)
    y1 = y.ravel()[idx].astype(float, copy=False)
    return x1, y1


def _fit_affine(x: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
    """Fit y ≈ a*x + b (least squares)."""
    if x.size < 2:
        return (float("nan"), float("nan"))
    A = np.column_stack([x, np.ones_like(x)])
    try:
        sol, *_ = np.linalg.lstsq(A, y, rcond=None)
        a = float(sol[0])
        b = float(sol[1])
        return a, b
    except Exception:
        return (float("nan"), float("nan"))


def _corr(x: np.ndarray, y: np.ndarray) -> float:
    if x.size < 2:
        return float("nan")
    x0 = x - float(np.nanmean(x))
    y0 = y - float(np.nanmean(y))
    den = float(np.sqrt(np.nansum(x0 * x0) * np.nansum(y0 * y0)))
    if den <= 0:
        return float("nan")
    return float(np.nansum(x0 * y0) / den)


def _metrics(ref: np.ndarray, pb: np.ndarray, *, mask: np.ndarray, a: float = 1.0, b: float = 0.0) -> Dict[str, Any]:
    m = mask
    if not np.any(m):
        return {"n": 0}
    r = ref[m]
    p = pb[m]
    pred = a * r + b
    err = p - pred
    out: Dict[str, Any] = {
        "n": int(r.size),
        "ref_mean": float(np.nanmean(r)),
        "pb_mean": float(np.nanmean(p)),
        "mae": float(np.nanmean(np.abs(err))),
        "rmse": float(np.sqrt(np.nanmean(err * err))),
        "ref_p99": float(np.nanpercentile(r, 99)),
        "pb_p99": float(np.nanpercentile(p, 99)),
    }
    return out


def _robust_mask(ref: np.ndarray, pb: np.ndarray) -> np.ndarray:
    m = _finite_mask(ref, pb)
    if not np.any(m):
        return m
    # Drop obvious background.
    m &= (ref != 0) | (pb != 0)
    if not np.any(m):
        return m

    # Trim extreme tails to reduce sensitivity to outliers (esp. MTT).
    r = ref[m]
    p = pb[m]
    try:
        r_lo, r_hi = np.nanpercentile(r, [1, 99])
        p_lo, p_hi = np.nanpercentile(p, [1, 99])
        m &= (ref >= r_lo) & (ref <= r_hi) & (pb >= p_lo) & (pb <= p_hi)
    except Exception:
        pass
    return m


def _save_montage(
    *,
    out_png: Path,
    title: str,
    ref: np.ndarray,
    pb: np.ndarray,
    diff: np.ndarray,
    mask: np.ndarray,
) -> None:
    os.environ.setdefault("MPLBACKEND", "Agg")
    import matplotlib.pyplot as plt

    # Scale per-panel from masked voxels.
    m = mask
    if np.any(m):
        rvals = ref[m]
        pvals = pb[m]
        dvals = diff[m]
        rvmin, rvmax = np.nanpercentile(rvals, [2, 98])
        pvmin, pvmax = np.nanpercentile(pvals, [2, 98])
        dv = float(np.nanpercentile(np.abs(dvals), 98))
    else:
        rvmin, rvmax = float(np.nanmin(ref)), float(np.nanmax(ref))
        pvmin, pvmax = float(np.nanmin(pb)), float(np.nanmax(pb))
        dv = float(np.nanmax(np.abs(diff)))

    if not np.isfinite(dv) or dv <= 0:
        dv = 1.0

    nz = int(ref.shape[2])
    ncols = min(nz, 10)
    # 3 rows: ref, pb, diff
    fig, axes = plt.subplots(3, ncols, figsize=(1.6 * ncols, 5.2), constrained_layout=True)
    fig.suptitle(title, fontsize=12)

    # Choose evenly spaced slices if > 10.
    if nz <= ncols:
        zs = list(range(nz))
    else:
        zs = [int(round(z)) for z in np.linspace(0, nz - 1, ncols)]

    for col, z in enumerate(zs):
        ax = axes[0, col]
        ax.imshow(ref[:, :, z].T, cmap="inferno", origin="lower", vmin=rvmin, vmax=rvmax)
        ax.set_title(f"z={z}", fontsize=9)
        ax.axis("off")

        ax = axes[1, col]
        ax.imshow(pb[:, :, z].T, cmap="inferno", origin="lower", vmin=pvmin, vmax=pvmax)
        ax.axis("off")

        ax = axes[2, col]
        ax.imshow(diff[:, :, z].T, cmap="coolwarm", origin="lower", vmin=-dv, vmax=dv)
        ax.axis("off")

    # Row labels
    axes[0, 0].set_ylabel("ref", fontsize=10)
    axes[1, 0].set_ylabel("p-brain", fontsize=10)
    axes[2, 0].set_ylabel("diff", fontsize=10)

    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=160)
    plt.close(fig)


def _apply_xy_transform(vol: np.ndarray, name: str) -> np.ndarray:
    """Apply an in-plane (x/y) transform slice-wise, preserving vol shape."""

    name = (name or "id").strip().lower()
    if name in {"id", "identity", "none"}:
        return vol

    def tx2(a2: np.ndarray) -> np.ndarray:
        if name in {"rot90", "r90"}:
            return np.rot90(a2, 1)
        if name in {"rot180", "r180"}:
            return np.rot90(a2, 2)
        if name in {"rot270", "r270"}:
            return np.rot90(a2, 3)
        if name in {"fliplr", "flipx"}:
            return np.fliplr(a2)
        if name in {"flipud", "flipy"}:
            return np.flipud(a2)
        if name in {"transpose", "t"}:
            return a2.T
        if name in {"rot90_fliplr"}:
            return np.fliplr(np.rot90(a2, 1))
        if name in {"rot90_flipud"}:
            return np.flipud(np.rot90(a2, 1))
        if name in {"rot180_fliplr"}:
            return np.fliplr(np.rot90(a2, 2))
        if name in {"rot180_flipud"}:
            return np.flipud(np.rot90(a2, 2))
        if name in {"rot270_fliplr"}:
            return np.fliplr(np.rot90(a2, 3))
        if name in {"rot270_flipud"}:
            return np.flipud(np.rot90(a2, 3))
        if name in {"transpose_fliplr"}:
            return np.fliplr(a2.T)
        if name in {"transpose_flipud"}:
            return np.flipud(a2.T)
        raise ValueError(f"Unknown transform: {name}")

    out = np.empty_like(vol)
    for z in range(vol.shape[2]):
        out[:, :, z] = tx2(vol[:, :, z])
    return out


def _best_transform_for_key(ref_vol: np.ndarray, pb_vol: np.ndarray) -> Tuple[str, float]:
    """Choose a reference->pbrain in-plane transform maximizing correlation."""

    candidates = [
        "id",
        "rot90",
        "rot180",
        "rot270",
        "fliplr",
        "flipud",
        "rot90_fliplr",
        "rot90_flipud",
        "rot180_fliplr",
        "rot180_flipud",
        "rot270_fliplr",
        "rot270_flipud",
        "transpose",
        "transpose_fliplr",
        "transpose_flipud",
    ]

    best_name = "id"
    best_corr = -1e9
    for name in candidates:
        try:
            r2 = _apply_xy_transform(ref_vol, name)
        except Exception:
            continue
        m = _robust_mask(r2, pb_vol)
        x, y = _sample_xy(r2, pb_vol, m)
        c = _corr(x, y)
        if not np.isfinite(c):
            continue
        if c > best_corr:
            best_corr = c
            best_name = name

    if best_corr == -1e9:
        return ("id", float("nan"))
    return (best_name, float(best_corr))


def _save_triptych_slices(
    *,
    out_png: Path,
    title: str,
    ref: np.ndarray,
    pb: np.ndarray,
    diff: np.ndarray,
    mask: np.ndarray,
    max_rows: int = 10,
) -> None:
    """Save a (rows=slices, cols=3) image: ref | pb | diff."""

    os.environ.setdefault("MPLBACKEND", "Agg")
    import matplotlib.pyplot as plt

    nz = int(ref.shape[2])
    nrows = min(nz, int(max_rows))
    if nz <= nrows:
        zs = list(range(nz))
    else:
        zs = [int(round(z)) for z in np.linspace(0, nz - 1, nrows)]

    # Per-map scaling derived from masked values.
    m = mask
    if np.any(m):
        rvals = ref[m]
        pvals = pb[m]
        dvals = diff[m]
        rvmin, rvmax = np.nanpercentile(rvals, [2, 98])
        pvmin, pvmax = np.nanpercentile(pvals, [2, 98])
        dv = float(np.nanpercentile(np.abs(dvals), 98))
    else:
        rvmin, rvmax = float(np.nanmin(ref)), float(np.nanmax(ref))
        pvmin, pvmax = float(np.nanmin(pb)), float(np.nanmax(pb))
        dv = float(np.nanmax(np.abs(diff)))

    if not np.isfinite(dv) or dv <= 0:
        dv = 1.0

    fig, axes = plt.subplots(nrows, 3, figsize=(9.2, 2.5 * nrows), constrained_layout=True)
    fig.suptitle(title, fontsize=12)
    if nrows == 1:
        axes = np.expand_dims(axes, axis=0)

    col_titles = ["MATLAB (ref)", "p-Brain", "diff (pb - scaled_ref)"]
    for j in range(3):
        axes[0, j].set_title(col_titles[j], fontsize=10)

    for i, z in enumerate(zs):
        axes[i, 0].imshow(ref[:, :, z], cmap="inferno", origin="lower", vmin=rvmin, vmax=rvmax)
        axes[i, 1].imshow(pb[:, :, z], cmap="inferno", origin="lower", vmin=pvmin, vmax=pvmax)
        axes[i, 2].imshow(diff[:, :, z], cmap="coolwarm", origin="lower", vmin=-dv, vmax=dv)

        for j in range(3):
            axes[i, j].axis("off")

        axes[i, 0].set_ylabel(f"z={z}", fontsize=9)

    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=160)
    plt.close(fig)


def _abs_pct_diff(pb: np.ndarray, pred: np.ndarray, *, denom_min: float = 1e-8, clip_max: float = 100.0) -> np.ndarray:
    """Absolute percent difference between pb and pred, clipped to [0, clip_max]."""

    denom = np.maximum(np.abs(pred), float(denom_min))
    out = 100.0 * np.abs(pb - pred) / denom
    out = np.asarray(out, dtype=float)
    if np.isfinite(clip_max):
        out = np.clip(out, 0.0, float(clip_max))
    return out


def _hist_counts(values: np.ndarray, *, bins: list[float]) -> Dict[str, Any]:
    """Histogram counts for values in [bins[0], bins[-1]] with inclusive last edge."""

    b = np.asarray(bins, dtype=float)
    if b.ndim != 1 or b.size < 2:
        return {"bins": list(map(float, bins)), "counts": [], "fractions": [], "n": 0}

    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return {"bins": b.tolist(), "counts": [0] * (b.size - 1), "fractions": [0.0] * (b.size - 1), "n": 0}

    v = np.clip(v, float(b[0]), float(b[-1]))
    counts, edges = np.histogram(v, bins=b)
    n = int(v.size)
    fracs = (counts / max(n, 1)).astype(float)
    return {
        "bins": edges.tolist(),
        "counts": counts.tolist(),
        "fractions": fracs.tolist(),
        "n": n,
    }


def _save_triptych_abs_pct_discrete(
    *,
    out_png: Path,
    title: str,
    ref: np.ndarray,
    pb: np.ndarray,
    abs_pct: np.ndarray,
    mask: np.ndarray,
    max_rows: int = 10,
    bins: list[float] | None = None,
) -> None:
    """Save a (rows=slices, cols=3) image: ref | pb | abs%diff (discrete bins)."""

    os.environ.setdefault("MPLBACKEND", "Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import BoundaryNorm

    if bins is None:
        bins = list(ABS_PCT_BINS_DEFAULT)

    nz = int(ref.shape[2])
    nrows = min(nz, int(max_rows))
    if nz <= nrows:
        zs = list(range(nz))
    else:
        zs = [int(round(z)) for z in np.linspace(0, nz - 1, nrows)]

    # Per-map scaling derived from masked values (ref/pb only).
    m = mask
    if np.any(m):
        rvals = ref[m]
        pvals = pb[m]
        rvmin, rvmax = np.nanpercentile(rvals, [2, 98])
        pvmin, pvmax = np.nanpercentile(pvals, [2, 98])
    else:
        rvmin, rvmax = float(np.nanmin(ref)), float(np.nanmax(ref))
        pvmin, pvmax = float(np.nanmin(pb)), float(np.nanmax(pb))

    bounds = np.asarray(bins, dtype=float)
    ncolors = max(int(bounds.size - 1), 2)
    cmap = plt.get_cmap("viridis", ncolors)
    norm = BoundaryNorm(bounds, ncolors=cmap.N, clip=True)

    fig, axes = plt.subplots(nrows, 3, figsize=(9.6, 2.5 * nrows), constrained_layout=True)
    fig.suptitle(title, fontsize=12)
    if nrows == 1:
        axes = np.expand_dims(axes, axis=0)

    col_titles = ["MATLAB (ref)", "p-Brain", "abs % diff (|pb - scaled_ref| / |scaled_ref|)"]
    for j in range(3):
        axes[0, j].set_title(col_titles[j], fontsize=10)

    last_im = None
    for i, z in enumerate(zs):
        axes[i, 0].imshow(ref[:, :, z], cmap="inferno", origin="lower", vmin=rvmin, vmax=rvmax)
        axes[i, 1].imshow(pb[:, :, z], cmap="inferno", origin="lower", vmin=pvmin, vmax=pvmax)
        last_im = axes[i, 2].imshow(abs_pct[:, :, z], cmap=cmap, norm=norm, origin="lower")

        for j in range(3):
            axes[i, j].axis("off")
        axes[i, 0].set_ylabel(f"z={z}", fontsize=9)

    # Discrete colorbar with bin edges.
    if last_im is not None:
        cax_axes = [axes[i, 2] for i in range(nrows)]
        cbar = fig.colorbar(last_im, ax=cax_axes, fraction=0.03, pad=0.02)
        cbar.set_label("abs % diff", fontsize=9)
        cbar.set_ticks(bounds)
        cbar.ax.tick_params(labelsize=8)

    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=160)
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--subject-dir", required=True, help="Subject directory (contains Analysis/ and Images/)")
    ap.add_argument("--ref-mat", required=True, help="Reference .mat path")
    ap.add_argument("--out-dir", default=None, help="Output directory (default: <subject>/Images/compare_reference)")
    ap.add_argument("--prefer", choices=["tikhonov", "patlak"], default="tikhonov")
    ap.add_argument(
        "--ref-transform",
        default="rot270",
        help="In-plane transform for reference (default: rot270). Supported: id|rot90|rot180|rot270|fliplr|flipud|transpose|...",
    )
    ap.add_argument("--max-slices", type=int, default=10, help="Max number of slices to show per plot")
    ap.add_argument(
        "--abs-pct-bins",
        default=",".join(str(x) for x in ABS_PCT_BINS_DEFAULT),
        help="Comma-separated bin edges for abs %% diff (default: 0,2,5,10,25,50,100)",
    )
    args = ap.parse_args()

    subject_dir = Path(args.subject_dir).expanduser().resolve()
    ref_mat = Path(args.ref_mat).expanduser().resolve()

    analysis_dir = subject_dir / "Analysis"
    images_dir = subject_dir / "Images"

    out_dir = Path(args.out_dir).expanduser().resolve() if args.out_dir else (images_dir / "compare_reference")
    out_dir.mkdir(parents=True, exist_ok=True)

    ref = _load_mat(ref_mat)

    # Map: reference key -> p-brain file
    prefer = str(args.prefer).strip().lower()
    suf = "_tikhonov" if prefer == "tikhonov" else "_patlak"

    pb_paths = {
        "CBF": analysis_dir / "CBF_per_voxel_tikhonov.nii.gz",
        "MTT": analysis_dir / "mtt_map.nii.gz",
        "CTH": analysis_dir / "cth_map.nii.gz",
        "CBKi": analysis_dir / f"Ki_per_voxel{suf}.nii.gz",
        "CBV_p": analysis_dir / f"vp_per_voxel{suf}.nii.gz",
        # Optional: CBV (total) is not always present in p-brain.
    }

    pb: Dict[str, np.ndarray] = {}
    missing_pb = []
    for k, p in pb_paths.items():
        if not p.exists():
            missing_pb.append(str(p))
            continue
        pb[k] = _load_nii(p)

    metrics: Dict[str, Any] = {
        "subject_dir": str(subject_dir),
        "ref_mat": str(ref_mat),
        "prefer": prefer,
        "missing_pb": missing_pb,
        "maps": {},
        "notes": [],
        "ref_transform": None,
    }

    ref_transform = str(args.ref_transform).strip().lower() or "rot270"
    metrics["ref_transform"] = {"name": ref_transform, "corr": None}

    try:
        abs_pct_bins = [float(x.strip()) for x in str(args.abs_pct_bins).split(",") if x.strip()]
        abs_pct_bins = sorted(set(abs_pct_bins))
        if len(abs_pct_bins) < 2:
            raise ValueError
    except Exception:
        abs_pct_bins = list(ABS_PCT_BINS_DEFAULT)
        metrics["notes"].append("Invalid --abs-pct-bins; using default 0,2,5,10,25,50,100")

    for key, pb_vol in pb.items():
        if key not in ref:
            if key == "CTH":
                metrics["notes"].append("Reference .mat does not contain a CTH volume key; CTH comparison skipped.")
            else:
                metrics["notes"].append(f"Reference .mat missing key {key!r}; skipped.")
            continue

        try:
            ref_vol = _apply_xy_transform(ref[key], ref_transform)
        except Exception as exc:
            metrics["notes"].append(f"Failed to apply ref transform '{ref_transform}' for {key}: {exc}")
            ref_vol = ref[key]
        if ref_vol.shape != pb_vol.shape:
            metrics["notes"].append(
                f"Shape mismatch for {key}: ref={tuple(ref_vol.shape)} pb={tuple(pb_vol.shape)}; skipped."
            )
            continue

        # Robust mask for fitting/metrics.
        m = _robust_mask(ref_vol, pb_vol)

        # Fit scaling (affine) from ref -> pb.
        x, y = _sample_xy(ref_vol, pb_vol, m)
        a, b = _fit_affine(x, y)
        c = _corr(x, y)

        entry: Dict[str, Any] = {
            "shape": list(ref_vol.shape),
            "corr": c,
            "fit": {"a": a, "b": b},
            "raw": _metrics(ref_vol, pb_vol, mask=m, a=1.0, b=0.0),
            "affine_scaled": _metrics(ref_vol, pb_vol, mask=m, a=a, b=b),
        }
        metrics["maps"][key] = entry

        # Create montage using scaled-diff (pb - (a*ref+b))
        if math.isfinite(a) and math.isfinite(b):
            pred = a * ref_vol + b
        else:
            pred = ref_vol
        diff = pb_vol - pred

        abs_pct = _abs_pct_diff(pb_vol, pred)
        entry["abs_pct_diff"] = {
            "bins": abs_pct_bins,
            "hist": _hist_counts(abs_pct[m], bins=abs_pct_bins),
        }

        # Triptych: matlab | p-brain | diff (rows=slices)
        out_png = out_dir / f"{key}_triptych.png"
        try:
            _save_triptych_slices(
                out_png=out_png,
                title=(
                    f"{key} | ref_transform={ref_transform} | corr={c:.3f} | fit: pb≈{a:.3g}*ref+{b:.3g}"
                ),
                ref=ref_vol,
                pb=pb_vol,
                diff=diff,
                mask=m,
                max_rows=int(args.max_slices),
            )
            entry["montage_png"] = str(out_png)
        except Exception as exc:
            metrics["notes"].append(f"Failed to write montage for {key}: {exc}")

        # Discrete abs-% diff triptych
        out_png2 = out_dir / f"{key}_triptych_abs_pct.png"
        try:
            _save_triptych_abs_pct_discrete(
                out_png=out_png2,
                title=(
                    f"{key} | ref_transform={ref_transform} | corr={c:.3f} | fit: pb≈{a:.3g}*ref+{b:.3g}"
                ),
                ref=ref_vol,
                pb=pb_vol,
                abs_pct=abs_pct,
                mask=m,
                max_rows=int(args.max_slices),
                bins=abs_pct_bins,
            )
            entry["abs_pct_triptych_png"] = str(out_png2)
        except Exception as exc:
            metrics["notes"].append(f"Failed to write abs%%diff triptych for {key}: {exc}")

    out_json = out_dir / "reference_compare_metrics.json"
    out_json.write_text(json.dumps(metrics, indent=2, sort_keys=True), encoding="utf-8")
    print(f"Wrote: {out_json}")
    print(f"Wrote montages under: {out_dir}")
    if metrics["missing_pb"]:
        print("Missing p-brain volumes:")
        for p in metrics["missing_pb"]:
            print(f"- {p}")
    if metrics["notes"]:
        print("Notes:")
        for n in metrics["notes"]:
            print(f"- {n}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
