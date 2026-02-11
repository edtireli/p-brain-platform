#!/usr/bin/env python3
"""End-to-end hemiSure comparison runner.

Creates, for a given subject directory:

1) Side-by-side PK map comparisons (MATLAB ref vs p-Brain vs diff) by invoking
   `compare_reference_pk_maps.py`.
2) TSCC max curve comparison: p-Brain "max TSCC" (slice inferred from the
   MATLAB shifted concentration file) vs MATLAB shifted `c_input`.
3) Intensity curves (ITC) for the selected arterial+venous slices.
4) AIF + VIF concentration curves before/after shifting+rescaling
   (dashed grey vs red, alpha=0.5).

All outputs are written under: <subject>/Images/compare_reference/

This script is intentionally non-interactive and uses only existing saved
pipeline artifacts.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True)
class CurvePair:
    before: np.ndarray
    after: np.ndarray


def _load_mat_time_and_curve(mat_path: Path) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    from scipy.io import loadmat

    md = loadmat(str(mat_path))
    meta: dict[str, Any] = {}

    def req(key: str) -> np.ndarray:
        if key not in md:
            raise KeyError(f"{mat_path.name} missing key {key!r}")
        return np.squeeze(np.asarray(md[key], dtype=float))

    time_s = req("time")
    curve = req("c_input")

    # Optional slice metadata.
    if "slice_c_input" in md:
        try:
            meta["slice_c_input"] = int(np.squeeze(np.asarray(md["slice_c_input"])) )
        except Exception:
            pass

    return time_s, curve, meta


def _safe_load_npy(path: Path) -> np.ndarray:
    arr = np.load(path)
    return np.asarray(arr, dtype=float)


def _align_len(time_s: np.ndarray, *curves: np.ndarray) -> tuple[np.ndarray, list[np.ndarray]]:
    n = int(min(time_s.size, *(c.size for c in curves)))
    t = np.asarray(time_s[:n], dtype=float)
    out = [np.asarray(c[:n], dtype=float).reshape(-1) for c in curves]
    return t, out


def _pick_max_tscc_series(subject_dir: Path, *, slice_no: int) -> Path:
    """Pick the p-Brain TSCC series for a slice that truly has the highest max.

    Preference order:
    - Analysis/TSCC Data/Max/TSCC_slice_<slice>_*.npy (if present)
    - Else: any Analysis/TSCC Data/**/TSCC_slice_<slice>_*.npy

    Ignores AppleDouble files (._*).
    """

    slice_no = int(slice_no)
    tscc_root = subject_dir / "Analysis" / "TSCC Data"
    if not tscc_root.exists():
        raise FileNotFoundError(f"Missing TSCC folder: {tscc_root}")

    candidates: list[Path] = []
    max_dir = tscc_root / "Max"
    if max_dir.exists():
        candidates.extend(sorted(max_dir.glob(f"TSCC_slice_{slice_no}_*.npy")))

    if not candidates:
        candidates.extend(sorted(tscc_root.rglob(f"TSCC_slice_{slice_no}_*.npy")))

    candidates = [p for p in candidates if p.is_file() and not p.name.startswith("._")]
    if not candidates:
        raise FileNotFoundError(f"No TSCC series found for slice {slice_no} under {tscc_root}")

    best_p = candidates[0]
    best_max = -np.inf
    for p in candidates:
        try:
            c = _safe_load_npy(p).reshape(-1)
            m = float(np.nanmax(c))
        except Exception:
            continue
        if np.isfinite(m) and m > best_max:
            best_max = m
            best_p = p

    return best_p


def _pick_vein_name(subject_dir: Path) -> str:
    vein_root = subject_dir / "Analysis" / "CTC Data" / "Vein"
    if not vein_root.exists():
        raise FileNotFoundError(f"Missing: {vein_root}")
    names = [p.name for p in vein_root.iterdir() if p.is_dir() and not p.name.startswith(".")]
    if not names:
        raise FileNotFoundError(f"No vein subfolders under: {vein_root}")
    if "Sinus Sagittalis" in names:
        return "Sinus Sagittalis"
    return sorted(names)[0]


def _load_ctc_pair(subject_dir: Path, *, kind: str, name: str, slice_no: int) -> CurvePair:
    ctc_root = subject_dir / "Analysis" / "CTC Data" / kind / name
    before_p = ctc_root / f"CTC_slice_{int(slice_no)}.npy"
    after_p = ctc_root / f"CTC_shifted_slice_{int(slice_no)}.npy"
    if not before_p.exists():
        raise FileNotFoundError(f"Missing CTC before: {before_p}")
    if not after_p.exists():
        raise FileNotFoundError(f"Missing CTC after: {after_p}")
    return CurvePair(before=_safe_load_npy(before_p).reshape(-1), after=_safe_load_npy(after_p).reshape(-1))


def _load_itc(subject_dir: Path, *, kind: str, name: str, slice_no: int) -> np.ndarray:
    itc_root = subject_dir / "Analysis" / "ITC Data" / kind / name
    p = itc_root / f"ITC_slice_{int(slice_no)}.npy"
    if not p.exists():
        raise FileNotFoundError(f"Missing ITC: {p}")
    return _safe_load_npy(p).reshape(-1)


def _plot_tscc_max_vs_matlab(
    *,
    out_png: Path,
    time_s: np.ndarray,
    matlab_shifted: np.ndarray,
    pbrain_tscc: np.ndarray,
    title: str,
) -> dict[str, float]:
    os.environ.setdefault("MPLBACKEND", "Agg")
    import matplotlib.pyplot as plt

    t, (m, pb) = _align_len(time_s, matlab_shifted, pbrain_tscc)
    diff = pb - m

    m_max = float(np.nanmax(m))
    pb_max = float(np.nanmax(pb))
    delta = pb_max - m_max
    ratio = pb_max / m_max if m_max else float("nan")

    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True, constrained_layout=True)
    axes[0].plot(t, m, label=f"MATLAB shifted c_input (max={m_max:.4g})", lw=2)
    axes[0].plot(t, pb, label=f"p-Brain TSCC max (max={pb_max:.4g})", lw=2)
    axes[0].set_ylabel("Concentration (mM)")
    axes[0].set_title(f"{title} | Δmax={delta:.4g} | ratio={ratio:.4g}")
    axes[0].grid(True, alpha=0.25)
    axes[0].legend(frameon=False)

    axes[1].plot(t, diff, color="black", lw=1.5)
    axes[1].axhline(0, color="gray", lw=1)
    axes[1].set_xlabel("Time (s)")
    axes[1].set_ylabel("p-Brain − MATLAB (mM)")
    axes[1].grid(True, alpha=0.25)

    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=180)
    plt.close(fig)

    return {
        "matlab_max": m_max,
        "pbrain_max": pb_max,
        "delta_max": float(delta),
        "ratio": float(ratio) if np.isfinite(ratio) else float("nan"),
    }


def _plot_itc_curves(
    *,
    out_png: Path,
    time_s: np.ndarray,
    artery_itc: np.ndarray,
    vein_itc: np.ndarray,
    artery_label: str,
    vein_label: str,
) -> None:
    os.environ.setdefault("MPLBACKEND", "Agg")
    import matplotlib.pyplot as plt

    t, (a, v) = _align_len(time_s, artery_itc, vein_itc)
    fig, ax = plt.subplots(1, 1, figsize=(10, 3.8), constrained_layout=True)
    ax.plot(t, a, lw=2, label=artery_label)
    ax.plot(t, v, lw=2, label=vein_label)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Signal intensity")
    ax.set_title("Intensity curves (ITC)")
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False)

    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=180)
    plt.close(fig)


def _plot_aif_vif_shift_effect(
    *,
    out_png: Path,
    time_s: np.ndarray,
    aif: CurvePair,
    vif: CurvePair,
    aif_label: str,
    vif_label: str,
) -> None:
    os.environ.setdefault("MPLBACKEND", "Agg")
    import matplotlib.pyplot as plt

    t, (a0, a1, v0, v1) = _align_len(time_s, aif.before, aif.after, vif.before, vif.after)

    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True, constrained_layout=True)

    # Before: dashed grey; After: solid red; both alpha=0.5.
    axes[0].plot(t, a0, "--", color="gray", alpha=0.5, lw=2, label="before")
    axes[0].plot(t, a1, "-", color="red", alpha=0.5, lw=2, label="after")
    axes[0].set_title(f"AIF shift/rescale effect: {aif_label}")
    axes[0].set_ylabel("Concentration (mM)")
    axes[0].grid(True, alpha=0.25)
    axes[0].legend(frameon=False)

    axes[1].plot(t, v0, "--", color="gray", alpha=0.5, lw=2, label="before")
    axes[1].plot(t, v1, "-", color="red", alpha=0.5, lw=2, label="after")
    axes[1].set_title(f"VIF shift/rescale effect: {vif_label}")
    axes[1].set_xlabel("Time (s)")
    axes[1].set_ylabel("Concentration (mM)")
    axes[1].grid(True, alpha=0.25)

    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=180)
    plt.close(fig)


def _run_map_comparisons(*, subject_dir: Path, ref_mat: Path, prefer: str, out_dir: Path) -> None:
    script = Path(__file__).parent / "compare_reference_pk_maps.py"
    cmd = [
        "python3",
        str(script),
        "--subject-dir",
        str(subject_dir),
        "--ref-mat",
        str(ref_mat),
        "--prefer",
        str(prefer),
        "--out-dir",
        str(out_dir),
    ]
    subprocess.run(cmd, check=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--subject-dir", required=True)
    ap.add_argument(
        "--ref-mat",
        default=None,
        help="MATLAB PK map reference .mat (default: first *CBF_maps_method4_tik*offset1*.mat in subject dir)",
    )
    ap.add_argument(
        "--matlab-shifted-conc-mat",
        default=None,
        help="MATLAB shifted conc .mat (default: conc_methodT1_map_input_MRsignal_M_fix_slice4_slice4_shifted.mat in subject dir)",
    )
    ap.add_argument("--prefer", choices=["tikhonov", "patlak"], default="tikhonov")
    ap.add_argument("--out-dir", default=None)
    args = ap.parse_args()

    subject_dir = Path(args.subject_dir).expanduser().resolve()
    if not subject_dir.exists():
        raise FileNotFoundError(subject_dir)

    images_dir = subject_dir / "Images"
    out_dir = Path(args.out_dir).expanduser().resolve() if args.out_dir else (images_dir / "compare_reference")
    out_dir.mkdir(parents=True, exist_ok=True)

    # Defaults for the two MATLAB inputs.
    if args.ref_mat:
        ref_mat = Path(args.ref_mat).expanduser().resolve()
    else:
        cands = [p for p in sorted(subject_dir.glob("*CBF_maps_method4_tik*offset1*.mat")) if not p.name.startswith("._")]
        if not cands:
            raise FileNotFoundError("No default ref-mat found; pass --ref-mat")
        ref_mat = cands[0]

    if args.matlab_shifted_conc_mat:
        shifted_conc_mat = Path(args.matlab_shifted_conc_mat).expanduser().resolve()
    else:
        shifted_conc_mat = subject_dir / "conc_methodT1_map_input_MRsignal_M_fix_slice4_slice4_shifted.mat"
        if not shifted_conc_mat.exists():
            raise FileNotFoundError("No default shifted conc .mat found; pass --matlab-shifted-conc-mat")

    # Load selection metadata (arterial/venous slice numbers).
    selection_path = subject_dir / "Analysis" / "tscc_selection.json"
    selection = json.loads(selection_path.read_text(encoding="utf-8")) if selection_path.exists() else {}
    artery_name = str(selection.get("artery_subtype", "Right Interior Carotid"))
    arterial_slice = int(selection.get("arterial_slice", 2))
    venous_slice = int(selection.get("venous_slice", 4))

    # MATLAB shifted conc provides the time axis and (optionally) the slice number.
    time_s, matlab_shifted_c, meta = _load_mat_time_and_curve(shifted_conc_mat)
    matlab_slice = int(meta.get("slice_c_input", 4))

    # 1) Map comparisons (writes triptychs + discrete abs% diff maps).
    _run_map_comparisons(subject_dir=subject_dir, ref_mat=ref_mat, prefer=args.prefer, out_dir=out_dir)

    # 2) TSCC max vs MATLAB shifted.
    tscc_path = _pick_max_tscc_series(subject_dir, slice_no=matlab_slice)
    pbrain_tscc = _safe_load_npy(tscc_path).reshape(-1)
    tscc_plot = out_dir / "TSCC_max_vs_matlab_shifted.png"
    tscc_stats = _plot_tscc_max_vs_matlab(
        out_png=tscc_plot,
        time_s=time_s,
        matlab_shifted=matlab_shifted_c,
        pbrain_tscc=pbrain_tscc,
        title=f"Slice {matlab_slice} TSCC max vs MATLAB shifted",
    )

    # 3) Intensity curves (arterial + venous slices).
    vein_name = _pick_vein_name(subject_dir)
    artery_itc = _load_itc(subject_dir, kind="Artery", name=artery_name, slice_no=arterial_slice)
    vein_itc = _load_itc(subject_dir, kind="Vein", name=vein_name, slice_no=venous_slice)
    itc_plot = out_dir / "ITC_artery_vein.png"
    _plot_itc_curves(
        out_png=itc_plot,
        time_s=time_s,
        artery_itc=artery_itc,
        vein_itc=vein_itc,
        artery_label=f"Artery ITC: {artery_name} slice {arterial_slice}",
        vein_label=f"Vein ITC: {vein_name} slice {venous_slice}",
    )

    # 4) AIF + VIF before/after shifting+rescaling.
    aif = _load_ctc_pair(subject_dir, kind="Artery", name=artery_name, slice_no=arterial_slice)
    vif = _load_ctc_pair(subject_dir, kind="Vein", name=vein_name, slice_no=venous_slice)
    aifvif_plot = out_dir / "AIF_VIF_shift_rescale_effect.png"
    _plot_aif_vif_shift_effect(
        out_png=aifvif_plot,
        time_s=time_s,
        aif=aif,
        vif=vif,
        aif_label=f"{artery_name} slice {arterial_slice}",
        vif_label=f"{vein_name} slice {venous_slice}",
    )

    summary = {
        "subject_dir": str(subject_dir),
        "out_dir": str(out_dir),
        "ref_mat": str(ref_mat),
        "matlab_shifted_conc_mat": str(shifted_conc_mat),
        "matlab_shifted_slice": matlab_slice,
        "pbrain_tscc_path": str(tscc_path),
        "tscc_max_stats": tscc_stats,
        "selection": {
            "artery_name": artery_name,
            "arterial_slice": arterial_slice,
            "vein_name": vein_name,
            "venous_slice": venous_slice,
            "raw": selection,
        },
        "plots": {
            "tscc_max": str(tscc_plot),
            "itc": str(itc_plot),
            "aif_vif_shift": str(aifvif_plot),
        },
    }
    (out_dir / "comparison_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(f"Wrote outputs under: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
