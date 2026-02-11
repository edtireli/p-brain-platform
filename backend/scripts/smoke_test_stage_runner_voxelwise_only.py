#!/usr/bin/env python3
"""Smoke test: stage runner supports voxelwise-only (no T1/T2/FLAIR).

This is a lightweight regression check that does not import the backend app
(or any of its runtime dependencies). It simply inspects the stage runner
template embedded in `backend/app.py`.

Run:
  python3 backend/scripts/smoke_test_stage_runner_voxelwise_only.py
"""

from __future__ import annotations

import sys
from pathlib import Path


def main() -> int:
    app_py = Path(__file__).resolve().parents[1] / "app.py"
    text = app_py.read_text(encoding="utf-8", errors="replace")

    must_contain = [
        # Guard against None filenames.
        "t1_path = os.path.join(nifti_directory, t1_3D_filename) if t1_3D_filename else None",
        "t2_path = os.path.join(nifti_directory, axial_t2_2D_filename) if axial_t2_2D_filename else None",
        # Segmentation skip when no structural.
        "[segmentation] Missing T1 structural",
        # Tissue curves skip when segmentation missing.
        "[tissue_ctc] Missing segmentation output",
        # Modelling fallback.
        "voxelwise-only modelling",
        "voxelwise_only=True",
    ]

    missing = [s for s in must_contain if s not in text]
    if missing:
        sys.stderr.write("Missing expected stage-runner template markers:\n")
        for s in missing:
            sys.stderr.write(f"- {s!r}\n")
        return 1

    print("OK: stage runner template includes voxelwise-only safeguards")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
