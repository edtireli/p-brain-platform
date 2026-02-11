import argparse
import os
from pathlib import Path
from typing import Optional

import numpy as np
import nibabel as nib
from PIL import Image


DEFAULT_NIFTI_CANDIDATES = (
    "WIPhperf120long.nii",
    "WIPDelRec-hperf120long.nii",
)

def _pick_nifti_path(logfolder_path: Path, nifti_path: Optional[Path]) -> Path:
    if nifti_path is not None:
        if not nifti_path.exists():
            raise FileNotFoundError(f"NIfTI file not found: {nifti_path}")
        return nifti_path

    nifti_dir = logfolder_path / "NIfTI"
    for name in DEFAULT_NIFTI_CANDIDATES:
        candidate = nifti_dir / name
        if candidate.exists():
            return candidate

    # Fallback: pick the first 4D NIfTI under NIfTI/
    for candidate in sorted(nifti_dir.glob("*.nii")):
        try:
            img = nib.load(str(candidate))
        except Exception:
            continue
        shape = tuple(img.shape)
        if len(shape) == 4 and shape[-1] > 1:
            return candidate

    raise FileNotFoundError(
        f"No suitable 4D NIfTI found under {nifti_dir}. "
        f"Tried {', '.join(DEFAULT_NIFTI_CANDIDATES)} and then any 4D *.nii."
    )


def load_data(logfolder_path: Path, roi_relpath: str, nifti_path: Optional[Path]):
    logfolder_path = Path(logfolder_path)
    data_path = _pick_nifti_path(logfolder_path, nifti_path)

    roi_path = logfolder_path / roi_relpath
    if not roi_path.exists():
        raise FileNotFoundError(f"ROI folder not found: {roi_path}")

    mri_data = np.rot90(nib.load(str(data_path)).get_fdata(), k=-1, axes=(0, 1))

    roi_files = [f for f in os.listdir(roi_path) if f.startswith("ROI_voxels_slice")]
    roi_data = {}
    for roi_file in roi_files:
        slice_index = int(roi_file.split("_")[-1].split(".")[0]) - 1  # 0-based indexing
        roi_voxels = np.load(os.path.join(roi_path, roi_file))
        roi_data[slice_index] = roi_voxels
    
    return mri_data, roi_data

def _iter_logfolders(input_path: Path) -> list[Path]:
    input_path = Path(input_path)
    if (input_path / "NIfTI").exists():
        return [input_path]

    logfolders: list[Path] = []
    for child in sorted(input_path.iterdir()):
        if not child.is_dir():
            continue
        if (child / "NIfTI").exists():
            logfolders.append(child)
    return logfolders


def save_slices_as_png(input_path: Path, output_dir: Path, roi_relpath: str, nifti_path: Optional[Path] = None):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logfolders = _iter_logfolders(Path(input_path))
    if not logfolders:
        raise FileNotFoundError(
            f"No logfolders found under {input_path}. "
            f"Provide a dataset folder containing NIfTI/ or a parent folder containing multiple such datasets."
        )

    for logfolder_path in logfolders:
        logfolder = logfolder_path.name
        mri_data, roi_data = load_data(logfolder_path, roi_relpath=roi_relpath, nifti_path=nifti_path)
        
        # Compute time-averaged intensity across all time frames
        time_averaged_data = np.mean(mri_data, axis=-1)
        
        num_slices = time_averaged_data.shape[2]
        out_subdir = output_dir / logfolder
        out_subdir.mkdir(parents=True, exist_ok=True)
        for slice_index in range(num_slices):
            # Extract the specific slice
            slice_data = time_averaged_data[:, :, slice_index]

            # Normalize the slice data
            denom = float(slice_data.max() - slice_data.min())
            if denom <= 0:
                slice_data_normalized = np.zeros(slice_data.shape, dtype=np.uint8)
            else:
                slice_data_normalized = ((slice_data - slice_data.min()) / denom * 255).astype(np.uint8)

            # Save the slice as a PNG image
            slice_image = Image.fromarray(slice_data_normalized)
            slice_filename = out_subdir / f"slice_{slice_index + 1:03d}.png"
            slice_image.save(str(slice_filename), format="PNG")

            # Create and save the ROI mask as a PNG image
            mask = np.zeros(slice_data.shape, dtype=np.uint8)
            if slice_index in roi_data:
                for roi in roi_data[slice_index]:
                    x, y = int(roi[0]), int(roi[1])
                    if 0 <= x < mask.shape[0] and 0 <= y < mask.shape[1]:
                        mask[x, y] = 255

            mask_image = Image.fromarray(mask)
            mask_filename = out_subdir / f"mask_{slice_index + 1:03d}.png"
            mask_image.save(str(mask_filename), format="PNG")

        print(f"Wrote {num_slices} slice PNGs + masks for {logfolder} -> {out_subdir}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Export time-averaged DCE slices and ROI masks to PNG.")
    parser.add_argument(
        "--input",
        required=True,
        help="Dataset folder containing NIfTI/ (single case) OR parent folder containing multiple datasets.",
    )
    parser.add_argument(
        "--output-dir",
        default="/Users/edt/Desktop/p-brain/output_pngs_rica",
        help="Output directory (subfolder per dataset).",
    )
    parser.add_argument(
        "--roi-relpath",
        default=os.path.join("Analysis", "ROI Data", "Artery", "Right Interior Carotid"),
        help="ROI folder path relative to each dataset folder.",
    )
    parser.add_argument(
        "--nifti",
        default=None,
        help="Optional explicit 4D NIfTI path; overrides auto-detection.",
    )
    args = parser.parse_args()

    nifti_path = Path(args.nifti) if args.nifti else None
    save_slices_as_png(Path(args.input), Path(args.output_dir), roi_relpath=args.roi_relpath, nifti_path=nifti_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
