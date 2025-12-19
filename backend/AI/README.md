# AI models (local)

This folder is used by the local runner/worker to provide the ROI extraction models used by `p-brain` (`modules/AI_input_functions.py`).

The **model binaries are intentionally not committed** (they are very large).

## What to copy

From your `p-brain/AI` folder, copy these into `p-brain-web/backend/AI`:

- `slice_classifier_model_rica.keras`
- `rica_roi_model.keras`
- `ss_slice_classifier.keras`
- `ss_roi_model.keras`

Optionally copy any additional model folders/files you use.

## How the worker finds them

The Supabase worker sets p-brain’s env overrides:

- `SLICE_CLASSIFIER_RICA_MODEL`
- `RICA_ROI_MODEL`
- `SLICE_CLASSIFIER_SS_MODEL`
- `SS_ROI_MODEL`

You can override the folder by setting `PBRAIN_AI_DIR=/path/to/AI`.
