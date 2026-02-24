# Changelog

## v1.2.5 (2026-02-24)

- Make Folder Structure Configuration dialog scrollable on all tabs.
- Remove the “Delete Analysis + Images” action from the configuration dialog.
- Improve Preview to show per-subject matched filenames (T1/DCE/Diffusion/T2/FLAIR).

## v1.2.0 (2026-02-23)

- Make modelling metric rendering null-safe (prevents UI crashes on empty ROIs).
- Add project controls for T1/M0 refit behavior and headless/stable p-brain execution defaults.
- Add a one-click action to delete derived outputs (Analysis/Images) per project.

## v1.0.0 (2026-01-09)

- Desktop app (Tauri) bundles the web UI + local FastAPI bridge.
- QC/review UI for DCE-MRI and diffusion outputs (including tractography).
- Integration pointers to the core `p-brain` pipeline and CNN bundle on Zenodo.

See: `docs/release/v1.0.0.md`
