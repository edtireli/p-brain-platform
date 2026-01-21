# Cohort analysis (project-wide)

This document describes how to do **cohort-level** (project-wide) analysis using outputs already produced by p-brain, and outlines a future in-app “Analysis” view.

## What “cohort analysis” means here

Cohort analysis uses subject-level outputs to answer questions like:
- Do two cohorts differ (e.g. control vs intervention) in **Ki**, **vp**, **MTT**, **CTH**, **FA**, etc.?
- Within a cohort, are two biomarkers correlated (e.g. **CTH vs FA**) and how strong is the association?
- Do effects differ by **tissue** (GM/WM), **segment/parcel**, or **whole brain**?

## Where the data comes from

p-brain writes quantitative outputs per subject under the subject’s `Analysis/` directory. The UI already reads many of these to populate subject-level tables.

Common examples:
- DCE model summaries (whole brain / tissue / parcels): `Analysis/AI_values_*.json`
- Atlas/parcel summaries: `Analysis/*_values_atlas_*.json`
- Diffusion summaries: typically under `Analysis/diffusion/` (e.g. FA summaries)

Exact filenames can vary slightly by pipeline version/model; prefer reading what the UI already uses for subject “Tables”.

## Recommended workflow today (no in-app cohort view)

### 1) Define cohorts and covariates

Create a cohort manifest (CSV/TSV/JSON) with at least:
- `subject_id` (must match the subject folder name or the platform subject id you are joining on)
- covariates like `age`, `sex`, and a grouping variable like `group` (e.g. `control` / `intervention`)

Example (TSV):

```text
subject_id	group	age	sex
20250408x3	control	61	F
20250217x4	intervention	58	M
```

### 2) Export subject-level metrics

Pick a “level”:
- **Whole-brain**: one value per subject per metric
- **Tissue**: one value per subject per tissue per metric
- **Parcel/segment**: one value per subject per parcel per metric

Then extract those values from each subject’s `Analysis/` JSON(s) into a single tidy table.

A tidy table is easiest to analyze:

```text
subject_id,group,metric,region,value
...,control,CTH,whole_brain,3.12
...,control,FA,whole_brain,0.41
...,control,CTH,Left-Hippocampus,3.44
```

### 3) Run the stats

Use standard statistical tooling (Python/R/JASP). At minimum:
- visualize distributions and outliers
- check normality/residuals (especially for parametric regression)
- use regression/ANCOVA when covariates matter

Practical defaults:
- **Two-group comparison** (continuous outcome): linear model `value ~ group + age + sex`
- **Within-cohort association**: `y ~ x + age + sex` (and consider robust regression if outliers)
- **Parcel-wise**: correct for multiple comparisons (e.g. FDR)

## Normality and alternative models

No single test is perfect; normality checks should be treated as guidance.

Common approach:
- Inspect histogram + Q–Q plot
- Fit a linear model; check residual Q–Q + residual-vs-fitted
- If assumptions are poor:
  - consider transforming the outcome (e.g. log) if domain-appropriate
  - consider robust regression
  - consider nonparametric alternatives for simple comparisons

## Future in-app “Analysis” view (design notes)

A project-wide Analysis view would ideally:
- compare two cohorts within a project (or across projects)
- support whole-brain, tissue, and parcel/segment resolution
- allow covariates (age/sex/group) and categorical stratification
- expose normality diagnostics and recommend/offer alternative models
- export results (tables + plots) for manuscripts

This is intentionally not implemented yet; this document exists so users can perform cohort analysis now using existing outputs.
