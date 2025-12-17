# p-brain Local Studio - Product Requirements Document

A local-first neuroimaging analysis platform that replicates the behavioral UX of the p-brain pipeline. All data and computation remain on the user's machine with no cloud dependencies.

**Core Mission**: Enable neuroscientists to perform comprehensive DCE-MRI and structural analysis locally with an intuitive web-based interface backed by a local Python compute engine.

**Experience Qualities**:
1. **Scientific** - Provides rigorous computational methods with transparent parameters and reproducible outputs
2. **Self-contained** - Operates entirely offline with zero external dependencies once installed
3. **Responsive** - Real-time job monitoring with interactive visualization that feels immediate despite heavy computation

**Complexity Level**: Complex Application (advanced functionality, likely with multiple views)
This is a sophisticated neuroimaging analysis suite with multiple computational stages, real-time job orchestration, interactive 3D/4D volume rendering, physiological modeling, and comprehensive data visualization across projects and subjects.

## Essential Features

### 1. Project Management
- **Functionality**: Create, list, and manage analysis projects with local storage paths
- **Purpose**: Organize multi-subject neuroimaging studies with persistent configuration
- **Trigger**: User clicks "New Project" or selects existing from list
- **Progression**: Name entry → Storage folder selection → Copy-data toggle → Project created → Dashboard view
- **Success criteria**: Project created with valid storage structure; accessible from project list; persists across app restarts

### 2. Subject Import & Indexing
- **Functionality**: Discover and validate NIfTI files, DCE series, T1/IR sequences from local folders
- **Purpose**: Prepare subject data for pipeline execution with clear requirements validation
- **Trigger**: User clicks "Add Subjects" and selects folder(s)
- **Progression**: Folder selection → File discovery → Validation checks → Missing data warnings → Confirm import → Subject added to cohort
- **Success criteria**: Subject appears in dashboard grid; required/optional data clearly indicated; invalid subjects rejected with actionable feedback

### 3. Full Auto Pipeline Execution
- **Functionality**: Execute all 10 stages sequentially with dependency management
- **Purpose**: Automate complete analysis workflow from raw data to quantitative metrics
- **Trigger**: User clicks "Run Full Auto Pipeline" with subjects selected
- **Progression**: Config validation → Job queue creation → Stage execution (import → T1 fit → concentration → AIF/VIF → time shift → segmentation → tissue curves → modeling → diffusion → QC) → Real-time progress → Completion notification
- **Success criteria**: All stages complete successfully for valid data; artifacts written to project storage; stage grid shows green checkmarks; failures logged with clear error messages

### 4. Interactive Volume Viewer
- **Functionality**: Render NIfTI volumes with slice navigation, time-series playback, overlays, window/level, colormaps
- **Purpose**: Visualize structural and functional data with segmentation and ROI overlays
- **Trigger**: User navigates to Subject Detail → Viewer tab
- **Progression**: Volume loaded → Slice slider interaction → Time slider for 4D → Window/level adjustment → Colormap selection → Overlay toggle → Alpha blending control
- **Success criteria**: Smooth slice navigation; 4D time playback works; overlays render correctly; colormap changes apply instantly; no lag on interaction

### 5. Physiological Modeling & Metrics
- **Functionality**: Patlak, Extended Tofts, and Deconvolution analysis with Ki, vp, Ktrans, ve, CBF, MTT, CTH computation
- **Purpose**: Quantify tissue perfusion and permeability from DCE-MRI data
- **Trigger**: Stage 8 job execution after AIF/VIF extraction
- **Progression**: Concentration curves loaded → Model selection → Parameter bounds applied → Nonlinear fitting → Regularization (deconvolution) → Metrics computed → Maps/tables written → Interactive plots rendered
- **Success criteria**: Patlak plot shows x-y scatter with fit line; Tofts curves match measured data; Residue and h(t) curves display correctly; Metrics tables show GM/WM/parcel values; CTH computed accurately

### 6. Real-time Job Monitoring
- **Functionality**: Background job execution with progress tracking, log streaming, cancel/retry controls
- **Purpose**: Provide transparency into long-running computational tasks
- **Trigger**: Any pipeline stage initiated
- **Progression**: Job created → Queued → Running → Progress updates via WebSocket → Log stream display → Completion/failure → Artifact indexing
- **Success criteria**: Progress percentage updates in real-time; logs stream to UI; cancel immediately stops job; retry restarts from clean state; multiple jobs queue properly

### 7. Input Function Management (AIF/VIF)
- **Functionality**: Manual ROI drawing on concentration maps to extract arterial/venous curves
- **Purpose**: Define input functions required for all pharmacokinetic models
- **Trigger**: Stage 4 execution or manual ROI editor invocation
- **Progression**: Concentration map displayed → User draws ROI on vessel → Curve extracted → Time-shift analysis → Venous-to-arterial adjustment → Final AIF/VIF saved
- **Success criteria**: ROI drawing responsive; curves extract correctly; cross-correlation shift computed; adjusted curves stored; used in downstream modeling

### 8. Cohort Status Dashboard
- **Functionality**: Grid view with subjects as rows, stages as columns, color-coded status indicators
- **Purpose**: At-a-glance understanding of analysis progress across entire cohort
- **Trigger**: User opens project
- **Progression**: Project loaded → Subject list fetched → Stage status queried → Grid rendered → Status updates via WebSocket → Cell clicked → Stage detail modal
- **Success criteria**: Grid renders quickly for 50+ subjects; status colors accurate (grey/yellow/green/red); clicking cell shows logs and artifacts; updates in real-time during job execution

### 9. Segmentation & Tissue ROIs
- **Functionality**: FastSurfer integration with fallback to threshold-based masks; registration to DCE space
- **Purpose**: Define anatomical regions for parcel-level analysis
- **Trigger**: Stage 6 execution
- **Progression**: T1 map loaded → FastSurfer invoked (if available) → Segmentation produced → Registration to DCE space via flirt/affine → Masks extracted → Parcel stats initialized
- **Success criteria**: FastSurfer runs successfully when installed; fallback masks created when not; segmentation aligns with DCE volumes; GM/WM/parcels defined; registration quality acceptable

### 10. Interactive Curve & Plot Visualization
- **Functionality**: Plotly-based rendering of concentration curves, model fits, residue functions, transit-time distributions
- **Purpose**: Enable interactive exploration of physiological time-series and model quality
- **Trigger**: User navigates to Curves/Maps tabs in Subject Detail
- **Progression**: Artifact JSON loaded → Plotly data formatted → Interactive plot rendered → Hover tooltips → Zoom/pan → Legend toggle → Export option (local save)
- **Success criteria**: All curves render smoothly; zoom/pan responsive; tooltips show precise values; multiple curves distinguishable; plots cache for instant re-display

## Edge Case Handling

- **Missing dependencies** - Validate environment on startup; show clear warnings for missing external tools (FastSurfer, dcm2niix, flirt); pipeline runs with degraded functionality
- **Corrupted NIfTI files** - Catch parsing errors during import; mark subject as invalid; log specific file issues; allow manual override
- **Job crashes** - Capture stack traces; write to job log; mark as failed; enable retry with same/different parameters
- **Incomplete data** - Clearly indicate optional vs required sequences; allow partial pipeline runs; skip unavailable stages gracefully
- **Concurrent job conflicts** - Queue jobs properly; prevent simultaneous writes to same artifacts; lock project storage during writes
- **Large datasets** - Stream slices on-demand; cache rendered tiles; lazy-load curves; paginate tables; show progress for slow operations
- **Permission errors** - Detect read/write access issues; prompt for folder re-selection; provide actionable error messages
- **Browser compatibility** - Gracefully degrade File System Access API; fallback to folder path string input; warn about persistence limitations

## Design Direction

The design should evoke **precision scientific instrumentation** combined with **modern web application fluidity**. Think medical imaging workstation meets contemporary data analysis platform - authoritative, data-dense, yet approachable. The interface should feel like a professional research tool that respects the complexity of the domain while remaining visually clean and spatially organized.

## Color Selection

A scientific color scheme with clinical authority and visual clarity for complex data visualization.

- **Primary Color**: Deep medical blue `oklch(0.45 0.12 250)` - Conveys scientific authority and clinical trust
- **Secondary Colors**: 
  - Charcoal gray `oklch(0.35 0.01 260)` - Grounding neutral for panels and controls
  - Soft slate `oklch(0.65 0.02 250)` - Muted backgrounds for data tables
- **Accent Color**: Vibrant cyan `oklch(0.70 0.15 210)` - Attention for interactive elements, running jobs, active selections
- **Foreground/Background Pairings**:
  - Background (Off-white `oklch(0.97 0.005 250)`): Dark text `oklch(0.25 0.01 260)` - Ratio 12.1:1 ✓
  - Primary (Medical blue `oklch(0.45 0.12 250)`): White text `oklch(0.99 0 0)` - Ratio 8.2:1 ✓
  - Accent (Cyan `oklch(0.70 0.15 210)`): Dark text `oklch(0.25 0.01 260)` - Ratio 7.8:1 ✓
  - Card (Pure white `oklch(1 0 0)`): Dark text `oklch(0.25 0.01 260)` - Ratio 14.5:1 ✓
- **Status Colors**:
  - Success green `oklch(0.65 0.15 145)` with white text - Ratio 5.1:1 ✓
  - Warning yellow `oklch(0.75 0.14 85)` with dark text - Ratio 8.5:1 ✓
  - Error red `oklch(0.55 0.22 25)` with white text - Ratio 6.2:1 ✓
  - Muted grey `oklch(0.70 0.005 250)` with dark text - Ratio 6.8:1 ✓

## Font Selection

Typefaces should balance technical precision with contemporary readability, suitable for dense data displays and extended reading of scientific content.

- **Primary**: **IBM Plex Sans** - Technical clarity with warmth; excellent for UI labels, buttons, and data tables
- **Secondary**: **JetBrains Mono** - For numerical data, parameters, file paths, and log displays where monospace aids scanning

**Typographic Hierarchy**:
- H1 (Page Title): IBM Plex Sans SemiBold / 32px / tight (-0.02em) / line-height 1.2
- H2 (Section Header): IBM Plex Sans Medium / 24px / tight (-0.01em) / line-height 1.3
- H3 (Subsection): IBM Plex Sans Medium / 18px / normal / line-height 1.4
- Body (Default): IBM Plex Sans Regular / 15px / normal / line-height 1.6
- Caption (Metadata): IBM Plex Sans Regular / 13px / normal / line-height 1.5 / color muted
- Code/Data: JetBrains Mono Regular / 14px / normal / line-height 1.5
- Table Headers: IBM Plex Sans SemiBold / 14px / wide (0.02em) / uppercase

## Animations

Animations serve functional purposes - reinforcing spatial relationships during navigation, providing feedback for computational state changes, and guiding attention to job status updates. Avoid gratuitous motion. Use subtle spring physics for natural feel.

- **Job status transitions**: Smooth color fade (300ms ease-out) when stage moves from pending → running → complete
- **Modal/drawer entry**: Slide-in with slight fade (250ms ease-out) for context retention
- **Hover states**: Gentle elevation shadow on cards (150ms ease-out) to indicate interactivity
- **Volume slice scrubbing**: Instant response with 60fps target; no animation delay
- **Progress bars**: Smooth interpolation (200ms linear) between progress updates
- **Notifications**: Toast slide-up from bottom (300ms spring) with auto-dismiss fade (200ms)

## Component Selection

**Components**: 
- **Table** (shadcn) - Cohort dashboard grid with custom cell renderers for status badges
- **Card** (shadcn) - Project cards, subject overview panels, metric summary cards
- **Tabs** (shadcn) - Subject detail navigation (Overview/Viewer/Curves/Maps/Tables/Logs)
- **Dialog** (shadcn) - Project creation, configuration modals
- **Sheet** (shadcn) - Sliding panels for job queue, stage logs
- **Button** (shadcn) - All CTAs with variants (primary/secondary/destructive)
- **Input** (shadcn) - Text fields for project names, parameters
- **Select** (shadcn) - Dropdown for colormap selection, model choice
- **Progress** (shadcn) - Linear progress bars for job completion percentage
- **Badge** (shadcn) - Status indicators (Not Run/Running/Done/Failed)
- **Separator** (shadcn) - Visual dividers in dense layouts
- **Slider** (shadcn) - Volume slice navigation, time scrubbing, window/level, overlay alpha
- **Switch** (shadcn) - Feature toggles (overlay visibility, voxelwise output)
- **Accordion** (shadcn) - Collapsible sections for advanced parameters
- **ScrollArea** (shadcn) - Log streaming, long tables

**Customizations**:
- **VolumeViewer** (custom) - Canvas-based slice renderer with WebGL for performance; controls overlay with sliders and colormap selector
- **CurveChart** (Plotly wrapper) - Standardized Plotly configuration for concentration curves, model fits, residue functions
- **StatusGrid** (custom) - Specialized table with efficient rendering for large cohorts; WebSocket-driven updates
- **ROIDrawer** (custom) - Canvas-based freehand/circle ROI tool with live curve preview
- **JobMonitor** (custom) - Real-time job list with expandable log viewers

**States**:
- Buttons: Default (solid primary) / Hover (slight darken + shadow) / Active (inner shadow) / Disabled (muted + no pointer) / Loading (spinner + disabled)
- Inputs: Default (border-input) / Focus (border-accent + ring) / Error (border-destructive) / Disabled (bg-muted)
- Status badges: Not Run (grey fill) / Running (yellow + pulse) / Done (green + checkmark) / Failed (red + X icon)

**Icon Selection** (Phosphor):
- Projects: Folders, FolderPlus
- Pipeline: Play, Pause, ArrowsClockwise (retry)
- Jobs: ListChecks, Spinner, CheckCircle, XCircle
- Viewer: Eye, Stack, Palette, SliderHorizontal
- Data: ChartLine, Table, FileText
- Navigation: CaretLeft, CaretRight, CaretDown, House
- Actions: Plus, Trash, Download, Upload (despite no actual upload)
- Status: Check, X, Warning, Info

**Spacing**:
- Container padding: 6 (24px) on desktop, 4 (16px) on mobile
- Card internal padding: 6 (24px)
- Section gaps: 8 (32px) between major sections, 4 (16px) within sections
- Element gaps: 3 (12px) for related form fields, 2 (8px) for tight groupings
- Grid gaps: 4 (16px) for cards, 2 (8px) for status grid cells
- Button padding: px-4 py-2 (standard), px-6 py-3 (large CTA)

**Mobile**:
- Cohort grid becomes scrollable table; subjects stack vertically; stage columns horizontal scroll
- Subject detail tabs convert to full-width stacked sections with sticky tab bar
- Volume viewer controls move to bottom sheet drawer instead of sidebar
- Job monitor becomes full-screen modal instead of side sheet
- Two-column layouts collapse to single column with logical reading order
- Increase touch targets to minimum 44px for sliders and buttons
