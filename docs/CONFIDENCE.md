# Reconstruction Confidence

This feature measures how strongly saved evidence supports a reconstruction. The number is an engineering heuristic, **not a probability, survey accuracy, or ground-truth error**. High scores can coexist with missing surfaces, distorted boundaries, systematic calibration errors or poor textures.

## Use in the MVP

1. Analyze the footage as usual. Dashboard shows **Capture Suitability** separately, using the existing accepted-frame quality score. Its tooltip shows accepted/sample counts; **View details** exposes individual measurements. Motion magnitude is available, but estimated overlap and a validated motion-diversity category are N/A. Suitability is not a calibrated success predictor.
2. Generate a reconstruction. After geometry completes, the backend measures evidence automatically and saves sidecars. If confidence processing fails, it logs N/A and retains completed reconstruction outputs.
3. Dashboard shows **Reconstruction Confidence**, registration counts, median reprojection error, normalized sparse support and sampled depth agreement. The card describes the selected model's reconstruction component; with no selection it uses the last visible model with a report. **View details** exposes raw evidence/configuration, and **Why this score?** gives a deterministic explanation. No LLM is used.
4. **Metric Status** is separate. GPS availability never changes the geometric score. Successful alignment is recognized only from the existing verified `georeference.json`, not from a settings checkbox. Residuals are agreement with telemetry, not independent survey validation.
5. For old projects click **Measure existing evidence**. This reads existing associated workspaces without rerunning COLMAP/OpenMVS. It does not assign the current video analysis to an old reconstruction. A missing historical frame snapshot means registration denominator and frame quality remain N/A. Imported meshes without a reconstruction workspace cannot be measured.
6. In Scene choose **Confidence**, **Provenance**, **Wireframe**, **Texture**, or the existing **Relative height**. Show Cameras displays centers and short viewing-direction markers. Show Camera Path joins filename-ordered centers (staged frame filenames follow video time). Clicking a surface in Confidence/Provenance mode shows approximate regional sparse support, when available.

The overlays do not change the original files, textures, transforms, or reconstruction score. LOD changes sample the same original evidence grid at the displayed mesh coordinates. Simplification never increases the score. Camera markers follow the object's transform; image-derived camera ordering in externally supplied workspaces may not be chronological.

## Formula and configuration

All defaults live in `ConfidenceConfig` in `src/drone3d_studio/reconstruction/confidence.py`. Every report includes its effective configuration.

| Component | Weight | Normalization |
|---|---:|---|
| Registration | 0.25 | registered images / accepted keyframes |
| Reprojection | 0.25 | clamp((4.0 − median stored error in px) / (4.0 − 0.5), 0, 1) |
| Multi-view support | 0.20 | mean(clamp((distinct track image count − 1) / (5 − 1), 0, 1)) |
| Depth consistency | 0.20 | agreeing evaluated sample/view pairs / all evaluated sample/view pairs |
| Frame quality | 0.10 | mean(existing accepted frame quality_score) / 100 |

For the available component set A:

```text
effective_weight_i = weight_i / sum(weight_j for j in A)
overall_score = 100 × sum(effective_weight_i × component_score_i for i in A)
```

Missing values are JSON `null`, displayed as **N/A**. They are excluded and weights are renormalized; they are never replaced with zero. A measured zero remains zero. If all components are absent, or only frame quality exists, reconstruction confidence is N/A. `available_weight` records how much of the original evidence mix was measured, so a score based on fewer signals is auditable.

Default labels are HIGH ≥85, GOOD ≥65, LIMITED ≥40, LOW <40. Reprojection good/weak ranges, support normalization, labels, warning thresholds, grid size and depth sample limits are configurable engineering choices. They are not universal pixel thresholds for accuracy.

To use configuration overrides without changing code:

```powershell
venv\Scripts\python scripts\measure_confidence.py output\MyProject --config confidence-settings.json
```

For example, `confidence-settings.json` may contain:

```json
{"reprojection_good_px": 0.75, "reprojection_weak_px": 5.0}
```

If changing weights, supply all five weights summing to one. Changing heuristic settings changes interpretation; retain the report's configuration when comparing runs. This command updates the project's report references and does not reconstruct geometry.

## Actual evidence and limitations

**COLMAP:** reads `images.bin` and `points3D.bin`, preserving non-contiguous identifiers. Camera centers come from stored poses. Mean/median/p95 refer to COLMAP's stored **per-point ERROR values**, not quantiles of every individual observation residual. These stored errors are updated by COLMAP after global bundle adjustment. Tracks supply distinct observation image counts. Triangulated-keypoint fraction measures how many registered-image keypoints reference a 3D point; it is not a whole-scene surface coverage fraction. See the [official COLMAP output format](https://colmap.github.io/format.html).

**Depth consistency:** reuses `ai_depth.consistent_support`, the existing 8% relative-depth comparison used by reconstruction. It evaluates at most 12 evenly distributed available depth views, with approximately 2,048 grid samples per view, against up to four nearest selected cameras. It uses valid positive projected depth pairs; missing pixels are not disagreements. It records numerator, denominator, valid samples, support histogram and sample survival. The 256 MB depth budget, unavailable maps, unsupported camera models or dimension mismatch result in N/A with a reason. Sampled values are not exhaustive coverage of the entire mesh. COLMAP geometric maps and saved calibrated Depth Anything maps are supported; native OpenMVS `.dmap` decoding is not implemented.

**AI:** calibration anchor counts, median relative inverse-depth errors and raw per-frame calibration reports are preserved when available. Newly generated fusion reports record actual pre-voxel/pre-triangulation sample survival. Old workspaces may only have the explicitly labelled bounded re-evaluation. Calibration anchors can reuse a point across frames; survival is not a unique surface-area fraction. These AI diagnostics are separate from sparse COLMAP evidence and are not treated as metric truth. Depth Anything and MiDaS retain `AI_ASSISTED` provenance regardless of score. MiDaS without multi-view measurements receives N/A geometric confidence. No new AI model is installed.

**OpenMVS:** available PLY header counts and native stage statuses are recorded as context. Completion is never added to the score. Native texture coverage, per-mesh visibility/support and surface-area provenance fractions remain N/A because they are not reliably measured by this adapter. Vertex counts are not coverage.

**Single pass:** checks use a bounded sample of first/last track-camera ray angles, adjacent filename-ordered camera rotation, camera-span/scene-distance ratio, stored blur/duplicate rejections and registration ratio. These checks generate warnings based on measured values. A single pass receives no penalty for being a single pass. Verified vertical-surface coverage and pre-reconstruction overlap are unavailable; the system never invents a vertical coverage warning.

## Local confidence and provenance

`local-confidence.npz` contains a bounded 32³-cell grid plus binary sparse-point IDs, positions, reprojection errors and actual observation counts. Raw point arrays scale with the sparse reconstruction, not millions of dense mesh vertices. The report JSON contains summaries and frame diagnostics, not vertex arrays.

Grid bounds use the 1st/99th coordinate percentiles to limit extreme sparse outliers. Each occupied cell holds the mean of `(normalized point reprojection + normalized point support) / 2`, and its mean actual sparse track support. Unoccupied cells and points outside the grid are N/A. This spatial approximation does **not** establish that nearby mesh vertices have those exact observation counts. Full sparse points remain in the sidecar even when outside the grid.

- Confidence: red → yellow → green for low → medium → high; gray where evidence is unavailable.
- Provenance: photogrammetric regions use those colours, AI-assisted models are purple independently of numerical confidence, and unavailable photogrammetric regions remain gray/insufficient evidence.
- Original photographic GLBs remain untouched. No confidence colours are baked into normal exports.
- The MVP does not combine AI and photogrammetry into one mixed mesh, so model provenance identifies its class. Surface-area percentages remain N/A instead of claiming invented percentages.

## Storage and manual verification

New runs save:

```text
reconstruction/<run>/metadata/
    frame-evidence.json
    sources.json
    confidence.json
    component-<index>-<provenance>/
        confidence.json
        local-confidence.npz
```

Each model stores only a portable `confidence_path`. Project import/export rebases it; project duplication copies sidecars. A report's component is a reconstructed coordinate system. Disconnected components or different provenance classes have separate reports; the run-level report is an index with overall N/A when multiple scopes exist. It does not mix coordinate systems or double-count registered images into a misleading single score.

To verify a score:

1. Open the selected model's `confidence.json` and read `sources` and `configuration`.
2. Check accepted frames in the run's immutable analysis snapshot and count registered COLMAP images. Old runs without that snapshot deliberately lack the denominator.
3. Inspect `point_errors` and `point_observations` in `local-confidence.npz` with NumPy (`allow_pickle=False`). Recompute the median error and mean normalized support using the formulas above.
4. Divide `agreeing_pairs` by `evaluable_pairs` for depth consistency; check the listed sampled views and its limitations.
5. Apply `overall.effective_weights` to the available component scores and multiply by 100. Do not insert zeros for null components.
6. Read `metric_status` separately. Confirm provenance and warnings before interpreting the result.

The existing monument project was measured without rerunning reconstruction: 65 cameras, 65,419 sparse points, median error 0.660736 px, mean track support 6.229 views, and 52,837 / 61,072 agreeing depth pairs. Registration denominator and frame quality are unavailable in this reused project. Effective weights are 0.25/0.65, 0.20/0.65 and 0.20/0.65 for reprojection/support/depth. The resulting heuristic is approximately **88.33 / HIGH**, based on only **65% of the default evidence weight**. This is not an accuracy result and does not remove the previously observed boundary distortion.

## Validation

Tests cover ratios, missing depth renormalization, no-evidence states, configurable ranges/labels, warnings, deterministic explanation, strict JSON serialization, GPS independence, AI provenance, binary parsing and truncation, cancellation, local unknown regions, measured depth agreement/disagreement, sidecar rebasing, shared LOD evidence and all new viewer modes without changing source mesh bytes. See `docs/VALIDATION.md` for the final suite result.
