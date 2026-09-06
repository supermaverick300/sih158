# Seven-stage reconstruction setup

The app implements the seven stages in the requested diagram. GPS stages are optional because they require actual flight telemetry. AI depth is calibrated against COLMAP geometry and can reject incompatible footage instead of producing a misleading cloud.

## Downloads and installation

On the current development laptop, these are already installed:

- COLMAP 4.2.0 CUDA: `D:\DOWNLOADS\colmap-x64-windows-cuda\bin\colmap.exe`. Keep the complete extracted folder where it is; do not move only the executable.
- Main Python environment: `D:\PROJECTS\sih158\venv`.
- AI dependencies: PyTorch 2.8.0 CPU, Transformers 4.57.6, Pillow 11.3.0 and safetensors 0.7.0.
- Official Depth Anything V2 Small weights: `D:\PROJECTS\sih158\models\depth-anything-v2-small`. The downloaded revision and SHA-256 are recorded in `download-manifest.json`.

**No more model downloads are needed on this laptop.** Open3D is optional for inspecting PLY files outside the app. FFmpeg, Blender, a Depth Anything API key and a Hugging Face account are not required by this pipeline.

For another computer, first follow the main README's Python/app setup. Download and extract [COLMAP's Windows release](https://github.com/colmap/colmap/releases), then run from the repository directory:

```cmd
scripts\setup_ai_windows.cmd
```

This explicitly installs CPU PyTorch and downloads the approximately 100 MB [Depth Anything V2 Small model](https://huggingface.co/depth-anything/Depth-Anything-V2-Small-hf). Allow roughly 2 GB of free space for the additional libraries, weights and installer cache, plus space for project outputs. The initial PyTorch wheel on the test machine was about 619 MB. Subsequent reconstruction reads local weights and does not download them automatically. Small is Apache-2.0 licensed; the official model card describes it as relative-depth estimation, not guaranteed metric depth.

CPU inference is tested. A CUDA PyTorch build is optional and separate from COLMAP's CUDA support. Use the matching installation command from the [official PyTorch instructions](https://pytorch.org/get-started/previous-versions/) if you later want GPU AI inference. The installed CPU build works even though the laptop has an NVIDIA GPU. Select CPU or Auto in the app; explicitly selecting CUDA with a CPU build produces an error. Installing the CUDA Toolkit is not needed for the tested CPU AI path.

## Open and configure the app

In PowerShell:

```powershell
Set-Location D:\PROJECTS\sih158
.\venv\Scripts\python.exe main.py
```

1. In **Projects**, create a project or open its `project.drone3d.json`.
2. In **Video**, import the survey video. Replacing a video clears its frame records and telemetry association; source files remain on disk.
3. In **Settings**, choose the COLMAP executable and click **Check COLMAP installation**.
4. Scroll to **1–3 · Adaptive sampling, quality and GPS keyframes**. Choose **Adaptive**, minimum features **40**, maximum clipped fraction **0.35**. On Video, start with base interval **0.5 s**, sharpness **35**, duplicate threshold **2.5**, maximum samples **150**. These are starting settings, not guaranteed capture-independent thresholds.
5. Without telemetry, use GPS spacing **0** and leave alignment unchecked. With matching telemetry, follow the next section before analyzing.
6. Under **4–7 · Reconstruction pipeline**, choose **Depth Anything V2**, the local weights folder above, and **CPU**. Click **Check AI installation**. Start with pixel stride **4** and voxel size **0.05**; voxel size is in arbitrary SfM units without GPS, or meters after successful alignment. Increase it if fusion reaches the 500,000-voxel memory cap.
7. Click **Save settings**. Depth method and fusion settings belong to the current project. General settings and extraction settings also become defaults for new projects.
8. On **Video**, click **Analyze**. Hover a frame for quality score, feature count, clipped pixels, motion and GPS. Review rejected-frame reasons. The sample cap limits the number of candidate frames, and may stop analysis before the end of a long video.
9. Click **Reconstruct footage** on Models, or **Generate scene** on Dashboard. The app runs SfM, optional GPS alignment, undistortion, AI inference, calibration, consistency filtering and colored point-cloud fusion.
10. Open **Scene** and use **Frame all**, orbit and zoom. AI output is a point cloud, not a watertight or textured mesh. Choose **COLMAP stereo** and **Dense mesh (CUDA)** instead when you want the existing stereo/Poisson surface workflow.

## Telemetry needed for stages 3 and 6

You must supply positions recorded with the same flight/video. Downloading a generic GPS file cannot georeference the scene. The current stock village/highway clip has no matching telemetry supplied, so geographic alignment is unavailable for it.

The current importer supports CSV with this header:

```csv
time_s,latitude,longitude,altitude_m
```

Each row must contain:

- `time_s`: increasing elapsed seconds in the telemetry recording; no duplicates. With zero offset, zero is the video start. The app samples telemetry at `video time + offset`.
- `latitude`, `longitude`: WGS84 decimal degrees.
- `altitude_m`: camera altitude in meters above the WGS84 ellipsoid. Convert orthometric/sea-level heights using the relevant geoid model before importing. Height above takeoff alone is insufficient for correct absolute geographic height.

Use the drone's original flight-log export. DJI SRT and manufacturer-specific CSV formats must be converted to these four columns first; direct SRT parsing is not included. Verify the altitude datum and timing in the source export rather than guessing them.

1. Import the video first, then **Settings → Import telemetry CSV**. The CSV is copied into the project and validated. Importing telemetry also tags existing frame timestamps; re-analyze to apply GPS spacing.
2. Set the time offset if recording start times differ. Default maximum interpolation gap is **2 seconds**; the app does not extrapolate beyond the supplied log.
3. Set GPS spacing to a modest distance appropriate to altitude and overlap, such as **1–2 m** for an initial low-altitude test. Zero disables the spacing filter but still allows GPS tagging. Excessive spacing can remove necessary overlap.
4. Check **Align SfM to GPS before depth fusion**, set the residual threshold appropriate to the telemetry, save settings, analyze and reconstruct.

At least three registered, GPS-tagged views with a non-collinear trajectory are required; more are strongly preferable. Alignment rejects inadequate spread and verifies camera residuals. At least three and at least half the registered reference positions must pass. All components share one local East/North/Up origin. `georeference.json` records that WGS84 origin, meter units, residuals and inlier RMSE; `sfm-to-enu.txt` records the transform. These values do not establish survey accuracy.

## How the seven stages are implemented

| Diagram stage | Current implementation |
| --- | --- |
| Adaptive frame extraction | Optical-flow displacement adjusts the next candidate interval within one-quarter to four times the base interval. Fixed sampling remains available. |
| Quality assessment | Laplacian sharpness, ORB feature count, dark/bright clipping fraction, weighted quality score and rejection reasons. |
| Keyframe selection | Quality thresholds, image-difference duplicate rejection, optional distance from previously accepted GPS positions. This is a greedy heuristic, not a proven globally optimal selector. |
| Structure from Motion | Actual COLMAP feature extraction, sequential matching, mapping, camera poses and triangulated sparse points. |
| AI depth estimation | Local Depth Anything V2 Small inference on undistorted images; robust affine inverse-depth calibration against positive-depth SfM observations. |
| Geospatial alignment | WGS84-to-ENU conversion, actual COLMAP similarity alignment and residual verification. Runs before depth calibration so fusion uses the aligned scale. |
| Dense reconstruction | Backprojection through calibrated cameras, agreement with neighboring depth maps, voxel averaging of coordinates/colors and PLY point-cloud output. |

The AI and stereo workflows are alternatives. Selecting Depth Anything V2 overrides the COLMAP sparse/dense output dropdown. AI calibration rejects reversed depth ordering, insufficient anchors and excessive error; fusion requires at least three accepted views and at least 100 consistent voxels. There is no procedural fallback.

## Outputs and troubleshooting

Use **Models → Reconstruction files**. Each reconstruction has its own run directory:

- `pipeline-report.json`: configuration, final success/failure/cancellation, last event and output paths.
- `sparse/`: original camera/point models, retained even if later processing fails.
- `component-*/georeference.json` and `sfm-to-enu.txt`: present after verified GPS alignment.
- `ai-dense-*/ai-depth/*.relative.npy`: raw relative AI predictions.
- `*.depth.npy`: calibrated depth maps for frames that passed checks.
- `*.preview.jpg`: colored relative-depth previews for inspection.
- `calibration.json`: model provenance and per-frame calibration/rejection details.
- `ai-dense-*/ai-fused.ply`: final colored cloud, created only after successful fusion.

The existing user drone project remains at `output/real-drone/project.drone3d.json`. Its COLMAP mesh is retained. AI inference produced 30 depth maps for this clip, but all disagreed with the existing SfM near/far ordering; **no AI cloud was accepted for this footage**. Better coverage and known camera calibration may be needed. A visually plausible AI depth preview does not prove a consistent 3D reconstruction.

A separate actual AI test on 12 official COLMAP South Building reference photographs produced **131,926 colored points**. Its local inspectable project is `output/AI-Reference-Validation/project.drone3d.json`. This reference scene is not the user's video and is not georeferenced. GPS alignment was separately exercised with the real executable against a synthetic known-transform fixture; no real flight GPS accuracy is claimed.

For repeatable CLI runs, `scripts/reconstruct_video.py` supports `--adaptive --ai --telemetry flight.csv --gps-spacing 2 --align-gps`. Use a new output directory. `scripts/refine_with_ai.py` can reuse an existing COLMAP undistorted workspace to avoid repeating SfM, but it does not verify the geographic provenance of that workspace. Full AI inference was tested from source; the optional packaged-executable build has not been validated with these AI dependencies.
