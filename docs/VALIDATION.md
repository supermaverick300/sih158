# Development validation

## MiDaS ONNX relief — September 7, 2026

Added a distinct estimated single-frame 2.5D relief mode, local MiDaS v2.1 Small ONNX inference on CPU, colored grid PLY export, a complete coarser preview, source/depth previews and provenance reports. Selected accepted frames are honored in the GUI; otherwise the best quality accepted frame is used. GPS alignment is explicitly rejected for this uncalibrated mode. Old reconstruction objects are hidden only after successful results; their records/files remain. Colored selections retain their image colors.

Final verification: **50 tests passed in 17.00 seconds**, startup smoke test exited **0**, and `pip check` found no broken requirements. New tests cover filled grid geometry, relative near/far direction, invalid constant depth, COLMAP bypass, output units, cancellation, GPS rejection, UI settings and old-cloud visibility. The actual ONNX model ran on a frame from the user's WebM video and exported **68,340 full faces / 11,550 preview faces**. The actual Qt app reopened and rendered the output. See [MiDaS usage, provenance and limitations](MIDAS_RELIEF.md). No metric accuracy, segmentation, unseen geometry or full-flight fusion is claimed.

## Single-pass workflow — September 7, 2026

New projects and the video CLI default to Single pass. Added timeline-spanning sampling, local sequential matching without loop detection, forward-motion initialization, optional fixed pinhole FOV calibration, and depth-consistent open surface patches instead of Poisson closure. General mode retains compatibility. The viewer supports previews up to 12,000 faces.

- `venv\Scripts\python -m pytest`: **47 passed in 16.04 seconds**.
- `venv\Scripts\python main.py --smoke-test`: **exit 0**.
- Tests cover depth binary decoding, discontinuities/holes, a controlled plane seen by three collinear translating cameras without rotation, cancellation, configuration compatibility, timeline coverage, and mocked single-pass COLMAP command orchestration. Synthetic and mocked tests are not real-world accuracy evidence.
- Actual COLMAP reran feature extraction, sequential matching and sparse reconstruction on all **30 saved frames** of the original straight-pass village clip. All 30 registered, yielding **26,601 sparse points**. Output: `output/Single-Pass-Flight/reconstruction/cc3b42936720`. The original video's previously supplied path is now missing, so this was saved-frame processing, not a fresh decode of that video.
- Separately, `scripts/rebuild_visible_surface.py` processed the earlier stereo depths and camera poses for those 30 frames. It produced **312,383 vertices / 552,528 faces**, with a **12,000-face** preview. Output: `output/real-drone/reconstruction/visible-59eadd056fad/visible-surface.ply`. The actual app reopened and rendered it. This reused old dense results; it was not a new end-to-end dense run with the new mapper settings.
- Visual inspection still shows distortion/noise in the original clip's geometry. More faces do not prove accuracy. The open surface builder changes representation and preserves missing coverage; it does not fix unreliable camera estimates. No exact scene, complete unseen surfaces, metric accuracy or successful AI calibration for this clip is claimed.

The original models and external files were preserved. See [single-pass usage](SINGLE_PASS.md). Full fresh-video dense validation, real calibrated single-pass accuracy measurements and the new AI surface path on real flight data remain unverified.

## Seven-stage integration — September 6, 2026

Added adaptive sampling, feature/exposure scoring, timestamped GPS CSV import and spacing, local Depth Anything V2 Small inference, SfM inverse-depth calibration, verified GPS-to-ENU alignment and multi-view colored voxel fusion. The Settings UI persists these controls in project JSON; imported/exported telemetry references are rebased. Existing projects retain compatible defaults. Real reconstruction never falls back to demo geometry.

AI setup was executed on Python 3.10.11: PyTorch **2.8.0+cpu**, Transformers **4.57.6**, Pillow **11.3.0**, safetensors **0.7.0**. The official Small model downloaded successfully at revision `5426e4f0f36572d16453bbda7a8389317b1bef99`; its local download manifest records the SHA-256. `pip check` reported no broken requirements.

- **Automated suite:** 41 passed in 14.31 seconds. Covers motion-adaptive intervals, quality rejection, telemetry gaps/dateline/spacing, ENU axes, robust calibration with outliers, camera poses, multi-view rejection and colored fusion, JSON round trips, GUI-to-backend configuration, AI orchestration and persisted failure reports. The broader pre-existing tests also pass.
- **Startup:** `venv\Scripts\python main.py --smoke-test` exited 0 after permission to write the normal application log. An initial sandboxed attempt failed at that log file; it was not an application startup defect.
- **User stock footage:** reused its existing COLMAP undistorted workspace, avoiding repeat SfM. All **30** actual neural depth maps were generated. All failed the positive inverse-depth slope check against the existing SfM geometry. No AI cloud was accepted and the existing project/mesh was preserved. This is a demonstrated limitation of this input/reconstruction pairing, not a successful AI reconstruction claim.
- **Real reference photographs:** undistorted the official COLMAP South Building model at a maximum 1000 pixels and selected the first 12 images by filename. All **12** neural depth maps passed calibration. Multi-view fusion produced **131,926** colored points. Source: the previously downloaded official South Building archive. The app loaded the resulting PLY and rendered it successfully. This is a partial reference-scene cloud, not the user's drone video, a complete building, or a metric survey.
- **Real GPS-aligner executable:** ran COLMAP `model_aligner` against a clearly synthetic fixture with five known WGS84 camera locations and a known 3× similarity transform. All five residuals passed, RMSE approximately **2.5e-10 m**. This measures numerical recovery of a constructed fixture; no real flight telemetry or geographic accuracy was validated.
- **UI:** rendered and visually inspected the scrollable pipeline Settings page and actual AI reference cloud. Saved screenshots and test assets remain in ignored `output/`.

Local artifacts: `output/real-drone/reconstruction/ai-refine-8ab29a72ec85/` contains the rejected user-video depth maps and calibration report; `output/ai-reference/dense/ai-fused.ply` is the successful reference cloud; `output/AI-Reference-Validation/project.drone3d.json` opens it in the app; `output/gps-validation/alignment/georeference.json` is the synthetic GPS validation report.

CPU AI inference was validated; CUDA PyTorch inference, direct DJI SRT parsing, real GPS accuracy and packaging AI into the optional executable were not validated or implemented. See [setup and limitations](PIPELINE_SETUP.md).

Commands actually executed for this integration (in addition to the ignored reference/GPS/UI validation scripts described above):

```powershell
venv\Scripts\python -m pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cpu
venv\Scripts\python -m pip install -r requirements-ai.txt
venv\Scripts\python scripts\download_depth_model.py
venv\Scripts\python scripts\refine_with_ai.py output\real-drone output\real-drone\reconstruction\fbb5ea4885f0\dense-0 --colmap D:\DOWNLOADS\colmap-x64-windows-cuda\bin\colmap.exe
venv\Scripts\python -m pip check
venv\Scripts\python -m pytest
venv\Scripts\python main.py --smoke-test
```

The refine command exited 1 for the documented calibration disagreement; dependency installation, download, dependency check, final tests and final startup exited 0.

## Real-reconstruction upgrade

The main reconstruction action now always calls COLMAP, including when opening older projects that saved Demo as their mode. Demo generation is an explicit separate Models action. Added actual dense stages (undistortion, CUDA stereo, fusion, Poisson surface generation and inspection simplification), child-only Windows plugin paths, bounded processing settings, executable/version probing, colored geometry display and a diagnostic video runner.

Commands run after these changes:

```cmd
venv\Scripts\python -m pytest
venv\Scripts\python main.py --smoke-test
```

Final result: **31 passed in 9.82 s**. The expanded tests verify dense command orchestration, isolated child-process environments, measured-color preservation and no demo fallback after real reconstruction failure. The startup check exited 0 after the final viewer fixes (normal application log access required sandbox permission). These tests do not substitute for a real reconstruction run.

Installed COLMAP was verified by running its actual `bin\colmap.exe -h`: version **4.2.0 with CUDA**. NVIDIA tools reported an **RTX 3050 Laptop GPU with 4096 MiB**. Dense-command flags were checked against this executable's help output.

An initial six-second portrait aircraft/flag clip was processed with the real executable and failed sparse registration. Its Failed state and log were retained; no placeholder scene was generated.

### Successful real drone-video run

The user supplied a 15-second 1280 × 720, 25 fps aerial village/highway clip. The following real pipeline was executed against that video (all generated data stays in ignored `output/`):

```cmd
venv\Scripts\python scripts\reconstruct_video.py "D:\DOWNLOADS\vidssave.com 4K Cinematic Drone view village Highway l Free  Drone Video l Free stock footage l Copyright free 720P.mp4" output\real-drone --colmap "D:\DOWNLOADS\colmap-x64-windows-cuda\bin\colmap.exe" --interval 0.5 --max-frames 30
```

Observed result: **exit code 0**, 30 accepted frames and all 30 views registered. COLMAP generated **26,685 sparse points**, **111,812 fused dense points**, and a full-resolution surface with **13,330 vertices / 23,031 faces**. Its inspection version has **1,154 vertices / 1,600 faces**. Processing took approximately 13 minutes on the test laptop; dense stereo used about 0.5 GB GPU memory in a sampled reading, not a measured peak.

The installed simplifier dropped the mesh's PLY color properties. A bounded-memory color transfer from the full-resolution surface was added and run on the inspection output; geometry was unchanged. The saved project was reopened through the actual Qt app offscreen, both objects loaded with Ready status, and the rendered screenshot was visually inspected. A principal-surface camera action was added after the initial view was almost edge-on.

Result location on this development machine: `output/real-drone/project.drone3d.json`. Its `reconstruction/fbb5ea4885f0/dense-0/` contains `mesh.ply`, `fused.ply` and `inspection-mesh.ply`. These assets and the user's source video are not committed.

**Quality limit:** the short shot reconstructed a partial terrain strip, not a complete village or a survey-grade model. Successful processing does not imply complete geometry, accurate scale or high-quality capture. No texture atlas was generated. The official COLMAP South Building archive was downloaded as a fallback test source but was not needed once the user supplied the drone clip.

The sections below record the original MVP validation before this upgrade. Their original “not executed” entries describe that earlier stage.

Environment: Windows, tested with Python 3.10.11 and Python 3.12.14. Runtime pins are in `requirements.txt`; test pins are in `requirements-dev.txt`.

## Commands actually run

From the repository directory, using PowerShell:

```powershell
git switch -c feature/drone-3d-studio
python -m venv venv
venv\Scripts\python -m pip install -r requirements-dev.txt
venv\Scripts\python scripts\make_sample.py
venv\Scripts\python -m pip check
venv\Scripts\python -m pytest -q
venv\Scripts\python main.py --smoke-test --screenshot output\startup.png
venv\Scripts\python scripts\capture_screenshots.py
```

The first dependency installation was blocked by the execution sandbox's network rules. The same command was rerun with permission and completed. The first startup attempt was blocked while creating the normal local application log directory. The same command was rerun with permission and exited 0.

The system Python launcher had no 3.12 installation registered. A bundled Python 3.12 runtime was found and used to create the second environment:

```powershell
& 'C:\Users\ASUS\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe' -m venv venv312
venv312\Scripts\python -m pip install -r requirements-dev.txt
venv312\Scripts\python -m pytest -q
venv312\Scripts\python -m pip check
venv312\Scripts\python main.py --smoke-test --screenshot output\startup312.png
```

The Python 3.12 startup command was also rerun with permission after the sandbox blocked writing the normal log file. It exited 0. These environment paths describe the development machine, not an installation requirement; users should use the standard Python launcher commands in the README.

## Observed results

| Check | Result |
| --- | --- |
| Runtime and test dependency install, Python 3.10 | Succeeded |
| Runtime and test dependency install, Python 3.12 | Succeeded |
| `pip check`, both environments | No broken requirements found |
| Initial suite, Python 3.10 | 23 passed in 9.27 s |
| Expanded final suite, Python 3.10 | 27 passed in 8.12 s |
| Expanded final suite, Python 3.12 | 27 passed in 9.98 s |
| Offscreen source startup, both environments | Exit code 0 |
| Sample project generation | Succeeded; tiny JSON and PLY files included |
| Actual UI screenshots | Generated and visually inspected; offscreen font issue fixed |

The GUI workflow test creates a project, generates/imports a synthetic video, analyzes four frames, generates a procedural scene, changes transforms and visibility, saves, closes and reopens the project. All seven pages are rendered offscreen. No permanent GUI window is left running by tests.

Service tests also exercise real child processes for stdout capture, nonzero exit codes and cancellation. A fake-runner test checks the COLMAP command contract and output import; **it is not evidence that a real reconstruction succeeded**.

## Not executed or not established

- Real COLMAP reconstruction: no external COLMAP installation or suitable real flight footage was supplied.
- PyInstaller executable build: the optional script is provided, but no packaged binary is claimed.
- Python 3.11: dependency support is declared; this runtime was not present for a test run.
- Long-duration, high-resolution video and very large meshes: performance remains bounded by decoding and import memory; small synthetic assets were used for automated validation.
- Visual interaction on every Windows DPI/driver combination: offscreen rendering and Qt interactions were tested; this is not broad hardware certification.
