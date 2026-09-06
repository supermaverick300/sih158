# Reconstructing one continuous drone pass

New projects default to **Single pass**. One flight is enough input: the drone must translate while successive frames overlap. An orbit, return flight or separate set of photographs is not required. Pure rotation, distant scenery with negligible parallax, motion blur and moving subjects can still prevent reliable geometry. The app cannot guarantee an exact model from arbitrary footage.

## On this laptop

No additional downloads are needed for COLMAP stereo. Start the app from `D:\PROJECTS\sih158`:

```powershell
venv\Scripts\python main.py
```

1. Create a project and import the original drone video.
2. In Settings, select **Single pass**, **COLMAP stereo**, and **Dense mesh (CUDA)**. Set COLMAP to `D:\DOWNLOADS\colmap-x64-windows-cuda\bin\colmap.exe`.
3. Leave horizontal FOV at **0** unless you know the calibrated horizontal FOV for this exact rectified video and crop. A supplied value fixes pinhole intrinsics; a guessed value can make geometry worse. Leave GPS alignment off unless you have suitable synchronized telemetry.
4. Save settings. In Video, choose Adaptive sampling, start with a 0.5-second interval and a maximum of 150 samples, then Analyze. The sample budget covers the whole pass. For longer footage, increase the budget to retain overlap; covering the timeline does not guarantee overlap.
5. Review accepted frames for sharpness and continuity, then Generate scene. Follow the Dashboard log. Successful camera registration alone does not prove accurate geometry.
6. Open the Scene preview or use Models → Reconstruction files to inspect the full PLY in Open3D.

Existing projects keep **General** mode for compatibility; change and save it explicitly before analyzing again.

## What changed

- Sequential matching uses nearby frames without loop detection. Single-pass initialization permits forward motion while retaining triangulation checks.
- Accepted stereo depth samples must agree with neighboring camera views. Triangles connect local depth samples, leaving holes and depth discontinuities open. The process does not run Poisson meshing in Single pass mode.
- The full output is `visible-surface.ply`; its JSON sidecar describes construction and limitations. The app keeps a colored preview of up to 12,000 faces. View patches may overlap; this is not a watertight solid or a texture atlas.
- Optional Depth Anything V2 uses the same visible-surface builder after calibration and consistency checks. Its depth is estimated and may be rejected. It cannot repair arbitrary incorrect camera poses or reveal unseen surfaces. Its output is `ai-visible-surface.ply`.
- General mode retains the previous Poisson mesh / AI point-cloud workflows.

Only surfaces visible with usable evidence can be reconstructed. Undersides and unseen backs remain missing. Scale remains arbitrary unless valid alignment succeeds. A straight, collinear GPS track alone cannot fully constrain the existing alignment method; GPS is not required to generate a model.

## Command line

Use a new output directory:

```powershell
venv\Scripts\python scripts\reconstruct_video.py "D:\path\flight.mp4" output\MySinglePass --adaptive --interval 0.5 --max-frames 150 --colmap "D:\DOWNLOADS\colmap-x64-windows-cuda\bin\colmap.exe"
```

Single pass is the CLI default. `--general` restores the older workflow; `--horizontal-fov` supplies known calibration; `--sparse` only produces a sparse cloud.

To rebuild an existing stereo workspace as an open surface:

```powershell
venv\Scripts\python scripts\rebuild_visible_surface.py output\real-drone output\real-drone\reconstruction\fbb5ea4885f0\dense-0
```

This utility reuses existing camera poses and depths. It preserves prior model files and records, hides their scene objects, and adds the new preview. It does not improve camera calibration or establish geographic accuracy.

See [validation evidence](VALIDATION.md) and [optional AI/GPS setup](PIPELINE_SETUP.md). Camera transforms and depth files follow the [official COLMAP format](https://colmap.github.io/format.html).
