# MiDaS Small ONNX colored relief

The sparse COLMAP view contains triangulated feature points, so it looks like dots. A dense surface needs successful depth reconstruction. MiDaS provides a different output: a filled colored **2.5D relief of one video frame**, useful for a recognizable presentation when camera recovery is unreliable. It does not reconstruct the full flight, segment objects into buildings/turbines, create unseen sides, or provide metric geometry.

## Use it

MiDaS ONNX and its CPU runtime are installed on this laptop. Restart the app, open your project, and:

1. In Settings choose **MiDaS ONNX relief** under Depth method. Leave GPS alignment off. Save settings.
2. Analyze the video. Select an accepted frame in the Video frame list. Without a selected accepted frame, the highest quality accepted frame is used.
3. Click Generate scene. This method uses CPU and does not launch COLMAP. The sparse/dense dropdown, FOV and stereo/AI fusion stride do not affect this relief.
4. The filled preview opens from the front. Orbit slightly to inspect estimated depth. On reopening a project, choose **Front** for the source-facing view. Large viewing changes expose stretching; unseen surfaces do not exist.
5. Models → Reconstruction files contains the textured full `midas-relief.glb` and lightweight `midas-relief-preview.glb`, colored PLY alternatives, `source.jpg`, `depth-preview.jpg`, `relative-depth.npy`, and reports. GLB embeds the source image and opens in Blender or other compatible viewers.

Previously generated COLMAP/AI scene objects are hidden after successful generation, while their files and records remain available. Selecting colored geometry now preserves its colors.

The full mesh uses a grid of up to 256 pixels on its longer side. The app now uses a textured preview with at most 48 grid samples along its longer side, preserving image detail independently of face count. Both connect the whole grid instead of discarding triangles. GLBs embed the source image; PLY alternatives retain vertex colors. Normalized inverse depth is extruded into presentation coordinates; connecting depth boundaries can stretch foreground objects into backgrounds. Unchanged frame/model inputs reuse cached depth. See the [reference review and measured optimizations](REFERENCE_REVIEW.md).

## Installation on another machine

```powershell
venv\Scripts\python -m pip install -r requirements-midas.txt
venv\Scripts\python scripts\download_midas.py
```

The local ONNX path is editable in Settings. Normal generation does not download weights. The download script records the resolved Hugging Face revision and SHA-256. Weights are excluded from Git.

The [Hugging Face mirror](https://huggingface.co/Heliosoph/midas-small-onnx) distributes the upstream MiDaS v2.1 Small export. Preprocessing follows [upstream ONNX inference](https://github.com/isl-org/MiDaS/blob/master/tf/run_onnx.py): RGB floats in [0,1], resized to 256×256. Inspection of the downloaded graph confirmed embedded ImageNet subtraction/division. Do not apply the mirror card's extra BGR conversion/normalization. MiDaS is a relative-depth estimator, not a complete 3D-model generator.

## Actual validation

On September 7, 2026, `D:\DOWNLOADS\videoplaybackreal.webm` was decoded and sampled into 12 accepted frames. The selected frame at 26.28 seconds produced **68,340 full faces**. The app reopened and rendered the colored relief. This validates inference and surface export, not scene accuracy or multi-frame fusion. Local project: `output/MiDaS-Video-Relief/project.drone3d.json`.

Download revision: `f3a5a5e4c852c7bfac10517767e5b0636cb42b92`; SHA-256: `b0a5b3f12625137e626805167907fe0410665bec671685d59daaa2daab19f977`. Runtime: ONNX Runtime 1.23.2 on CPU. ONNX 1.20.1 was used for development graph inspection and is not required for inference.

Reproduce with a new destination:

```powershell
venv\Scripts\python scripts\reconstruct_video.py "D:\DOWNLOADS\videoplaybackreal.webm" output\MyRelief --midas --max-frames 12 --interval 1
```
