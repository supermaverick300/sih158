# Textured photogrammetry with OpenMVS and Blender

Select **OpenMVS textured mesh** in Settings to run:

```text
Drone video → adaptive quality-filtered keyframes
→ COLMAP feature matching / camera poses / sparse points
→ COLMAP image undistortion
→ optional COLMAP PatchMatch + stereo fusion
→ InterfaceCOLMAP → DensifyPointCloud
→ ReconstructMesh → RefineMesh → TextureMesh
→ textured OBJ + MTL + atlas images
→ Blender headless
→ scene_high.glb / scene_medium.glb / scene_low.glb
→ low LOD in Drone 3D Studio
```

This is multi-view photogrammetry using successive frames of one flight. It is separate from MiDaS's single-frame estimated relief. The footage still needs sharp overlapping views, translation and a largely static scene; neither meshing nor texturing can guarantee exact geometry from inadequate observations. Unseen surfaces are not established by this pipeline. Hole closure is disabled in the reconstruction and texturing commands.

## Installed tools on this laptop

- COLMAP: `D:\DOWNLOADS\colmap-x64-windows-cuda\bin\colmap.exe`.
- OpenMVS 2.4.0 CPU: `D:\PROJECTS\sih158\output\tools\OpenMVS\vc17\x64\Release`.
- Blender 4.5.3 LTS: `D:\PROJECTS\sih158\output\tools\Blender\blender-4.5.3-windows-x64\blender.exe`.

Keep each extracted distribution together, including DLLs. These portable tools are not committed and are discovered automatically at these locations. Removing `output/tools` removes the installations. Custom locations can be entered in Settings.

For another machine, download the [official OpenMVS Windows package](https://github.com/cdcseacave/openMVS/releases/expanded_assets/v2.4.0) and [Blender portable Windows build](https://download.blender.org/release/Blender4.5/). The tested OpenMVS CPU ZIP's SHA-256 is `0c31660c15c9ebc4c106873cf67564d9570d404aef7a6403451da1b6178b2167`; it was verified after download. CPU OpenMVS avoids an additional CUDA compatibility requirement. Enabling the preceding COLMAP stereo pass still requires CUDA.

## Run from the app

1. Restart the app, create/open a project and import your video.
2. In Settings choose **Single pass** and **OpenMVS textured mesh**. Enter paths if using custom installation locations. **Check OpenMVS and Blender paths** checks file availability, not successful reconstruction.
3. Leave **Run COLMAP dense stereo before OpenMVS** unchecked for the efficient route. Checking it adds COLMAP PatchMatch/fusion before conversion; OpenMVS still runs its own densifier, so this can duplicate work. It is available to reproduce the supplied diagram explicitly.
4. Leave GPS alignment off unless synchronized valid telemetry is available. Save settings.
5. Analyze using adaptive sampling, review accepted frames, then Generate scene.
6. On success the low LOD opens as a textured model. Models → Reconstruction files contains full exports. Previous scene objects/files are retained; earlier reconstruction objects are hidden after success.

The OpenMVS mode overrides the older Sparse cloud / Dense mesh selection. Tools are checked before expensive camera recovery. Failure or cancellation stops downstream work, retains intermediate files and records the failing stage; it never substitutes demo geometry or reports a successful texture when atlas files are absent.

## Output details

Global and local seam leveling are disabled: this Windows build produced black atlas patches on the validation mesh when they were enabled. This preserves source image colors but can leave exposure differences between patches. The corrected exports were visually inspected in the app.

Each reconstruction component gets its own `openmvs-N/` directory. It contains `.mvs` intermediate scenes, dense points, reconstructed/refined meshes, `scene_textured.obj`, `scene_textured.mtl` and atlas images. Atlas dimensions are capped at 4096; OpenMVS can generate multiple atlases.

Blender imports materials and UVs, packs the real atlas images into GLBs, and exports each LOD independently from the full source mesh:

- `scene_high.glb`: full triangulated source geometry.
- `scene_medium.glb`: approximately half the faces, capped at 100,000.
- `scene_low.glb`: approximately 10,000 faces or fewer for the CPU viewport.

Decimation targets are approximate; small inputs can have identical LOD counts. UV boundaries can introduce extra GLB vertices. All LODs retain atlas resolution; this first implementation reduces geometry, not texture memory. The low LOD is the only one loaded automatically, to avoid loading all high-resolution assets into the CPU viewport. The exports preserve source coordinates and arbitrary SfM scale unless verified GPS alignment succeeded.

`textured-report.json` records executable paths, exact stage commands, elapsed times, stage status and outputs. `lod-report.json` records actual Blender face counts and file sizes. OpenMVS writes detailed native logs beside its outputs; the app's log also records stage starts and process exits. Re-running creates new output directories. Automatic stage resumption is not implemented.

## CLI

Fresh video to a new project:

```powershell
venv\Scripts\python scripts\reconstruct_video.py "D:\path\flight.mp4" output\TexturedFlight --openmvs --adaptive --interval 0.5 --max-frames 150
```

Add `--openmvs-dir "D:\tools\OpenMVS" --blender "D:\tools\Blender\blender.exe"` for custom locations. Add `--colmap-stereo-first` only when the additional COLMAP dense stage is desired.

Continue an existing undistorted COLMAP workspace into a **new** project:

```powershell
venv\Scripts\python scripts\texture_colmap_workspace.py output\Podgaric-Video\reconstruction\7baaf3a1d466\dense-0 output\TexturedMonument
```

This reuses existing cameras/images/depth data and does not claim a fresh video-to-camera run or verify geographic provenance. Source files remain untouched. The downstream tools may recompute dense estimates.

Stage interfaces follow [OpenMVS usage](https://github.com/cdcseacave/openMVS/wiki/Usage). Actual local validation is recorded in [VALIDATION.md](VALIDATION.md).
