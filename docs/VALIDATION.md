# Development validation

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
