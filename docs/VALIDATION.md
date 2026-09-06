# Development validation

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
