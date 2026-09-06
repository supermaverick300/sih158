# Drone 3D Studio development

- Keep UI, persisted models, video services, viewer and external processes separate.
- Demo geometry must always be labelled; never describe it as reconstruction.
- Do not put Qt objects or meshes in project JSON.
- Preserve external source files. Only delete a project directory after an explicit user confirmation.
- Run `venv\Scripts\python -m pytest` and `venv\Scripts\python main.py --smoke-test` after functional changes.
- Never commit generated user projects, virtual environments or large videos.
