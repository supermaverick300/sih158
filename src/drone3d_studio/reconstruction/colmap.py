"""Real sparse and CUDA dense COLMAP reconstruction. No fabricated output."""
import os
import json
from pathlib import Path
import queue
import shutil
import subprocess
import threading
from typing import Protocol
from uuid import uuid4

from drone3d_studio.services.video import Cancelled
from drone3d_studio.services.meshes import model_record, transfer_vertex_colors
from drone3d_studio.persistence.store import resolve, atomic_write


class Backend(Protocol):
    def run(self, root, frames, cancel, progress): ...


def detect(configured=""):
    candidate = configured or os.environ.get("COLMAP_EXECUTABLE") or shutil.which("colmap.exe") or shutil.which("colmap")
    if not candidate:
        # Search conventional user extraction locations, without storing machine paths.
        downloads = [Path.home() / "Downloads"]
        if os.name == "nt":
            downloads.extend(Path(f"{letter}:/Downloads") for letter in "CDEFG" if Path(f"{letter}:/Downloads").is_dir())
            import winreg
            try:
                with winreg.OpenKey(winreg.HKEY_CURRENT_USER, r"Software\Microsoft\Windows\CurrentVersion\Explorer\User Shell Folders") as key:
                    downloads.append(Path(os.path.expandvars(winreg.QueryValueEx(key, "{374DE290-123F-4565-9164-39C4925E467B}")[0])))
            except OSError:
                pass
        for folder in downloads:
            candidate = next(folder.glob("colmap*/bin/colmap.exe"), None)
            if candidate:
                break
    if not candidate or not Path(candidate).is_file():
        raise ValueError("COLMAP was not found. In Settings choose the extracted bin/colmap.exe. Keep its bin and plugins folders together. Choose Sparse cloud for CPU processing or Dense mesh for CUDA processing.")
    if Path(candidate).suffix.lower() in (".bat", ".cmd"):
        raise ValueError("Choose the actual colmap.exe in the extracted distribution, not a batch launcher.")
    return str(Path(candidate).resolve())


def run_command(args, cancel, log, cwd):
    flags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
    env = os.environ.copy()
    binary = Path(args[0]).resolve().parent
    plugins = binary.parent / "plugins"
    if plugins.is_dir():
        env["PATH"] = str(binary) + os.pathsep + env.get("PATH", "")
        env["QT_PLUGIN_PATH"] = str(plugins)
        env["QT_QPA_PLATFORM_PLUGIN_PATH"] = str(plugins / "platforms")
        env.pop("QT_QPA_PLATFORM", None)
    process = subprocess.Popen(args, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                               text=True, encoding="utf-8", errors="replace", creationflags=flags, env=env)
    lines = queue.Queue()
    def reader():
        for line in process.stdout:
            lines.put(line.rstrip())
        lines.put(None)
    thread = threading.Thread(target=reader, daemon=True)
    thread.start()
    from collections import deque
    recent = deque(maxlen=8)
    try:
        while True:
            if cancel.is_set():
                process.terminate()
                try:
                    process.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    process.kill()
                raise Cancelled("COLMAP cancelled")
            try:
                line = lines.get(timeout=.1)
            except queue.Empty:
                continue
            if line is None:
                break
            recent.append(line)
            log(line)
        code = process.wait()
        log(f"Exit code: {code}")
        if code != 0:
            raise RuntimeError(f"COLMAP {args[1] if len(args) > 1 else ''} exited with code {code}. See project logs for details.\n" + "\n".join(recent))
    finally:
        if process.poll() is None:
            process.kill()
        process.wait()
        thread.join(timeout=2)
        process.stdout.close()


class ColmapBackend:
    def __init__(self, executable="", output="Sparse cloud (CPU)", pipeline=None):
        from drone3d_studio.domain.models import PipelineConfig
        self.executable = executable
        self.output = output
        self.pipeline = pipeline or PipelineConfig()

    def run(self, root, frames, cancel, progress):
        self.work = None
        report = {"status": "Running", "configuration": self.pipeline.model_dump(), "accepted_frames": sum(f.accepted for f in frames), "gps_alignment": "Requested" if self.pipeline.align_gps else "Skipped: no metric/geographic claim", "last_event": "Preflight"}
        def tracking(value, message):
            report["last_event"] = message
            progress(value, message)
        try:
            result = self._run(root, frames, cancel, tracking)
            report["status"] = "Succeeded"
            report["units"] = "Local ENU metres" if self.pipeline.align_gps else "Arbitrary SfM units"
            report["gps_alignment"] = "Succeeded" if self.pipeline.align_gps else "Skipped"
            report["outputs"] = [{"path": model.path, "origin": model.origin, "vertices": model.vertices, "faces": model.faces} for model, mesh in result]
            return result
        except Exception as exc:
            report["status"] = "Cancelled" if isinstance(exc, Cancelled) else "Failed"
            report["error"] = str(exc)
            raise
        finally:
            if self.work:
                atomic_write(self.work / "pipeline-report.json", json.dumps(report, indent=2))

    def _run(self, root, frames, cancel, progress):
        executable = detect(self.executable)
        ai = self.pipeline.depth_method == "Depth Anything V2"
        weights = Path(self.pipeline.weights_path)
        if not weights.is_absolute():
            weights = Path(__file__).resolve().parents[3] / weights
        if ai:
            from drone3d_studio.reconstruction.ai_depth import check_installation
            check_installation(weights)
        accepted = [f for f in frames if f.accepted]
        if len(accepted) < 3:
            raise ValueError("COLMAP requires at least three accepted overlapping frames; usually many more.")
        work = root / "reconstruction" / uuid4().hex[:12]
        self.work = work
        images, sparse = work / "images", work / "sparse"
        images.mkdir(parents=True)
        sparse.mkdir()
        if self.pipeline.align_gps and sum(f.gps is not None for f in accepted) < 3:
            raise ValueError("GPS alignment enabled but fewer than three keyframes have GPS. Import synchronized telemetry and re-analyze; disabling alignment keeps arbitrary SfM coordinates.")
        for i, frame in enumerate(accepted):
            if cancel.is_set():
                raise Cancelled("COLMAP cancelled while staging frames")
            shutil.copy2(resolve(root, frame.path), images / f"{i:06d}.jpg")
        database = str(work / "database.db")
        with (root / "logs" / "colmap.log").open("a", encoding="utf-8") as logfile:
            def log(line):
                logfile.write(line + "\n")
                logfile.flush()
                progress(-1, line)
            # Query installed option names: COLMAP renamed these between releases.
            help_lines = []
            run_command([executable, "feature_extractor", "-h"], cancel, help_lines.append, work)
            extraction = "FeatureExtraction" if any("FeatureExtraction.use_gpu" in x for x in help_lines) else "SiftExtraction"
            help_lines = []
            run_command([executable, "sequential_matcher", "-h"], cancel, help_lines.append, work)
            matching = "FeatureMatching" if any("FeatureMatching.use_gpu" in x for x in help_lines) else "SiftMatching"
            commands = [
                [executable, "feature_extractor", "--database_path", database, "--image_path", str(images), "--ImageReader.single_camera", "1", f"--{extraction}.use_gpu", "0"],
                [executable, "sequential_matcher", "--database_path", database, f"--{matching}.use_gpu", "0"],
                [executable, "mapper", "--database_path", database, "--image_path", str(images), "--output_path", str(sparse)],
            ]
            # Keep extraction bounded on typical 8–16 GB laptops.
            commands[0][2:2] = [f"--{extraction}.max_image_size", "1600", f"--{extraction}.num_threads", "4"]
            if matching == "FeatureMatching":
                commands[1][2:2] = ["--FeatureMatching.num_threads", "4"]
            for command in commands:
                log("Running " + command[1])
                run_command(command, cancel, log, work)
            outputs = sorted(p.parent for p in sparse.glob("*/cameras.bin"))
            if not outputs:
                raise RuntimeError("COLMAP could not register a scene. Try sharper footage with more overlap and viewpoint variation.")
            result = []
            for index, output in enumerate(outputs):
                component = work / f"component-{index}"
                component.mkdir()
                if self.pipeline.align_gps:
                    from drone3d_studio.reconstruction.alignment import align
                    output = align(output, component, accepted, executable, self.pipeline, cancel, log, run_command)
                else:
                    log("GPS alignment skipped: output coordinates have arbitrary SfM scale.")
                ply = work / f"sparse-{index}.ply"
                run_command([executable, "model_converter", "--input_path", str(output), "--output_path", str(ply), "--output_type", "PLY"], cancel, log, work)
                record, mesh = model_record(root, ply, "COLMAP")
                record.name = f"COLMAP sparse cloud {index + 1}"
                result.append((record, mesh))
                if ai:
                    from drone3d_studio.reconstruction.ai_depth import run_depth_pipeline
                    dense = work / f"ai-dense-{index}"
                    dense.mkdir()
                    run_command([executable, "image_undistorter", "--image_path", str(images), "--input_path", str(output), "--output_path", str(dense), "--output_type", "COLMAP", "--max_image_size", "1000"], cancel, log, work)
                    text_model = dense / "text-model"
                    text_model.mkdir()
                    run_command([executable, "model_converter", "--input_path", str(dense / "sparse"), "--output_path", str(text_model), "--output_type", "TXT"], cancel, log, work)
                    cloud_path = run_depth_pipeline(dense, text_model, weights, self.pipeline, cancel, progress)
                    ai_record, ai_cloud = model_record(root, cloud_path, "Depth Anything V2")
                    ai_record.name = "Depth Anything V2 fused cloud" + (" · ENU metres" if self.pipeline.align_gps else " · SfM units")
                    record.visible = False
                    result.append((ai_record, ai_cloud))
                    continue
                if self.output == "Dense mesh (CUDA)":
                    dense = work / f"dense-{index}"
                    dense.mkdir()
                    fused = dense / "fused.ply"
                    surface = dense / "mesh.ply"
                    dense_commands = [
                        [executable, "image_undistorter", "--image_path", str(images), "--input_path", str(output), "--output_path", str(dense), "--output_type", "COLMAP", "--max_image_size", "1000"],
                        [executable, "patch_match_stereo", "--workspace_path", str(dense), "--workspace_format", "COLMAP", "--PatchMatchStereo.max_image_size", "1000", "--PatchMatchStereo.cache_size", "1", "--PatchMatchStereo.num_threads", "4", "--PatchMatchStereo.geom_consistency", "1"],
                        [executable, "stereo_fusion", "--workspace_path", str(dense), "--workspace_format", "COLMAP", "--input_type", "geometric", "--output_path", str(fused), "--StereoFusion.max_image_size", "1000", "--StereoFusion.cache_size", "1", "--StereoFusion.use_cache", "1", "--StereoFusion.num_threads", "4"],
                        [executable, "poisson_mesher", "--input_path", str(fused), "--output_path", str(surface), "--PoissonMeshing.depth", "9", "--PoissonMeshing.num_threads", "4"],
                    ]
                    for command in dense_commands:
                        log("Running " + command[1])
                        try:
                            run_command(command, cancel, log, work)
                        except RuntimeError as exc:
                            raise RuntimeError(f"Dense stage {command[1]} failed. Sparse output is retained at {ply}. Import it from Models or choose Sparse cloud (CPU). {exc}") from exc
                    dense_record, dense_mesh = model_record(root, surface, "COLMAP")
                    if dense_record.faces == 0:
                        raise RuntimeError("COLMAP produced no surface faces. Try more overlapping, sharper footage. Sparse and dense point files are retained in reconstruction.")
                    if dense_record.faces > 1600:
                        # Simplify the surface coherently rather than dropping random
                        # triangles in the CPU viewer. Keep full resolution on disk.
                        preview = dense / "inspection-mesh.ply"
                        log("Running mesh_simplifier for the CPU inspection viewport")
                        run_command([executable, "mesh_simplifier", "--input_path", str(surface), "--output_path", str(preview), "--MeshSimplification.target_face_ratio", str(1600 / dense_record.faces)], cancel, log, work)
                        preview_record, preview_mesh = model_record(root, preview, "COLMAP")
                        transfer_vertex_colors(dense_mesh, preview_mesh, cancel)
                        preview_mesh.export(str(preview))
                        dense_record, dense_mesh = preview_record, preview_mesh
                    dense_record.name = f"Reconstructed surface {index + 1}"
                    log(f"Full-resolution surface: {surface}")
                    record.visible = False
                    result.append((dense_record, dense_mesh))
            return result


def probe(executable, cancel, progress, cwd):
    path = detect(executable)
    lines = []
    run_command([path, "-h"], cancel, lines.append, cwd)
    version = next((line for line in lines if "COLMAP" in line), "COLMAP responded successfully")
    return f"{version}\nExecutable: {path}\nReady for real reconstruction. Dense mesh additionally requires working CUDA support."
