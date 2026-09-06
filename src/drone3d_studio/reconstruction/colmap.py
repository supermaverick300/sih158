"""Optional CPU sparse reconstruction. No fabricated output or shell commands."""
import os
from pathlib import Path
import queue
import shutil
import subprocess
import threading
from typing import Protocol
from uuid import uuid4

from drone3d_studio.services.video import Cancelled
from drone3d_studio.services.meshes import model_record
from drone3d_studio.persistence.store import resolve


class Backend(Protocol):
    def run(self, root, frames, cancel, progress): ...


def detect(configured=""):
    candidate = configured or shutil.which("colmap.exe") or shutil.which("colmap")
    if not candidate or not Path(candidate).is_file():
        raise ValueError("COLMAP is not installed or configured. Download the Windows release from https://github.com/colmap/colmap/releases, extract it, and set colmap.exe in Settings. CPU sparse reconstruction is supported; dense reconstruction is not bundled.")
    if Path(candidate).suffix.lower() in (".bat", ".cmd"):
        raise ValueError("Choose the actual colmap.exe in the extracted distribution, not a batch launcher.")
    return str(Path(candidate).resolve())


def run_command(args, cancel, log, cwd):
    flags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
    process = subprocess.Popen(args, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                               text=True, encoding="utf-8", errors="replace", creationflags=flags)
    lines = queue.Queue()
    def reader():
        for line in process.stdout:
            lines.put(line.rstrip())
        lines.put(None)
    thread = threading.Thread(target=reader, daemon=True)
    thread.start()
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
            log(line)
        code = process.wait()
        log(f"Exit code: {code}")
        if code != 0:
            raise RuntimeError(f"COLMAP exited with code {code}. See project logs for details.")
    finally:
        if process.poll() is None:
            process.kill()
        process.wait()
        thread.join(timeout=2)
        process.stdout.close()


class ColmapBackend:
    def __init__(self, executable=""):
        self.executable = executable

    def run(self, root, frames, cancel, progress):
        executable = detect(self.executable)
        accepted = [f for f in frames if f.accepted]
        if len(accepted) < 3:
            raise ValueError("COLMAP requires at least three accepted overlapping frames; usually many more.")
        work = root / "reconstruction" / uuid4().hex[:12]
        images, sparse = work / "images", work / "sparse"
        images.mkdir(parents=True)
        sparse.mkdir()
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
            for command in commands:
                log("Running " + command[1])
                run_command(command, cancel, log, work)
            outputs = sorted(p.parent for p in sparse.glob("*/cameras.bin"))
            if not outputs:
                raise RuntimeError("COLMAP could not register a scene. Try sharper footage with more overlap and viewpoint variation.")
            result = []
            for index, output in enumerate(outputs):
                ply = work / f"sparse-{index}.ply"
                run_command([executable, "model_converter", "--input_path", str(output), "--output_path", str(ply), "--output_type", "PLY"], cancel, log, work)
                record, mesh = model_record(root, ply, "COLMAP")
                record.name = f"COLMAP sparse cloud {index + 1}"
                result.append((record, mesh))
            return result
