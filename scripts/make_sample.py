"""Generate a small, portable sample scene."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from drone3d_studio.services.samples import make_sample

if __name__ == "__main__":
    make_sample(Path(sys.argv[1] if len(sys.argv) > 1 else "data/samples/Harbor-Demo").resolve())
