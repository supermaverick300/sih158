"""Generate a four-second synthetic clip; this is not real drone footage."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from drone3d_studio.services.video import synthetic_video

if __name__ == "__main__":
    destination = Path(sys.argv[1] if len(sys.argv) > 1 else "output/synthetic.avi").resolve()
    synthetic_video(destination)
    print(destination)
