"""Source checkout launcher: python main.py."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from drone3d_studio.application import main

if __name__ == "__main__":
    raise SystemExit(main())
