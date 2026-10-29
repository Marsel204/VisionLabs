"""Package this checkout for Colab without credentials or datasets."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app.services.colab_bundle import build_source_bundle  # noqa: E402, F401

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path, default=Path("artifacts/colab/visionlab-colab-source.zip")
    )
    print(build_source_bundle(parser.parse_args().output))
