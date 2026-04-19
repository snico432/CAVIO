#!/usr/bin/env python3
"""
Download pretrained weights from Google Drive related to the following repo:

https://github.com/mingyuyng/Visual-Selective-VIO

using gdown.

Install:
    pip install gdown

Default URL points to the weights also used in the VIFT paper.
The file is saved under the chosen directory using **Google Drive's filename** (gdown default).

Keep ``model.tester.wrapper_weights_path`` in sync with that name (expected:
``pretrained_models/vf_512_if_256_3e-05.model`` for the default URL).

Usage:
    python download_model.py
    python download_model.py -o /path/to/dir
    python download_model.py --url "https://drive.google.com/file/d/FILE_ID/view?usp=sharing"
"""

import argparse
from pathlib import Path

import gdown

# Visual-Selective-VIO: https://github.com/mingyuyng/Visual-Selective-VIO
DEFAULT_URL = (
    "https://drive.google.com/file/d/18tfw94asjStribqA8SLvaacrUwT86HsT/view?usp=sharing"
)
SCRIPT_DIR = Path(__file__).resolve().parent


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download a file from Google Drive using gdown (remote filename preserved)."
    )
    parser.add_argument(
        "--url",
        default=DEFAULT_URL,
        help="Google Drive file (or folder) URL.",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        type=Path,
        default=None,
        metavar="DIR",
        help="Directory to write into (default: directory containing this script).",
    )
    args = parser.parse_args()

    dest = (args.output_dir or SCRIPT_DIR).expanduser().resolve()
    dest.mkdir(parents=True, exist_ok=True)
    # Trailing slash => gdown keeps the file name from Google Drive
    gdown.download(args.url, str(dest).rstrip("/") + "/", quiet=False)


if __name__ == "__main__":
    main()
