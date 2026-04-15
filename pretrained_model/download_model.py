#!/usr/bin/env python3
"""
Download pretrained weights from Google Drive related to the following repo:

https://github.com/mingyuyng/Visual-Selective-VIO

using gdown.

Install:
    pip install gdown

Default URL points to the weights also used in the VIFT paper.
The model will be saved in the ./pretrained_models folder with the name vf_512_if_256_3e-05.model
unless overridden with -o.

Usage:
    python download_model.py
    python download_model.py -o ./pretrained_models/encoder_weights.pt
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
        description="Download a file from Google Drive using gdown."
    )
    parser.add_argument(
        "--url",
        default=DEFAULT_URL,
        help="Google Drive file (or folder) URL.",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        metavar="PATH",
        help=(
            "Destination: a file path, or a directory. "
            "Default: this folder; Drive's original filename is kept."
        ),
    )
    args = parser.parse_args()

    if args.output is None:
        # Directory with trailing slash -> gdown keeps the remote filename
        output = str(SCRIPT_DIR) + "/"
    else:
        out = args.output.expanduser().resolve()
        if out.exists() and out.is_dir():
            output = str(out).rstrip("/") + "/"
        else:
            parent = out.parent
            if not parent.exists():
                parent.mkdir(parents=True, exist_ok=True)
            output = str(out)

    gdown.download(args.url, output, fuzzy=True, quiet=False)


if __name__ == "__main__":
    main()