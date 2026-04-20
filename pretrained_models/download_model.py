#!/usr/bin/env python3
"""
Download the **Visual-Selective-VIO** frozen encoder weights from Google Drive:

  https://github.com/mingyuyng/Visual-Selective-VIO

This is the same artifact the VIFT repository documents placing under ``pretrained_models/``
manually (see https://github.com/ybkurt/vift); CAVIO automates the download with ``gdown``.

Dependency (also listed in ``requirements.txt``):

  pip install gdown

Behavior: by default, writes into **this script's directory** (the ``pretrained_models/`` folder
next to this file). ``gdown`` keeps the **remote file name**; the default Drive link should
produce ``vf_512_if_256_3e-05.model``. If the on-disk name differs, rename it or update
``model.tester.wrapper_weights_path`` in ``configs/model/*.yaml`` (defaults use
``${paths.root_dir}/pretrained_models/vf_512_if_256_3e-05.model``).

From the **repository root**:

  python pretrained_models/download_model.py
  python pretrained_models/download_model.py -o /path/to/dir
  python pretrained_models/download_model.py --url "https://drive.google.com/file/d/FILE_ID/view?usp=sharing"
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
        description=(
            "Download VS-VIO encoder weights from Google Drive (gdown; remote filename preserved)."
        )
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
