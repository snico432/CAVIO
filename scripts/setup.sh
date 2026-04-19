#!/bin/bash

##########################################################################
# Make sure to create a virtual environment and install the requirements #
# before running this script                                             #
##########################################################################

set -euo pipefail

# Run from repo root (CAVIO/) so paths work no matter where this is invoked from
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$ROOT"

# Download the pretrained model
python3 ./pretrained_models/download_model.py

# Download the KITTI dataset
bash ./data/data_prep.sh
