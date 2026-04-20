#!/bin/bash

##########################################################################
# Prerequisites: create a venv, ``pip install -r requirements.txt``       #
# (needs ``gdown`` for the encoder weights). Network access required.      #
# System tools: ``wget`` and ``unzip`` for KITTI zips (large downloads).   #
##########################################################################

set -euo pipefail

# Resolve CAVIO repo root so this works when invoked as ``bash scripts/setup.sh``
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$ROOT"

# Visual-Selective-VIO encoder checkpoint -> pretrained_models/ (see download_model.py)
python3 ./pretrained_models/download_model.py

# KITTI odometry color + poses -> data/kitti_data/ (VIFT-identical script; cwd must be data/)
( cd "$ROOT/data" && bash ./data_prep.sh )
