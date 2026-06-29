#!/usr/bin/env bash
# Clones the three baseline backbones extended by models/*_semantic.py next to
# this repo and prints the PYTHONPATH export needed to use them.
set -euo pipefail

BASELINE_DIR="${1:-../baselines}"
mkdir -p "$BASELINE_DIR"
cd "$BASELINE_DIR"

[ -d Fine-grained-Multimodal-DeepFake-Classification ] || \
    git clone https://github.com/yinql1995/Fine-grained-Multimodal-DeepFake-Classification.git
[ -d FGI ] || git clone https://github.com/aseuteurideu/FGI.git
[ -d MRDF ] || git clone https://github.com/Vincent-ZHQ/MRDF.git

cd - >/dev/null
pip install -r requirements.txt
pip install git+https://github.com/facebookresearch/ImageBind.git

ABS_BASELINE_DIR="$(cd "$BASELINE_DIR" && pwd)"
echo
echo "Baselines cloned to: $ABS_BASELINE_DIR"
echo "Add them to PYTHONPATH before running scripts/, e.g.:"
echo "  export PYTHONPATH=\$PYTHONPATH:$ABS_BASELINE_DIR/Fine-grained-Multimodal-DeepFake-Classification:$ABS_BASELINE_DIR/FGI:$ABS_BASELINE_DIR/MRDF"
