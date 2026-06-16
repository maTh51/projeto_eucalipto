#!/usr/bin/env bash
# Eucalipto Environment Setup & Weights Downloader Script
set -euo pipefail

echo "============================================="
# 1) Git Submodules Setup
echo "=== [1/5] Initializing & updating Git submodules ==="
# Initialize and update submodules recursively
git submodule update --init --recursive
echo "Submodules updated successfully."

echo "============================================="
# 2) Python Virtual Environment Setup
echo "=== [2/5] Setting up local Python Virtual Environment (.venv) ==="
if [ ! -d ".venv" ]; then
    echo "Creating virtual environment (.venv) using python3..."
    python3 -m venv .venv
else
    echo "Virtual environment (.venv) already exists."
fi

echo "Activating virtual environment and installing packages..."
# Activate and upgrade pip/packaging tools
source .venv/bin/activate
pip install --upgrade pip wheel setuptools

# Install host orchestrator dependencies in editable mode
pip install -e .
echo "Python environment setup completed successfully."

echo "============================================="
# 3) Target weights directories creation
echo "=== [3/5] Creating directories for weights on host ==="
FF3D_WEIGHTS_DIR="third_party/FF3D_inference/ff3d_forestsens/work_dirs/clean_forestformer"
LEAFWOOD_WEIGHTS_DIR="third_party/leaf-wood-segmentation-with-deep-learning/model_weights"

mkdir -p "${FF3D_WEIGHTS_DIR}"
mkdir -p "${LEAFWOOD_WEIGHTS_DIR}"
echo "Weights directories verified."

echo "============================================="
# 4) Download and Extract ForestFormer3D model weights
echo "=== [4/5] Checking ForestFormer3D model weights ==="
FF3D_WEIGHTS_FILE="${FF3D_WEIGHTS_DIR}/epoch_3000_fix.pth"

if [ ! -f "${FF3D_WEIGHTS_FILE}" ]; then
    echo "Downloading ForestFormer3D weights from Zenodo..."
    wget -O clean_forestformer.zip "https://zenodo.org/api/records/16742708/files/clean_forestformer.zip/content"
    
    echo "Extracting weights using python..."
    python3 -c "
import zipfile, os, shutil
with zipfile.ZipFile('clean_forestformer.zip', 'r') as zf:
    for member in zf.infolist():
        filename = os.path.basename(member.filename)
        if filename == 'epoch_3000_fix.pth':
            source = zf.open(member)
            target = open(os.path.join('${FF3D_WEIGHTS_DIR}', filename), 'wb')
            with source, target:
                shutil.copyfileobj(source, target)
            print('✓ Extracted epoch_3000_fix.pth')
"
    rm -f clean_forestformer.zip
    echo "ForestFormer3D weights installation completed."
else
    echo "ForestFormer3D weights epoch_3000_fix.pth already exists."
fi

echo "============================================="
# 5) Download and Extract Leaf-wood segmentation model weights
echo "=== [5/5] Checking Leaf-wood segmentation weights ==="
LEAFWOOD_WEIGHTS_FILE="${LEAFWOOD_WEIGHTS_DIR}/weights_randlanet.pth"

if [ ! -f "${LEAFWOOD_WEIGHTS_FILE}" ]; then
    echo "Downloading Leaf-wood weights from Zenodo..."
    wget -O model_weights.zip "https://zenodo.org/api/records/13767795/files/model_weights.zip/content"
    
    echo "Extracting weights using python..."
    python3 -c "
import zipfile, os, shutil
with zipfile.ZipFile('model_weights.zip', 'r') as zf:
    for member in zf.infolist():
        filename = os.path.basename(member.filename)
        if filename in ['weights_randlanet.pth', 'weights_kpconv.pth', 'weights_pointtransformer.pth']:
            source = zf.open(member)
            target = open(os.path.join('${LEAFWOOD_WEIGHTS_DIR}', filename), 'wb')
            with source, target:
                shutil.copyfileobj(source, target)
            print(f'✓ Extracted {filename}')
"
    rm -f model_weights.zip
    echo "Leaf-wood weights installation completed."
else
    echo "Leaf-wood segmentation weights already exist."
fi

echo "============================================="
echo "🎉 Setup successfully completed!"
echo "To activate the environment: source .venv/bin/activate"
echo "============================================="
