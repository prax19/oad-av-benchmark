#!/usr/bin/env bash
set -euo pipefail

# --- system deps ---
apt update
apt install -y gh git python3.11 python3.11-venv

# --- GitHub auth + git identity (maps commits to your GitHub account via noreply email) ---
gh auth login -h github.com -w

login="$(gh api user -q .login)"
id="$(gh api user -q .id)"
noreply="${id}+${login}@users.noreply.github.com"

git config --global user.name "$login"
git config --global user.email "$noreply"

echo "Git identity set to: $(git config --global user.name) <$(git config --global user.email)>"

# --- python venv ---
python3.11 -m venv /workspace/venv
source /workspace/venv/bin/activate
python -m pip install -U pip

# --- pytorch (CUDA 12.8 wheels) ---
pip install torch==2.9.0 torchvision==0.24.0 torchaudio==2.9.0 \
  --index-url https://download.pytorch.org/whl/cu128

# --- repo clone & submodules ---
mkdir -p /workspace
cd /workspace

if [ ! -d "oad-av-benchmark" ]; then
  git clone https://github.com/prax19/oad-av-benchmark
fi

cd oad-av-benchmark
git submodule update --init --recursive

# --- project setup ---
export PYTHONPATH="/workspace/oad-av-benchmark"
python setup/setup_dependencies.py

# --- run ---
# python setup/get_road.py
# python -m extraction.extract_dataset_features
