#!/bin/bash
set -e

echo "🚀 Starting Setup..."

# 1. System Dependencies
echo "🛠️ Updating System..."
export DEBIAN_FRONTEND=noninteractive
apt-get update && apt-get install -y libgl1 unzip zip tmux

# 2. Python Dependencies
echo "📦 Installing Python Libraries..."
pip install --upgrade pip
pip install -r requirements.txt

# 3. Kaggle CLI
echo "🔑 Installing Kaggle API..."
pip install kaggle

# 4. Kaggle Auth Check (ENV VAR MODE)
if [ -z "$KAGGLE_USERNAME" ] || [ -z "$KAGGLE_KEY" ]; then
    echo "❌ KAGGLE_USERNAME or KAGGLE_KEY is not set"
    echo "👉 Export them before running setup.sh"
    exit 1
fi

# 5. Dataset Download
DATA_DIR="kaggle_data"

if [ ! -d "$DATA_DIR" ]; then
    echo "📂 Creating data directory..."
    mkdir -p "$DATA_DIR"
    cd "$DATA_DIR"

    echo "⬇️ Downloading datasets from Kaggle..."
    kaggle datasets download -d nguyenlecao/t31tfm-1618

    FREE_GB=$(df -BG . | tail -1 | awk '{print $4}' | tr -d 'G')
    if [ "$FREE_GB" -lt 25 ]; then
        echo "❌ Not enough disk space (${FREE_GB}G free). Abort."
        exit 1
    fi

    echo "📦 Unzipping safely..."
    shopt -s nullglob
    for zipfile in *.zip; do
        echo "➡️ Extracting $zipfile"
        unzip -o "$zipfile"
        echo "🧹 Removing $zipfile"
        rm "$zipfile"
        df -h .
        echo "----------------------"
    done

    cd ..
else
    echo "✅ Data directory already exists. Skipping download."
fi

echo "📁 Creating checkpoints dir"
mkdir -p checkpoints

echo "✅ Setup Complete! Ready to train 🚀"

