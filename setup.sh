#!/bin/bash
set -e

echo "🚀 Starting Setup..."

# 1. System Dependencies
echo "🛠️ Updating System..."
export DEBIAN_FRONTEND=noninteractive
apt-get update && apt-get install -y libgl1-mesa-glx unzip zip tmux

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
DATA_DIR="data"

if [ ! -d "$DATA_DIR" ]; then
    echo "📂 Creating data directory..."
    mkdir -p "$DATA_DIR"
    cd "$DATA_DIR"

    echo "⬇️ Downloading datasets from Kaggle..."

    kaggle datasets download -d hukibeginner2/pastis-pkl-firsthalf
    kaggle datasets download -d nguyenlecao/pastis-pkl

    echo "📦 Unzipping..."
    unzip -q -o "*.zip"
    rm *.zip

    cd ..
else
    echo "✅ Data directory already exists. Skipping download."
fi

echo "✅ Setup Complete! Ready to train 🚀"
