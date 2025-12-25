#!/bin/bash
set -e

echo "🚀 Starting Setup..."

# 1. System Dependencies
echo "🛠️ Updating System..."
export DEBIAN_FRONTEND=noninteractive
apt-get update && apt-get install -y --no-install-recommends \
    libgl1 unzip zip tmux

# --- FIX: Create 8GB Swap File (Prevents 'Killed' Error) ---
if [ ! -f /swapfile ]; then
    echo "🧠 Creating 8GB Swap File to prevent crashes..."
    fallocate -l 8G /swapfile
    chmod 600 /swapfile
    mkswap /swapfile
    swapon /swapfile
    echo "✅ Swap enabled."
else
    echo "✅ Swap already exists."
fi
# -----------------------------------------------------------

# 2. Python Dependencies
echo "📦 Installing Python Libraries..."
pip install --upgrade pip
pip install -r requirements.txt

# 3. Kaggle API
echo "🔑 Installing Kaggle API..."
pip install kaggle

# 4. Auth Check
if [ -z "$KAGGLE_USERNAME" ] || [ -z "$KAGGLE_KEY" ]; then
    echo "❌ KAGGLE_USERNAME or KAGGLE_KEY is not set"
    exit 1
fi

# 5. Optimized Download Loop
DATA_DIR="kaggle_data"
mkdir -p "$DATA_DIR"
cd "$DATA_DIR"

# Define datasets to download sequentially
# Format: "Owner/DatasetName"
DATASETS=(
    "hukibeginner2/pastis-pkl-firsthalf"
    "nguyenlecao/pastis-pkl"
)

for HANDLE in "${DATASETS[@]}"; do
    echo "---------------------------------------------------"
    echo "⬇️  Downloading: $HANDLE"
    
    # Download with --force to overwrite if needed
    kaggle datasets download -d "$HANDLE" --force

    echo "📦 Unzipping..."
    # Unzip quietly (-q) to prevent terminal crash, overwrite (-o)
    unzip -q -o "*.zip"
    
    echo "🧹 Cleaning up zip..."
    rm *.zip
    
    # Show disk space after this step
    df -h . | awk 'NR==2 {print "💾 Disk Used: " $5 " | Free: " $4}'
done

cd ..

echo "---------------------------------------------------"
echo "📁 Creating checkpoints dir..."
mkdir -p checkpoints

echo "✅ Setup Complete! Ready to train 🚀"

