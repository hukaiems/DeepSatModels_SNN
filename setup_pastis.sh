#!/bin/bash
set -e

echo "🚀 Starting Setup..."

# 1. System Dependencies
echo "🛠️ Updating System..."
export DEBIAN_FRONTEND=noninteractive
apt-get update && apt-get install -y --no-install-recommends \
    libgl1 unzip zip tmux

# 2. Python Dependencies
echo "📦 Installing Python Libraries..."
python3 -m pip install --upgrade pip
python3 -m pip install -r requirements.txt

# 3. Kaggle CLI
echo "🔑 Installing Kaggle API..."
python3 -m pip install kaggle

# 4. Auth Check
if [ -z "$KAGGLE_USERNAME" ] || [ -z "$KAGGLE_KEY" ]; then
    echo "❌ KAGGLE_USERNAME or KAGGLE_KEY is not set"
    echo "👉 Export them before running setup.sh"
    exit 1
fi

# 5. Optimized Download Loop (Saves Disk Space)
DATA_DIR="kaggle_data"
mkdir -p "$DATA_DIR"
cd "$DATA_DIR"

# List datasets here to process one by one
DATASETS=(
    "hukibeginner2/pastis-pkl-firsthalf"
    "nguyenlecao/pastis-pkl"
)

for HANDLE in "${DATASETS[@]}"; do
    echo "---------------------------------------------------"
    echo "⬇️  Downloading: $HANDLE"
    
    # Download JUST this one file (Force overwrite if exists)
    kaggle datasets download -d "$HANDLE" --force

    echo "➡️ Extracting..."
    # Unzip quietly (> /dev/null) so it doesn't spam/crash terminal
    if unzip -o -q "*.zip"; then
        echo "✅ Extracted successfully."
        
        echo "🧹 Deleting zip to save space..."
        rm *.zip
        
        # Check disk space to confirm we are safe
        df -h . | awk 'NR==2 {print "💾 Space Left: " $4}'
    else
        echo "❌ Failed to extract $HANDLE"
        exit 1
    fi
done

cd ..

echo "---------------------------------------------------"
echo "📁 Creating checkpoints dir..."
mkdir -p checkpoints

echo "✅ Setup Complete! Ready to train 🚀"