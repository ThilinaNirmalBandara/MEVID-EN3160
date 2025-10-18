#!/bin/bash

# MEvid Hybrid Search Setup Script
# This script sets up the environment and prepares the system

set -e  # Exit on error

echo "=================================================="
echo "MEvid Hybrid CLIP + Video ReID Setup"
echo "=================================================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check Python version
echo -e "\n${YELLOW}[1/6] Checking Python version...${NC}"
python_version=$(python3 --version 2>&1 | awk '{print $2}')
echo "Found Python $python_version"

if ! python3 -c "import sys; sys.exit(0 if sys.version_info >= (3, 7) else 1)"; then
    echo -e "${RED}Error: Python 3.7 or higher required${NC}"
    exit 1
fi
echo -e "${GREEN}✓ Python version OK${NC}"

# Install dependencies
echo -e "\n${YELLOW}[2/6] Installing dependencies...${NC}"
pip install --upgrade pip

# Core dependencies
echo "Installing PyTorch..."
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118  # CUDA 11.8

echo "Installing CLIP..."
pip install git+https://github.com/openai/CLIP.git

echo "Installing other packages..."
pip install faiss-cpu numpy pillow scikit-learn tqdm

echo -e "${GREEN}✓ Dependencies installed${NC}"

# Check CUDA availability
echo -e "\n${YELLOW}[3/6] Checking CUDA availability...${NC}"
python3 << EOF
import torch
if torch.cuda.is_available():
    print(f"CUDA available: {torch.cuda.get_device_name(0)}")
else:
    print("CUDA not available, will use CPU (slower)")
EOF

# Create directory structure
echo -e "\n${YELLOW}[4/6] Creating directories...${NC}"
mkdir -p artifacts
mkdir -p artifacts/thumbs
mkdir -p logs
mkdir -p checkpoints
echo -e "${GREEN}✓ Directories created${NC}"

# Verify MEvid dataset
echo -e "\n${YELLOW}[5/6] Verifying MEvid dataset...${NC}"
python3 << 'EOF'
import sys
from pathlib import Path

# Try to import config
try:
    from config import MEVID_ROOT
    
    if not MEVID_ROOT.exists():
        print(f"Warning: MEVID_ROOT not found at {MEVID_ROOT}")
        print("Please update config.py with correct path")
        sys.exit(1)
    
    # Check for required files
    required_files = [
        "test_name.txt",
        "track_test_info.txt",
        "query_IDX.txt"
    ]
    
    missing = []
    for f in required_files:
        if not (MEVID_ROOT / f).exists():
            missing.append(f)
    
    if missing:
        print(f"Warning: Missing files in MEvid dataset: {', '.join(missing)}")
        sys.exit(1)
    
    print(f"✓ MEvid dataset found at {MEVID_ROOT}")
    
except Exception as e:
    print(f"Error checking dataset: {e}")
    print("Please ensure config.py is set up correctly")
    sys.exit(1)
EOF

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ MEvid dataset verified${NC}"
else
    echo -e "${YELLOW}⚠ Please configure MEVID_ROOT in config.py${NC}"
fi

# Parse dataset and build index
echo -e "\n${YELLOW}[6/6] Building search index...${NC}"
echo "This may take several minutes..."

echo "  - Parsing MEvid annotations..."
python3 scripts/parse_mevid.py

echo "  - Extracting CLIP features..."
python3 scripts/extract_vecs.py

echo "  - Building FAISS index..."
python3 scripts/build_index.py

echo "  - Generating thumbnails..."
python3 scripts/make_thumbs.py

echo -e "${GREEN}✓ Index built successfully${NC}"

# Summary
echo -e "\n=================================================="
echo -e "${GREEN}Setup Complete!${NC}"
echo "=================================================="
echo ""
echo "Quick Start:"
echo "  1. Text search:"
echo "     python scripts/search_hybrid.py --query 'man in black jacket'"
echo ""
echo "  2. Train ReID model (optional):"
echo "     python scripts/train_reid.py --epochs 50 --batch_size 8"
echo ""
echo "  3. Visual demo:"
echo "     python scripts/demo_visual.py --query 'your query' --compare"
echo ""
echo "  4. Evaluate system:"
echo "     python scripts/eval_hybrid.py --eval_reid --eval_text"
echo ""
echo "For more options, see README.md"
echo "=================================================="