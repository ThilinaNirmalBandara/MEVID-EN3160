#!/usr/bin/env python3
"""
Setup verification script
Checks if all dependencies and data are properly configured
"""
import sys, os
from pathlib import Path
import json

# ensure we can import config + package when running as: python scripts/parse_mevid.py
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import sys
from pathlib import Path

# ANSI color codes
GREEN = '\033[92m'
RED = '\033[91m'
YELLOW = '\033[93m'
BLUE = '\033[94m'
RESET = '\033[0m'

def check_mark(success):
    return f"{GREEN}✓{RESET}" if success else f"{RED}✗{RESET}"

def print_header(text):
    print(f"\n{BLUE}{'='*70}")
    print(f"{text}")
    print(f"{'='*70}{RESET}\n")

def check_python_version():
    """Check Python version"""
    print("Checking Python version...", end=" ")
    version = sys.version_info
    success = version >= (3, 7)
    print(f"{check_mark(success)} Python {version.major}.{version.minor}.{version.micro}")
    if not success:
        print(f"{RED}  Error: Python 3.7+ required{RESET}")
    return success

def check_dependencies():
    """Check if all required packages are installed"""
    print("\nChecking dependencies:")
    
    packages = {
        'torch': 'PyTorch',
        'torchvision': 'TorchVision',
        'clip': 'CLIP',
        'faiss': 'FAISS',
        'PIL': 'Pillow',
        'numpy': 'NumPy',
        'sklearn': 'scikit-learn',
        'tqdm': 'tqdm'
    }
    
    all_ok = True
    for module, name in packages.items():
        try:
            __import__(module)
            print(f"  {check_mark(True)} {name}")
        except ImportError:
            print(f"  {check_mark(False)} {name} {RED}(missing){RESET}")
            all_ok = False
    
    return all_ok

def check_cuda():
    """Check CUDA availability"""
    print("\nChecking CUDA:")
    try:
        import torch
        if torch.cuda.is_available():
            device_name = torch.cuda.get_device_name(0)
            print(f"  {check_mark(True)} CUDA available: {device_name}")
            print(f"  {GREEN}  CUDA version: {torch.version.cuda}{RESET}")
            return True
        else:
            print(f"  {check_mark(False)} CUDA not available (CPU mode)")
            print(f"  {YELLOW}  Training and search will be slower on CPU{RESET}")
            return False
    except Exception as e:
        print(f"  {check_mark(False)} Error checking CUDA: {e}")
        return False

def check_config():
    """Check config.py settings"""
    print("\nChecking configuration:")
    try:
        from config import MEVID_ROOT, ARTIFACTS, THUMBS_DIR
        
        print(f"  {check_mark(True)} config.py found")
        print(f"    MEVID_ROOT: {MEVID_ROOT}")
        print(f"    ARTIFACTS:  {ARTIFACTS}")
        print(f"    THUMBS_DIR: {THUMBS_DIR}")
        
        # Check if directories exist
        if MEVID_ROOT.exists():
            print(f"  {check_mark(True)} MEVID_ROOT exists")
        else:
            print(f"  {check_mark(False)} MEVID_ROOT not found")
            return False
        
        if ARTIFACTS.exists():
            print(f"  {check_mark(True)} ARTIFACTS directory exists")
        else:
            print(f"  {check_mark(False)} ARTIFACTS directory not found (will be created)")
            ARTIFACTS.mkdir(parents=True, exist_ok=True)
        
        return True
    except Exception as e:
        print(f"  {check_mark(False)} Error loading config: {e}")
        return False

def check_mevid_dataset():
    """Check MEvid dataset files"""
    print("\nChecking MEvid dataset:")
    
    try:
        from config import MEVID_ROOT
        
        required_files = {
            'test_name.txt': 'Test frame names',
            'track_test_info.txt': 'Test tracklet info',
            'query_IDX.txt': 'Query indices'
        }
        
        optional_files = {
            'train_name.txt': 'Train frame names',
            'track_train_info.txt': 'Train tracklet info'
        }
        
        all_ok = True
        for filename, description in required_files.items():
            filepath = MEVID_ROOT / filename
            exists = filepath.exists()
            print(f"  {check_mark(exists)} {description}: {filename}")
            if not exists:
                all_ok = False
        
        print("\n  Optional files (for training):")
        for filename, description in optional_files.items():
            filepath = MEVID_ROOT / filename
            exists = filepath.exists()
            status = "found" if exists else "not found"
            print(f"    {check_mark(exists)} {description}: {filename} ({status})")
        
        # Check for image directory
        bbox_test = MEVID_ROOT / "bbox_test"
        if bbox_test.exists():
            print(f"\n  {check_mark(True)} bbox_test directory found")
            # Count some images
            image_files = list(bbox_test.rglob("*.jpg"))
            print(f"    Found {len(image_files)} .jpg files")
        else:
            print(f"\n  {check_mark(False)} bbox_test directory not found")
            all_ok = False
        
        return all_ok
    except Exception as e:
        print(f"  {check_mark(False)} Error checking dataset: {e}")
        return False

def check_artifacts():
    """Check if preprocessed artifacts exist"""
    print("\nChecking preprocessed artifacts:")
    
    try:
        from config import ARTIFACTS
        
        artifacts = {
            'tracks_test.jsonl': 'Parsed tracklets',
            'query_idx.json': 'Query indices',
            'vecs_test.npy': 'CLIP features',
            'meta_test.pkl': 'Metadata',
            'faiss.index': 'FAISS index'
        }
        
        all_exist = True
        for filename, description in artifacts.items():
            filepath = ARTIFACTS / filename
            exists = filepath.exists()
            print(f"  {check_mark(exists)} {description}: {filename}")
            if not exists:
                all_exist = False
        
        if not all_exist:
            print(f"\n  {YELLOW}Run these scripts to build artifacts:")
            print(f"    python scripts/parse_mevid.py")
            print(f"    python scripts/extract_vecs.py")
            print(f"    python scripts/build_index.py{RESET}")
        
        # Check thumbnails
        thumbs_dir = ARTIFACTS / "thumbs"
        if thumbs_dir.exists():
            thumb_count = len(list(thumbs_dir.glob("*.gif")))
            print(f"\n  {check_mark(thumb_count > 0)} Thumbnails: {thumb_count} GIFs")
        else:
            print(f"\n  {check_mark(False)} Thumbnails directory not found")
            print(f"  {YELLOW}Run: python scripts/make_thumbs.py{RESET}")
        
        return all_exist
    except Exception as e:
        print(f"  {check_mark(False)} Error checking artifacts: {e}")
        return False

def check_models():
    """Check for trained models"""
    print("\nChecking trained models:")
    
    try:
        from config import ARTIFACTS
        
        reid_model = ARTIFACTS / "reid_model.pth"
        if reid_model.exists():
            import os
            size_mb = os.path.getsize(reid_model) / (1024 * 1024)
            print(f"  {check_mark(True)} Video ReID model: reid_model.pth ({size_mb:.1f} MB)")
        else:
            print(f"  {check_mark(False)} Video ReID model not found")
            print(f"  {YELLOW}Optional: Train with 'python scripts/train_reid.py'{RESET}")
        
        return True
    except Exception as e:
        print(f"  {check_mark(False)} Error checking models: {e}")
        return False

def print_summary(checks):
    """Print summary of checks"""
    print_header("SUMMARY")
    
    total = len(checks)
    passed = sum(checks.values())
    
    print(f"Checks passed: {passed}/{total}")
    
    if passed == total:
        print(f"\n{GREEN}✓ All checks passed! System is ready.{RESET}")
        print(f"\nTry running:")
        print(f"  python scripts/search_hybrid.py --query 'man in black jacket'")
    elif passed >= total - 2:
        print(f"\n{YELLOW}⚠ Most checks passed. System should work with minor issues.{RESET}")
    else:
        print(f"\n{RED}✗ Several checks failed. Please fix issues above.{RESET}")

def main():
    print_header("MEVID HYBRID SEARCH - SETUP VERIFICATION")
    
    checks = {}
    
    # Run all checks
    checks['python'] = check_python_version()
    checks['dependencies'] = check_dependencies()
    checks['cuda'] = check_cuda()
    checks['config'] = check_config()
    checks['dataset'] = check_mevid_dataset()
    checks['artifacts'] = check_artifacts()
    checks['models'] = check_models()
    
    # Print summary
    print_summary(checks)
    
    # Exit code
    sys.exit(0 if all(checks.values()) else 1)

if __name__ == "__main__":
    main()