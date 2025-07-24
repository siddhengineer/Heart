#!/usr/bin/env python3
"""
Setup and run script for CTENN Heart Sound Classification
This script checks dependencies and runs the training
"""

import sys
import os
import subprocess
import pkg_resources

def check_dependencies():
    """Check if all required packages are installed"""
    required_packages = [
        'torch>=1.9.0',
        'torchaudio>=0.9.0',
        'numpy>=1.19.0',
        'scikit-learn>=0.24.0',
        'matplotlib>=3.3.0',
        'seaborn>=0.11.0',
        'tqdm>=4.60.0',
        'scipy>=1.6.0',
        'pandas>=1.2.0'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            pkg_name = package.split('>=')[0]
            __import__(pkg_name)
            print(f"✅ {pkg_name} is installed")
        except ImportError:
            missing_packages.append(package)
            print(f"❌ {pkg_name} is missing")
    
    if missing_packages:
        print(f"\n🔧 Installing missing packages: {', '.join(missing_packages)}")
        for package in missing_packages:
            try:
                subprocess.check_call([sys.executable, "-m", "pip", "install", package])
                print(f"✅ Installed {package}")
            except subprocess.CalledProcessError as e:
                print(f"❌ Failed to install {package}: {e}")
                return False
    
    return True

def check_files():
    """Check if all required files exist"""
    required_files = [
        'config.py',
        'model.py',
        'dataset.py',
        'data_loader.py',
        'utils.py',
        'train.py'
    ]
    
    missing_files = []
    for file in required_files:
        if not os.path.exists(file):
            missing_files.append(file)
            print(f"❌ {file} is missing")
        else:
            print(f"✅ {file} found")
    
    if missing_files:
        print(f"\n❌ Missing files: {', '.join(missing_files)}")
        print("Please make sure all files are in the current directory")
        return False
    
    return True

def check_data_paths():
    """Check if data paths in config are valid"""
    try:
        from config import PHYSIONET_PATH, KAGGLE_PATH
        
        paths_exist = True
        if not os.path.exists(PHYSIONET_PATH):
            print(f"❌ PhysioNet path not found: {PHYSIONET_PATH}")
            paths_exist = False
        else:
            print(f"✅ PhysioNet path found: {PHYSIONET_PATH}")
            
        if not os.path.exists(KAGGLE_PATH):
            print(f"❌ Kaggle path not found: {KAGGLE_PATH}")
            paths_exist = False
        else:
            print(f"✅ Kaggle path found: {KAGGLE_PATH}")
            
        if not paths_exist:
            print("\n⚠️  Please update the paths in config.py to match your data location")
            
        return paths_exist
        
    except ImportError as e:
        print(f"❌ Could not import config: {e}")
        return False

def main():
    """Main setup and run function"""
    print("🚀 CTENN Heart Sound Classification Setup")
    print("=" * 50)
    
    # Check Python version
    if sys.version_info < (3, 7):
        print("❌ Python 3.7 or higher is required")
        sys.exit(1)
    else:
        print(f"✅ Python {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}")
    
    print("\n📦 Checking dependencies...")
    if not check_dependencies():
        print("❌ Dependency check failed")
        sys.exit(1)
    
    print("\n📁 Checking files...")
    if not check_files():
        print("❌ File check failed")
        sys.exit(1)
        
    print("\n📊 Checking data paths...")
    if not check_data_paths():
        print("⚠️  Data path check failed - please verify your data paths in config.py")
        response = input("Continue anyway? (y/n): ")
        if response.lower() != 'y':
            sys.exit(1)
    
    print("\n🎯 All checks passed!")
    print("🚀 Starting training...")
    print("=" * 50)
    
    try:
        # Import and run training
        from train import main as train_main
        train_main()
    except Exception as e:
        print(f"❌ Training failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()