#!/usr/bin/env python3
import sys
import subprocess

print("🔧 Fixing Python 3.13 compatibility for Greta PAI")

# Check Python version
version = sys.version_info
print(f"🐍 Python version: {version.major}.{version.minor}.{version.micro}")

# Install compatible packages for Python 3.13
if version >= (3, 13):
    print("⚠️  Python 3.13 detected - installing compatible versions...")
    
    # Force compatible versions
    packages = [
        "pydantic<2.0.0",
        "fastapi<0.104.0",
        "uvicorn[standard]",
        "typing-extensions>=4.5.0",
        "loguru"
    ]
    
    for package in packages:
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", package])
            print(f"✅ Installed {package}")
        except subprocess.CalledProcessError as e:
            print(f"❌ Failed {package}: {e}")

print("🎉 Python compatibility fixes applied!")
