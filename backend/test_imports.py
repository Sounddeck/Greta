#!/usr/bin/env python3
"""Test if basic imports work"""
import sys
print("Python version:", sys.version)

try:
    import fastapi
    print("✓ fastapi imported")
except Exception as e:
    print(f"✗ fastapi failed: {e}")
    sys.exit(1)

try:
    import uvicorn
    print("✓ uvicorn imported")
except Exception as e:
    print(f"✗ uvicorn failed: {e}")
    sys.exit(1)

try:
    import pydantic
    print("✓ pydantic imported")
except Exception as e:
    print(f"✗ pydantic failed: {e}")
    sys.exit(1)

try:
    import motor
    print("✓ motor imported")
except Exception as e:
    print(f"✗ motor failed: {e}")
    sys.exit(1)

try:
    import slowapi
    print("✓ slowapi imported")
except Exception as e:
    print(f"✗ slowapi failed: {e}")
    sys.exit(1)

print("\nAll imports successful!")

