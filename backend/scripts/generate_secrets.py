#!/usr/bin/env python3
"""
CPAS5 Security - Secret Generation Script
Generates secure secrets for JWT, encryption, and API keys

Usage:
    python scripts/generate_secrets.py
    
This will generate:
- JWT_SECRET (64 bytes)
- JWT_REFRESH_SECRET (64 bytes)
- ENCRYPTION_KEY (32 bytes)
- API_KEY (32 bytes)

The secrets will be printed to stdout. Copy them to your .env file.
"""

import secrets
import sys
from pathlib import Path


def generate_secret(length: int = 64) -> str:
    """Generate a URL-safe secret token"""
    return secrets.token_urlsafe(length)


def generate_hex_secret(length: int = 32) -> str:
    """Generate a hexadecimal secret"""
    return secrets.token_hex(length)


def main():
    """Generate all required secrets"""
    print("=" * 80)
    print("CPAS5 Security - Secret Generation")
    print("=" * 80)
    print()
    print("⚠️  IMPORTANT: Keep these secrets secure!")
    print("⚠️  Never commit these secrets to version control!")
    print("⚠️  Store them in environment variables or secure configuration files!")
    print()
    print("=" * 80)
    print()
    
    # Generate JWT secrets
    jwt_secret = generate_secret(64)
    jwt_refresh_secret = generate_secret(64)
    
    # Generate encryption key
    encryption_key = generate_secret(32)
    
    # Generate API key
    api_key = generate_secret(32)
    
    # Print secrets
    print("# JWT Configuration")
    print(f"JWT_SECRET={jwt_secret}")
    print(f"JWT_REFRESH_SECRET={jwt_refresh_secret}")
    print()
    
    print("# Encryption Configuration")
    print(f"ENCRYPTION_KEY={encryption_key}")
    print()
    
    print("# API Configuration")
    print(f"API_KEY={api_key}")
    print()
    
    print("=" * 80)
    print()
    print("✅ Secrets generated successfully!")
    print()
    print("📋 Next steps:")
    print("1. Copy the secrets above to your .env file")
    print("2. For production, set these as environment variables")
    print("3. For macOS app, add to ~/Library/Application Support/CPAS5/config/.env")
    print("4. Restart the backend server")
    print()
    print("=" * 80)
    
    # Optionally write to a file
    output_file = Path(__file__).parent.parent / ".env.secrets"
    
    print()
    response = input("Would you like to save these secrets to .env.secrets? (y/N): ")
    
    if response.lower() == 'y':
        with open(output_file, 'w') as f:
            f.write("# CPAS5 Security Secrets\n")
            f.write("# Generated: " + secrets.token_hex(8) + "\n")
            f.write("# IMPORTANT: Do not commit this file to version control!\n")
            f.write("\n")
            f.write("# JWT Configuration\n")
            f.write(f"JWT_SECRET={jwt_secret}\n")
            f.write(f"JWT_REFRESH_SECRET={jwt_refresh_secret}\n")
            f.write("\n")
            f.write("# Encryption Configuration\n")
            f.write(f"ENCRYPTION_KEY={encryption_key}\n")
            f.write("\n")
            f.write("# API Configuration\n")
            f.write(f"API_KEY={api_key}\n")
        
        print(f"✅ Secrets saved to {output_file}")
        print(f"⚠️  Remember to add .env.secrets to .gitignore!")
    else:
        print("ℹ️  Secrets not saved to file. Copy them manually to your .env file.")


if __name__ == "__main__":
    main()

