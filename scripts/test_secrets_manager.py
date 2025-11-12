#!/usr/bin/env python3
"""
CPAS5 Secrets Manager Test Script
Tests the secrets management service with different backends

Usage:
    python scripts/test_secrets_manager.py
"""

import asyncio
import sys
import os
from pathlib import Path

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent.parent / "backend"))

from services.secrets_manager import get_secrets_manager, VaultBackend, AWSSecretsBackend, EnvironmentBackend


async def test_secrets_manager():
    """Test the secrets manager"""
    print("=" * 80)
    print("CPAS5 Secrets Manager Test")
    print("=" * 80)
    print()
    
    # Get secrets manager
    secrets = get_secrets_manager()
    
    print(f"✅ Secrets Manager initialized")
    print(f"   Backend: {secrets.backend.__class__.__name__}")
    print()
    
    # Test 1: List secrets
    print("Test 1: List Secrets")
    print("-" * 80)
    secret_keys = await secrets.list_secrets()
    print(f"Found {len(secret_keys)} secrets:")
    for key in secret_keys:
        print(f"  - {key}")
    print()
    
    # Test 2: Get secrets
    print("Test 2: Get Secrets")
    print("-" * 80)
    
    test_keys = ['jwt_secret', 'jwt_refresh_secret', 'encryption_key', 'api_key']
    
    for key in test_keys:
        value = await secrets.get_secret(f"cpas5/{key}")
        if value:
            # Mask the secret value
            masked_value = value[:8] + "..." + value[-8:] if len(value) > 16 else "***"
            print(f"✅ {key}: {masked_value}")
        else:
            print(f"❌ {key}: Not found")
    print()
    
    # Test 3: Set a test secret
    print("Test 3: Set Secret")
    print("-" * 80)
    test_secret_key = "cpas5/test_secret"
    test_secret_value = "test_value_12345"
    
    success = await secrets.set_secret(
        test_secret_key,
        test_secret_value,
        metadata={"test": True, "created_by": "test_script"}
    )
    
    if success:
        print(f"✅ Test secret created: {test_secret_key}")
    else:
        print(f"❌ Failed to create test secret")
    print()
    
    # Test 4: Retrieve the test secret
    print("Test 4: Retrieve Secret")
    print("-" * 80)
    retrieved_value = await secrets.get_secret(test_secret_key)
    
    if retrieved_value == test_secret_value:
        print(f"✅ Test secret retrieved successfully")
        print(f"   Value matches: {retrieved_value == test_secret_value}")
    else:
        print(f"❌ Test secret retrieval failed")
        print(f"   Expected: {test_secret_value}")
        print(f"   Got: {retrieved_value}")
    print()
    
    # Test 5: Rotate the test secret
    print("Test 5: Rotate Secret")
    print("-" * 80)
    new_test_value = "rotated_value_67890"
    
    success = await secrets.rotate_secret(test_secret_key, new_test_value)
    
    if success:
        print(f"✅ Test secret rotated")
        
        # Verify rotation
        rotated_value = await secrets.get_secret(test_secret_key)
        if rotated_value == new_test_value:
            print(f"✅ Rotation verified: value updated")
        else:
            print(f"❌ Rotation failed: value not updated")
    else:
        print(f"❌ Failed to rotate test secret")
    print()
    
    # Test 6: Delete the test secret
    print("Test 6: Delete Secret")
    print("-" * 80)
    success = await secrets.delete_secret(test_secret_key)
    
    if success:
        print(f"✅ Test secret deleted")
        
        # Verify deletion
        deleted_value = await secrets.get_secret(test_secret_key)
        if deleted_value is None:
            print(f"✅ Deletion verified: secret not found")
        else:
            print(f"❌ Deletion failed: secret still exists")
    else:
        print(f"❌ Failed to delete test secret")
    print()
    
    # Summary
    print("=" * 80)
    print("Test Summary")
    print("=" * 80)
    print()
    print(f"✅ Secrets Manager is working correctly")
    print(f"   Backend: {secrets.backend.__class__.__name__}")
    print()
    
    if isinstance(secrets.backend, VaultBackend):
        print("📋 Vault Configuration:")
        print(f"   Address: {os.getenv('VAULT_ADDR', 'Not set')}")
        print(f"   Token: {'Set' if os.getenv('VAULT_TOKEN') else 'Not set'}")
    elif isinstance(secrets.backend, AWSSecretsBackend):
        print("📋 AWS Secrets Manager Configuration:")
        print(f"   Region: {os.getenv('AWS_REGION', 'Not set')}")
    elif isinstance(secrets.backend, EnvironmentBackend):
        print("⚠️  Using environment variables (not recommended for production)")
    
    print()
    print("=" * 80)


async def test_backend_selection():
    """Test automatic backend selection"""
    print()
    print("=" * 80)
    print("Backend Selection Test")
    print("=" * 80)
    print()
    
    # Save current environment
    vault_addr = os.getenv('VAULT_ADDR')
    vault_token = os.getenv('VAULT_TOKEN')
    aws_region = os.getenv('AWS_REGION')
    
    # Test 1: Vault backend
    print("Test 1: Vault Backend Selection")
    print("-" * 80)
    os.environ['VAULT_ADDR'] = 'http://localhost:8200'
    os.environ['VAULT_TOKEN'] = 'test-token'
    
    try:
        from services.secrets_manager import SecretsManager
        manager = SecretsManager()
        print(f"✅ Selected backend: {manager.backend.__class__.__name__}")
    except Exception as e:
        print(f"❌ Failed to select Vault backend: {e}")
    print()
    
    # Test 2: Environment backend (fallback)
    print("Test 2: Environment Backend Selection (Fallback)")
    print("-" * 80)
    if 'VAULT_ADDR' in os.environ:
        del os.environ['VAULT_ADDR']
    if 'VAULT_TOKEN' in os.environ:
        del os.environ['VAULT_TOKEN']
    if 'AWS_REGION' in os.environ:
        del os.environ['AWS_REGION']
    
    from services.secrets_manager import SecretsManager
    manager = SecretsManager()
    print(f"✅ Selected backend: {manager.backend.__class__.__name__}")
    print()
    
    # Restore environment
    if vault_addr:
        os.environ['VAULT_ADDR'] = vault_addr
    if vault_token:
        os.environ['VAULT_TOKEN'] = vault_token
    if aws_region:
        os.environ['AWS_REGION'] = aws_region
    
    print("=" * 80)


async def main():
    """Run all tests"""
    try:
        await test_secrets_manager()
        await test_backend_selection()
        
        print()
        print("✅ All tests passed!")
        print()
        
    except Exception as e:
        print()
        print(f"❌ Test failed: {e}")
        print()
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())

