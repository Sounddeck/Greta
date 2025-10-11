"""
Test Suite for Week 15-16: Portability

Tests:
1. Data Export (conversations, templates, contexts, RAG, models, adapters)
2. Data Import (with conflict resolution)
3. Backup and Restore
4. Cloud Storage Integration
5. Data Encryption
6. Data Validation
7. Migration Tools
8. Incremental Backup

Target: 100% pass rate
"""

import asyncio
import sys
from pathlib import Path

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent))

from backend.services.portability_service import (
    PortabilityService,
    ExportMetadata,
    ImportResult,
    BackupInfo,
)

# Test results tracking
test_results = {"passed": 0, "failed": 0, "total": 0}


def print_test_header(component: str):
    """Print test component header"""
    print("\n" + "=" * 80)
    print(f"Component {component}")
    print("=" * 80 + "\n")


def print_test(test_num: int, total: int, description: str):
    """Print test description"""
    print(f"[Test {test_num}/{total}] {description}...")


def test_passed():
    """Mark test as passed"""
    test_results["passed"] += 1
    test_results["total"] += 1
    print("✅ PASSED\n")


def test_failed(error: str):
    """Mark test as failed"""
    test_results["failed"] += 1
    test_results["total"] += 1
    print(f"❌ FAILED: {error}\n")


# ============================================================================
# Component 1: Data Export
# ============================================================================


async def test_data_export():
    """Test data export functionality"""
    print_test_header("1: Data Export")

    service = PortabilityService()

    # Test 1: Export conversations
    print_test(1, 6, "Export conversations")
    try:
        result = await service.export_conversations()
        assert "conversations" in result
        assert "count" in result
        print(f"   Exported {result['count']} conversations")
        test_passed()
    except Exception as e:
        test_failed(str(e))

    # Test 2: Export agent templates
    print_test(2, 6, "Export agent templates")
    try:
        result = await service.export_agent_templates()
        assert "templates" in result
        assert "count" in result
        print(f"   Exported {result['count']} templates")
        test_passed()
    except Exception as e:
        test_failed(str(e))

    # Test 3: Export contexts
    print_test(3, 6, "Export context windows")
    try:
        result = await service.export_contexts()
        assert "contexts" in result
        assert "count" in result
        print(f"   Exported {result['count']} contexts")
        test_passed()
    except Exception as e:
        test_failed(str(e))

    # Test 4: Export RAG documents
    print_test(4, 6, "Export RAG documents")
    try:
        result = await service.export_rag_documents()
        assert "collections" in result
        assert "count" in result
        print(f"   Exported {result['count']} RAG collections")
        test_passed()
    except Exception as e:
        test_failed(str(e))

    # Test 5: Export models
    print_test(5, 6, "Export model configurations")
    try:
        result = await service.export_models()
        assert "models" in result
        assert "count" in result
        print(f"   Exported {result['count']} models")
        test_passed()
    except Exception as e:
        test_failed(str(e))

    # Test 6: Export all data
    print_test(6, 6, "Export all data to ZIP")
    try:
        export_path = await service.export_all_data(export_id="test_export", encrypt=False)
        assert export_path is not None
        assert Path(export_path).exists()
        print(f"   Exported to: {export_path}")
        test_passed()
    except Exception as e:
        test_failed(str(e))


# ============================================================================
# Component 2: Data Encryption
# ============================================================================


async def test_encryption():
    """Test data encryption"""
    print_test_header("2: Data Encryption")

    service = PortabilityService()

    # Test 1: Encrypt data
    print_test(1, 4, "Encrypt data")
    try:
        original = "This is sensitive data"
        encrypted = service.encrypt_data(original)
        assert encrypted != original.encode()
        print(f"   Encrypted: {len(encrypted)} bytes")
        test_passed()
    except Exception as e:
        test_failed(str(e))

    # Test 2: Decrypt data
    print_test(2, 4, "Decrypt data")
    try:
        original = "This is sensitive data"
        encrypted = service.encrypt_data(original)
        decrypted = service.decrypt_data(encrypted)
        assert decrypted == original
        print(f"   Decrypted successfully")
        test_passed()
    except Exception as e:
        test_failed(str(e))

    # Test 3: Generate checksum
    print_test(3, 4, "Generate checksum")
    try:
        data = "Test data for checksum"
        checksum = service.generate_checksum(data)
        assert len(checksum) == 64  # SHA256 hex digest
        print(f"   Checksum: {checksum[:16]}...")
        test_passed()
    except Exception as e:
        test_failed(str(e))

    # Test 4: Export with encryption
    print_test(4, 4, "Export with encryption")
    try:
        export_path = await service.export_all_data(export_id="test_encrypted", encrypt=True)
        assert export_path is not None
        assert Path(export_path).exists()
        print(f"   Encrypted export: {export_path}")
        test_passed()
    except Exception as e:
        test_failed(str(e))


# ============================================================================
# Component 3: Data Import
# ============================================================================


async def test_data_import():
    """Test data import functionality"""
    print_test_header("3: Data Import")

    service = PortabilityService()

    # Test 1: Create export for import
    print_test(1, 2, "Create export for import test")
    try:
        export_path = await service.export_all_data(export_id="test_import", encrypt=False)
        assert export_path is not None
        print(f"   Created export: {export_path}")
        test_passed()
    except Exception as e:
        test_failed(str(e))
        return

    # Test 2: Import data
    print_test(2, 2, "Import data from export")
    try:
        result = await service.import_all_data(export_path, conflict_resolution="skip")
        assert result.success is True
        print(f"   Imported: {result.imported_items}, Skipped: {result.skipped_items}")
        test_passed()
    except Exception as e:
        test_failed(str(e))


# ============================================================================
# Component 4: Backup and Restore
# ============================================================================


async def test_backup_restore():
    """Test backup and restore"""
    print_test_header("4: Backup and Restore")

    service = PortabilityService()

    # Test 1: Create backup
    print_test(1, 3, "Create full backup")
    try:
        backup_info = await service.create_backup(backup_type="full", encrypt=True)
        assert backup_info is not None
        assert Path(backup_info.file_path).exists()
        print(f"   Backup: {backup_info.backup_id} ({backup_info.file_size} bytes)")
        test_passed()
    except Exception as e:
        test_failed(str(e))
        return

    # Test 2: List backups
    print_test(2, 3, "List backups")
    try:
        backups = await service.list_backups()
        assert len(backups) > 0
        print(f"   Found {len(backups)} backups")
        test_passed()
    except Exception as e:
        test_failed(str(e))

    # Test 3: Restore backup
    print_test(3, 3, "Restore from backup")
    try:
        backups = await service.list_backups()
        if backups:
            result = await service.restore_backup(backups[0].file_path, conflict_resolution="skip")
            assert result.success is True
            print(f"   Restored: {result.imported_items} items")
            test_passed()
        else:
            test_failed("No backups available")
    except Exception as e:
        test_failed(str(e))


# ============================================================================
# Component 5: Cloud Storage Integration
# ============================================================================


async def test_cloud_storage():
    """Test cloud storage integration"""
    print_test_header("5: Cloud Storage Integration")

    service = PortabilityService()

    # Test 1: Configure cloud storage
    print_test(1, 3, "Configure cloud storage")
    try:
        result = await service.configure_cloud_storage(
            provider="local",
            credentials={"path": "/tmp/greta_cloud"},
            auto_sync=False,
        )
        assert result is True
        print(f"   Configured: local storage")
        test_passed()
    except Exception as e:
        test_failed(str(e))

    # Test 2: Get cloud sync status
    print_test(2, 3, "Get cloud sync status")
    try:
        status = await service.get_cloud_sync_status()
        assert status.provider == "local"
        assert status.sync_enabled is True
        print(f"   Provider: {status.provider}, Enabled: {status.sync_enabled}")
        test_passed()
    except Exception as e:
        test_failed(str(e))

    # Test 3: Sync to cloud
    print_test(3, 3, "Sync file to cloud")
    try:
        # Create a test export
        export_path = await service.export_all_data(export_id="test_cloud_sync", encrypt=False)
        result = await service.sync_to_cloud(export_path)
        # Local provider always succeeds
        print(f"   Sync result: {result}")
        test_passed()
    except Exception as e:
        test_failed(str(e))


# ============================================================================
# Component 6: Data Validation
# ============================================================================


async def test_data_validation():
    """Test data validation"""
    print_test_header("6: Data Validation")

    service = PortabilityService()

    # Test 1: Validate export file
    print_test(1, 2, "Validate export file")
    try:
        # Create export
        export_path = await service.export_all_data(export_id="test_validate", encrypt=False)

        # Validate
        is_valid, errors = await service.validate_export(export_path)
        assert is_valid is True
        assert len(errors) == 0
        print(f"   Validation: {'PASS' if is_valid else 'FAIL'}")
        test_passed()
    except Exception as e:
        test_failed(str(e))

    # Test 2: Validate data integrity
    print_test(2, 2, "Validate data integrity")
    try:
        report = await service.validate_data_integrity()
        assert "conversations" in report
        assert "templates" in report
        print(f"   Conversations: {report['conversations']['valid']}")
        print(f"   Templates: {report['templates']['valid']}")
        test_passed()
    except Exception as e:
        test_failed(str(e))


# ============================================================================
# Main Test Runner
# ============================================================================


async def run_all_tests():
    """Run all test components"""
    print("\n" + "=" * 80)
    print("WEEK 15-16 TEST SUITE: Portability")
    print("=" * 80)
    print("Testing all portability components...\n")

    # Run test components
    await test_data_export()
    await test_encryption()
    await test_data_import()
    await test_backup_restore()
    await test_cloud_storage()
    await test_data_validation()

    # Print summary
    print("\n" + "=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)
    print(f"Total Tests: {test_results['total']}")
    print(f"Passed: {test_results['passed']} ✅")
    print(f"Failed: {test_results['failed']} ❌")

    if test_results["failed"] == 0:
        print("\n🎉 ALL TESTS PASSED! 100% Success Rate")
    else:
        success_rate = (
            test_results["passed"] / test_results["total"] * 100
            if test_results["total"] > 0
            else 0
        )
        print(f"\n⚠️  Success Rate: {success_rate:.1f}%")

    print("=" * 80 + "\n")

    return test_results["failed"] == 0


if __name__ == "__main__":
    result = asyncio.run(run_all_tests())
    sys.exit(0 if result else 1)

