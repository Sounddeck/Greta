"""
Google Drive Module Tests
Tests for Drive models, service, and router
"""

import sys
import os
import asyncio
import base64

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from database import Database
from services.drive_service import DriveService
from services.oauth_service import OAuthService
from models.drive_models import (
    DriveFileUpload, DriveFileListRequest,
    DriveShareRequest, DriveFolderCreate,
    PermissionRole, PermissionType
)


class DriveModuleTests:
    """Drive module test suite"""
    
    def __init__(self):
        self.db = Database()
        self.oauth_service = OAuthService(self.db)
        self.service = DriveService(self.db, self.oauth_service)
        self.test_user_id = "test_drive_user"
    
    async def test_1_service_initialization(self):
        """Test 1: Service initialization"""
        print("\n" + "="*80)
        print("TEST 1: Drive Service Initialization")
        print("="*80)
        
        assert self.service is not None
        assert self.service.mock_mode is True
        
        print("✅ Drive service initialized successfully")
        print(f"   - Mock mode: {self.service.mock_mode}")
    
    async def test_2_upload_file(self):
        """Test 2: Upload file to Drive"""
        print("\n" + "="*80)
        print("TEST 2: Upload File to Drive")
        print("="*80)
        
        content = "Hello, this is a test file!"
        encoded_content = base64.b64encode(content.encode()).decode()
        
        request = DriveFileUpload(
            name="test_document.txt",
            mime_type="text/plain",
            content=encoded_content,
            description="Test document for upload"
        )
        
        response = await self.service.upload_file(self.test_user_id, request)
        
        assert response.success is True
        assert response.file.name == "test_document.txt"
        assert response.file.mime_type == "text/plain"
        
        print("✅ File uploaded successfully")
        print(f"   - File ID: {response.file.file_id}")
        print(f"   - Name: {response.file.name}")
        print(f"   - Size: {response.file.size} bytes")
        print(f"   - Web view link: {response.file.web_view_link}")
    
    async def test_3_download_file(self):
        """Test 3: Download file from Drive"""
        print("\n" + "="*80)
        print("TEST 3: Download File from Drive")
        print("="*80)
        
        file_id = "test_file_123"
        response = await self.service.download_file(self.test_user_id, file_id)
        
        assert response.success is True
        assert response.file_id == file_id
        assert response.content is not None
        
        # Decode content
        decoded_content = base64.b64decode(response.content).decode()
        
        print("✅ File downloaded successfully")
        print(f"   - File ID: {response.file_id}")
        print(f"   - Name: {response.name}")
        print(f"   - Size: {response.size} bytes")
        print(f"   - Content preview: {decoded_content[:50]}...")
    
    async def test_4_list_files(self):
        """Test 4: List files from Drive"""
        print("\n" + "="*80)
        print("TEST 4: List Files from Drive")
        print("="*80)
        
        request = DriveFileListRequest(
            max_results=5
        )
        
        response = await self.service.list_files(self.test_user_id, request)
        
        assert len(response.files) > 0
        assert response.total_results > 0
        
        print("✅ Files listed successfully")
        print(f"   - Files count: {len(response.files)}")
        print(f"   - Total results: {response.total_results}")
        
        for i, file in enumerate(response.files[:3], 1):
            print(f"\n   File {i}:")
            print(f"   - ID: {file.file_id}")
            print(f"   - Name: {file.name}")
            print(f"   - Size: {file.size} bytes")
            print(f"   - Type: {file.file_type}")
    
    async def test_5_get_file_metadata(self):
        """Test 5: Get file metadata"""
        print("\n" + "="*80)
        print("TEST 5: Get File Metadata")
        print("="*80)
        
        file_id = "test_file_123"
        file = await self.service.get_file(self.test_user_id, file_id)
        
        assert file.file_id == file_id
        assert file.name is not None
        assert file.mime_type is not None
        
        print("✅ File metadata retrieved successfully")
        print(f"   - File ID: {file.file_id}")
        print(f"   - Name: {file.name}")
        print(f"   - MIME type: {file.mime_type}")
        print(f"   - Size: {file.size} bytes")
        print(f"   - Starred: {file.starred}")
        print(f"   - Owner: {file.owner_email}")
    
    async def test_6_delete_file(self):
        """Test 6: Delete file from Drive"""
        print("\n" + "="*80)
        print("TEST 6: Delete File from Drive")
        print("="*80)
        
        file_id = "test_file_to_delete"
        response = await self.service.delete_file(self.test_user_id, file_id)
        
        assert response.success is True
        assert response.file_id == file_id
        
        print("✅ File deleted successfully")
        print(f"   - File ID: {response.file_id}")
        print(f"   - Message: {response.message}")
    
    async def test_7_share_file(self):
        """Test 7: Share file with permissions"""
        print("\n" + "="*80)
        print("TEST 7: Share File with Permissions")
        print("="*80)
        
        file_id = "test_file_123"
        request = DriveShareRequest(
            email="colleague@example.com",
            role=PermissionRole.WRITER,
            type=PermissionType.USER,
            send_notification=True,
            message="Please review this document"
        )
        
        response = await self.service.share_file(self.test_user_id, file_id, request)
        
        assert response.success is True
        assert response.permission.email == "colleague@example.com"
        assert response.permission.role == PermissionRole.WRITER
        
        print("✅ File shared successfully")
        print(f"   - Permission ID: {response.permission.permission_id}")
        print(f"   - Email: {response.permission.email}")
        print(f"   - Role: {response.permission.role}")
        print(f"   - Type: {response.permission.type}")
    
    async def test_8_create_folder(self):
        """Test 8: Create folder in Drive"""
        print("\n" + "="*80)
        print("TEST 8: Create Folder in Drive")
        print("="*80)
        
        request = DriveFolderCreate(
            name="Project Documents",
            description="All project-related documents"
        )
        
        response = await self.service.create_folder(self.test_user_id, request)
        
        assert response.success is True
        assert response.folder.name == "Project Documents"
        assert response.folder.file_type.value == "folder"
        
        print("✅ Folder created successfully")
        print(f"   - Folder ID: {response.folder.file_id}")
        print(f"   - Name: {response.folder.name}")
        print(f"   - Web view link: {response.folder.web_view_link}")
    
    async def test_9_drive_health(self):
        """Test 9: Drive service health check"""
        print("\n" + "="*80)
        print("TEST 9: Drive Service Health Check")
        print("="*80)
        
        response = await self.service.get_health(self.test_user_id)
        
        assert response.status is not None
        assert response.timestamp is not None
        
        print("✅ Drive health check successful")
        print(f"   - Status: {response.status}")
        print(f"   - Message: {response.message}")
        print(f"   - Authenticated: {response.authenticated}")
        if response.storage_quota:
            print(f"   - Storage quota: {response.storage_quota}")
    
    async def run_all_tests(self):
        """Run all tests"""
        print("\n" + "="*80)
        print("GOOGLE DRIVE MODULE TESTS")
        print("="*80)
        
        tests = [
            self.test_1_service_initialization,
            self.test_2_upload_file,
            self.test_3_download_file,
            self.test_4_list_files,
            self.test_5_get_file_metadata,
            self.test_6_delete_file,
            self.test_7_share_file,
            self.test_8_create_folder,
            self.test_9_drive_health
        ]
        
        passed = 0
        failed = 0
        
        for test in tests:
            try:
                await test()
                passed += 1
            except AssertionError as e:
                print(f"❌ Test failed: {e}")
                failed += 1
            except Exception as e:
                print(f"❌ Test error: {e}")
                failed += 1
        
        print("\n" + "="*80)
        print("DRIVE MODULE TESTS COMPLETE")
        print("="*80)
        print(f"\n{'='*80}")
        if failed == 0:
            print(f"✅ ALL TESTS PASSED ({passed}/{passed + failed})")
        else:
            print(f"❌ SOME TESTS FAILED ({passed}/{passed + failed})")
        print(f"{'='*80}\n")


def main():
    """Main test runner"""
    tests = DriveModuleTests()
    asyncio.get_event_loop().run_until_complete(tests.run_all_tests())


if __name__ == "__main__":
    main()

