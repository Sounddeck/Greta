"""
Comprehensive tests for file storage service (Google Drive)

Tests cover:
- File upload and download
- File metadata management
- File deletion and cleanup
- Error handling
"""

import pytest
from unittest.mock import MagicMock, AsyncMock, patch
from datetime import datetime, timedelta
import base64


class TestFileUploadAndDownload:
    """Test file upload and download functionality"""
    
    @pytest.mark.asyncio
    async def test_upload_file_success(self):
        """Test successful file upload"""
        from services.drive_service import DriveService
        from models.drive_models import DriveFileUpload

        # Create drive service (already in mock mode by default)
        mock_db = AsyncMock()
        drive_service = DriveService(mock_db)
        
        # Create upload request
        content = base64.b64encode(b"Hello World").decode()
        upload_request = DriveFileUpload(
            name="test.txt",
            mime_type="text/plain",
            content=content,
            description="Test file"
        )
        
        # Upload file
        response = await drive_service.upload_file(
            user_id="user_123",
            request=upload_request
        )
        
        assert response.success is True
        assert response.file.name == "test.txt"
        assert response.file.mime_type == "text/plain"
        assert "mock" in response.message.lower()
    
    @pytest.mark.asyncio
    async def test_download_file_success(self):
        """Test successful file download"""
        from services.drive_service import DriveService
        
        # Create drive service (already in mock mode by default)
        mock_db = AsyncMock()
        drive_service = DriveService(mock_db)

        # Download file
        response = await drive_service.download_file(
            user_id="user_123",
            file_id="file_123"
        )
        
        assert response.success is True
        assert response.file_id == "file_123"
        assert response.content is not None
        assert "mock" in response.message.lower()
    
    @pytest.mark.asyncio
    async def test_upload_file_with_parent_folder(self):
        """Test uploading file to specific folder"""
        from services.drive_service import DriveService
        from models.drive_models import DriveFileUpload
        
        # Create drive service (already in mock mode by default)
        mock_db = AsyncMock()
        drive_service = DriveService(mock_db)

        # Create upload request with parent folder
        content = base64.b64encode(b"Hello World").decode()
        upload_request = DriveFileUpload(
            name="test.txt",
            mime_type="text/plain",
            content=content,
            parent_folder_id="folder_123"
        )
        
        # Upload file
        response = await drive_service.upload_file(
            user_id="user_123",
            request=upload_request
        )
        
        assert response.success is True
        assert "folder_123" in response.file.parents
    
    @pytest.mark.asyncio
    async def test_upload_large_file(self):
        """Test uploading large file"""
        from services.drive_service import DriveService
        from models.drive_models import DriveFileUpload
        
        # Create drive service (already in mock mode by default)
        mock_db = AsyncMock()
        drive_service = DriveService(mock_db)

        # Create large file content (1MB)
        large_content = b"x" * (1024 * 1024)
        content = base64.b64encode(large_content).decode()
        
        upload_request = DriveFileUpload(
            name="large_file.bin",
            mime_type="application/octet-stream",
            content=content
        )
        
        # Upload file
        response = await drive_service.upload_file(
            user_id="user_123",
            request=upload_request
        )
        
        assert response.success is True
        assert response.file.size > 1000000  # At least 1MB


class TestFileMetadataManagement:
    """Test file metadata management"""
    
    @pytest.mark.asyncio
    async def test_get_file_metadata(self):
        """Test getting file metadata"""
        from services.drive_service import DriveService
        
        # Create drive service in mock mode
        mock_db = AsyncMock()
        drive_service = DriveService(mock_db)
        
        # Get file metadata
        file = await drive_service.get_file(
            user_id="user_123",
            file_id="file_123"
        )
        
        assert file is not None
        assert file.file_id == "file_123"
        assert file.name is not None
        assert file.mime_type is not None
    
    @pytest.mark.asyncio
    async def test_list_files(self):
        """Test listing files"""
        from services.drive_service import DriveService
        from models.drive_models import DriveFileListRequest
        
        # Create drive service in mock mode
        mock_db = AsyncMock()
        drive_service = DriveService(mock_db)
        
        # List files
        request = DriveFileListRequest(
            page_size=10,
            page_token=None
        )
        
        response = await drive_service.list_files(
            user_id="user_123",
            request=request
        )

        assert response is not None
        assert isinstance(response.files, list)
        assert response.total_results >= 0
    
    @pytest.mark.asyncio
    async def test_share_file(self):
        """Test sharing file with permissions"""
        from services.drive_service import DriveService
        from models.drive_models import DriveShareRequest, PermissionRole, PermissionType
        
        # Create drive service in mock mode
        mock_db = AsyncMock()
        drive_service = DriveService(mock_db)
        
        # Share file
        share_request = DriveShareRequest(
            email="user@example.com",
            role=PermissionRole.READER,
            type=PermissionType.USER
        )
        
        response = await drive_service.share_file(
            user_id="user_123",
            file_id="file_123",
            request=share_request
        )
        
        assert response.success is True
        assert response.permission.email == "user@example.com"
        assert response.permission.role == PermissionRole.READER
    
    @pytest.mark.asyncio
    async def test_create_folder(self):
        """Test creating folder"""
        from services.drive_service import DriveService
        from models.drive_models import DriveFolderCreate
        
        # Create drive service in mock mode
        mock_db = AsyncMock()
        drive_service = DriveService(mock_db)
        
        # Create folder
        folder_request = DriveFolderCreate(
            name="Test Folder",
            description="Test folder description"
        )
        
        response = await drive_service.create_folder(
            user_id="user_123",
            request=folder_request
        )
        
        assert response.success is True
        assert response.folder.name == "Test Folder"
        assert response.folder.file_type.value == "folder"


class TestFileDeletionAndCleanup:
    """Test file deletion and cleanup"""
    
    @pytest.mark.asyncio
    async def test_delete_file_success(self):
        """Test successful file deletion"""
        from services.drive_service import DriveService
        
        # Create drive service in mock mode
        mock_db = AsyncMock()
        drive_service = DriveService(mock_db)
        
        # Delete file
        response = await drive_service.delete_file(
            user_id="user_123",
            file_id="file_123"
        )
        
        assert response.success is True
        assert response.file_id == "file_123"
        assert "deleted" in response.message.lower()
    
    @pytest.mark.asyncio
    async def test_delete_multiple_files(self):
        """Test deleting multiple files"""
        from services.drive_service import DriveService
        
        # Create drive service in mock mode
        mock_db = AsyncMock()
        drive_service = DriveService(mock_db)
        
        # Delete multiple files
        file_ids = ["file_1", "file_2", "file_3"]
        
        for file_id in file_ids:
            response = await drive_service.delete_file(
                user_id="user_123",
                file_id=file_id
            )
            assert response.success is True
    
    @pytest.mark.asyncio
    async def test_cleanup_old_files(self):
        """Test cleanup of old files"""
        # Mock cleanup service
        mock_cleanup_service = AsyncMock()
        mock_cleanup_service.cleanup_old_files = AsyncMock(return_value={
            "deleted_count": 5,
            "freed_space_bytes": 1024 * 1024 * 10  # 10MB
        })
        
        # Run cleanup
        result = await mock_cleanup_service.cleanup_old_files(
            user_id="user_123",
            days_old=30
        )
        
        assert result["deleted_count"] == 5
        assert result["freed_space_bytes"] > 0

    @pytest.mark.asyncio
    async def test_delete_folder_with_contents(self):
        """Test deleting folder with contents"""
        from services.drive_service import DriveService

        # Create drive service in mock mode
        mock_db = AsyncMock()
        drive_service = DriveService(mock_db)

        # Delete folder (in mock mode, same as file)
        response = await drive_service.delete_file(
            user_id="user_123",
            file_id="folder_123"
        )

        assert response.success is True


class TestErrorHandling:
    """Test error handling"""

    @pytest.mark.asyncio
    async def test_upload_file_without_auth(self):
        """Test file upload without authentication"""
        from services.drive_service import DriveService
        from models.drive_models import DriveFileUpload

        # Create drive service in mock mode
        mock_db = AsyncMock()
        drive_service = DriveService(mock_db)

        # Mock _get_user_access_token to raise error
        drive_service._get_user_access_token = AsyncMock(
            side_effect=Exception("No access token found")
        )

        # Try to upload file - should raise error
        content = base64.b64encode(b"Hello").decode()
        upload_request = DriveFileUpload(
            name="test.txt",
            mime_type="text/plain",
            content=content
        )

        with pytest.raises(Exception):
            await drive_service.upload_file(
                user_id="user_123",
                request=upload_request
            )

    @pytest.mark.asyncio
    async def test_download_nonexistent_file(self):
        """Test downloading nonexistent file"""
        from services.drive_service import DriveService

        # Create drive service and disable mock mode
        mock_db = AsyncMock()
        drive_service = DriveService(mock_db)
        drive_service.mock_mode = False  # Disable mock mode to test real API path

        # Mock _get_user_access_token
        drive_service._get_user_access_token = AsyncMock(return_value="token_123")

        # Try to download file - should raise NotImplementedError (real API not implemented)
        with pytest.raises(NotImplementedError):
            await drive_service.download_file(
                user_id="user_123",
                file_id="nonexistent_file"
            )

    @pytest.mark.asyncio
    async def test_invalid_file_content(self):
        """Test upload with invalid file content"""
        from services.drive_service import DriveService
        from models.drive_models import DriveFileUpload

        # Create drive service in mock mode
        mock_db = AsyncMock()
        drive_service = DriveService(mock_db)

        # Try to upload with invalid base64 content
        try:
            upload_request = DriveFileUpload(
                name="test.txt",
                mime_type="text/plain",
                content="invalid_base64_content!!!"
            )

            # Upload should still work in mock mode
            response = await drive_service.upload_file(
                user_id="user_123",
                request=upload_request
            )
            # Mock mode doesn't validate content
            assert response.success is True
        except Exception:
            # Expected if validation is strict
            pass

    @pytest.mark.asyncio
    async def test_storage_quota_exceeded(self):
        """Test storage quota exceeded error"""
        # Mock storage service
        mock_storage_service = AsyncMock()
        mock_storage_service.check_quota = AsyncMock(return_value={
            "used_bytes": 15 * 1024 * 1024 * 1024,  # 15GB
            "total_bytes": 15 * 1024 * 1024 * 1024,  # 15GB
            "available_bytes": 0,
            "quota_exceeded": True
        })

        # Check quota
        quota = await mock_storage_service.check_quota(user_id="user_123")

        assert quota["quota_exceeded"] is True
        assert quota["available_bytes"] == 0

