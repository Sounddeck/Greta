"""
File Management Router - Download and manage generated files

This router provides endpoints for:
- Downloading generated files (PDF, Excel, CSV, charts)
- Listing all generated files
- Deleting generated files
- Getting file metadata
"""

from fastapi import APIRouter, HTTPException, Response
from fastapi.responses import FileResponse, JSONResponse
from pathlib import Path
from typing import List, Dict, Any, Optional
import os
import mimetypes
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/files", tags=["File Management"])

# Base directory for generated files
GENERATED_FILES_DIR = Path(__file__).parent.parent / "generated_files"

# Ensure the directory exists
GENERATED_FILES_DIR.mkdir(exist_ok=True)

# MIME type mappings
MIME_TYPES = {
    '.pdf': 'application/pdf',
    '.xlsx': 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
    '.csv': 'text/csv',
    '.png': 'image/png',
    '.jpg': 'image/jpeg',
    '.jpeg': 'image/jpeg',
    '.html': 'text/html',
    '.json': 'application/json',
}


@router.get("/download/{filename}")
async def download_file(
    filename: str,
    inline: bool = False
):
    """
    Download a generated file by filename.
    
    Args:
        filename: Name of the file to download
        inline: If True, display inline in browser; if False, force download
        
    Returns:
        FileResponse with the requested file
        
    Raises:
        HTTPException: If file not found or invalid filename
    """
    try:
        # Security: Prevent directory traversal attacks
        if '..' in filename or '/' in filename or '\\' in filename:
            logger.warning(f"⚠️ Invalid filename attempted: {filename}")
            raise HTTPException(status_code=400, detail="Invalid filename")
        
        # Construct file path
        file_path = GENERATED_FILES_DIR / filename
        
        # Check if file exists
        if not file_path.exists():
            logger.warning(f"⚠️ File not found: {file_path}")
            raise HTTPException(status_code=404, detail=f"File not found: {filename}")
        
        # Check if it's actually a file (not a directory)
        if not file_path.is_file():
            logger.warning(f"⚠️ Path is not a file: {file_path}")
            raise HTTPException(status_code=400, detail="Invalid file path")
        
        # Get MIME type
        file_extension = file_path.suffix.lower()
        mime_type = MIME_TYPES.get(file_extension, 'application/octet-stream')
        
        # Determine content disposition
        disposition = "inline" if inline else "attachment"
        
        logger.info(f"📥 Serving file: {filename} ({mime_type}, {disposition})")
        
        return FileResponse(
            path=str(file_path),
            media_type=mime_type,
            filename=filename,
            headers={
                "Content-Disposition": f'{disposition}; filename="{filename}"'
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Error downloading file {filename}: {e}")
        raise HTTPException(status_code=500, detail=f"Error downloading file: {str(e)}")


@router.get("/list")
async def list_files(
    file_type: Optional[str] = None,
    limit: int = 100,
    offset: int = 0
) -> Dict[str, Any]:
    """
    List all generated files with metadata.
    
    Args:
        file_type: Filter by file extension (e.g., 'pdf', 'csv', 'xlsx')
        limit: Maximum number of files to return
        offset: Number of files to skip
        
    Returns:
        Dictionary with file list and metadata
    """
    try:
        # Get all files in the directory
        all_files = []
        
        for file_path in GENERATED_FILES_DIR.iterdir():
            if file_path.is_file():
                # Filter by file type if specified
                if file_type and not file_path.suffix.lower().endswith(f'.{file_type.lower()}'):
                    continue
                
                # Get file stats
                stats = file_path.stat()
                
                all_files.append({
                    'filename': file_path.name,
                    'size': stats.st_size,
                    'size_human': _format_file_size(stats.st_size),
                    'created_at': datetime.fromtimestamp(stats.st_ctime).isoformat(),
                    'modified_at': datetime.fromtimestamp(stats.st_mtime).isoformat(),
                    'extension': file_path.suffix.lower(),
                    'mime_type': MIME_TYPES.get(file_path.suffix.lower(), 'application/octet-stream'),
                    'download_url': f'/api/v1/files/download/{file_path.name}',
                    'path': str(file_path.absolute())
                })
        
        # Sort by modified time (newest first)
        all_files.sort(key=lambda x: x['modified_at'], reverse=True)
        
        # Apply pagination
        total_count = len(all_files)
        paginated_files = all_files[offset:offset + limit]
        
        logger.info(f"📋 Listed {len(paginated_files)} files (total: {total_count})")
        
        return {
            'files': paginated_files,
            'total_count': total_count,
            'limit': limit,
            'offset': offset,
            'storage_path': str(GENERATED_FILES_DIR.absolute())
        }
        
    except Exception as e:
        logger.error(f"❌ Error listing files: {e}")
        raise HTTPException(status_code=500, detail=f"Error listing files: {str(e)}")


@router.get("/info/{filename}")
async def get_file_info(filename: str) -> Dict[str, Any]:
    """
    Get detailed information about a specific file.
    
    Args:
        filename: Name of the file
        
    Returns:
        Dictionary with file metadata
    """
    try:
        # Security: Prevent directory traversal attacks
        if '..' in filename or '/' in filename or '\\' in filename:
            raise HTTPException(status_code=400, detail="Invalid filename")
        
        file_path = GENERATED_FILES_DIR / filename
        
        if not file_path.exists():
            raise HTTPException(status_code=404, detail=f"File not found: {filename}")
        
        if not file_path.is_file():
            raise HTTPException(status_code=400, detail="Invalid file path")
        
        stats = file_path.stat()
        
        return {
            'filename': file_path.name,
            'size': stats.st_size,
            'size_human': _format_file_size(stats.st_size),
            'created_at': datetime.fromtimestamp(stats.st_ctime).isoformat(),
            'modified_at': datetime.fromtimestamp(stats.st_mtime).isoformat(),
            'accessed_at': datetime.fromtimestamp(stats.st_atime).isoformat(),
            'extension': file_path.suffix.lower(),
            'mime_type': MIME_TYPES.get(file_path.suffix.lower(), 'application/octet-stream'),
            'download_url': f'/api/v1/files/download/{file_path.name}',
            'path': str(file_path.absolute()),
            'is_readable': os.access(file_path, os.R_OK),
            'is_writable': os.access(file_path, os.W_OK)
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Error getting file info for {filename}: {e}")
        raise HTTPException(status_code=500, detail=f"Error getting file info: {str(e)}")


@router.delete("/delete/{filename}")
async def delete_file(filename: str) -> Dict[str, Any]:
    """
    Delete a generated file.
    
    Args:
        filename: Name of the file to delete
        
    Returns:
        Success message
    """
    try:
        # Security: Prevent directory traversal attacks
        if '..' in filename or '/' in filename or '\\' in filename:
            raise HTTPException(status_code=400, detail="Invalid filename")
        
        file_path = GENERATED_FILES_DIR / filename
        
        if not file_path.exists():
            raise HTTPException(status_code=404, detail=f"File not found: {filename}")
        
        if not file_path.is_file():
            raise HTTPException(status_code=400, detail="Invalid file path")
        
        # Delete the file
        file_path.unlink()
        
        logger.info(f"🗑️ Deleted file: {filename}")
        
        return {
            'success': True,
            'message': f'File deleted successfully: {filename}',
            'filename': filename
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Error deleting file {filename}: {e}")
        raise HTTPException(status_code=500, detail=f"Error deleting file: {str(e)}")


@router.get("/storage-info")
async def get_storage_info() -> Dict[str, Any]:
    """
    Get information about the file storage directory.
    
    Returns:
        Dictionary with storage information
    """
    try:
        # Count files by type
        file_counts = {}
        total_size = 0
        
        for file_path in GENERATED_FILES_DIR.iterdir():
            if file_path.is_file():
                ext = file_path.suffix.lower()
                file_counts[ext] = file_counts.get(ext, 0) + 1
                total_size += file_path.stat().st_size
        
        return {
            'storage_path': str(GENERATED_FILES_DIR.absolute()),
            'total_files': sum(file_counts.values()),
            'total_size': total_size,
            'total_size_human': _format_file_size(total_size),
            'file_counts_by_type': file_counts,
            'exists': GENERATED_FILES_DIR.exists(),
            'is_writable': os.access(GENERATED_FILES_DIR, os.W_OK)
        }
        
    except Exception as e:
        logger.error(f"❌ Error getting storage info: {e}")
        raise HTTPException(status_code=500, detail=f"Error getting storage info: {str(e)}")


def _format_file_size(size_bytes: int) -> str:
    """Format file size in human-readable format."""
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.2f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.2f} PB"

