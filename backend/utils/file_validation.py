"""
File Validation Utilities for GRETA PAI
Validates uploaded files for security and format compliance
"""

import logging
from typing import Dict, Any, Optional
from fastapi import UploadFile
import mimetypes

logger = logging.getLogger(__name__)


class ValidationError(Exception):
    """Custom exception for file validation errors"""
    pass


async def validate_audio_file(
    file: UploadFile,
    max_size_mb: int = 50,
    allowed_formats: Optional[list] = None
) -> Dict[str, Any]:
    """
    Validate an uploaded audio file
    
    Args:
        file: The uploaded file to validate
        max_size_mb: Maximum file size in megabytes
        allowed_formats: List of allowed MIME types (default: common audio formats)
        
    Returns:
        Dict with validation results
        
    Raises:
        ValidationError: If validation fails
    """
    if allowed_formats is None:
        allowed_formats = [
            'audio/mpeg',
            'audio/mp3',
            'audio/wav',
            'audio/wave',
            'audio/x-wav',
            'audio/ogg',
            'audio/webm',
            'audio/flac',
            'audio/aac',
            'audio/m4a',
            'audio/x-m4a'
        ]
    
    try:
        # Check if file exists
        if not file:
            raise ValidationError("No file provided")
        
        # Check filename
        if not file.filename:
            raise ValidationError("File has no filename")
        
        # Get file extension
        file_ext = file.filename.split('.')[-1].lower() if '.' in file.filename else ''
        
        # Check content type
        content_type = file.content_type
        if content_type not in allowed_formats:
            # Try to guess from filename
            guessed_type, _ = mimetypes.guess_type(file.filename)
            if guessed_type and guessed_type in allowed_formats:
                content_type = guessed_type
            else:
                raise ValidationError(
                    f"Invalid audio format: {content_type}. "
                    f"Allowed formats: {', '.join(allowed_formats)}"
                )
        
        # Read file to check size
        contents = await file.read()
        file_size_mb = len(contents) / (1024 * 1024)
        
        # Reset file pointer for later use
        await file.seek(0)
        
        # Check file size
        if file_size_mb > max_size_mb:
            raise ValidationError(
                f"File too large: {file_size_mb:.2f}MB. "
                f"Maximum allowed: {max_size_mb}MB"
            )
        
        # Check if file is empty
        if len(contents) == 0:
            raise ValidationError("File is empty")
        
        logger.info(
            f"✅ Audio file validated: {file.filename} "
            f"({file_size_mb:.2f}MB, {content_type})"
        )
        
        return {
            "valid": True,
            "filename": file.filename,
            "content_type": content_type,
            "size_mb": file_size_mb,
            "extension": file_ext
        }
        
    except ValidationError:
        raise
    except Exception as e:
        logger.error(f"File validation error: {e}")
        raise ValidationError(f"File validation failed: {str(e)}")


async def validate_image_file(
    file: UploadFile,
    max_size_mb: int = 10,
    allowed_formats: Optional[list] = None
) -> Dict[str, Any]:
    """
    Validate an uploaded image file
    
    Args:
        file: The uploaded file to validate
        max_size_mb: Maximum file size in megabytes
        allowed_formats: List of allowed MIME types (default: common image formats)
        
    Returns:
        Dict with validation results
        
    Raises:
        ValidationError: If validation fails
    """
    if allowed_formats is None:
        allowed_formats = [
            'image/jpeg',
            'image/jpg',
            'image/png',
            'image/gif',
            'image/webp',
            'image/svg+xml'
        ]
    
    try:
        # Check if file exists
        if not file:
            raise ValidationError("No file provided")
        
        # Check filename
        if not file.filename:
            raise ValidationError("File has no filename")
        
        # Get file extension
        file_ext = file.filename.split('.')[-1].lower() if '.' in file.filename else ''
        
        # Check content type
        content_type = file.content_type
        if content_type not in allowed_formats:
            # Try to guess from filename
            guessed_type, _ = mimetypes.guess_type(file.filename)
            if guessed_type and guessed_type in allowed_formats:
                content_type = guessed_type
            else:
                raise ValidationError(
                    f"Invalid image format: {content_type}. "
                    f"Allowed formats: {', '.join(allowed_formats)}"
                )
        
        # Read file to check size
        contents = await file.read()
        file_size_mb = len(contents) / (1024 * 1024)
        
        # Reset file pointer for later use
        await file.seek(0)
        
        # Check file size
        if file_size_mb > max_size_mb:
            raise ValidationError(
                f"File too large: {file_size_mb:.2f}MB. "
                f"Maximum allowed: {max_size_mb}MB"
            )
        
        # Check if file is empty
        if len(contents) == 0:
            raise ValidationError("File is empty")
        
        logger.info(
            f"✅ Image file validated: {file.filename} "
            f"({file_size_mb:.2f}MB, {content_type})"
        )
        
        return {
            "valid": True,
            "filename": file.filename,
            "content_type": content_type,
            "size_mb": file_size_mb,
            "extension": file_ext
        }
        
    except ValidationError:
        raise
    except Exception as e:
        logger.error(f"File validation error: {e}")
        raise ValidationError(f"File validation failed: {str(e)}")


async def validate_document_file(
    file: UploadFile,
    max_size_mb: int = 25,
    allowed_formats: Optional[list] = None
) -> Dict[str, Any]:
    """
    Validate an uploaded document file
    
    Args:
        file: The uploaded file to validate
        max_size_mb: Maximum file size in megabytes
        allowed_formats: List of allowed MIME types (default: common document formats)
        
    Returns:
        Dict with validation results
        
    Raises:
        ValidationError: If validation fails
    """
    if allowed_formats is None:
        allowed_formats = [
            'application/pdf',
            'application/msword',
            'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
            'application/vnd.ms-excel',
            'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
            'text/plain',
            'text/markdown',
            'application/json'
        ]
    
    try:
        # Check if file exists
        if not file:
            raise ValidationError("No file provided")
        
        # Check filename
        if not file.filename:
            raise ValidationError("File has no filename")
        
        # Get file extension
        file_ext = file.filename.split('.')[-1].lower() if '.' in file.filename else ''
        
        # Check content type
        content_type = file.content_type
        if content_type not in allowed_formats:
            # Try to guess from filename
            guessed_type, _ = mimetypes.guess_type(file.filename)
            if guessed_type and guessed_type in allowed_formats:
                content_type = guessed_type
            else:
                raise ValidationError(
                    f"Invalid document format: {content_type}. "
                    f"Allowed formats: PDF, Word, Excel, Text, Markdown, JSON"
                )
        
        # Read file to check size
        contents = await file.read()
        file_size_mb = len(contents) / (1024 * 1024)
        
        # Reset file pointer for later use
        await file.seek(0)
        
        # Check file size
        if file_size_mb > max_size_mb:
            raise ValidationError(
                f"File too large: {file_size_mb:.2f}MB. "
                f"Maximum allowed: {max_size_mb}MB"
            )
        
        # Check if file is empty
        if len(contents) == 0:
            raise ValidationError("File is empty")
        
        logger.info(
            f"✅ Document file validated: {file.filename} "
            f"({file_size_mb:.2f}MB, {content_type})"
        )
        
        return {
            "valid": True,
            "filename": file.filename,
            "content_type": content_type,
            "size_mb": file_size_mb,
            "extension": file_ext
        }
        
    except ValidationError:
        raise
    except Exception as e:
        logger.error(f"File validation error: {e}")
        raise ValidationError(f"File validation failed: {str(e)}")

