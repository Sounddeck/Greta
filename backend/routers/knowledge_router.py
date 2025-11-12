"""
CPAS4 UI Knowledge Router
Provides knowledge base and file upload endpoints for the CPAS4 UI
"""

from fastapi import APIRouter, HTTPException, UploadFile, File, Form
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime
import logging
import os
import shutil

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/knowledge", tags=["Knowledge"])

# Pydantic models
class DocumentResponse(BaseModel):
    """Document upload response"""
    id: str
    filename: str
    file_type: str
    file_size: int
    collection: Optional[str] = None
    uploaded_at: str
    metadata: Dict[str, Any] = {}

@router.post("/upload")
async def upload_files(
    files: List[UploadFile] = File(...),
    collection: Optional[str] = Form(None)
):
    """
    Upload files to the knowledge base
    
    Supports multiple file uploads for document processing and knowledge extraction.
    """
    try:
        logger.info(f"Uploading {len(files)} file(s)")
        
        # Create uploads directory if it doesn't exist
        upload_dir = "data/uploads"
        os.makedirs(upload_dir, exist_ok=True)
        
        uploaded_files = []
        
        for file in files:
            try:
                # Generate unique filename
                timestamp = datetime.utcnow().timestamp()
                safe_filename = f"{timestamp}_{file.filename}"
                file_path = os.path.join(upload_dir, safe_filename)
                
                # Save file
                with open(file_path, "wb") as buffer:
                    shutil.copyfileobj(file.file, buffer)
                
                # Get file info
                file_size = os.path.getsize(file_path)
                file_type = file.content_type or "application/octet-stream"
                
                # Create document response
                doc_id = f"doc_{timestamp}"
                
                document = {
                    "id": doc_id,
                    "filename": file.filename,
                    "file_type": file_type,
                    "file_size": file_size,
                    "collection": collection,
                    "uploaded_at": datetime.utcnow().isoformat(),
                    "metadata": {
                        "original_filename": file.filename,
                        "saved_path": file_path,
                        "content_type": file_type
                    }
                }
                
                uploaded_files.append(document)
                
                logger.info(f"Uploaded file: {file.filename} ({file_size} bytes)")
                
            except Exception as file_error:
                logger.error(f"Failed to upload file {file.filename}: {file_error}")
                # Continue with other files
                continue
        
        if not uploaded_files:
            raise HTTPException(
                status_code=400,
                detail="No files were successfully uploaded"
            )
        
        return {
            "success": True,
            "files": uploaded_files,
            "total": len(uploaded_files),
            "message": f"Successfully uploaded {len(uploaded_files)} file(s)"
        }
        
    except Exception as e:
        logger.error(f"File upload failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"File upload failed: {str(e)}"
        )

@router.get("/documents")
async def list_documents(
    collection: Optional[str] = None,
    limit: int = 100,
    offset: int = 0
):
    """List all documents in the knowledge base"""
    try:
        # This is a simplified implementation
        # In a real system, you would query a database
        
        upload_dir = "data/uploads"
        if not os.path.exists(upload_dir):
            return {
                "documents": [],
                "total": 0,
                "limit": limit,
                "offset": offset
            }
        
        files = os.listdir(upload_dir)
        documents = []
        
        for filename in files[offset:offset + limit]:
            file_path = os.path.join(upload_dir, filename)
            if os.path.isfile(file_path):
                file_size = os.path.getsize(file_path)
                file_stat = os.stat(file_path)
                
                documents.append({
                    "id": f"doc_{filename}",
                    "filename": filename,
                    "file_type": "unknown",
                    "file_size": file_size,
                    "collection": collection,
                    "uploaded_at": datetime.fromtimestamp(file_stat.st_ctime).isoformat(),
                    "metadata": {}
                })
        
        return {
            "documents": documents,
            "total": len(files),
            "limit": limit,
            "offset": offset
        }
        
    except Exception as e:
        logger.error(f"Failed to list documents: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to list documents: {str(e)}"
        )

@router.get("/documents/{document_id}")
async def get_document(document_id: str):
    """Get document details"""
    return {
        "id": document_id,
        "message": "Document retrieval not fully implemented yet"
    }

@router.delete("/documents/{document_id}")
async def delete_document(document_id: str):
    """Delete a document"""
    return {
        "id": document_id,
        "deleted": True,
        "message": "Document deletion not fully implemented yet"
    }

@router.post("/search")
async def search_knowledge(query: str, limit: int = 10):
    """Search the knowledge base"""
    return {
        "query": query,
        "results": [],
        "total": 0,
        "message": "Knowledge search not fully implemented yet"
    }

@router.get("/collections")
async def list_collections():
    """List all collections"""
    return {
        "collections": [],
        "total": 0,
        "message": "Collections not fully implemented yet"
    }

