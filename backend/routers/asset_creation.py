"""
GRETA Digital Asset Creation Router
API endpoints for creating any digital product
"""

import logging
from typing import Dict, Any, List
from fastapi import APIRouter, HTTPException, BackgroundTasks, File, UploadFile
from fastapi.responses import FileResponse
from pathlib import Path

from ..services.asset_creation import asset_creation_engine, AssetRequest, AssetResponse

logger = logging.getLogger(__name__)
router = APIRouter()

@router.post("/create", response_model=AssetResponse)
async def create_asset(request: AssetRequest, background_tasks: BackgroundTasks):
    """
    Create any digital asset: websites, apps, documents, marketing materials, etc.
    """
    try:
        logger.info(f"Creating {request.asset_type} asset")
        
        # Create asset
        result = await asset_creation_engine.create_asset(request)
        
        if result.success:
            logger.info(f"✅ Asset created successfully: {result.asset_id}")
        else:
            logger.error(f"❌ Asset creation failed: {result.metadata.get('error')}")
        
        return result
        
    except Exception as e:
        logger.error(f"Asset creation endpoint failed: {e}")
        raise HTTPException(status_code=500, detail=f"Asset creation failed: {e}")

@router.get("/types")
async def get_supported_asset_types():
    """Get list of supported asset types and their specifications"""
    return {
        "supported_types": [
            {
                "type": "website",
                "description": "Complete website with HTML, CSS, and JavaScript",
                "required_specs": ["title", "description"],
                "optional_specs": ["sections", "company_name", "custom_css", "custom_js", "pages"]
            },
            {
                "type": "mobile_app", 
                "description": "Mobile app project structure (React Native, Flutter, etc.)",
                "required_specs": ["app_name"],
                "optional_specs": ["platform", "description", "features"]
            },
            {
                "type": "document",
                "description": "Documents in various formats (PDF, Markdown, Word)",
                "required_specs": ["title", "content"],
                "optional_specs": ["format", "author", "template"]
            },
            {
                "type": "business_plan",
                "description": "Comprehensive business plan document",
                "required_specs": ["company_name", "executive_summary"],
                "optional_specs": ["market_analysis", "financial_projections", "include_pdf"]
            },
            {
                "type": "api_documentation",
                "description": "API documentation with endpoints and examples",
                "required_specs": ["api_name", "endpoints"],
                "optional_specs": ["authentication", "overview", "error_codes"]
            },
            {
                "type": "presentation",
                "description": "Presentation slides (PowerPoint, Google Slides)",
                "required_specs": ["title", "slides"],
                "optional_specs": ["theme", "template", "speaker_notes"]
            },
            {
                "type": "marketing_material",
                "description": "Marketing materials (flyers, ads, social media posts)",
                "required_specs": ["campaign_name", "target_audience"],
                "optional_specs": ["brand_colors", "images", "call_to_action"]
            },
            {
                "type": "brochure",
                "description": "Product or service brochures",
                "required_specs": ["product_name", "description"],
                "optional_specs": ["features", "pricing", "contact_info"]
            },
            {
                "type": "user_manual",
                "description": "User manuals and documentation",
                "required_specs": ["product_name", "sections"],
                "optional_specs": ["version", "screenshots", "troubleshooting"]
            },
            {
                "type": "code_project",
                "description": "Complete code project with structure and files",
                "required_specs": ["project_name", "language"],
                "optional_specs": ["framework", "features", "dependencies"]
            }
        ]
    }

@router.get("/templates")
async def get_available_templates():
    """Get list of available templates for each asset type"""
    return {
        "templates": {
            "website": [
                {"name": "business", "description": "Professional business website"},
                {"name": "portfolio", "description": "Personal portfolio website"},
                {"name": "ecommerce", "description": "E-commerce store website"},
                {"name": "blog", "description": "Blog or news website"}
            ],
            "mobile_app": [
                {"name": "social", "description": "Social media app template"},
                {"name": "productivity", "description": "Productivity app template"},
                {"name": "ecommerce", "description": "Shopping app template"},
                {"name": "utility", "description": "Utility app template"}
            ],
            "document": [
                {"name": "report", "description": "Professional report template"},
                {"name": "proposal", "description": "Business proposal template"},
                {"name": "manual", "description": "User manual template"},
                {"name": "whitepaper", "description": "Technical whitepaper template"}
            ],
            "business_plan": [
                {"name": "startup", "description": "Startup business plan template"},
                {"name": "traditional", "description": "Traditional business plan template"},
                {"name": "lean", "description": "Lean canvas business plan"},
                {"name": "nonprofit", "description": "Nonprofit organization plan"}
            ]
        }
    }

@router.get("/download/{asset_id}")
async def download_asset(asset_id: str):
    """Download created asset by ID"""
    try:
        # Find asset file
        asset_files = list(Path("generated_assets").glob(f"*{asset_id}*"))
        
        if not asset_files:
            raise HTTPException(status_code=404, detail="Asset not found")
        
        asset_file = asset_files[0]
        
        if not asset_file.exists():
            raise HTTPException(status_code=404, detail="Asset file not found")
        
        return FileResponse(
            path=str(asset_file),
            filename=asset_file.name,
            media_type='application/octet-stream'
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Asset download failed: {e}")
        raise HTTPException(status_code=500, detail=f"Download failed: {e}")

@router.get("/status/{asset_id}")
async def get_asset_status(asset_id: str):
    """Get status and metadata of created asset"""
    try:
        # Find asset file
        asset_files = list(Path("generated_assets").glob(f"*{asset_id}*"))
        
        if not asset_files:
            return {"status": "not_found", "asset_id": asset_id}
        
        asset_file = asset_files[0]
        
        return {
            "status": "completed",
            "asset_id": asset_id,
            "file_name": asset_file.name,
            "file_size": asset_file.stat().st_size,
            "created_at": asset_file.stat().st_ctime,
            "download_url": f"/api/v1/assets/download/{asset_id}"
        }
        
    except Exception as e:
        logger.error(f"Asset status check failed: {e}")
        return {"status": "error", "asset_id": asset_id, "error": str(e)}

@router.get("/list")
async def list_created_assets():
    """List all created assets"""
    try:
        assets_dir = Path("generated_assets")
        if not assets_dir.exists():
            return {"assets": []}
        
        assets = []
        for asset_file in assets_dir.iterdir():
            if asset_file.is_file():
                # Extract asset ID from filename
                asset_id = None
                for part in asset_file.stem.split('_'):
                    if len(part) == 36 and '-' in part:  # UUID format
                        asset_id = part
                        break
                
                assets.append({
                    "asset_id": asset_id,
                    "file_name": asset_file.name,
                    "file_size": asset_file.stat().st_size,
                    "created_at": asset_file.stat().st_ctime,
                    "download_url": f"/api/v1/assets/download/{asset_id}" if asset_id else None
                })
        
        return {"assets": assets, "total": len(assets)}
        
    except Exception as e:
        logger.error(f"Asset listing failed: {e}")
        raise HTTPException(status_code=500, detail=f"Listing failed: {e}")

@router.delete("/delete/{asset_id}")
async def delete_asset(asset_id: str):
    """Delete created asset"""
    try:
        # Find and delete asset file
        asset_files = list(Path("generated_assets").glob(f"*{asset_id}*"))
        
        if not asset_files:
            raise HTTPException(status_code=404, detail="Asset not found")
        
        deleted_files = []
        for asset_file in asset_files:
            asset_file.unlink()
            deleted_files.append(asset_file.name)
        
        return {
            "success": True,
            "asset_id": asset_id,
            "deleted_files": deleted_files
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Asset deletion failed: {e}")
        raise HTTPException(status_code=500, detail=f"Deletion failed: {e}")

@router.post("/batch-create")
async def batch_create_assets(requests: List[AssetRequest], background_tasks: BackgroundTasks):
    """Create multiple assets in batch"""
    try:
        results = []
        
        for request in requests:
            result = await asset_creation_engine.create_asset(request)
            results.append(result)
        
        successful = sum(1 for r in results if r.success)
        
        return {
            "total_requested": len(requests),
            "successful": successful,
            "failed": len(requests) - successful,
            "results": results
        }
        
    except Exception as e:
        logger.error(f"Batch asset creation failed: {e}")
        raise HTTPException(status_code=500, detail=f"Batch creation failed: {e}")

@router.post("/upload-template")
async def upload_custom_template(
    template_name: str,
    asset_type: str,
    file: UploadFile = File(...)
):
    """Upload custom template for asset creation"""
    try:
        # Save uploaded template
        templates_dir = Path("templates")
        templates_dir.mkdir(exist_ok=True)
        
        template_path = templates_dir / f"{asset_type}_{template_name}.{file.filename.split('.')[-1]}"
        
        with open(template_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)
        
        return {
            "success": True,
            "template_name": template_name,
            "asset_type": asset_type,
            "file_path": str(template_path),
            "file_size": len(content)
        }
        
    except Exception as e:
        logger.error(f"Template upload failed: {e}")
        raise HTTPException(status_code=500, detail=f"Template upload failed: {e}")

@router.get("/examples")
async def get_creation_examples():
    """Get example requests for different asset types"""
    return {
        "examples": {
            "website": {
                "asset_type": "website",
                "specifications": {
                    "title": "My Business Website",
                    "description": "Professional website for my business",
                    "company_name": "Acme Corp",
                    "sections": [
                        {"title": "About Us", "content": "We are a leading company..."},
                        {"title": "Services", "content": "We offer the following services..."},
                        {"title": "Contact", "content": "Get in touch with us..."}
                    ]
                }
            },
            "mobile_app": {
                "asset_type": "mobile_app",
                "specifications": {
                    "app_name": "My Mobile App",
                    "platform": "react-native",
                    "description": "A mobile app for my business"
                }
            },
            "business_plan": {
                "asset_type": "business_plan",
                "specifications": {
                    "company_name": "Startup Inc",
                    "executive_summary": "We are building the next big thing...",
                    "market_analysis": "The market size is $10B...",
                    "financial_projections": "We expect $1M revenue in year 1..."
                }
            }
        }
    }
