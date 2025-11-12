"""
GRETA Digital Asset Creation Engine
Create any digital product: apps, websites, documents, marketing materials, etc.
"""

import asyncio
import logging
import json
import os
import tempfile
import zipfile
from typing import Dict, Any, List, Optional, Union
from datetime import datetime
from pathlib import Path
from pydantic import BaseModel, Field
import uuid

# Template and generation dependencies
try:
    from jinja2 import Environment, FileSystemLoader, Template
    import markdown
    from reportlab.pdfgen import canvas
    from reportlab.lib.pagesizes import letter, A4
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
    DOCUMENT_GENERATION_AVAILABLE = True
except ImportError:
    logging.warning("Document generation dependencies not available. Install: pip install jinja2 markdown reportlab")
    DOCUMENT_GENERATION_AVAILABLE = False

# Web development dependencies
try:
    import requests
    WEB_GENERATION_AVAILABLE = True
except ImportError:
    WEB_GENERATION_AVAILABLE = False

logger = logging.getLogger(__name__)

class AssetRequest(BaseModel):
    asset_type: str = Field(..., description="Type of asset to create")
    specifications: Dict[str, Any] = Field(..., description="Asset specifications and requirements")
    template: Optional[str] = Field(None, description="Template to use")
    output_format: str = Field(default="zip", description="Output format")
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict)

class AssetResponse(BaseModel):
    success: bool
    asset_id: str
    asset_type: str
    file_path: Optional[str]
    download_url: Optional[str]
    metadata: Dict[str, Any]
    creation_time: float
    size_bytes: Optional[int]

class AssetCreationEngine:
    """Engine for creating digital assets and products"""
    
    def __init__(self):
        self.output_dir = Path("generated_assets")
        self.output_dir.mkdir(exist_ok=True)
        
        self.templates_dir = Path("templates")
        self.templates_dir.mkdir(exist_ok=True)
        
        # Initialize template environment
        if DOCUMENT_GENERATION_AVAILABLE:
            self.jinja_env = Environment(loader=FileSystemLoader(str(self.templates_dir)))
        
        # Asset type handlers
        self.asset_handlers = {
            "website": self._create_website,
            "mobile_app": self._create_mobile_app,
            "document": self._create_document,
            "presentation": self._create_presentation,
            "marketing_material": self._create_marketing_material,
            "brochure": self._create_brochure,
            "business_plan": self._create_business_plan,
            "api_documentation": self._create_api_docs,
            "user_manual": self._create_user_manual,
            "code_project": self._create_code_project
        }
        
        # Initialize default templates
        self._setup_default_templates()
    
    def _setup_default_templates(self):
        """Setup default templates for common asset types"""
        templates = {
            "website_index.html": """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{{ title }}</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 0; padding: 20px; }
        .container { max-width: 1200px; margin: 0 auto; }
        .header { text-align: center; margin-bottom: 40px; }
        .section { margin: 30px 0; }
        .footer { text-align: center; margin-top: 40px; color: #666; }
    </style>
</head>
<body>
    <div class="container">
        <header class="header">
            <h1>{{ title }}</h1>
            <p>{{ description }}</p>
        </header>
        
        {% for section in sections %}
        <div class="section">
            <h2>{{ section.title }}</h2>
            <p>{{ section.content }}</p>
        </div>
        {% endfor %}
        
        <footer class="footer">
            <p>&copy; {{ year }} {{ company_name }}. All rights reserved.</p>
        </footer>
    </div>
</body>
</html>
            """,
            
            "business_plan.md": """
# {{ company_name }} Business Plan

## Executive Summary
{{ executive_summary }}

## Company Description
{{ company_description }}

## Market Analysis
{{ market_analysis }}

## Organization & Management
{{ organization_management }}

## Service or Product Line
{{ product_line }}

## Marketing & Sales
{{ marketing_sales }}

## Funding Request
{{ funding_request }}

## Financial Projections
{{ financial_projections }}

## Appendix
{{ appendix }}
            """,
            
            "api_docs.md": """
# {{ api_name }} API Documentation

## Overview
{{ overview }}

## Authentication
{{ authentication }}

## Endpoints

{% for endpoint in endpoints %}
### {{ endpoint.method }} {{ endpoint.path }}
{{ endpoint.description }}

**Parameters:**
{% for param in endpoint.parameters %}
- `{{ param.name }}` ({{ param.type }}): {{ param.description }}
{% endfor %}

**Response:**
```json
{{ endpoint.response_example }}
```
{% endfor %}

## Error Codes
{{ error_codes }}
            """
        }
        
        # Save templates to disk
        for template_name, content in templates.items():
            template_path = self.templates_dir / template_name
            if not template_path.exists():
                template_path.write_text(content.strip())
    
    async def create_asset(self, request: AssetRequest) -> AssetResponse:
        """Create digital asset based on request"""
        start_time = datetime.now()
        asset_id = str(uuid.uuid4())
        
        try:
            # Validate asset type
            if request.asset_type not in self.asset_handlers:
                raise ValueError(f"Unsupported asset type: {request.asset_type}")
            
            logger.info(f"Creating {request.asset_type} asset: {asset_id}")
            
            # Create asset using appropriate handler
            handler = self.asset_handlers[request.asset_type]
            result = await handler(asset_id, request.specifications, request.template)
            
            creation_time = (datetime.now() - start_time).total_seconds()
            
            # Get file size if file was created
            file_size = None
            if result.get("file_path") and Path(result["file_path"]).exists():
                file_size = Path(result["file_path"]).stat().st_size
            
            return AssetResponse(
                success=True,
                asset_id=asset_id,
                asset_type=request.asset_type,
                file_path=result.get("file_path"),
                download_url=result.get("download_url"),
                metadata={
                    **request.metadata,
                    **result.get("metadata", {}),
                    "specifications": request.specifications
                },
                creation_time=creation_time,
                size_bytes=file_size
            )
            
        except Exception as e:
            logger.error(f"Asset creation failed: {e}")
            return AssetResponse(
                success=False,
                asset_id=asset_id,
                asset_type=request.asset_type,
                file_path=None,
                download_url=None,
                metadata={"error": str(e)},
                creation_time=(datetime.now() - start_time).total_seconds(),
                size_bytes=None
            )
    
    async def _create_website(self, asset_id: str, specs: Dict[str, Any], template: Optional[str]) -> Dict[str, Any]:
        """Create a complete website"""
        if not DOCUMENT_GENERATION_AVAILABLE:
            raise RuntimeError("Document generation dependencies not available")
        
        # Create website directory
        website_dir = self.output_dir / f"website_{asset_id}"
        website_dir.mkdir(exist_ok=True)
        
        # Generate HTML files
        template_obj = self.jinja_env.get_template("website_index.html")
        
        # Default website structure
        pages = specs.get("pages", [{"name": "index", "title": specs.get("title", "My Website")}])
        
        for page in pages:
            html_content = template_obj.render(
                title=page.get("title", "Page"),
                description=specs.get("description", "Welcome to our website"),
                sections=specs.get("sections", []),
                year=datetime.now().year,
                company_name=specs.get("company_name", "Your Company")
            )
            
            page_file = website_dir / f"{page['name']}.html"
            page_file.write_text(html_content)
        
        # Create CSS file
        css_content = specs.get("custom_css", "/* Custom styles */")
        (website_dir / "styles.css").write_text(css_content)
        
        # Create JavaScript file
        js_content = specs.get("custom_js", "// Custom JavaScript")
        (website_dir / "script.js").write_text(js_content)
        
        # Create ZIP archive
        zip_path = self.output_dir / f"website_{asset_id}.zip"
        with zipfile.ZipFile(zip_path, 'w') as zipf:
            for file_path in website_dir.rglob('*'):
                if file_path.is_file():
                    zipf.write(file_path, file_path.relative_to(website_dir))
        
        return {
            "file_path": str(zip_path),
            "download_url": f"/api/v1/assets/download/{asset_id}",
            "metadata": {
                "pages_created": len(pages),
                "website_directory": str(website_dir)
            }
        }
    
    async def _create_mobile_app(self, asset_id: str, specs: Dict[str, Any], template: Optional[str]) -> Dict[str, Any]:
        """Create a mobile app project structure"""
        app_dir = self.output_dir / f"mobile_app_{asset_id}"
        app_dir.mkdir(exist_ok=True)
        
        # Create React Native project structure
        app_name = specs.get("app_name", "MyApp")
        platform = specs.get("platform", "react-native")
        
        if platform == "react-native":
            # Create basic React Native structure
            (app_dir / "src").mkdir(exist_ok=True)
            (app_dir / "src" / "components").mkdir(exist_ok=True)
            (app_dir / "src" / "screens").mkdir(exist_ok=True)
            
            # Package.json
            package_json = {
                "name": app_name.lower().replace(" ", "-"),
                "version": "1.0.0",
                "main": "index.js",
                "dependencies": {
                    "react": "^18.0.0",
                    "react-native": "^0.72.0"
                }
            }
            (app_dir / "package.json").write_text(json.dumps(package_json, indent=2))
            
            # Main App component
            app_js = f"""
import React from 'react';
import {{ View, Text, StyleSheet }} from 'react-native';

const App = () => {{
  return (
    <View style={{styles.container}}>
      <Text style={{styles.title}}>{app_name}</Text>
      <Text style={{styles.subtitle}}>{specs.get('description', 'Welcome to your app!')}</Text>
    </View>
  );
}};

const styles = StyleSheet.create({{
  container: {{
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
    backgroundColor: '#f0f0f0',
  }},
  title: {{
    fontSize: 24,
    fontWeight: 'bold',
    marginBottom: 10,
  }},
  subtitle: {{
    fontSize: 16,
    textAlign: 'center',
    margin: 20,
  }},
}});

export default App;
            """
            (app_dir / "App.js").write_text(app_js.strip())
        
        # Create ZIP archive
        zip_path = self.output_dir / f"mobile_app_{asset_id}.zip"
        with zipfile.ZipFile(zip_path, 'w') as zipf:
            for file_path in app_dir.rglob('*'):
                if file_path.is_file():
                    zipf.write(file_path, file_path.relative_to(app_dir))
        
        return {
            "file_path": str(zip_path),
            "download_url": f"/api/v1/assets/download/{asset_id}",
            "metadata": {
                "platform": platform,
                "app_name": app_name
            }
        }
    
    async def _create_document(self, asset_id: str, specs: Dict[str, Any], template: Optional[str]) -> Dict[str, Any]:
        """Create a document (PDF, Word, etc.)"""
        if not DOCUMENT_GENERATION_AVAILABLE:
            raise RuntimeError("Document generation dependencies not available")
        
        doc_type = specs.get("format", "pdf")
        title = specs.get("title", "Document")
        content = specs.get("content", "Document content goes here.")
        
        if doc_type == "pdf":
            pdf_path = self.output_dir / f"document_{asset_id}.pdf"
            
            # Create PDF
            doc = SimpleDocTemplate(str(pdf_path), pagesize=letter)
            styles = getSampleStyleSheet()
            story = []
            
            # Add title
            story.append(Paragraph(title, styles['Title']))
            story.append(Spacer(1, 12))
            
            # Add content
            for paragraph in content.split('\n\n'):
                if paragraph.strip():
                    story.append(Paragraph(paragraph, styles['Normal']))
                    story.append(Spacer(1, 12))
            
            doc.build(story)
            
            return {
                "file_path": str(pdf_path),
                "download_url": f"/api/v1/assets/download/{asset_id}",
                "metadata": {"format": "pdf", "title": title}
            }
        
        elif doc_type == "markdown":
            md_path = self.output_dir / f"document_{asset_id}.md"
            
            # Create Markdown
            md_content = f"# {title}\n\n{content}"
            md_path.write_text(md_content)
            
            return {
                "file_path": str(md_path),
                "download_url": f"/api/v1/assets/download/{asset_id}",
                "metadata": {"format": "markdown", "title": title}
            }
    
    async def _create_business_plan(self, asset_id: str, specs: Dict[str, Any], template: Optional[str]) -> Dict[str, Any]:
        """Create a comprehensive business plan"""
        if not DOCUMENT_GENERATION_AVAILABLE:
            raise RuntimeError("Document generation dependencies not available")
        
        template_obj = self.jinja_env.get_template("business_plan.md")
        
        # Generate business plan content
        business_plan = template_obj.render(**specs)
        
        # Save as Markdown
        md_path = self.output_dir / f"business_plan_{asset_id}.md"
        md_path.write_text(business_plan)
        
        # Convert to PDF if requested
        if specs.get("include_pdf", True):
            html_content = markdown.markdown(business_plan)
            pdf_path = self.output_dir / f"business_plan_{asset_id}.pdf"
            
            # Simple PDF generation (would need more sophisticated formatting in production)
            doc = SimpleDocTemplate(str(pdf_path), pagesize=A4)
            styles = getSampleStyleSheet()
            story = [Paragraph(html_content, styles['Normal'])]
            doc.build(story)
        
        return {
            "file_path": str(md_path),
            "download_url": f"/api/v1/assets/download/{asset_id}",
            "metadata": {
                "company_name": specs.get("company_name", "Unknown"),
                "includes_pdf": specs.get("include_pdf", True)
            }
        }
    
    # Placeholder implementations for other asset types
    async def _create_presentation(self, asset_id: str, specs: Dict[str, Any], template: Optional[str]) -> Dict[str, Any]:
        """Create a presentation"""
        # TODO: Implement presentation creation (PowerPoint, Google Slides, etc.)
        return {"file_path": None, "metadata": {"status": "not_implemented"}}
    
    async def _create_marketing_material(self, asset_id: str, specs: Dict[str, Any], template: Optional[str]) -> Dict[str, Any]:
        """Create marketing materials"""
        # TODO: Implement marketing material creation
        return {"file_path": None, "metadata": {"status": "not_implemented"}}
    
    async def _create_brochure(self, asset_id: str, specs: Dict[str, Any], template: Optional[str]) -> Dict[str, Any]:
        """Create a brochure"""
        # TODO: Implement brochure creation
        return {"file_path": None, "metadata": {"status": "not_implemented"}}
    
    async def _create_api_docs(self, asset_id: str, specs: Dict[str, Any], template: Optional[str]) -> Dict[str, Any]:
        """Create API documentation"""
        if not DOCUMENT_GENERATION_AVAILABLE:
            raise RuntimeError("Document generation dependencies not available")
        
        template_obj = self.jinja_env.get_template("api_docs.md")
        api_docs = template_obj.render(**specs)
        
        md_path = self.output_dir / f"api_docs_{asset_id}.md"
        md_path.write_text(api_docs)
        
        return {
            "file_path": str(md_path),
            "download_url": f"/api/v1/assets/download/{asset_id}",
            "metadata": {"api_name": specs.get("api_name", "API")}
        }
    
    async def _create_user_manual(self, asset_id: str, specs: Dict[str, Any], template: Optional[str]) -> Dict[str, Any]:
        """Create a user manual"""
        # TODO: Implement user manual creation
        return {"file_path": None, "metadata": {"status": "not_implemented"}}
    
    async def _create_code_project(self, asset_id: str, specs: Dict[str, Any], template: Optional[str]) -> Dict[str, Any]:
        """Create a code project"""
        # TODO: Implement code project generation
        return {"file_path": None, "metadata": {"status": "not_implemented"}}

# Global service instance
asset_creation_engine = AssetCreationEngine()
