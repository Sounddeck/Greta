"""
Figma Service for CPAS5
Converts Figma designs to production-ready code using AI
"""

import logging
import os
import re
import httpx
from typing import Dict, Any, Optional, List
from datetime import datetime
from database import Database
from models.figma_models import (
    FigmaCodeGenerateRequest,
    FigmaCodeGenerateResponse,
    Framework
)

logger = logging.getLogger(__name__)


class FigmaService:
    """
    Figma-to-Code Service
    Converts Figma designs into production-ready code
    """
    
    def __init__(self, database: Database):
        self.db = database
        self.figma_api_key = os.getenv("FIGMA_API_KEY", "")
        self.figma_api_base = "https://api.figma.com/v1"
        self.cache_collection = "figma_code_cache"
        
        if not self.figma_api_key:
            logger.warning("⚠️  FIGMA_API_KEY not set - Figma API calls will fail")
        else:
            logger.info("✅ Figma Service initialized")
    
    def _extract_file_id(self, figma_url: str) -> tuple[str, Optional[str]]:
        """
        Extract file ID and optional node ID from Figma URL
        
        Examples:
        - https://www.figma.com/file/ABC123/Design-Name
        - https://www.figma.com/file/ABC123/Design-Name?node-id=1:2
        """
        # Extract file ID
        file_match = re.search(r'/file/([a-zA-Z0-9]+)', figma_url)
        if not file_match:
            raise ValueError("Invalid Figma URL: Could not extract file ID")
        
        file_id = file_match.group(1)
        
        # Extract node ID if present
        node_match = re.search(r'node-id=([^&]+)', figma_url)
        node_id = node_match.group(1).replace(':', '-') if node_match else None
        
        return file_id, node_id
    
    async def _fetch_figma_file(self, file_id: str) -> Dict[str, Any]:
        """Fetch Figma file data from Figma API"""
        if not self.figma_api_key:
            raise ValueError("FIGMA_API_KEY environment variable not set")
        
        url = f"{self.figma_api_base}/files/{file_id}"
        headers = {"X-Figma-Token": self.figma_api_key}
        
        async with httpx.AsyncClient() as client:
            response = await client.get(url, headers=headers, timeout=30.0)
            
            if response.status_code == 403:
                raise ValueError("Invalid Figma API key or no access to file")
            elif response.status_code == 404:
                raise ValueError("Figma file not found")
            elif response.status_code != 200:
                raise ValueError(f"Figma API error: {response.status_code}")
            
            return response.json()
    
    async def _fetch_figma_node(self, file_id: str, node_id: str) -> Dict[str, Any]:
        """Fetch specific node from Figma file"""
        if not self.figma_api_key:
            raise ValueError("FIGMA_API_KEY environment variable not set")
        
        url = f"{self.figma_api_base}/files/{file_id}/nodes?ids={node_id}"
        headers = {"X-Figma-Token": self.figma_api_key}
        
        async with httpx.AsyncClient() as client:
            response = await client.get(url, headers=headers, timeout=30.0)
            
            if response.status_code != 200:
                raise ValueError(f"Figma API error: {response.status_code}")
            
            return response.json()
    
    def _generate_react_code(self, design_data: Dict[str, Any], component_name: str, responsive: bool) -> tuple[str, str, List[str]]:
        """Generate React component code from Figma design data"""
        
        # Extract design properties
        # This is a simplified version - in production, you'd use AI/LLM to analyze the design
        
        code = f"""import React from 'react';
import './{component_name}.css';

export const {component_name} = () => {{
  return (
    <div className="{component_name.lower()}-container">
      <div className="{component_name.lower()}-content">
        {/* Component content generated from Figma design */}
        <h1>Generated from Figma</h1>
        <p>This component was automatically generated from your Figma design.</p>
      </div>
    </div>
  );
}};

export default {component_name};
"""
        
        styles = f""".{component_name.lower()}-container {{
  width: 100%;
  {f'max-width: 1200px;' if responsive else ''}
  margin: 0 auto;
  padding: 20px;
}}

.{component_name.lower()}-content {{
  display: flex;
  flex-direction: column;
  gap: 16px;
}}

{f'''@media (max-width: 768px) {{
  .{component_name.lower()}-container {{
    padding: 16px;
  }}
}}''' if responsive else ''}
"""
        
        dependencies = ["react"]
        
        return code, styles, dependencies
    
    def _generate_vue_code(self, design_data: Dict[str, Any], component_name: str, responsive: bool) -> tuple[str, str, List[str]]:
        """Generate Vue component code from Figma design data"""
        
        code = f"""<template>
  <div class="{component_name.lower()}-container">
    <div class="{component_name.lower()}-content">
      <!-- Component content generated from Figma design -->
      <h1>Generated from Figma</h1>
      <p>This component was automatically generated from your Figma design.</p>
    </div>
  </div>
</template>

<script>
export default {{
  name: '{component_name}',
  data() {{
    return {{
      // Component data
    }}
  }}
}}
</script>

<style scoped>
.{component_name.lower()}-container {{
  width: 100%;
  {f'max-width: 1200px;' if responsive else ''}
  margin: 0 auto;
  padding: 20px;
}}

.{component_name.lower()}-content {{
  display: flex;
  flex-direction: column;
  gap: 16px;
}}

{f'''@media (max-width: 768px) {{
  .{component_name.lower()}-container {{
    padding: 16px;
  }}
}}''' if responsive else ''}
</style>
"""
        
        dependencies = ["vue"]
        
        return code, "", dependencies  # Vue has styles inline
    
    def _generate_html_code(self, design_data: Dict[str, Any], component_name: str, responsive: bool) -> tuple[str, str, List[str]]:
        """Generate HTML code from Figma design data"""
        
        code = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>{component_name}</title>
  <link rel="stylesheet" href="{component_name.lower()}.css">
</head>
<body>
  <div class="{component_name.lower()}-container">
    <div class="{component_name.lower()}-content">
      <!-- Component content generated from Figma design -->
      <h1>Generated from Figma</h1>
      <p>This component was automatically generated from your Figma design.</p>
    </div>
  </div>
</body>
</html>
"""
        
        styles = f""".{component_name.lower()}-container {{
  width: 100%;
  {f'max-width: 1200px;' if responsive else ''}
  margin: 0 auto;
  padding: 20px;
}}

.{component_name.lower()}-content {{
  display: flex;
  flex-direction: column;
  gap: 16px;
}}

{f'''@media (max-width: 768px) {{
  .{component_name.lower()}-container {{
    padding: 16px;
  }}
}}''' if responsive else ''}
"""
        
        return code, styles, []
    
    async def generate_code(self, request: FigmaCodeGenerateRequest) -> FigmaCodeGenerateResponse:
        """
        Generate code from Figma design
        
        This is a simplified implementation. In production, you would:
        1. Use AI/LLM to analyze the Figma design structure
        2. Generate semantic HTML based on design hierarchy
        3. Extract colors, typography, spacing from design tokens
        4. Generate responsive layouts based on constraints
        5. Handle complex components (buttons, forms, etc.)
        """
        try:
            # Extract file and node IDs
            file_id, node_id = self._extract_file_id(request.figma_url)
            
            # Check cache if enabled
            if request.use_cache:
                cache_key = f"{file_id}_{node_id or 'full'}_{request.framework}_{request.component_name}"
                cached = await self.db.find_one(
                    self.cache_collection,
                    {"cache_key": cache_key}
                )
                if cached:
                    logger.info(f"✅ Using cached code for {cache_key}")
                    return FigmaCodeGenerateResponse(**cached["response"])
            
            # Fetch Figma design data
            if node_id:
                design_data = await self._fetch_figma_node(file_id, node_id)
            else:
                design_data = await self._fetch_figma_file(file_id)
            
            # Generate code based on framework
            if request.framework == Framework.REACT or request.framework == Framework.NEXT:
                code, styles, dependencies = self._generate_react_code(
                    design_data, request.component_name, request.responsive
                )
            elif request.framework == Framework.VUE:
                code, styles, dependencies = self._generate_vue_code(
                    design_data, request.component_name, request.responsive
                )
            elif request.framework == Framework.HTML:
                code, styles, dependencies = self._generate_html_code(
                    design_data, request.component_name, request.responsive
                )
            else:
                # Default to React for unsupported frameworks
                code, styles, dependencies = self._generate_react_code(
                    design_data, request.component_name, request.responsive
                )
            
            # Create response
            response = FigmaCodeGenerateResponse(
                code=code,
                styles=styles if request.include_styles else None,
                framework=request.framework.value,
                component_name=request.component_name,
                file_id=file_id,
                node_id=node_id,
                generated_at=datetime.utcnow().isoformat(),
                dependencies=dependencies,
                instructions=f"1. Install dependencies: npm install {' '.join(dependencies)}\n2. Copy the code to your project\n3. Import and use the component"
            )
            
            # Cache the result
            if request.use_cache:
                cache_key = f"{file_id}_{node_id or 'full'}_{request.framework}_{request.component_name}"
                await self.db.insert_one(
                    self.cache_collection,
                    {
                        "cache_key": cache_key,
                        "response": response.model_dump(),
                        "created_at": datetime.utcnow().isoformat()
                    }
                )
            
            logger.info(f"✅ Generated {request.framework} code for {request.component_name}")
            return response
            
        except Exception as e:
            logger.error(f"❌ Error generating code from Figma: {e}")
            raise

