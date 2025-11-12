"""
Prompt Library Service
Manages prompt templates with CRUD operations, versioning, and search
"""

import logging
from typing import Dict, Any, Optional, List
from datetime import datetime

from models.prompt_template_models import (
    PromptTemplate,
    PromptTemplateCreate,
    PromptTemplateUpdate,
    PromptCategory,
    DEFAULT_TEMPLATES
)

logger = logging.getLogger(__name__)


class PromptLibraryService:
    """Service for managing prompt templates"""

    def __init__(self, database):
        """Initialize with Database instance"""
        self.database = database
        self.collection_name = "prompt_templates"
        
    async def initialize_default_templates(self) -> Dict[str, Any]:
        """Initialize database with default system templates"""
        try:
            # Check if templates already exist
            existing = await self.database.find_many(self.collection_name, {"is_system": True}, limit=100)
            existing_count = len(existing)

            if existing_count > 0:
                logger.info(f"✅ {existing_count} system templates already exist")
                return {
                    "success": True,
                    "message": f"System templates already initialized ({existing_count} templates)",
                    "count": existing_count
                }

            # Insert default templates
            inserted_count = 0
            for template_data in DEFAULT_TEMPLATES:
                template = PromptTemplate(
                    id=f"template_{datetime.utcnow().timestamp()}_{template_data['name'].lower().replace(' ', '_')}",
                    **template_data
                )
                await self.database.insert_one(self.collection_name, template.model_dump())
                inserted_count += 1

            logger.info(f"✅ Initialized {inserted_count} default templates")

            return {
                "success": True,
                "message": f"Initialized {inserted_count} default templates",
                "count": inserted_count
            }

        except Exception as e:
            logger.error(f"Failed to initialize default templates: {e}")
            return {
                "success": False,
                "message": f"Failed to initialize templates: {str(e)}",
                "count": 0
            }
    
    async def create_template(self, template_data: PromptTemplateCreate, user_id: Optional[str] = None) -> Dict[str, Any]:
        """Create a new prompt template"""
        try:
            # Generate template ID
            template_id = f"template_{datetime.utcnow().timestamp()}_{template_data.name.lower().replace(' ', '_')}"

            # Create template
            template = PromptTemplate(
                id=template_id,
                name=template_data.name,
                description=template_data.description,
                category=template_data.category,
                template=template_data.template,
                variables=template_data.variables,
                tags=template_data.tags,
                metadata=template_data.metadata,
                created_by=user_id,
                is_system=False
            )

            # Insert into database
            await self.database.insert_one(self.collection_name, template.model_dump())

            logger.info(f"✅ Created template: {template.name} (ID: {template_id})")

            return {
                "success": True,
                "template": template,
                "message": f"Template '{template.name}' created successfully"
            }

        except Exception as e:
            logger.error(f"Failed to create template: {e}")
            return {
                "success": False,
                "template": None,
                "message": f"Failed to create template: {str(e)}"
            }
    
    async def get_template(self, template_id: str) -> Optional[PromptTemplate]:
        """Get a template by ID"""
        try:
            template_data = await self.database.find_one(self.collection_name, {"id": template_id})

            if template_data:
                return PromptTemplate(**template_data)

            return None

        except Exception as e:
            logger.error(f"Failed to get template {template_id}: {e}")
            return None
    
    async def list_templates(
        self,
        category: Optional[PromptCategory] = None,
        tags: Optional[List[str]] = None,
        is_active: bool = True,
        page: int = 1,
        page_size: int = 20
    ) -> Dict[str, Any]:
        """List templates with filtering and pagination"""
        try:
            # Build query
            query = {"is_active": is_active}

            if category:
                query["category"] = category

            if tags:
                # For simple implementation, check if any tag matches
                query["tags"] = {"$in": tags}

            # Get all matching templates
            all_templates = await self.database.find_many(self.collection_name, query, limit=1000)
            total = len(all_templates)

            # Sort by created_at (newest first)
            all_templates.sort(key=lambda x: x.get("created_at", datetime.min), reverse=True)

            # Apply pagination
            skip = (page - 1) * page_size
            paginated_templates = all_templates[skip:skip + page_size]

            templates = [PromptTemplate(**t) for t in paginated_templates]

            return {
                "success": True,
                "templates": templates,
                "total": total,
                "page": page,
                "page_size": page_size,
                "message": f"Found {len(templates)} templates"
            }

        except Exception as e:
            logger.error(f"Failed to list templates: {e}")
            return {
                "success": False,
                "templates": [],
                "total": 0,
                "page": page,
                "page_size": page_size,
                "message": f"Failed to list templates: {str(e)}"
            }
    
    async def search_templates(self, query: str, category: Optional[PromptCategory] = None) -> List[PromptTemplate]:
        """Search templates by name, description, or tags"""
        try:
            # Get all active templates
            search_filter = {"is_active": True}
            if category:
                search_filter["category"] = category

            all_templates = await self.database.find_many(self.collection_name, search_filter, limit=1000)

            # Filter by search query (case-insensitive)
            query_lower = query.lower()
            matching_templates = []

            for template_data in all_templates:
                name_match = query_lower in template_data.get("name", "").lower()
                desc_match = query_lower in template_data.get("description", "").lower()
                tags_match = any(query_lower in tag.lower() for tag in template_data.get("tags", []))

                if name_match or desc_match or tags_match:
                    matching_templates.append(template_data)

            # Sort by usage_count (most used first)
            matching_templates.sort(key=lambda x: x.get("usage_count", 0), reverse=True)

            # Limit to 50 results
            matching_templates = matching_templates[:50]

            templates = [PromptTemplate(**t) for t in matching_templates]

            logger.info(f"Found {len(templates)} templates matching '{query}'")
            return templates

        except Exception as e:
            logger.error(f"Failed to search templates: {e}")
            return []
    
    async def update_template(self, template_id: str, update_data: PromptTemplateUpdate) -> Dict[str, Any]:
        """Update a template"""
        try:
            # Check if template exists
            existing = await self.database.find_one(self.collection_name, {"id": template_id})

            if not existing:
                return {
                    "success": False,
                    "template": None,
                    "message": f"Template {template_id} not found"
                }

            # Don't allow updating system templates
            if existing.get("is_system", False):
                return {
                    "success": False,
                    "template": None,
                    "message": "Cannot update system templates"
                }

            # Build update dict (only include non-None fields)
            update_dict = {}
            for field, value in update_data.model_dump(exclude_unset=True).items():
                if value is not None:
                    update_dict[field] = value

            # Add updated_at timestamp
            update_dict["updated_at"] = datetime.utcnow()

            # Update in database
            await self.database.update_one(
                self.collection_name,
                {"id": template_id},
                {"$set": update_dict}
            )

            # Merge with existing data for response
            updated_data = {**existing, **update_dict}

            # Get updated template
            updated_template = PromptTemplate(**updated_data)

            logger.info(f"✅ Updated template: {template_id}")

            return {
                "success": True,
                "template": updated_template,
                "message": f"Template updated successfully"
            }

        except Exception as e:
            logger.error(f"Failed to update template {template_id}: {e}")
            return {
                "success": False,
                "template": None,
                "message": f"Failed to update template: {str(e)}"
            }
    
    async def delete_template(self, template_id: str) -> Dict[str, Any]:
        """Delete a template (soft delete by setting is_active=False)"""
        try:
            # Check if template exists
            existing = await self.database.find_one(self.collection_name, {"id": template_id})

            if not existing:
                return {
                    "success": False,
                    "message": f"Template {template_id} not found"
                }

            # Don't allow deleting system templates
            if existing.get("is_system", False):
                return {
                    "success": False,
                    "message": "Cannot delete system templates"
                }

            # Soft delete - update the template
            await self.database.update_one(
                self.collection_name,
                {"id": template_id},
                {"$set": {"is_active": False, "updated_at": datetime.utcnow()}}
            )

            logger.info(f"✅ Deleted template: {template_id}")

            return {
                "success": True,
                "message": f"Template deleted successfully"
            }

        except Exception as e:
            logger.error(f"Failed to delete template {template_id}: {e}")
            return {
                "success": False,
                "message": f"Failed to delete template: {str(e)}"
            }
    
    async def render_template(self, template_id: str, variables: Dict[str, str]) -> Dict[str, Any]:
        """Render a template with provided variables"""
        try:
            # Get template
            template_data = await self.database.find_one(self.collection_name, {"id": template_id})

            if not template_data:
                return {
                    "success": False,
                    "rendered_prompt": "",
                    "message": f"Template {template_id} not found"
                }

            template = PromptTemplate(**template_data)

            # Render template
            rendered = template.template
            for var_name, var_value in variables.items():
                placeholder = "{" + var_name + "}"
                rendered = rendered.replace(placeholder, var_value)

            # Increment usage count
            await self.database.update_one(
                self.collection_name,
                {"id": template_id},
                {"$set": {"usage_count": template.usage_count + 1}}
            )

            return {
                "success": True,
                "template_id": template_id,
                "rendered_prompt": rendered,
                "variables_used": variables,
                "message": "Template rendered successfully"
            }

        except Exception as e:
            logger.error(f"Failed to render template {template_id}: {e}")
            return {
                "success": False,
                "rendered_prompt": "",
                "message": f"Failed to render template: {str(e)}"
            }
    
    async def rate_template(self, template_id: str, rating: float, feedback: Optional[str] = None) -> Dict[str, Any]:
        """Rate a template"""
        try:
            # Get current template
            template_data = await self.database.find_one(self.collection_name, {"id": template_id})

            if not template_data:
                return {
                    "success": False,
                    "message": f"Template {template_id} not found"
                }

            template = PromptTemplate(**template_data)

            # Calculate new average rating
            current_rating = template.rating
            current_count = template.usage_count
            new_rating = ((current_rating * current_count) + rating) / (current_count + 1) if current_count > 0 else rating

            # Update template
            await self.database.update_one(
                self.collection_name,
                {"id": template_id},
                {"$set": {"rating": new_rating, "updated_at": datetime.utcnow()}}
            )

            logger.info(f"✅ Rated template {template_id}: {rating}/5.0 (new average: {new_rating:.2f})")

            return {
                "success": True,
                "message": f"Template rated successfully",
                "new_rating": new_rating
            }

        except Exception as e:
            logger.error(f"Failed to rate template {template_id}: {e}")
            return {
                "success": False,
                "message": f"Failed to rate template: {str(e)}"
            }

