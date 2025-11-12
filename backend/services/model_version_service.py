"""
Model Version Service
Manages model versions, deployment, and rollback
"""

import logging
import os
from typing import Optional, List
from datetime import datetime
from pathlib import Path

from models.training_models import (
    ModelVersion,
    DeploymentStatus,
    PerformanceMetrics
)
from database import database

logger = logging.getLogger(__name__)


class ModelVersionService:
    """Service for managing model versions and deployment"""

    def __init__(self):
        """Initialize model version service"""
        self.db = database
        self.versions_collection = "model_versions"
        self.config_collection = "system_config"
        self.active_model_symlink = Path(os.getenv(
            "ACTIVE_MODEL_PATH",
            "./models/active_model.gguf"
        ))
        logger.info("✅ Model Version Service initialized")
    
    async def get_version(self, version_id: str) -> Optional[ModelVersion]:
        """
        Get model version by ID
        
        Args:
            version_id: Model version ID
            
        Returns:
            ModelVersion or None
        """
        try:
            doc = await self.db.find_one(
                self.versions_collection,
                {"version_id": version_id}
            )
            
            if doc:
                return ModelVersion(**doc)
            return None
            
        except Exception as e:
            logger.error(f"❌ Failed to get model version: {e}")
            return None
    
    async def list_versions(
        self,
        status: Optional[DeploymentStatus] = None,
        limit: int = 100
    ) -> List[ModelVersion]:
        """
        List model versions
        
        Args:
            status: Filter by deployment status
            limit: Maximum number of versions to return
            
        Returns:
            List of model versions
        """
        try:
            query = {}
            if status:
                query["deployment_status"] = status.value
            
            cursor = self.db.find(
                self.versions_collection,
                query,
                sort=[("created_at", -1)],
                limit=limit
            )
            
            versions = []
            async for doc in cursor:
                versions.append(ModelVersion(**doc))
            
            return versions
            
        except Exception as e:
            logger.error(f"❌ Failed to list model versions: {e}")
            return []
    
    async def get_active_version(self) -> Optional[ModelVersion]:
        """
        Get currently active model version
        
        Returns:
            Active ModelVersion or None
        """
        try:
            doc = await self.db.find_one(
                self.versions_collection,
                {"is_active": True}
            )
            
            if doc:
                return ModelVersion(**doc)
            return None
            
        except Exception as e:
            logger.error(f"❌ Failed to get active model version: {e}")
            return None
    
    async def deploy_version(self, version_id: str) -> bool:
        """
        Deploy a model version
        
        Args:
            version_id: Model version ID to deploy
            
        Returns:
            True if successful
        """
        try:
            # Get model version
            version = await self.get_version(version_id)
            if not version:
                raise ValueError(f"Model version not found: {version_id}")
            
            logger.info(f"🚀 Deploying model version: {version_id}")
            
            # Determine which quantized model to use (prefer Q4_K_M)
            model_path = None
            for quant_type in ["gguf_q4_k_m", "gguf_q5_k_m", "gguf_q8_0", "gguf_f16"]:
                if quant_type in version.model_paths:
                    model_path = version.model_paths[quant_type]
                    break
            
            if not model_path:
                raise ValueError("No suitable model file found for deployment")
            
            # Verify model file exists
            if not Path(model_path).exists():
                raise FileNotFoundError(f"Model file not found: {model_path}")
            
            # Update active model pointer in config
            await self.db.update_one(
                self.config_collection,
                {"key": "active_model"},
                {
                    "$set": {
                        "value": model_path,
                        "version_id": version_id,
                        "updated_at": datetime.utcnow()
                    }
                },
                upsert=True
            )
            
            # Create symlink (optional, for convenience)
            try:
                self.active_model_symlink.unlink(missing_ok=True)
                self.active_model_symlink.symlink_to(Path(model_path).resolve())
                logger.info(f"✅ Created symlink: {self.active_model_symlink} -> {model_path}")
            except Exception as e:
                logger.warning(f"⚠️ Failed to create symlink: {e}")
            
            # Mark version as deployed and active
            await self.db.update_one(
                self.versions_collection,
                {"version_id": version_id},
                {
                    "$set": {
                        "deployment_status": DeploymentStatus.DEPLOYED.value,
                        "is_active": True,
                        "deployed_at": datetime.utcnow()
                    }
                }
            )
            
            # Deactivate other versions
            await self.db.update_many(
                self.versions_collection,
                {"version_id": {"$ne": version_id}},
                {"$set": {"is_active": False}}
            )
            
            logger.info(f"✅ Model version deployed: {version_id}")
            
            # TODO: Restart LLM service (Ollama/llama.cpp)
            # This requires service-specific implementation
            logger.info("⚠️ Note: LLM service restart required to use new model")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to deploy model version: {e}")
            raise
    
    async def rollback_to_version(self, version_id: str) -> bool:
        """
        Rollback to a previous model version
        
        Args:
            version_id: Model version ID to rollback to
            
        Returns:
            True if successful
        """
        try:
            logger.info(f"🔄 Rolling back to model version: {version_id}")
            
            # Deployment is the same as rollback for single-user system
            return await self.deploy_version(version_id)
            
        except Exception as e:
            logger.error(f"❌ Failed to rollback model version: {e}")
            raise
    
    async def archive_version(self, version_id: str) -> bool:
        """
        Archive a model version
        
        Args:
            version_id: Model version ID to archive
            
        Returns:
            True if successful
        """
        try:
            # Verify version exists and is not active
            version = await self.get_version(version_id)
            if not version:
                raise ValueError(f"Model version not found: {version_id}")
            
            if version.is_active:
                raise ValueError("Cannot archive active model version")
            
            # Update status to archived
            await self.db.update_one(
                self.versions_collection,
                {"version_id": version_id},
                {
                    "$set": {
                        "deployment_status": DeploymentStatus.ARCHIVED.value
                    }
                }
            )
            
            logger.info(f"✅ Model version archived: {version_id}")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to archive model version: {e}")
            raise
    
    async def delete_version(self, version_id: str, delete_files: bool = False) -> bool:
        """
        Delete a model version
        
        Args:
            version_id: Model version ID to delete
            delete_files: Whether to delete model files from disk
            
        Returns:
            True if successful
        """
        try:
            # Verify version exists and is not active
            version = await self.get_version(version_id)
            if not version:
                raise ValueError(f"Model version not found: {version_id}")
            
            if version.is_active:
                raise ValueError("Cannot delete active model version")
            
            # Delete files if requested
            if delete_files:
                logger.info(f"🗑️ Deleting model files for version: {version_id}")
                
                for path_key, path_value in version.model_paths.items():
                    try:
                        path_obj = Path(path_value)
                        if path_obj.exists():
                            if path_obj.is_file():
                                path_obj.unlink()
                            elif path_obj.is_dir():
                                import shutil
                                shutil.rmtree(path_obj)
                            logger.info(f"✅ Deleted: {path_value}")
                    except Exception as e:
                        logger.warning(f"⚠️ Failed to delete {path_value}: {e}")
            
            # Delete from database
            await self.db.delete_one(
                self.versions_collection,
                {"version_id": version_id}
            )
            
            logger.info(f"✅ Model version deleted: {version_id}")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to delete model version: {e}")
            raise
    
    async def update_performance_metrics(
        self,
        version_id: str,
        metrics: PerformanceMetrics
    ) -> bool:
        """
        Update performance metrics for a model version
        
        Args:
            version_id: Model version ID
            metrics: Performance metrics
            
        Returns:
            True if successful
        """
        try:
            await self.db.update_one(
                self.versions_collection,
                {"version_id": version_id},
                {
                    "$set": {
                        "performance_metrics": metrics.model_dump()
                    }
                }
            )
            
            logger.info(f"✅ Performance metrics updated for version: {version_id}")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to update performance metrics: {e}")
            raise
    
    async def add_tag(self, version_id: str, tag: str) -> bool:
        """
        Add a tag to a model version
        
        Args:
            version_id: Model version ID
            tag: Tag to add
            
        Returns:
            True if successful
        """
        try:
            await self.db.update_one(
                self.versions_collection,
                {"version_id": version_id},
                {"$addToSet": {"tags": tag}}
            )
            
            logger.info(f"✅ Tag added to version {version_id}: {tag}")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to add tag: {e}")
            raise
    
    async def remove_tag(self, version_id: str, tag: str) -> bool:
        """
        Remove a tag from a model version
        
        Args:
            version_id: Model version ID
            tag: Tag to remove
            
        Returns:
            True if successful
        """
        try:
            await self.db.update_one(
                self.versions_collection,
                {"version_id": version_id},
                {"$pull": {"tags": tag}}
            )
            
            logger.info(f"✅ Tag removed from version {version_id}: {tag}")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to remove tag: {e}")
            raise

