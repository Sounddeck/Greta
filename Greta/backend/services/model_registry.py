"""
Model Registry Service
Week 2 - Component 1: Model Registry System (MongoDB Version)

Tracks available models, metadata, and configurations.
Uses MongoDB for persistence with JSON fallback.
"""

import os
import json
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime
from pydantic import BaseModel, Field
from loguru import logger
from motor.motor_asyncio import AsyncIOMotorClient


class ModelMetadata(BaseModel):
    """Model metadata schema"""
    name: str
    version: str = "1.0.0"
    family: str  # e.g., "llama", "mistral", "phi"
    size: str  # e.g., "7B", "13B", "70B"
    quantization: str  # e.g., "Q4_K_M", "Q5_K_S", "Q8_0"
    file_name: str
    file_size_mb: float = 0.0
    context_window: int = 4096
    capabilities: List[str] = Field(default_factory=list)  # e.g., ["chat", "code", "reasoning"]
    download_url: Optional[str] = None
    huggingface_repo: Optional[str] = None
    status: str = "available"  # available, downloading, downloaded, error
    download_progress: float = 0.0
    created_at: str = Field(default_factory=lambda: datetime.utcnow().isoformat())
    updated_at: str = Field(default_factory=lambda: datetime.utcnow().isoformat())
    last_used_at: Optional[str] = None
    usage_count: int = 0
    tags: List[str] = Field(default_factory=list)
    notes: Optional[str] = None


class ModelRegistry:
    """
    Model Registry Service
    
    Manages model metadata, tracking, and discovery using MongoDB.
    """
    
    def __init__(self, models_dir: Optional[str] = None, db_client: Optional[AsyncIOMotorClient] = None):
        self.models_dir = Path(models_dir or os.getenv('CPAS4_MODELS', Path.home() / '.cpas4' / 'models'))
        self.models_dir.mkdir(parents=True, exist_ok=True)
        
        # MongoDB client and collection
        self.db_client = db_client
        self.collection = None
        self.use_mongodb = False
        
        # JSON fallback path
        self.json_path = self.models_dir.parent / 'model_registry.json'
        
        # In-memory cache
        self.cache: Dict[str, ModelMetadata] = {}
        
        logger.info(f"Model Registry initialized")
        logger.info(f"Models directory: {self.models_dir}")
    
    async def initialize(self, db_client: Optional[AsyncIOMotorClient] = None):
        """Initialize database and load existing models"""
        if db_client:
            self.db_client = db_client
        
        await self._init_mongodb()
        await self._load_from_database()
        await self._scan_models_directory()
        logger.info(f"Model Registry initialized with {len(self.cache)} models")
    
    async def _init_mongodb(self):
        """Initialize MongoDB collection"""
        if self.db_client:
            try:
                # Get database from client
                db = self.db_client.greta_pai
                self.collection = db.models
                
                # Create indexes
                await self.collection.create_index("name", unique=True)
                await self.collection.create_index("family")
                await self.collection.create_index("status")
                
                self.use_mongodb = True
                logger.success("MongoDB collection initialized for model registry")
            except Exception as e:
                logger.error(f"MongoDB initialization failed: {e}")
                logger.warning("Falling back to JSON storage")
                self.use_mongodb = False
        else:
            logger.warning("No MongoDB client provided, using JSON storage")
            self.use_mongodb = False
    
    async def _load_from_database(self):
        """Load models from MongoDB or JSON fallback"""
        if self.use_mongodb and self.collection:
            try:
                cursor = self.collection.find({})
                async for doc in cursor:
                    # Remove MongoDB _id field
                    doc.pop('_id', None)
                    metadata = ModelMetadata(**doc)
                    self.cache[metadata.name] = metadata
                logger.info(f"Loaded {len(self.cache)} models from MongoDB")
            except Exception as e:
                logger.warning(f"Failed to load from MongoDB: {e}")
                await self._load_from_json()
        else:
            await self._load_from_json()
    
    async def _load_from_json(self):
        """Load models from JSON fallback"""
        if self.json_path.exists():
            try:
                with open(self.json_path, 'r') as f:
                    data = json.load(f)
                    for model_data in data.get('models', []):
                        metadata = ModelMetadata(**model_data)
                        self.cache[metadata.name] = metadata
                logger.info(f"Loaded {len(self.cache)} models from JSON")
            except Exception as e:
                logger.error(f"Failed to load from JSON: {e}")
    
    async def _scan_models_directory(self):
        """Scan models directory and register any untracked models"""
        for file_path in self.models_dir.glob("*.gguf"):
            model_name = file_path.stem
            
            if model_name not in self.cache:
                # Auto-register discovered model
                metadata = self._infer_metadata_from_filename(file_path)
                await self.register_model(metadata)
                logger.info(f"Auto-registered model: {model_name}")
    
    def _infer_metadata_from_filename(self, file_path: Path) -> ModelMetadata:
        """Infer model metadata from filename"""
        file_name = file_path.name
        name = file_path.stem
        size_mb = file_path.stat().st_size / (1024 * 1024)
        
        # Try to parse filename (e.g., "llama-3-8b.Q4_K_M.gguf")
        parts = name.lower().split('-')
        
        family = parts[0] if parts else "unknown"
        size = "unknown"
        quantization = "unknown"
        
        # Extract size (e.g., "7b", "13b")
        for part in parts:
            if 'b' in part and any(c.isdigit() for c in part):
                size = part.upper()
                break
        
        # Extract quantization from filename
        if '.' in name:
            quant_part = name.split('.')[-1]
            if 'Q' in quant_part.upper():
                quantization = quant_part.upper()
        
        return ModelMetadata(
            name=name,
            family=family,
            size=size,
            quantization=quantization,
            file_name=file_name,
            file_size_mb=round(size_mb, 2),
            status="downloaded"
        )
    
    async def register_model(self, metadata: ModelMetadata) -> bool:
        """Register a new model"""
        try:
            # Add to cache
            self.cache[metadata.name] = metadata
            
            # Save to MongoDB
            if self.use_mongodb and self.collection:
                await self.collection.update_one(
                    {"name": metadata.name},
                    {"$set": metadata.model_dump()},
                    upsert=True
                )
            
            # Save to JSON fallback
            await self._save_to_json()
            
            logger.success(f"Registered model: {metadata.name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to register model {metadata.name}: {e}")
            return False
    
    async def _save_to_json(self):
        """Save registry to JSON fallback"""
        try:
            data = {
                'models': [model.model_dump() for model in self.cache.values()],
                'updated_at': datetime.utcnow().isoformat()
            }
            with open(self.json_path, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save to JSON: {e}")
    
    async def get_model(self, name: str) -> Optional[ModelMetadata]:
        """Get model metadata by name"""
        return self.cache.get(name)
    
    async def list_models(self, family: Optional[str] = None, status: Optional[str] = None) -> List[ModelMetadata]:
        """List all models with optional filters"""
        models = list(self.cache.values())
        
        if family:
            models = [m for m in models if m.family.lower() == family.lower()]
        
        if status:
            models = [m for m in models if m.status == status]
        
        return models
    
    async def update_model(self, name: str, **updates) -> bool:
        """Update model metadata"""
        if name not in self.cache:
            logger.error(f"Model not found: {name}")
            return False
        
        try:
            metadata = self.cache[name]
            
            # Update fields
            for key, value in updates.items():
                if hasattr(metadata, key):
                    setattr(metadata, key, value)
            
            metadata.updated_at = datetime.utcnow().isoformat()
            
            # Re-register to save changes
            await self.register_model(metadata)
            
            logger.info(f"Updated model: {name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to update model {name}: {e}")
            return False
    
    async def delete_model(self, name: str, delete_file: bool = False) -> bool:
        """Delete model from registry"""
        if name not in self.cache:
            logger.error(f"Model not found: {name}")
            return False
        
        try:
            metadata = self.cache[name]
            
            # Remove from cache
            del self.cache[name]
            
            # Remove from MongoDB
            if self.use_mongodb and self.collection:
                await self.collection.delete_one({"name": name})
            
            # Delete file if requested
            if delete_file:
                file_path = self.models_dir / metadata.file_name
                if file_path.exists():
                    file_path.unlink()
                    logger.info(f"Deleted model file: {file_path}")
            
            # Save to JSON
            await self._save_to_json()
            
            logger.success(f"Deleted model: {name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to delete model {name}: {e}")
            return False
    
    async def record_usage(self, name: str):
        """Record model usage"""
        if name in self.cache:
            await self.update_model(
                name,
                last_used_at=datetime.utcnow().isoformat(),
                usage_count=self.cache[name].usage_count + 1
            )


# Global instance
model_registry = ModelRegistry()
