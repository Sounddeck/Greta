"""
Training Data Service
Collects and formats conversation data for LLM training
"""

import logging
import json
import os
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
from pathlib import Path
import hashlib

from models.training_models import (
    TrainingDataset,
    TrainingDatasetCreate,
    DatasetStatus,
    ConversationSample,
    InstructionSample
)
from database import database

logger = logging.getLogger(__name__)


class TrainingDataService:
    """Service for managing training datasets"""

    def __init__(self):
        """Initialize training data service"""
        self.db = database
        self.datasets_collection = "training_datasets"
        self.output_dir = Path("./training_data")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        logger.info("✅ Training Data Service initialized")
    
    async def create_dataset(self, dataset_create: TrainingDatasetCreate) -> TrainingDataset:
        """
        Create a new training dataset
        
        Args:
            dataset_create: Dataset creation request
            
        Returns:
            TrainingDataset: Created dataset
        """
        try:
            # Generate dataset ID
            dataset_id = self._generate_dataset_id(dataset_create.name)
            
            # Create dataset object
            dataset = TrainingDataset(
                dataset_id=dataset_id,
                name=dataset_create.name,
                description=dataset_create.description,
                source_collections=dataset_create.source_collections,
                filters=dataset_create.filters,
                file_path=str(self.output_dir / f"{dataset_id}.jsonl"),
                status=DatasetStatus.PROCESSING
            )
            
            # Save to database
            await self.db.insert_one(
                self.datasets_collection,
                dataset.model_dump()
            )
            
            logger.info(f"✅ Dataset created: {dataset_id}")
            
            # Collect and export data asynchronously
            await self._collect_and_export(dataset, dataset_create.max_samples)
            
            return dataset
            
        except Exception as e:
            logger.error(f"❌ Failed to create dataset: {e}")
            raise
    
    async def _collect_and_export(self, dataset: TrainingDataset, max_samples: Optional[int]):
        """
        Collect data from MongoDB and export to JSONL
        
        Args:
            dataset: Dataset object
            max_samples: Maximum number of samples to collect
        """
        try:
            # Collect conversations
            conversations = await self._collect_conversations(
                dataset.source_collections,
                dataset.filters,
                max_samples
            )
            
            logger.info(f"📊 Collected {len(conversations)} conversations")
            
            # Export to JSONL
            file_size_mb = await self._export_to_jsonl(
                conversations,
                dataset.file_path
            )
            
            # Update dataset
            dataset.total_samples = len(conversations)
            dataset.file_size_mb = file_size_mb
            dataset.status = DatasetStatus.READY
            
            await self.db.update_one(
                self.datasets_collection,
                {"dataset_id": dataset.dataset_id},
                {
                    "$set": {
                        "total_samples": dataset.total_samples,
                        "file_size_mb": dataset.file_size_mb,
                        "status": dataset.status.value
                    }
                }
            )
            
            logger.info(f"✅ Dataset ready: {dataset.dataset_id} ({dataset.total_samples} samples, {file_size_mb:.2f} MB)")
            
        except Exception as e:
            logger.error(f"❌ Failed to collect and export data: {e}")
            
            # Update dataset with error
            await self.db.update_one(
                self.datasets_collection,
                {"dataset_id": dataset.dataset_id},
                {
                    "$set": {
                        "status": DatasetStatus.ERROR.value,
                        "error": str(e)
                    }
                }
            )
    
    async def _collect_conversations(
        self,
        source_collections: List[str],
        filters: Any,
        max_samples: Optional[int]
    ) -> List[Dict[str, Any]]:
        """
        Collect conversations from MongoDB collections
        
        Args:
            source_collections: Collections to collect from
            filters: Data filters
            max_samples: Maximum samples to collect
            
        Returns:
            List of conversation dictionaries
        """
        conversations = []
        seen_hashes = set()  # For deduplication
        
        for collection_name in source_collections:
            try:
                # Build query
                query = {}
                
                # Date range filter
                if filters.date_range:
                    start_date = datetime.fromisoformat(filters.date_range.get("start", "2020-01-01"))
                    end_date = datetime.fromisoformat(filters.date_range.get("end", datetime.utcnow().isoformat()))
                    query["timestamp"] = {"$gte": start_date, "$lte": end_date}
                
                # Rating filter
                if filters.min_rating is not None:
                    query["user_rating"] = {"$gte": filters.min_rating}
                
                # Confidence filter
                if filters.min_confidence is not None:
                    query["confidence"] = {"$gte": filters.min_confidence}
                
                # Success filter
                if filters.success_only:
                    query["success"] = True
                
                # Agent IDs filter
                if filters.agent_ids:
                    query["agent_id"] = {"$in": filters.agent_ids}
                
                # Fetch conversations
                cursor = self.db.find(
                    collection_name,
                    query,
                    sort=[("timestamp", -1)]
                )
                
                async for doc in cursor:
                    # Extract conversation data
                    conversation = self._extract_conversation(doc, collection_name)
                    
                    if conversation:
                        # Deduplication
                        if filters.exclude_duplicates:
                            conv_hash = self._hash_conversation(conversation)
                            if conv_hash in seen_hashes:
                                continue
                            seen_hashes.add(conv_hash)
                        
                        conversations.append(conversation)
                        
                        # Check max samples
                        if max_samples and len(conversations) >= max_samples:
                            break
                
                if max_samples and len(conversations) >= max_samples:
                    break
                    
            except Exception as e:
                logger.warning(f"⚠️ Failed to collect from {collection_name}: {e}")
                continue
        
        return conversations
    
    def _extract_conversation(self, doc: Dict[str, Any], collection_name: str) -> Optional[Dict[str, Any]]:
        """
        Extract conversation from document
        
        Args:
            doc: MongoDB document
            collection_name: Collection name
            
        Returns:
            Conversation dictionary or None
        """
        try:
            # Different extraction logic based on collection
            if collection_name == "agent_conversations":
                return {
                    "user_message": doc.get("user_message", ""),
                    "agent_response": doc.get("agent_response", ""),
                    "context": doc.get("context", {}),
                    "metadata": {
                        "agent_id": doc.get("agent_id"),
                        "rating": doc.get("user_rating"),
                        "timestamp": doc.get("timestamp")
                    }
                }
            elif collection_name == "interactions":
                return {
                    "user_message": doc.get("request", ""),
                    "agent_response": doc.get("response", ""),
                    "context": {},
                    "metadata": {
                        "model_used": doc.get("model_used"),
                        "confidence": doc.get("confidence"),
                        "timestamp": doc.get("timestamp")
                    }
                }
            elif collection_name == "pai_learning_interactions":
                return {
                    "user_message": doc.get("user_input", ""),
                    "agent_response": doc.get("assistant_response", ""),
                    "context": doc.get("context", {}),
                    "metadata": {
                        "response_quality": doc.get("response_quality"),
                        "user_satisfaction": doc.get("user_satisfaction"),
                        "timestamp": doc.get("timestamp")
                    }
                }
            else:
                return None
                
        except Exception as e:
            logger.warning(f"⚠️ Failed to extract conversation: {e}")
            return None
    
    def _hash_conversation(self, conversation: Dict[str, Any]) -> str:
        """Generate hash for conversation deduplication"""
        text = f"{conversation['user_message']}{conversation['agent_response']}"
        return hashlib.md5(text.encode()).hexdigest()
    
    async def _export_to_jsonl(self, conversations: List[Dict[str, Any]], file_path: str) -> float:
        """
        Export conversations to JSONL format
        
        Args:
            conversations: List of conversations
            file_path: Output file path
            
        Returns:
            File size in MB
        """
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                for conv in conversations:
                    # Format as conversation text
                    sample = ConversationSample(
                        text=f"User: {conv['user_message']}\nAssistant: {conv['agent_response']}",
                        metadata=conv.get('metadata')
                    )
                    
                    # Write as JSONL
                    f.write(json.dumps({"text": sample.text}) + '\n')
            
            # Calculate file size
            file_size_bytes = os.path.getsize(file_path)
            file_size_mb = file_size_bytes / (1024 * 1024)
            
            return file_size_mb
            
        except Exception as e:
            logger.error(f"❌ Failed to export to JSONL: {e}")
            raise
    
    async def get_dataset(self, dataset_id: str) -> Optional[TrainingDataset]:
        """Get dataset by ID"""
        try:
            doc = await self.db.find_one(self.datasets_collection, {"dataset_id": dataset_id})
            if doc:
                return TrainingDataset(**doc)
            return None
        except Exception as e:
            logger.error(f"❌ Failed to get dataset: {e}")
            return None
    
    async def list_datasets(
        self,
        status: Optional[DatasetStatus] = None,
        limit: int = 100
    ) -> List[TrainingDataset]:
        """List datasets with optional filters"""
        try:
            query = {}
            if status:
                query["status"] = status.value
            
            cursor = self.db.find(
                self.datasets_collection,
                query,
                sort=[("created_at", -1)],
                limit=limit
            )
            
            datasets = []
            async for doc in cursor:
                datasets.append(TrainingDataset(**doc))
            
            return datasets
            
        except Exception as e:
            logger.error(f"❌ Failed to list datasets: {e}")
            return []
    
    async def delete_dataset(self, dataset_id: str) -> bool:
        """Delete dataset and its file"""
        try:
            dataset = await self.get_dataset(dataset_id)
            if not dataset:
                return False
            
            # Delete file
            if os.path.exists(dataset.file_path):
                os.remove(dataset.file_path)
            
            # Delete from database
            await self.db.delete_one(self.datasets_collection, {"dataset_id": dataset_id})
            
            logger.info(f"✅ Dataset deleted: {dataset_id}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to delete dataset: {e}")
            return False
    
    def _generate_dataset_id(self, name: str) -> str:
        """Generate unique dataset ID"""
        timestamp = datetime.utcnow().strftime("%Y%m%d%H%M%S")
        name_hash = hashlib.md5(name.encode()).hexdigest()[:8]
        return f"ds-{timestamp}-{name_hash}"

