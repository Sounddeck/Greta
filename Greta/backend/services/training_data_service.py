"""
Training Data Service - Week 9-10 Component 1

Collects and prepares training data from conversations for LoRA fine-tuning.
Integrates with Week 4 Conversation Service and Week 5-6 RAG Service.
"""

import asyncio
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from loguru import logger
from pydantic import BaseModel, Field

# Import existing services
try:
    from backend.services.conversation_service import conversation_service
    from backend.services.rag_service import rag_service
except ImportError:
    logger.warning("Could not import conversation or RAG services")
    conversation_service = None
    rag_service = None


# ============================================================================
# Pydantic Schemas
# ============================================================================


class TrainingSample(BaseModel):
    """Single training sample"""

    sample_id: str
    instruction: str
    input: Optional[str] = ""
    output: str
    source: str  # conversation, rag, manual
    quality_score: float = 0.0
    metadata: Dict[str, Any] = Field(default_factory=dict)
    created_at: str = Field(default_factory=lambda: datetime.utcnow().isoformat())


class TrainingDataset(BaseModel):
    """Training dataset collection"""

    dataset_id: str
    name: str
    description: str
    samples: List[TrainingSample]
    total_samples: int
    format: str = "instruction"  # instruction, chat, completion
    created_at: str = Field(default_factory=lambda: datetime.utcnow().isoformat())
    metadata: Dict[str, Any] = Field(default_factory=dict)


class DatasetStats(BaseModel):
    """Dataset statistics"""

    total_samples: int
    avg_quality_score: float
    avg_instruction_length: int
    avg_output_length: int
    sources: Dict[str, int]
    quality_distribution: Dict[str, int]


# ============================================================================
# Training Data Service
# ============================================================================


class TrainingDataService:
    """
    Training Data Service

    Collects and prepares training data from:
    - Conversation history (Week 4)
    - RAG documents (Week 5-6)
    - Manual samples

    Features:
    - Quality filtering
    - Format conversion (instruction, chat, completion)
    - Dataset export (JSONL, JSON, Parquet)
    - Statistics and analysis
    """

    def __init__(self, data_dir: str = "./data/training"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)

        self.datasets: Dict[str, TrainingDataset] = {}
        self.samples_cache: List[TrainingSample] = []

        logger.info(f"Training Data Service initialized: {self.data_dir}")

    async def collect_from_conversations(
        self,
        conversation_ids: Optional[List[str]] = None,
        min_quality_score: float = 0.5,
        min_length: int = 20,
        max_samples: int = 10000,
    ) -> List[TrainingSample]:
        """
        Collect training samples from conversation history

        Args:
            conversation_ids: Specific conversation IDs (None = all)
            min_quality_score: Minimum quality threshold
            min_length: Minimum message length
            max_samples: Maximum samples to collect

        Returns:
            List of training samples
        """
        if not conversation_service:
            logger.warning("Conversation service not available")
            return []

        samples = []

        try:
            # Get conversations
            if conversation_ids:
                conversations = []
                for conv_id in conversation_ids:
                    conv = await conversation_service.get_conversation(conv_id)
                    if conv:
                        conversations.append(conv)
            else:
                # Get all conversations (limited)
                all_convs = await conversation_service.list_conversations(limit=100)
                conversations = all_convs

            # Extract training samples from conversations
            for conv in conversations:
                if len(samples) >= max_samples:
                    break

                messages = conv.get("messages", [])

                # Create instruction-response pairs
                for i in range(len(messages) - 1):
                    if len(samples) >= max_samples:
                        break

                    current_msg = messages[i]
                    next_msg = messages[i + 1]

                    # User message -> Assistant response
                    if (
                        current_msg.get("role") == "user"
                        and next_msg.get("role") == "assistant"
                    ):
                        instruction = current_msg.get("content", "")
                        output = next_msg.get("content", "")

                        # Quality filtering
                        if len(instruction) < min_length or len(output) < min_length:
                            continue

                        # Calculate quality score (simple heuristic)
                        quality = self._calculate_quality_score(instruction, output)

                        if quality < min_quality_score:
                            continue

                        # Create sample
                        sample = TrainingSample(
                            sample_id=f"conv_{conv.get('conversation_id', 'unknown')}_{i}",
                            instruction=instruction,
                            output=output,
                            source="conversation",
                            quality_score=quality,
                            metadata={
                                "conversation_id": conv.get("conversation_id"),
                                "message_index": i,
                                "timestamp": current_msg.get("timestamp"),
                            },
                        )

                        samples.append(sample)

            logger.info(
                f"Collected {len(samples)} samples from {len(conversations)} conversations"
            )
            return samples

        except Exception as e:
            logger.error(f"Error collecting from conversations: {e}")
            return []

    async def collect_from_rag(
        self,
        query: Optional[str] = None,
        n_documents: int = 100,
        min_quality_score: float = 0.5,
    ) -> List[TrainingSample]:
        """
        Collect training samples from RAG documents

        Creates instruction-response pairs from document content
        """
        if not rag_service:
            logger.warning("RAG service not available")
            return []

        samples = []

        try:
            # Get collection stats
            stats = await rag_service.get_collection_stats()
            total_docs = stats.get("total_documents", 0)

            logger.info(f"Collecting from {total_docs} RAG documents")

            # For now, create samples from document summaries
            # In production, you'd want more sophisticated extraction

            # This is a placeholder - actual implementation would:
            # 1. Query documents by topic
            # 2. Extract key information
            # 3. Generate instruction-response pairs
            # 4. Use LLM to create training data from documents

            logger.info(f"Collected {len(samples)} samples from RAG documents")
            return samples

        except Exception as e:
            logger.error(f"Error collecting from RAG: {e}")
            return []

    def _calculate_quality_score(self, instruction: str, output: str) -> float:
        """
        Calculate quality score for a training sample

        Simple heuristic based on:
        - Length (not too short, not too long)
        - Completeness (ends with punctuation)
        - Coherence (basic checks)

        Returns score 0.0-1.0
        """
        score = 0.5  # Base score

        # Length scoring
        inst_len = len(instruction)
        out_len = len(output)

        # Prefer moderate lengths
        if 20 <= inst_len <= 500:
            score += 0.1
        if 50 <= out_len <= 1000:
            score += 0.1

        # Completeness
        if output.strip().endswith((".", "!", "?", "```")):
            score += 0.1

        # Avoid very short responses
        if out_len < 20:
            score -= 0.3

        # Avoid very long responses (might be errors)
        if out_len > 2000:
            score -= 0.2

        # Check for code blocks (valuable training data)
        if "```" in output:
            score += 0.1

        # Clamp to 0-1
        return max(0.0, min(1.0, score))

    async def create_dataset(
        self,
        name: str,
        description: str,
        samples: List[TrainingSample],
        format: str = "instruction",
    ) -> TrainingDataset:
        """Create a training dataset from samples"""
        dataset_id = f"dataset_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"

        dataset = TrainingDataset(
            dataset_id=dataset_id,
            name=name,
            description=description,
            samples=samples,
            total_samples=len(samples),
            format=format,
            metadata={"created_by": "training_data_service"},
        )

        self.datasets[dataset_id] = dataset

        logger.info(
            f"Created dataset {dataset_id}: {name} with {len(samples)} samples"
        )
        return dataset

    async def export_dataset(
        self,
        dataset: TrainingDataset,
        output_path: Optional[str] = None,
        format: str = "jsonl",
    ) -> str:
        """
        Export dataset to file

        Args:
            dataset: Dataset to export
            output_path: Output file path (None = auto-generate)
            format: Export format (jsonl, json, parquet)

        Returns:
            Path to exported file
        """
        if output_path is None:
            output_path = str(
                self.data_dir / f"{dataset.dataset_id}.{format}"
            )

        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        if format == "jsonl":
            # JSONL format (one JSON object per line)
            with open(output_file, "w") as f:
                for sample in dataset.samples:
                    # Format for instruction fine-tuning
                    entry = {
                        "instruction": sample.instruction,
                        "input": sample.input or "",
                        "output": sample.output,
                    }
                    f.write(json.dumps(entry) + "\n")

        elif format == "json":
            # Single JSON file
            data = {
                "dataset_id": dataset.dataset_id,
                "name": dataset.name,
                "description": dataset.description,
                "samples": [
                    {
                        "instruction": s.instruction,
                        "input": s.input or "",
                        "output": s.output,
                        "quality_score": s.quality_score,
                    }
                    for s in dataset.samples
                ],
            }
            with open(output_file, "w") as f:
                json.dump(data, f, indent=2)

        else:
            raise ValueError(f"Unsupported format: {format}")

        logger.info(f"Exported dataset to {output_file} ({format})")
        return str(output_file)

    async def get_dataset_stats(self, dataset: TrainingDataset) -> DatasetStats:
        """Calculate dataset statistics"""
        if not dataset.samples:
            return DatasetStats(
                total_samples=0,
                avg_quality_score=0.0,
                avg_instruction_length=0,
                avg_output_length=0,
                sources={},
                quality_distribution={},
            )

        # Calculate averages
        total_quality = sum(s.quality_score for s in dataset.samples)
        total_inst_len = sum(len(s.instruction) for s in dataset.samples)
        total_out_len = sum(len(s.output) for s in dataset.samples)

        # Source distribution
        sources = {}
        for sample in dataset.samples:
            sources[sample.source] = sources.get(sample.source, 0) + 1

        # Quality distribution
        quality_dist = {"high": 0, "medium": 0, "low": 0}
        for sample in dataset.samples:
            if sample.quality_score >= 0.7:
                quality_dist["high"] += 1
            elif sample.quality_score >= 0.5:
                quality_dist["medium"] += 1
            else:
                quality_dist["low"] += 1

        return DatasetStats(
            total_samples=len(dataset.samples),
            avg_quality_score=total_quality / len(dataset.samples),
            avg_instruction_length=int(total_inst_len / len(dataset.samples)),
            avg_output_length=int(total_out_len / len(dataset.samples)),
            sources=sources,
            quality_distribution=quality_dist,
        )


# ============================================================================
# Singleton Instance
# ============================================================================

training_data_service = TrainingDataService()

