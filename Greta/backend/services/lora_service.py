"""
LoRA Fine-Tuning Service - Week 9-10 Component 2

Implements LoRA (Low-Rank Adaptation) fine-tuning for local LLMs.
Uses PEFT library for efficient parameter-efficient fine-tuning.
"""

import asyncio
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from loguru import logger
from pydantic import BaseModel, Field

# PEFT and transformers imports
try:
    from peft import (
        LoraConfig,
        TaskType,
        get_peft_model,
        prepare_model_for_kbit_training,
    )
    PEFT_AVAILABLE = True
    logger.info("PEFT library loaded successfully")
except ImportError as e:
    logger.warning(f"PEFT not available: {e}")
    PEFT_AVAILABLE = False
    # Create placeholder classes
    class LoraConfig:
        pass
    class TaskType:
        CAUSAL_LM = "CAUSAL_LM"

try:
    # Import only what we need to avoid pyarrow issues
    from transformers import AutoModelForCausalLM, AutoTokenizer
    TRANSFORMERS_AVAILABLE = True
    logger.info("Transformers library loaded successfully")

    # Try to import training components separately
    try:
        from transformers import TrainingArguments, Trainer
        TRAINING_AVAILABLE = True
    except Exception as e:
        logger.warning(f"Transformers training components not available: {e}")
        TRAINING_AVAILABLE = False
        # Create placeholders
        class TrainingArguments:
            pass
        class Trainer:
            pass
except ImportError as e:
    logger.warning(f"Transformers not available: {e}")
    TRANSFORMERS_AVAILABLE = False
    TRAINING_AVAILABLE = False
    # Create placeholders
    class AutoModelForCausalLM:
        pass
    class AutoTokenizer:
        pass
    class TrainingArguments:
        pass
    class Trainer:
        pass

try:
    from datasets import Dataset
    DATASETS_AVAILABLE = True
    logger.info("Datasets library loaded successfully")
except (ImportError, AttributeError, RuntimeError) as e:
    logger.warning(f"Datasets not available: {e}")
    DATASETS_AVAILABLE = False
    # Create placeholder
    class Dataset:
        @staticmethod
        def from_dict(data):
            """Placeholder Dataset.from_dict"""
            return None


# ============================================================================
# Pydantic Schemas
# ============================================================================


class LoRAConfiguration(BaseModel):
    """LoRA configuration"""

    rank: int = 8  # LoRA rank (r)
    alpha: int = 16  # LoRA alpha
    dropout: float = 0.1  # Dropout probability
    target_modules: List[str] = Field(
        default_factory=lambda: ["q_proj", "v_proj"]
    )  # Modules to apply LoRA
    bias: str = "none"  # Bias type
    task_type: str = "CAUSAL_LM"  # Task type


class TrainingConfiguration(BaseModel):
    """Training configuration"""

    num_epochs: int = 3
    batch_size: int = 4
    learning_rate: float = 2e-4
    warmup_steps: int = 100
    max_steps: int = -1  # -1 = use num_epochs
    gradient_accumulation_steps: int = 4
    max_grad_norm: float = 0.3
    logging_steps: int = 10
    save_steps: int = 100
    eval_steps: int = 100
    fp16: bool = False  # Use FP16 (not supported on Mac Metal)
    bf16: bool = False  # Use BF16


class TrainingResult(BaseModel):
    """Training result"""

    adapter_id: str
    base_model: str
    training_loss: float
    eval_loss: Optional[float] = None
    training_time: float  # seconds
    num_samples: int
    num_epochs: int
    output_dir: str
    metadata: Dict[str, Any] = Field(default_factory=dict)
    created_at: str = Field(default_factory=lambda: datetime.utcnow().isoformat())


class EvaluationMetrics(BaseModel):
    """Evaluation metrics"""

    perplexity: float
    loss: float
    num_samples: int
    metadata: Dict[str, Any] = Field(default_factory=dict)


# ============================================================================
# LoRA Service
# ============================================================================


class LoRAService:
    """
    LoRA Fine-Tuning Service

    Features:
    - LoRA configuration management
    - Model fine-tuning with PEFT
    - Training monitoring
    - Evaluation metrics
    - Checkpoint management
    - Metal GPU acceleration (Mac)
    """

    def __init__(
        self,
        models_dir: str = "./data/models",
        adapters_dir: str = "./data/adapters",
        checkpoints_dir: str = "./data/checkpoints",
    ):
        if not PEFT_AVAILABLE:
            raise ImportError("PEFT not installed. Install with: pip install peft")

        self.models_dir = Path(models_dir)
        self.adapters_dir = Path(adapters_dir)
        self.checkpoints_dir = Path(checkpoints_dir)

        # Create directories
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.adapters_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoints_dir.mkdir(parents=True, exist_ok=True)

        # Device configuration (Metal GPU on Mac)
        if torch.backends.mps.is_available():
            self.device = torch.device("mps")
            logger.info("Using Metal GPU (MPS) for training")
        elif torch.cuda.is_available():
            self.device = torch.device("cuda")
            logger.info("Using CUDA GPU for training")
        else:
            self.device = torch.device("cpu")
            logger.warning("Using CPU for training (slow)")

        self.current_model = None
        self.current_tokenizer = None

        logger.info(f"LoRA Service initialized")
        logger.info(f"  Models: {self.models_dir}")
        logger.info(f"  Adapters: {self.adapters_dir}")
        logger.info(f"  Device: {self.device}")

    def create_lora_config(
        self,
        rank: int = 8,
        alpha: int = 16,
        dropout: float = 0.1,
        target_modules: Optional[List[str]] = None,
    ) -> LoraConfig:
        """
        Create LoRA configuration

        Args:
            rank: LoRA rank (r) - lower = fewer parameters
            alpha: LoRA alpha - scaling factor
            dropout: Dropout probability
            target_modules: Modules to apply LoRA (None = default)

        Returns:
            LoraConfig object
        """
        if target_modules is None:
            # Default target modules for Llama-style models
            target_modules = ["q_proj", "v_proj", "k_proj", "o_proj"]

        config = LoraConfig(
            r=rank,
            lora_alpha=alpha,
            lora_dropout=dropout,
            target_modules=target_modules,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
        )

        logger.info(f"Created LoRA config: rank={rank}, alpha={alpha}")
        return config

    async def load_model(
        self, model_name_or_path: str, load_in_8bit: bool = False
    ) -> tuple:
        """
        Load base model and tokenizer

        Args:
            model_name_or_path: HuggingFace model name or local path
            load_in_8bit: Use 8-bit quantization (saves memory)

        Returns:
            (model, tokenizer) tuple
        """
        try:
            logger.info(f"Loading model: {model_name_or_path}")

            # Load tokenizer
            tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)

            # Add padding token if missing
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token

            # Load model
            # Note: load_in_8bit requires bitsandbytes which may not work on Mac
            model = AutoModelForCausalLM.from_pretrained(
                model_name_or_path,
                device_map="auto" if str(self.device) != "mps" else None,
                torch_dtype=torch.float32,  # Use float32 for Mac Metal
            )

            # Move to device if not using device_map
            if str(self.device) == "mps":
                model = model.to(self.device)

            self.current_model = model
            self.current_tokenizer = tokenizer

            logger.info(f"Model loaded successfully")
            return model, tokenizer

        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise

    async def prepare_dataset(
        self, samples: List[Dict[str, str]], tokenizer: Any, max_length: int = 512
    ) -> Dataset:
        """
        Prepare dataset for training

        Args:
            samples: List of {instruction, input, output} dicts
            tokenizer: Tokenizer
            max_length: Maximum sequence length

        Returns:
            HuggingFace Dataset
        """

        def format_sample(sample):
            """Format sample for instruction fine-tuning"""
            instruction = sample.get("instruction", "")
            input_text = sample.get("input", "")
            output = sample.get("output", "")

            # Alpaca-style prompt
            if input_text:
                prompt = f"### Instruction:\n{instruction}\n\n### Input:\n{input_text}\n\n### Response:\n{output}"
            else:
                prompt = f"### Instruction:\n{instruction}\n\n### Response:\n{output}"

            return prompt

        # Format all samples
        formatted_texts = [format_sample(s) for s in samples]

        # Tokenize
        def tokenize_function(examples):
            return tokenizer(
                examples["text"],
                truncation=True,
                max_length=max_length,
                padding="max_length",
            )

        # Create dataset
        dataset = Dataset.from_dict({"text": formatted_texts})
        tokenized_dataset = dataset.map(
            tokenize_function, batched=True, remove_columns=["text"]
        )

        logger.info(f"Prepared dataset with {len(tokenized_dataset)} samples")
        return tokenized_dataset

    async def fine_tune(
        self,
        base_model: str,
        dataset: Dataset,
        lora_config: LoraConfig,
        training_config: TrainingConfiguration,
        adapter_name: str = "default",
    ) -> TrainingResult:
        """
        Fine-tune model with LoRA

        Args:
            base_model: Base model name or path
            dataset: Training dataset
            lora_config: LoRA configuration
            training_config: Training configuration
            adapter_name: Name for the adapter

        Returns:
            TrainingResult
        """
        start_time = datetime.utcnow()

        try:
            # Load model if not already loaded
            if self.current_model is None:
                model, tokenizer = await self.load_model(base_model)
            else:
                model = self.current_model
                tokenizer = self.current_tokenizer

            # Apply LoRA
            logger.info("Applying LoRA to model...")
            model = get_peft_model(model, lora_config)
            model.print_trainable_parameters()

            # Output directory
            adapter_id = f"{adapter_name}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
            output_dir = self.adapters_dir / adapter_id

            # Training arguments
            training_args = TrainingArguments(
                output_dir=str(output_dir),
                num_train_epochs=training_config.num_epochs,
                per_device_train_batch_size=training_config.batch_size,
                gradient_accumulation_steps=training_config.gradient_accumulation_steps,
                learning_rate=training_config.learning_rate,
                warmup_steps=training_config.warmup_steps,
                max_steps=training_config.max_steps,
                logging_steps=training_config.logging_steps,
                save_steps=training_config.save_steps,
                max_grad_norm=training_config.max_grad_norm,
                fp16=training_config.fp16,
                bf16=training_config.bf16,
                logging_dir=str(output_dir / "logs"),
                save_total_limit=3,
                load_best_model_at_end=False,
                report_to="none",  # Disable wandb/tensorboard for now
            )

            # Create trainer
            trainer = Trainer(
                model=model,
                args=training_args,
                train_dataset=dataset,
            )

            # Train
            logger.info(f"Starting training: {adapter_id}")
            train_result = trainer.train()

            # Save adapter
            logger.info(f"Saving adapter to {output_dir}")
            model.save_pretrained(str(output_dir))
            tokenizer.save_pretrained(str(output_dir))

            # Calculate training time
            end_time = datetime.utcnow()
            training_time = (end_time - start_time).total_seconds()

            # Create result
            result = TrainingResult(
                adapter_id=adapter_id,
                base_model=base_model,
                training_loss=train_result.training_loss,
                training_time=training_time,
                num_samples=len(dataset),
                num_epochs=training_config.num_epochs,
                output_dir=str(output_dir),
                metadata={
                    "lora_rank": lora_config.r,
                    "lora_alpha": lora_config.lora_alpha,
                    "learning_rate": training_config.learning_rate,
                },
            )

            logger.info(f"Training complete: {adapter_id}")
            logger.info(f"  Loss: {result.training_loss:.4f}")
            logger.info(f"  Time: {result.training_time:.1f}s")

            return result

        except Exception as e:
            logger.error(f"Error during fine-tuning: {e}")
            raise


# ============================================================================
# Singleton Instance
# ============================================================================

lora_service: Optional[LoRAService] = None


def get_lora_service() -> LoRAService:
    """Get or create LoRA service instance"""
    global lora_service
    if lora_service is None:
        lora_service = LoRAService()
    return lora_service

