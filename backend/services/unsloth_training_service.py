"""
Unsloth Training Service
Manages LLM fine-tuning using Unsloth library
"""

import logging
import os
import asyncio
import json
import re
from typing import Dict, Any, Optional
from datetime import datetime
from pathlib import Path
import hashlib

from models.training_models import (
    TrainingJob,
    TrainingJobCreate,
    TrainingJobStatus,
    TrainingProgress
)
from database import database

logger = logging.getLogger(__name__)


class UnslothTrainingService:
    """Service for managing Unsloth training jobs"""

    def __init__(self):
        """Initialize Unsloth training service"""
        self.db = database
        self.jobs_collection = "training_jobs"
        self.output_dir = Path("./models/training_output")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.active_processes: Dict[str, asyncio.subprocess.Process] = {}
        logger.info("✅ Unsloth Training Service initialized")
    
    async def create_training_job(self, job_create: TrainingJobCreate) -> TrainingJob:
        """
        Create a new training job
        
        Args:
            job_create: Training job creation request
            
        Returns:
            TrainingJob: Created training job
        """
        try:
            # Generate job ID
            job_id = self._generate_job_id()
            
            # Create output paths
            output_paths = {
                "lora_adapter": str(self.output_dir / job_id / "lora_model"),
                "merged_model": str(self.output_dir / job_id / "merged_model"),
                "gguf_model": str(self.output_dir / job_id / "model.gguf"),
                "quantized_model": str(self.output_dir / job_id / f"model-{job_create.quantization_type.value}.gguf")
            }
            
            # Create training job
            job = TrainingJob(
                job_id=job_id,
                dataset_id=job_create.dataset_id,
                base_model=job_create.base_model,
                base_model_path=job_create.base_model_path,
                training_config=job_create.training_config,
                status=TrainingJobStatus.PENDING,
                output_paths=output_paths
            )
            
            # Save to database
            await self.db.insert_one(
                self.jobs_collection,
                job.model_dump()
            )
            
            logger.info(f"✅ Training job created: {job_id}")
            
            # Start training asynchronously
            asyncio.create_task(self._run_training(job))
            
            return job
            
        except Exception as e:
            logger.error(f"❌ Failed to create training job: {e}")
            raise
    
    async def _run_training(self, job: TrainingJob):
        """
        Run training job
        
        Args:
            job: Training job to run
        """
        try:
            # Update status to running
            job.status = TrainingJobStatus.RUNNING
            job.started_at = datetime.utcnow()
            
            await self.db.update_one(
                self.jobs_collection,
                {"job_id": job.job_id},
                {
                    "$set": {
                        "status": job.status.value,
                        "started_at": job.started_at
                    }
                }
            )
            
            logger.info(f"🚀 Starting training job: {job.job_id}")
            
            # Generate training script
            script_path = await self._generate_training_script(job)
            
            # Execute training script
            await self._execute_training_script(job, script_path)
            
            # If quantize is enabled, convert and quantize
            if hasattr(job, 'quantize') and job.quantize:
                await self._convert_and_quantize(job)
            
            # Update status to completed
            job.status = TrainingJobStatus.COMPLETED
            job.completed_at = datetime.utcnow()
            
            await self.db.update_one(
                self.jobs_collection,
                {"job_id": job.job_id},
                {
                    "$set": {
                        "status": job.status.value,
                        "completed_at": job.completed_at
                    }
                }
            )
            
            logger.info(f"✅ Training job completed: {job.job_id}")
            
        except Exception as e:
            logger.error(f"❌ Training job failed: {job.job_id} - {e}")
            
            # Update status to failed
            await self.db.update_one(
                self.jobs_collection,
                {"job_id": job.job_id},
                {
                    "$set": {
                        "status": TrainingJobStatus.FAILED.value,
                        "error": str(e),
                        "completed_at": datetime.utcnow()
                    }
                }
            )
    
    async def _generate_training_script(self, job: TrainingJob) -> str:
        """
        Generate Unsloth training script
        
        Args:
            job: Training job
            
        Returns:
            Path to generated script
        """
        script_dir = self.output_dir / job.job_id
        script_dir.mkdir(parents=True, exist_ok=True)
        script_path = script_dir / "train.py"
        
        # Get dataset path
        dataset_doc = await self.db.find_one("training_datasets", {"dataset_id": job.dataset_id})
        dataset_path = dataset_doc.get("file_path") if dataset_doc else None
        
        if not dataset_path:
            raise ValueError(f"Dataset not found: {job.dataset_id}")
        
        config = job.training_config
        
        script_content = f'''"""
Auto-generated Unsloth training script
Job ID: {job.job_id}
"""

import torch
from unsloth import FastLanguageModel
from datasets import load_dataset
from trl import SFTTrainer
from transformers import TrainingArguments
import json

# Load model
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="{job.base_model}",
    max_seq_length={config.ctx_size},
    dtype=None,
    load_in_4bit=True,
)

# Add LoRA adapters
model = FastLanguageModel.get_peft_model(
    model,
    r={config.lora_rank},
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_alpha={config.lora_alpha},
    lora_dropout=0,
    bias="none",
    use_gradient_checkpointing="unsloth",
    random_state=3407,
)

# Load dataset
dataset = load_dataset("json", data_files="{dataset_path}", split="train")

# Training arguments
training_args = TrainingArguments(
    per_device_train_batch_size={config.batch_size},
    gradient_accumulation_steps=4,
    warmup_steps={config.warmup_steps},
    num_train_epochs={config.epochs},
    learning_rate={config.learning_rate},
    fp16=not torch.cuda.is_bf16_supported(),
    bf16=torch.cuda.is_bf16_supported(),
    logging_steps=1,
    optim="adamw_8bit",
    weight_decay=0.01,
    lr_scheduler_type="linear",
    seed=3407,
    output_dir="{script_dir / 'outputs'}",
    save_steps={config.save_every},
    save_total_limit=3,
)

# Create trainer
trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    dataset_text_field="text",
    max_seq_length={config.ctx_size},
    args=training_args,
)

# Train
print("🚀 Starting training...")
trainer.train()

# Save LoRA adapter
print("💾 Saving LoRA adapter...")
model.save_pretrained("{job.output_paths['lora_adapter']}")
tokenizer.save_pretrained("{job.output_paths['lora_adapter']}")

print("✅ Training complete!")
'''
        
        with open(script_path, 'w') as f:
            f.write(script_content)
        
        logger.info(f"📝 Training script generated: {script_path}")
        return str(script_path)
    
    async def _execute_training_script(self, job: TrainingJob, script_path: str):
        """
        Execute training script and monitor progress
        
        Args:
            job: Training job
            script_path: Path to training script
        """
        try:
            # Create subprocess
            process = await asyncio.create_subprocess_exec(
                "python", script_path,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT
            )
            
            self.active_processes[job.job_id] = process
            
            # Monitor output
            while True:
                line = await process.stdout.readline()
                if not line:
                    break
                
                line_str = line.decode().strip()
                logger.info(f"[{job.job_id}] {line_str}")
                
                # Parse progress
                progress = self._parse_training_output(line_str)
                if progress:
                    await self._update_progress(job.job_id, progress)
                
                # Store log
                await self.db.update_one(
                    self.jobs_collection,
                    {"job_id": job.job_id},
                    {"$push": {"logs": line_str}}
                )
            
            # Wait for completion
            await process.wait()
            
            if process.returncode != 0:
                raise RuntimeError(f"Training script failed with code {process.returncode}")
            
            # Remove from active processes
            del self.active_processes[job.job_id]
            
        except Exception as e:
            logger.error(f"❌ Training execution failed: {e}")
            if job.job_id in self.active_processes:
                del self.active_processes[job.job_id]
            raise
    
    def _parse_training_output(self, line: str) -> Optional[Dict[str, Any]]:
        """
        Parse training output for progress
        
        Args:
            line: Output line
            
        Returns:
            Progress dictionary or None
        """
        try:
            # Look for JSON-like output from transformers
            if "{'loss':" in line or '{"loss":' in line:
                # Extract JSON
                match = re.search(r'\{[^}]+\}', line)
                if match:
                    data = eval(match.group())  # Safe here as we control the output
                    return {
                        "loss": data.get("loss"),
                        "learning_rate": data.get("learning_rate"),
                        "epoch": data.get("epoch")
                    }
        except Exception as e:
            logger.debug(f"Failed to parse line: {line} - {e}")
        
        return None
    
    async def _update_progress(self, job_id: str, progress_data: Dict[str, Any]):
        """Update job progress in database"""
        try:
            update_fields = {}
            
            if "loss" in progress_data:
                update_fields["progress.loss"] = progress_data["loss"]
            if "learning_rate" in progress_data:
                update_fields["progress.learning_rate"] = progress_data["learning_rate"]
            if "epoch" in progress_data:
                update_fields["progress.current_epoch"] = int(progress_data["epoch"])
            
            if update_fields:
                await self.db.update_one(
                    self.jobs_collection,
                    {"job_id": job_id},
                    {"$set": update_fields}
                )
        except Exception as e:
            logger.warning(f"Failed to update progress: {e}")
    
    async def _convert_and_quantize(self, job: TrainingJob):
        """Convert LoRA to GGUF and quantize"""
        # TODO: Implement in model conversion service
        logger.info(f"⏳ Model conversion not implemented yet for job: {job.job_id}")
    
    async def get_job(self, job_id: str) -> Optional[TrainingJob]:
        """Get training job by ID"""
        try:
            doc = await self.db.find_one(self.jobs_collection, {"job_id": job_id})
            if doc:
                return TrainingJob(**doc)
            return None
        except Exception as e:
            logger.error(f"❌ Failed to get job: {e}")
            return None
    
    async def list_jobs(self, status: Optional[TrainingJobStatus] = None, limit: int = 100) -> list:
        """List training jobs"""
        try:
            query = {}
            if status:
                query["status"] = status.value
            
            cursor = self.db.find(
                self.jobs_collection,
                query,
                sort=[("created_at", -1)],
                limit=limit
            )
            
            jobs = []
            async for doc in cursor:
                jobs.append(TrainingJob(**doc))
            
            return jobs
        except Exception as e:
            logger.error(f"❌ Failed to list jobs: {e}")
            return []
    
    async def cancel_job(self, job_id: str) -> bool:
        """Cancel running training job"""
        try:
            if job_id in self.active_processes:
                process = self.active_processes[job_id]
                process.terminate()
                await process.wait()
                del self.active_processes[job_id]
                
                await self.db.update_one(
                    self.jobs_collection,
                    {"job_id": job_id},
                    {
                        "$set": {
                            "status": TrainingJobStatus.CANCELLED.value,
                            "completed_at": datetime.utcnow()
                        }
                    }
                )
                
                logger.info(f"✅ Job cancelled: {job_id}")
                return True
            
            return False
        except Exception as e:
            logger.error(f"❌ Failed to cancel job: {e}")
            return False
    
    def _generate_job_id(self) -> str:
        """Generate unique job ID"""
        timestamp = datetime.utcnow().strftime("%Y%m%d%H%M%S")
        random_hash = hashlib.md5(str(datetime.utcnow()).encode()).hexdigest()[:8]
        return f"job-{timestamp}-{random_hash}"

