"""
Model Conversion Service
Handles LoRA merging, GGUF conversion, and model quantization
"""

import logging
import os
import asyncio
import shutil
from typing import Dict, Any, Optional, List
from datetime import datetime
from pathlib import Path

from models.training_models import (
    TrainingJob,
    ModelVersion,
    QuantizationType,
    DeploymentStatus
)
from database import database
from config import get_settings

logger = logging.getLogger(__name__)


class ModelConversionService:
    """Service for converting trained models to GGUF format"""

    def __init__(self):
        """Initialize model conversion service"""
        self.db = database
        self.config = get_settings()
        self.jobs_collection = "training_jobs"
        self.versions_collection = "model_versions"
        
        # llama.cpp paths
        self.llamacpp_dir = Path(os.getenv("LLAMACPP_DIR", "./llama.cpp"))
        self.convert_script = Path(os.getenv(
            "LLAMACPP_CONVERT_SCRIPT",
            str(self.llamacpp_dir / "convert_hf_to_gguf.py")
        ))
        self.quantize_bin = Path(os.getenv(
            "LLAMACPP_QUANTIZE_BIN",
            str(self.llamacpp_dir / "build" / "bin" / "llama-quantize")
        ))
        
        logger.info("✅ Model Conversion Service initialized")
        self._check_dependencies()
    
    def _check_dependencies(self):
        """Check if required tools are available"""
        if not self.convert_script.exists():
            logger.warning(f"⚠️ convert_hf_to_gguf.py not found at {self.convert_script}")
        
        if not self.quantize_bin.exists():
            logger.warning(f"⚠️ llama-quantize not found at {self.quantize_bin}")
    
    async def convert_job_to_model(
        self,
        job_id: str,
        output_name: str,
        description: str = "",
        quantization_types: Optional[List[QuantizationType]] = None
    ) -> ModelVersion:
        """
        Convert a completed training job to a deployable model
        
        Args:
            job_id: Training job ID
            output_name: Name for the model version
            description: Model description
            quantization_types: List of quantization types to create
            
        Returns:
            ModelVersion: Created model version
        """
        try:
            # Get training job
            job_doc = await self.db.find_one(self.jobs_collection, {"job_id": job_id})
            if not job_doc:
                raise ValueError(f"Training job not found: {job_id}")
            
            job = TrainingJob(**job_doc)
            
            # Verify job is completed
            if job.status.value != "completed":
                raise ValueError(f"Training job not completed: {job.status.value}")
            
            # Default quantization types
            if quantization_types is None:
                quantization_types = [QuantizationType.Q4_K_M]
            
            logger.info(f"🔄 Starting model conversion for job: {job_id}")
            
            # Step 1: Merge LoRA adapter
            merged_model_path = await self._merge_lora_adapter(job)
            
            # Step 2: Convert to GGUF (F16)
            gguf_f16_path = await self._convert_to_gguf(job, merged_model_path)
            
            # Step 3: Quantize
            quantized_paths = {}
            model_sizes = {}
            
            for quant_type in quantization_types:
                quant_path = await self._quantize_model(job, gguf_f16_path, quant_type)
                quantized_paths[quant_type.value] = str(quant_path)
                model_sizes[f"gguf_{quant_type.value}_mb"] = self._get_file_size_mb(quant_path)
            
            # Create model version
            version = await self._create_model_version(
                job=job,
                output_name=output_name,
                description=description,
                merged_model_path=merged_model_path,
                gguf_f16_path=gguf_f16_path,
                quantized_paths=quantized_paths,
                model_sizes=model_sizes
            )
            
            logger.info(f"✅ Model conversion complete: {version.version_id}")
            
            return version
            
        except Exception as e:
            logger.error(f"❌ Model conversion failed: {e}")
            raise
    
    async def _merge_lora_adapter(self, job: TrainingJob) -> Path:
        """
        Merge LoRA adapter with base model using Unsloth
        
        Args:
            job: Training job
            
        Returns:
            Path to merged model
        """
        try:
            logger.info(f"🔄 Merging LoRA adapter for job: {job.job_id}")
            
            lora_path = job.output_paths["lora_adapter"]
            merged_path = job.output_paths["merged_model"]
            
            # Create merge script
            script_path = Path(lora_path).parent / "merge.py"
            script_content = f'''"""
Auto-generated LoRA merge script
Job ID: {job.job_id}
"""

from unsloth import FastLanguageModel

# Load model with LoRA adapter
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="{lora_path}",
    max_seq_length={job.training_config.ctx_size},
    dtype=None,
    load_in_4bit=True,
)

# Merge LoRA weights
print("🔄 Merging LoRA adapter...")
model.save_pretrained_merged(
    "{merged_path}",
    tokenizer,
    save_method="merged_16bit"
)

print("✅ LoRA merge complete!")
'''
            
            with open(script_path, 'w') as f:
                f.write(script_content)
            
            # Execute merge script
            process = await asyncio.create_subprocess_exec(
                "python", str(script_path),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT
            )
            
            # Monitor output
            while True:
                line = await process.stdout.readline()
                if not line:
                    break
                logger.info(f"[merge] {line.decode().strip()}")
            
            await process.wait()
            
            if process.returncode != 0:
                raise RuntimeError(f"LoRA merge failed with code {process.returncode}")
            
            logger.info(f"✅ LoRA adapter merged: {merged_path}")
            
            return Path(merged_path)
            
        except Exception as e:
            logger.error(f"❌ LoRA merge failed: {e}")
            raise
    
    async def _convert_to_gguf(self, job: TrainingJob, merged_model_path: Path) -> Path:
        """
        Convert HuggingFace model to GGUF format
        
        Args:
            job: Training job
            merged_model_path: Path to merged model
            
        Returns:
            Path to GGUF model
        """
        try:
            logger.info(f"🔄 Converting to GGUF for job: {job.job_id}")
            
            gguf_path = Path(job.output_paths["gguf_model"])
            
            # Execute conversion
            process = await asyncio.create_subprocess_exec(
                "python",
                str(self.convert_script),
                str(merged_model_path),
                "--outfile", str(gguf_path),
                "--outtype", "f16",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT
            )
            
            # Monitor output
            while True:
                line = await process.stdout.readline()
                if not line:
                    break
                logger.info(f"[gguf] {line.decode().strip()}")
            
            await process.wait()
            
            if process.returncode != 0:
                raise RuntimeError(f"GGUF conversion failed with code {process.returncode}")
            
            logger.info(f"✅ GGUF conversion complete: {gguf_path}")
            
            return gguf_path
            
        except Exception as e:
            logger.error(f"❌ GGUF conversion failed: {e}")
            raise
    
    async def _quantize_model(
        self,
        job: TrainingJob,
        gguf_f16_path: Path,
        quant_type: QuantizationType
    ) -> Path:
        """
        Quantize GGUF model
        
        Args:
            job: Training job
            gguf_f16_path: Path to F16 GGUF model
            quant_type: Quantization type
            
        Returns:
            Path to quantized model
        """
        try:
            logger.info(f"🔄 Quantizing model to {quant_type.value} for job: {job.job_id}")
            
            # Determine output path
            output_dir = Path(job.output_paths["gguf_model"]).parent
            quant_path = output_dir / f"model-{quant_type.value.lower()}.gguf"
            
            # Execute quantization
            process = await asyncio.create_subprocess_exec(
                str(self.quantize_bin),
                str(gguf_f16_path),
                str(quant_path),
                quant_type.value,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT
            )
            
            # Monitor output
            while True:
                line = await process.stdout.readline()
                if not line:
                    break
                logger.info(f"[quant] {line.decode().strip()}")
            
            await process.wait()
            
            if process.returncode != 0:
                raise RuntimeError(f"Quantization failed with code {process.returncode}")
            
            logger.info(f"✅ Quantization complete: {quant_path}")
            
            return quant_path
            
        except Exception as e:
            logger.error(f"❌ Quantization failed: {e}")
            raise
    
    async def _create_model_version(
        self,
        job: TrainingJob,
        output_name: str,
        description: str,
        merged_model_path: Path,
        gguf_f16_path: Path,
        quantized_paths: Dict[str, str],
        model_sizes: Dict[str, float]
    ) -> ModelVersion:
        """Create model version record in database"""
        try:
            # Generate version ID
            version_id = self._generate_version_id()
            
            # Build model paths
            model_paths = {
                "lora_adapter": job.output_paths["lora_adapter"],
                "merged_model": str(merged_model_path),
                "gguf_f16": str(gguf_f16_path),
                **quantized_paths
            }
            
            # Calculate sizes
            all_sizes = {
                "lora_adapter_mb": self._get_file_size_mb(job.output_paths["lora_adapter"]),
                "merged_model_mb": self._get_file_size_mb(merged_model_path),
                "gguf_f16_mb": self._get_file_size_mb(gguf_f16_path),
                **model_sizes
            }
            
            # Extract performance metrics from job
            performance_metrics = {}
            if job.progress:
                performance_metrics["training_loss"] = job.progress.loss
            
            # Create model version
            version = ModelVersion(
                version_id=version_id,
                job_id=job.job_id,
                dataset_id=job.dataset_id,
                name=output_name,
                description=description,
                base_model=job.base_model,
                model_paths=model_paths,
                model_sizes=all_sizes,
                training_config=job.training_config.model_dump(),
                performance_metrics=performance_metrics,
                deployment_status=DeploymentStatus.TESTING,
                is_active=False
            )
            
            # Save to database
            await self.db.insert_one(
                self.versions_collection,
                version.model_dump()
            )
            
            logger.info(f"✅ Model version created: {version_id}")
            
            return version
            
        except Exception as e:
            logger.error(f"❌ Failed to create model version: {e}")
            raise
    
    def _get_file_size_mb(self, path: str) -> float:
        """Get file or directory size in MB"""
        try:
            path_obj = Path(path)
            
            if path_obj.is_file():
                return path_obj.stat().st_size / (1024 * 1024)
            elif path_obj.is_dir():
                total_size = sum(
                    f.stat().st_size
                    for f in path_obj.rglob('*')
                    if f.is_file()
                )
                return total_size / (1024 * 1024)
            else:
                return 0.0
        except Exception as e:
            logger.warning(f"Failed to get file size for {path}: {e}")
            return 0.0
    
    def _generate_version_id(self) -> str:
        """Generate unique version ID"""
        import hashlib
        timestamp = datetime.utcnow().strftime("%Y%m%d%H%M%S")
        random_hash = hashlib.md5(str(datetime.utcnow()).encode()).hexdigest()[:8]
        return f"v-{timestamp}-{random_hash}"

