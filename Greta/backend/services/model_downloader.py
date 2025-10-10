"""
Model Download Manager
Week 2 - Component 2: Model Download Manager

Handles model downloads from HuggingFace with progress tracking and resume capability.
"""

import os
import asyncio
from pathlib import Path
from typing import Optional, Callable, Dict
from huggingface_hub import hf_hub_download, list_repo_files
from huggingface_hub.utils import HfHubHTTPError
from loguru import logger
from tqdm import tqdm


class DownloadProgress:
    """Track download progress"""
    
    def __init__(self, model_name: str, total_size: int = 0):
        self.model_name = model_name
        self.total_size = total_size
        self.downloaded = 0
        self.progress_percent = 0.0
        self.status = "pending"  # pending, downloading, complete, error
        self.error_message: Optional[str] = None
        self.pbar: Optional[tqdm] = None
    
    def update(self, chunk_size: int):
        """Update progress"""
        self.downloaded += chunk_size
        if self.total_size > 0:
            self.progress_percent = (self.downloaded / self.total_size) * 100
        
        if self.pbar:
            self.pbar.update(chunk_size)
    
    def start(self):
        """Start progress tracking"""
        self.status = "downloading"
        if self.total_size > 0:
            self.pbar = tqdm(
                total=self.total_size,
                unit='B',
                unit_scale=True,
                desc=f"Downloading {self.model_name}"
            )
    
    def complete(self):
        """Mark as complete"""
        self.status = "complete"
        self.progress_percent = 100.0
        if self.pbar:
            self.pbar.close()
    
    def error(self, message: str):
        """Mark as error"""
        self.status = "error"
        self.error_message = message
        if self.pbar:
            self.pbar.close()


class ModelDownloader:
    """
    Model Download Manager
    
    Downloads models from HuggingFace Hub with progress tracking and resume capability.
    """
    
    def __init__(self, models_dir: Optional[str] = None):
        self.models_dir = Path(models_dir or os.getenv('CPAS4_MODELS', Path.home() / '.cpas4' / 'models'))
        self.models_dir.mkdir(parents=True, exist_ok=True)
        
        # Track active downloads
        self.active_downloads: Dict[str, DownloadProgress] = {}
        
        logger.info(f"Model Downloader initialized")
        logger.info(f"Download directory: {self.models_dir}")
    
    async def download_from_huggingface(
        self,
        repo_id: str,
        filename: str,
        model_name: Optional[str] = None,
        progress_callback: Optional[Callable[[float], None]] = None
    ) -> Optional[str]:
        """
        Download a model from HuggingFace Hub
        
        Args:
            repo_id: HuggingFace repository ID (e.g., "TheBloke/Llama-2-7B-GGUF")
            filename: File to download (e.g., "llama-2-7b.Q4_K_M.gguf")
            model_name: Optional custom name for tracking
            progress_callback: Optional callback for progress updates
        
        Returns:
            Path to downloaded file or None if failed
        """
        model_name = model_name or filename
        
        try:
            logger.info(f"Starting download: {repo_id}/{filename}")
            
            # Create progress tracker
            progress = DownloadProgress(model_name)
            self.active_downloads[model_name] = progress
            progress.start()
            
            # Download in thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            file_path = await loop.run_in_executor(
                None,
                lambda: hf_hub_download(
                    repo_id=repo_id,
                    filename=filename,
                    cache_dir=str(self.models_dir.parent / 'hf_cache'),
                    local_dir=str(self.models_dir),
                    local_dir_use_symlinks=False,
                    resume_download=True
                )
            )
            
            # Mark as complete
            progress.complete()
            logger.success(f"Download complete: {file_path}")
            
            # Cleanup
            del self.active_downloads[model_name]
            
            return file_path
            
        except HfHubHTTPError as e:
            error_msg = f"HuggingFace Hub error: {e}"
            logger.error(error_msg)
            if model_name in self.active_downloads:
                self.active_downloads[model_name].error(error_msg)
            return None
            
        except Exception as e:
            error_msg = f"Download failed: {e}"
            logger.error(error_msg)
            if model_name in self.active_downloads:
                self.active_downloads[model_name].error(error_msg)
            return None
    
    async def list_available_models(self, repo_id: str) -> list[str]:
        """List available model files in a HuggingFace repository"""
        try:
            loop = asyncio.get_event_loop()
            files = await loop.run_in_executor(
                None,
                lambda: list_repo_files(repo_id)
            )
            
            # Filter for GGUF files
            gguf_files = [f for f in files if f.endswith('.gguf')]
            
            logger.info(f"Found {len(gguf_files)} GGUF files in {repo_id}")
            return gguf_files
            
        except Exception as e:
            logger.error(f"Failed to list models from {repo_id}: {e}")
            return []
    
    def get_download_progress(self, model_name: str) -> Optional[DownloadProgress]:
        """Get download progress for a model"""
        return self.active_downloads.get(model_name)
    
    def cancel_download(self, model_name: str) -> bool:
        """Cancel an active download"""
        if model_name in self.active_downloads:
            progress = self.active_downloads[model_name]
            progress.error("Download cancelled by user")
            del self.active_downloads[model_name]
            logger.info(f"Cancelled download: {model_name}")
            return True
        return False
    
    async def download_recommended_models(self) -> Dict[str, Optional[str]]:
        """Download a set of recommended models"""
        recommended = {
            "llama-2-7b-q4": {
                "repo_id": "TheBloke/Llama-2-7B-GGUF",
                "filename": "llama-2-7b.Q4_K_M.gguf"
            },
            "mistral-7b-q4": {
                "repo_id": "TheBloke/Mistral-7B-Instruct-v0.2-GGUF",
                "filename": "mistral-7b-instruct-v0.2.Q4_K_M.gguf"
            },
            "phi-2-q4": {
                "repo_id": "TheBloke/phi-2-GGUF",
                "filename": "phi-2.Q4_K_M.gguf"
            }
        }
        
        results = {}
        
        for model_name, config in recommended.items():
            logger.info(f"Downloading recommended model: {model_name}")
            file_path = await self.download_from_huggingface(
                repo_id=config["repo_id"],
                filename=config["filename"],
                model_name=model_name
            )
            results[model_name] = file_path
        
        return results


# Global instance
model_downloader = ModelDownloader()
