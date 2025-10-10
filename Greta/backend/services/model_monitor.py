"""
Model Performance Monitoring
Week 2 - Component 4: Performance Monitoring System

Tracks inference speed, memory usage, and quality metrics.
"""

import time
import psutil
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime, timedelta
from pydantic import BaseModel
from loguru import logger
import json


class PerformanceMetrics(BaseModel):
    """Performance metrics for a single inference"""
    model_name: str
    timestamp: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    inference_time_ms: float
    tokens_per_second: float
    memory_used_mb: float
    gpu_memory_used_mb: float = 0.0
    temperature: float
    success: bool = True
    error_message: Optional[str] = None


class ModelPerformanceMonitor:
    """
    Model Performance Monitor
    
    Tracks and aggregates performance metrics for model inference.
    """
    
    def __init__(self, metrics_dir: Optional[str] = None):
        self.metrics_dir = Path(metrics_dir or Path.home() / '.cpas4' / 'metrics')
        self.metrics_dir.mkdir(parents=True, exist_ok=True)
        
        # In-memory metrics (last 1000 per model)
        self.metrics: Dict[str, List[PerformanceMetrics]] = {}
        self.max_metrics_per_model = 1000
        
        # Active inference tracking
        self.active_inferences: Dict[str, dict] = {}
        
        logger.info(f"Performance Monitor initialized")
        logger.info(f"Metrics directory: {self.metrics_dir}")
    
    def start_inference(self, model_name: str, prompt_tokens: int) -> str:
        """Start tracking an inference"""
        inference_id = f"{model_name}_{int(time.time() * 1000)}"
        
        self.active_inferences[inference_id] = {
            'model_name': model_name,
            'prompt_tokens': prompt_tokens,
            'start_time': time.time(),
            'start_memory': psutil.Process().memory_info().rss / (1024 * 1024)
        }
        
        return inference_id
    
    def end_inference(
        self,
        inference_id: str,
        completion_tokens: int,
        temperature: float,
        success: bool = True,
        error_message: Optional[str] = None
    ) -> Optional[PerformanceMetrics]:
        """End tracking an inference and record metrics"""
        if inference_id not in self.active_inferences:
            logger.warning(f"Inference ID not found: {inference_id}")
            return None
        
        inference_data = self.active_inferences[inference_id]
        end_time = time.time()
        end_memory = psutil.Process().memory_info().rss / (1024 * 1024)
        
        # Calculate metrics
        inference_time_ms = (end_time - inference_data['start_time']) * 1000
        total_tokens = inference_data['prompt_tokens'] + completion_tokens
        tokens_per_second = (completion_tokens / (inference_time_ms / 1000)) if inference_time_ms > 0 else 0
        memory_used_mb = end_memory - inference_data['start_memory']
        
        # Create metrics object
        metrics = PerformanceMetrics(
            model_name=inference_data['model_name'],
            timestamp=datetime.utcnow().isoformat(),
            prompt_tokens=inference_data['prompt_tokens'],
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            inference_time_ms=round(inference_time_ms, 2),
            tokens_per_second=round(tokens_per_second, 2),
            memory_used_mb=round(memory_used_mb, 2),
            temperature=temperature,
            success=success,
            error_message=error_message
        )
        
        # Store metrics
        model_name = inference_data['model_name']
        if model_name not in self.metrics:
            self.metrics[model_name] = []
        
        self.metrics[model_name].append(metrics)
        
        # Limit metrics per model
        if len(self.metrics[model_name]) > self.max_metrics_per_model:
            self.metrics[model_name] = self.metrics[model_name][-self.max_metrics_per_model:]
        
        # Save to disk periodically (every 10 inferences)
        if len(self.metrics[model_name]) % 10 == 0:
            self._save_metrics(model_name)
        
        # Cleanup
        del self.active_inferences[inference_id]
        
        logger.debug(f"Recorded metrics: {model_name} - {tokens_per_second:.2f} tok/s")
        
        return metrics
    
    def get_model_stats(self, model_name: str, hours: int = 24) -> Dict:
        """Get aggregated statistics for a model"""
        if model_name not in self.metrics:
            return {
                'model_name': model_name,
                'total_inferences': 0,
                'error': 'No metrics available'
            }
        
        # Filter by time window
        cutoff_time = datetime.utcnow() - timedelta(hours=hours)
        recent_metrics = [
            m for m in self.metrics[model_name]
            if datetime.fromisoformat(m.timestamp) > cutoff_time
        ]
        
        if not recent_metrics:
            return {
                'model_name': model_name,
                'total_inferences': 0,
                'error': f'No metrics in last {hours} hours'
            }
        
        # Calculate statistics
        successful = [m for m in recent_metrics if m.success]
        
        avg_tokens_per_second = sum(m.tokens_per_second for m in successful) / len(successful) if successful else 0
        avg_inference_time = sum(m.inference_time_ms for m in successful) / len(successful) if successful else 0
        avg_memory_used = sum(m.memory_used_mb for m in successful) / len(successful) if successful else 0
        
        total_tokens = sum(m.total_tokens for m in recent_metrics)
        success_rate = (len(successful) / len(recent_metrics)) * 100 if recent_metrics else 0
        
        return {
            'model_name': model_name,
            'time_window_hours': hours,
            'total_inferences': len(recent_metrics),
            'successful_inferences': len(successful),
            'failed_inferences': len(recent_metrics) - len(successful),
            'success_rate_percent': round(success_rate, 2),
            'total_tokens_processed': total_tokens,
            'avg_tokens_per_second': round(avg_tokens_per_second, 2),
            'avg_inference_time_ms': round(avg_inference_time, 2),
            'avg_memory_used_mb': round(avg_memory_used, 2),
            'min_tokens_per_second': round(min((m.tokens_per_second for m in successful), default=0), 2),
            'max_tokens_per_second': round(max((m.tokens_per_second for m in successful), default=0), 2)
        }
    
    def get_all_models_stats(self, hours: int = 24) -> List[Dict]:
        """Get statistics for all models"""
        return [self.get_model_stats(model_name, hours) for model_name in self.metrics.keys()]
    
    def _save_metrics(self, model_name: str):
        """Save metrics to disk"""
        try:
            metrics_file = self.metrics_dir / f"{model_name}_metrics.json"
            data = {
                'model_name': model_name,
                'metrics': [m.model_dump() for m in self.metrics[model_name]],
                'updated_at': datetime.utcnow().isoformat()
            }
            
            with open(metrics_file, 'w') as f:
                json.dump(data, f, indent=2)
            
            logger.debug(f"Saved metrics for {model_name}")
            
        except Exception as e:
            logger.error(f"Failed to save metrics for {model_name}: {e}")
    
    def load_metrics(self, model_name: str) -> bool:
        """Load metrics from disk"""
        try:
            metrics_file = self.metrics_dir / f"{model_name}_metrics.json"
            
            if not metrics_file.exists():
                return False
            
            with open(metrics_file, 'r') as f:
                data = json.load(f)
            
            self.metrics[model_name] = [
                PerformanceMetrics(**m) for m in data.get('metrics', [])
            ]
            
            logger.info(f"Loaded {len(self.metrics[model_name])} metrics for {model_name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load metrics for {model_name}: {e}")
            return False
    
    def clear_metrics(self, model_name: Optional[str] = None):
        """Clear metrics for a model or all models"""
        if model_name:
            if model_name in self.metrics:
                del self.metrics[model_name]
                logger.info(f"Cleared metrics for {model_name}")
        else:
            self.metrics.clear()
            logger.info("Cleared all metrics")


# Global instance
model_monitor = ModelPerformanceMonitor()
