"""
GRETA Predictive Analytics Router
Complete API endpoints for ML-powered analytics and predictions
"""

from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel, Field
from typing import Dict, Any, Optional, List
from datetime import datetime
import logging

from database import get_database
from services.llm_service import get_llm_service
from services.learning_service import LearningService
from services.predictive_analytics_service import PredictiveAnalyticsService

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/analytics", tags=["Predictive Analytics"])

# Global instance
analytics_instance: Optional[PredictiveAnalyticsService] = None

# Pydantic models
class DatasetIngestRequest(BaseModel):
    """Dataset ingestion request"""
    name: str = Field(..., description="Dataset name")
    description: Optional[str] = Field(None, description="Dataset description")
    data: List[Any] = Field(..., description="Dataset data")
    schema: Dict[str, Any] = Field(default_factory=dict, description="Data schema")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")

class ModelTrainRequest(BaseModel):
    """Model training request"""
    model_type: str = Field(..., description="Model type")
    dataset_id: str = Field(..., description="Dataset ID to train on")
    config: Dict[str, Any] = Field(default_factory=dict, description="Training configuration")

class PredictionRequest(BaseModel):
    """Prediction request"""
    model_type: str = Field(..., description="Model type (time_series, classification, regression, clustering, anomaly_detection)")
    data: List[float] = Field(..., description="Input data for prediction")
    forecast_periods: Optional[int] = Field(5, description="Number of periods to forecast")
    classes: Optional[List[str]] = Field(None, description="Classes for classification")
    num_clusters: Optional[int] = Field(3, description="Number of clusters")
    threshold: Optional[float] = Field(2.0, description="Threshold for anomaly detection")

async def get_analytics() -> PredictiveAnalyticsService:
    """Get or initialize Predictive Analytics instance"""
    global analytics_instance
    
    if analytics_instance is None:
        db = await get_database()
        llm_service = await get_llm_service()
        learning_service = LearningService(db.client)
        await learning_service.initialize()
        
        analytics_instance = PredictiveAnalyticsService(db, llm_service, learning_service)
        await analytics_instance.initialize()
    
    return analytics_instance

@router.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "Predictive Analytics Engine",
        "timestamp": datetime.utcnow().isoformat()
    }

@router.get("/metrics")
async def get_metrics(analytics: PredictiveAnalyticsService = Depends(get_analytics)):
    """Get analytics metrics"""
    try:
        metrics = await analytics.get_metrics()
        return metrics
    except Exception as e:
        logger.error(f"❌ Get metrics failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ==================== DATA INGESTION ====================

@router.post("/datasets")
async def ingest_data(
    request: DatasetIngestRequest,
    analytics: PredictiveAnalyticsService = Depends(get_analytics)
):
    """Ingest data for analytics"""
    try:
        result = await analytics.ingest_data({
            "name": request.name,
            "description": request.description,
            "data": request.data,
            "schema": request.schema,
            "metadata": request.metadata
        })

        if not result.get("success"):
            raise HTTPException(status_code=400, detail=result.get("error"))

        return result

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error ingesting data: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/datasets")
async def get_all_datasets(analytics: PredictiveAnalyticsService = Depends(get_analytics)):
    """Get all datasets"""
    try:
        result = await analytics.get_all_datasets()
        return result
    except Exception as e:
        logger.error(f"Error getting datasets: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/datasets/{dataset_id}")
async def get_dataset(
    dataset_id: str,
    analytics: PredictiveAnalyticsService = Depends(get_analytics)
):
    """Get a specific dataset"""
    try:
        result = await analytics.get_dataset(dataset_id)

        if not result.get("success"):
            raise HTTPException(status_code=404, detail=result.get("error"))

        return result

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting dataset: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.delete("/datasets/{dataset_id}")
async def delete_dataset(
    dataset_id: str,
    analytics: PredictiveAnalyticsService = Depends(get_analytics)
):
    """Delete a dataset"""
    try:
        result = await analytics.delete_dataset(dataset_id)

        if not result.get("success"):
            raise HTTPException(status_code=404, detail=result.get("error"))

        return result

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting dataset: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ==================== MODEL TRAINING ====================

@router.post("/models/train")
async def train_model(
    request: ModelTrainRequest,
    analytics: PredictiveAnalyticsService = Depends(get_analytics)
):
    """Train a model"""
    try:
        result = await analytics.train_model({
            "model_type": request.model_type,
            "dataset_id": request.dataset_id,
            **request.config
        })

        if not result.get("success"):
            raise HTTPException(status_code=400, detail=result.get("error"))

        return result

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error training model: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/models")
async def get_models(analytics: PredictiveAnalyticsService = Depends(get_analytics)):
    """Get available models"""
    try:
        result = await analytics.get_models()
        return result
    except Exception as e:
        logger.error(f"Error getting models: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/models/trained")
async def get_trained_models(analytics: PredictiveAnalyticsService = Depends(get_analytics)):
    """Get trained models"""
    try:
        result = await analytics.get_trained_models()
        return result
    except Exception as e:
        logger.error(f"Error getting trained models: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ==================== PREDICTIONS ====================

@router.post("/predict")
async def predict(
    request: PredictionRequest,
    analytics: PredictiveAnalyticsService = Depends(get_analytics)
):
    """Make a prediction using ML models"""
    try:
        prediction_config = {
            "model_type": request.model_type,
            "data": request.data,
            "forecast_periods": request.forecast_periods,
            "classes": request.classes,
            "num_clusters": request.num_clusters,
            "threshold": request.threshold
        }

        result = await analytics.predict(prediction_config)

        if not result.get("success"):
            raise HTTPException(status_code=400, detail=result.get("error"))

        return result

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error making prediction: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/predictions")
async def get_all_predictions(
    model_type: Optional[str] = None,
    analytics: PredictiveAnalyticsService = Depends(get_analytics)
):
    """Get all predictions"""
    try:
        result = await analytics.get_all_predictions(model_type)
        return result
    except Exception as e:
        logger.error(f"Error getting predictions: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/predictions/{prediction_id}")
async def get_prediction(
    prediction_id: str,
    analytics: PredictiveAnalyticsService = Depends(get_analytics)
):
    """Get a specific prediction"""
    try:
        result = await analytics.get_prediction(prediction_id)

        if not result.get("success"):
            raise HTTPException(status_code=404, detail=result.get("error"))

        return result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting prediction: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.delete("/predictions/{prediction_id}")
async def delete_prediction(
    prediction_id: str,
    analytics: PredictiveAnalyticsService = Depends(get_analytics)
):
    """Delete a prediction"""
    try:
        result = await analytics.delete_prediction(prediction_id)

        if not result.get("success"):
            raise HTTPException(status_code=404, detail=result.get("error"))

        return result

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting prediction: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ==================== ACTIVITY ====================

@router.get("/activity-log")
async def get_activity_log(
    limit: int = 100,
    analytics: PredictiveAnalyticsService = Depends(get_analytics)
):
    """Get activity log"""
    try:
        result = await analytics.get_activity_log(limit)
        return result
    except Exception as e:
        logger.error(f"Error getting activity log: {e}")
        raise HTTPException(status_code=500, detail=str(e))

