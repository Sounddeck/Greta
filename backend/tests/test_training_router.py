"""
API Endpoint Tests for Training Router
Tests all training API endpoints with FastAPI TestClient
"""

import pytest
import pytest_asyncio
from datetime import datetime
from typing import AsyncGenerator
from fastapi.testclient import TestClient
from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorDatabase
from unittest.mock import AsyncMock, MagicMock, patch
import json

from main import app
from database import Database


@pytest_asyncio.fixture
async def test_db() -> AsyncGenerator[AsyncIOMotorDatabase, None]:
    """Test database fixture"""
    client = AsyncIOMotorClient("mongodb://localhost:27017")
    db = client["greta_test_training"]
    
    # Clean up collections
    await db.drop_collection("training_datasets")
    await db.drop_collection("training_jobs")
    await db.drop_collection("model_versions")
    await db.drop_collection("agent_conversations")
    
    yield db
    
    # Cleanup
    await db.drop_collection("training_datasets")
    await db.drop_collection("training_jobs")
    await db.drop_collection("model_versions")
    await db.drop_collection("agent_conversations")
    client.close()


@pytest.fixture
def client():
    """FastAPI test client"""
    return TestClient(app)


@pytest_asyncio.fixture
async def sample_conversations(test_db):
    """Insert sample conversations"""
    conversations = [
        {
            "conversation_id": "conv1",
            "messages": [
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi!"}
            ],
            "rating": 5,
            "success": True,
            "created_at": datetime.utcnow()
        }
    ]
    await test_db.agent_conversations.insert_many(conversations)


class TestDatasetEndpoints:
    """Test dataset API endpoints"""
    
    def test_create_dataset(self, client, sample_conversations):
        """Test POST /api/v1/llm-training/datasets"""
        payload = {
            "name": "Test Dataset",
            "description": "Test description",
            "source_collections": ["agent_conversations"],
            "quality_filters": {
                "min_rating": 0,
                "min_confidence": 0.0,
                "success_only": False,
                "exclude_duplicates": False
            },
            "max_samples": 100
        }
        
        response = client.post("/api/v1/llm-training/datasets", json=payload)
        
        assert response.status_code == 200
        data = response.json()
        assert data["name"] == "Test Dataset"
        assert data["status"] == "pending"
    
    def test_create_dataset_invalid_data(self, client):
        """Test creating dataset with invalid data"""
        payload = {
            "name": "",  # Empty name
            "source_collections": []  # Empty collections
        }
        
        response = client.post("/api/v1/llm-training/datasets", json=payload)
        
        assert response.status_code == 422  # Validation error
    
    def test_list_datasets(self, client, sample_conversations):
        """Test GET /api/v1/llm-training/datasets"""
        # Create a dataset first
        payload = {
            "name": "Test Dataset",
            "description": "Test",
            "source_collections": ["agent_conversations"],
            "quality_filters": {
                "min_rating": 0,
                "min_confidence": 0.0,
                "success_only": False,
                "exclude_duplicates": False
            },
            "max_samples": 100
        }
        client.post("/api/v1/llm-training/datasets", json=payload)
        
        # List datasets
        response = client.get("/api/v1/llm-training/datasets")
        
        assert response.status_code == 200
        data = response.json()
        assert "datasets" in data
        assert len(data["datasets"]) >= 1
    
    def test_get_dataset_by_id(self, client, sample_conversations):
        """Test GET /api/v1/llm-training/datasets/{id}"""
        # Create a dataset
        payload = {
            "name": "Test Dataset",
            "description": "Test",
            "source_collections": ["agent_conversations"],
            "quality_filters": {
                "min_rating": 0,
                "min_confidence": 0.0,
                "success_only": False,
                "exclude_duplicates": False
            },
            "max_samples": 100
        }
        create_response = client.post("/api/v1/llm-training/datasets", json=payload)
        dataset_id = create_response.json()["id"]
        
        # Get dataset
        response = client.get(f"/api/v1/llm-training/datasets/{dataset_id}")
        
        assert response.status_code == 200
        data = response.json()
        assert data["id"] == dataset_id
        assert data["name"] == "Test Dataset"
    
    def test_get_nonexistent_dataset(self, client):
        """Test getting nonexistent dataset"""
        response = client.get("/api/v1/llm-training/datasets/nonexistent_id")
        
        assert response.status_code == 404
    
    def test_delete_dataset(self, client, sample_conversations):
        """Test DELETE /api/v1/llm-training/datasets/{id}"""
        # Create a dataset
        payload = {
            "name": "Test Dataset",
            "description": "Test",
            "source_collections": ["agent_conversations"],
            "quality_filters": {
                "min_rating": 0,
                "min_confidence": 0.0,
                "success_only": False,
                "exclude_duplicates": False
            },
            "max_samples": 100
        }
        create_response = client.post("/api/v1/llm-training/datasets", json=payload)
        dataset_id = create_response.json()["id"]
        
        # Delete dataset
        response = client.delete(f"/api/v1/llm-training/datasets/{dataset_id}")
        
        assert response.status_code == 200
        
        # Verify deleted
        get_response = client.get(f"/api/v1/llm-training/datasets/{dataset_id}")
        assert get_response.status_code == 404


class TestJobEndpoints:
    """Test training job API endpoints"""
    
    @pytest.mark.asyncio
    async def test_create_job(self, client, test_db):
        """Test POST /api/v1/llm-training/jobs"""
        # Create a dataset first
        dataset = {
            "dataset_id": "test_dataset",
            "name": "Test Dataset",
            "status": "ready",
            "jsonl_path": "/tmp/test.jsonl",
            "sample_count": 100,
            "created_at": datetime.utcnow()
        }
        await test_db.training_datasets.insert_one(dataset)
        
        payload = {
            "job_name": "Test Job",
            "dataset_id": "test_dataset",
            "base_model": "unsloth/llama-3-8b-bnb-4bit",
            "config": {
                "lora_rank": 16,
                "lora_alpha": 16,
                "learning_rate": 0.0002,
                "epochs": 3,
                "batch_size": 2,
                "ctx_size": 2048
            }
        }
        
        response = client.post("/api/v1/llm-training/jobs", json=payload)
        
        assert response.status_code == 200
        data = response.json()
        assert data["job_name"] == "Test Job"
        assert data["status"] == "pending"
    
    def test_create_job_invalid_config(self, client):
        """Test creating job with invalid configuration"""
        payload = {
            "job_name": "Test Job",
            "dataset_id": "test_dataset",
            "base_model": "unsloth/llama-3-8b-bnb-4bit",
            "config": {
                "lora_rank": -1,  # Invalid
                "learning_rate": 2.0  # Invalid
            }
        }
        
        response = client.post("/api/v1/llm-training/jobs", json=payload)
        
        assert response.status_code == 422
    
    @pytest.mark.asyncio
    async def test_list_jobs(self, client, test_db):
        """Test GET /api/v1/llm-training/jobs"""
        # Create a job
        job = {
            "job_id": "test_job",
            "job_name": "Test Job",
            "dataset_id": "test_dataset",
            "base_model": "unsloth/llama-3-8b-bnb-4bit",
            "status": "pending",
            "config": {
                "lora_rank": 16,
                "lora_alpha": 16,
                "learning_rate": 0.0002,
                "epochs": 3,
                "batch_size": 2,
                "ctx_size": 2048
            },
            "logs": [],
            "created_at": datetime.utcnow()
        }
        await test_db.training_jobs.insert_one(job)
        
        response = client.get("/api/v1/llm-training/jobs")
        
        assert response.status_code == 200
        data = response.json()
        assert "jobs" in data
        assert len(data["jobs"]) >= 1
    
    @pytest.mark.asyncio
    async def test_get_job_by_id(self, client, test_db):
        """Test GET /api/v1/llm-training/jobs/{id}"""
        job = {
            "job_id": "test_job_123",
            "job_name": "Test Job",
            "dataset_id": "test_dataset",
            "base_model": "unsloth/llama-3-8b-bnb-4bit",
            "status": "pending",
            "config": {
                "lora_rank": 16,
                "lora_alpha": 16,
                "learning_rate": 0.0002,
                "epochs": 3,
                "batch_size": 2,
                "ctx_size": 2048
            },
            "logs": [],
            "created_at": datetime.utcnow()
        }
        await test_db.training_jobs.insert_one(job)
        
        response = client.get("/api/v1/llm-training/jobs/test_job_123")
        
        assert response.status_code == 200
        data = response.json()
        assert data["id"] == "test_job_123"
    
    @pytest.mark.asyncio
    async def test_cancel_job(self, client, test_db):
        """Test POST /api/v1/llm-training/jobs/{id}/cancel"""
        job = {
            "job_id": "test_job_cancel",
            "job_name": "Test Job",
            "dataset_id": "test_dataset",
            "base_model": "unsloth/llama-3-8b-bnb-4bit",
            "status": "running",
            "config": {
                "lora_rank": 16,
                "lora_alpha": 16,
                "learning_rate": 0.0002,
                "epochs": 3,
                "batch_size": 2,
                "ctx_size": 2048
            },
            "logs": [],
            "created_at": datetime.utcnow()
        }
        await test_db.training_jobs.insert_one(job)
        
        response = client.post("/api/v1/llm-training/jobs/test_job_cancel/cancel")
        
        assert response.status_code == 200


class TestStatsEndpoints:
    """Test statistics and health endpoints"""
    
    def test_get_stats(self, client):
        """Test GET /api/v1/llm-training/stats"""
        response = client.get("/api/v1/llm-training/stats")
        
        assert response.status_code == 200
        data = response.json()
        assert "datasets" in data
        assert "jobs" in data
        assert "models" in data
    
    def test_get_health(self, client):
        """Test GET /api/v1/llm-training/health"""
        response = client.get("/api/v1/llm-training/health")
        
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert "components" in data

