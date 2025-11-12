"""
System Knowledge Service
Manages system knowledge initialization and caching for Master Agent
"""

import logging
from typing import List, Optional, Dict, Any
from datetime import datetime, timedelta
from motor.motor_asyncio import AsyncIOMotorDatabase

logger = logging.getLogger(__name__)


class SystemKnowledgeService:
    """Service for managing system knowledge base"""

    def __init__(self, db: AsyncIOMotorDatabase):
        self.db = db
        self.collection = db["system_knowledge"]
        self.cache = {}
        self.cache_expiry = {}
        self.cache_ttl = 300  # 5 minutes

    async def initialize(self):
        """Initialize knowledge base with seed data"""
        try:
            # Check if collection has data
            count = await self.collection.count_documents({})
            if count == 0:
                logger.info("🌱 Seeding knowledge base with initial content...")
                await self._seed_knowledge_base()
            else:
                logger.info(f"✅ Knowledge base already initialized with {count} entries")
        except Exception as e:
            logger.error(f"❌ Failed to initialize knowledge base: {e}")
            raise

    async def _seed_knowledge_base(self):
        """Seed knowledge base with comprehensive CPAS5 documentation"""
        seed_data = [
            {
                "category": "architecture",
                "title": "CPAS5 System Architecture Overview",
                "content": """# CPAS5 System Architecture

## Overview
CPAS5 (Complete Personal AI System) is a comprehensive, multi-agent AI platform built on modern technologies.

## Core Components

### Backend
- **Framework**: FastAPI 2.2.0 with Uvicorn ASGI server
- **Database**: MongoDB 7.0 with Motor 3.3.2 async driver
- **Language**: Python 3.11/3.13
- **Port**: 8000

### Frontend
- **Framework**: React 18 with TypeScript
- **Build Tool**: Vite 7.1.9
- **State Management**: Zustand with persistence
- **Routing**: React Router v6
- **UI Components**: Custom Shadcn/ui components
- **Port**: 5173

### Key Services
- **Master Agent**: Central AI coordination and chat interface
- **LLM Service**: Local and cloud-based language models
- **Learning Service**: Continuous learning and adaptation
- **Multi-Agent System**: Specialized agents for different tasks
""",
                "tags": ["architecture", "system-design", "overview", "core"],
                "priority": 10,
                "version": "1.0.0"
            },
            {
                "category": "features",
                "title": "CPAS5 Core Features",
                "content": """# CPAS5 Core Features

## 1. Master Agent
- Central AI coordination hub
- Multi-agent orchestration
- Conversation history management
- File upload and processing
- Context-aware responses

## 2. Agent Builder
- Visual agent creation interface
- Custom agent configuration
- Agent specialization
- Team collaboration

## 3. Workflow Builder
- Visual workflow designer with React Flow
- Node types: Trigger, Condition, Action, Agent
- Edge types: Default, Conditional
- Workflow execution and monitoring

## 4. LLM Training Automation
- Local model training with llama.cpp
- Dataset management
- Training job monitoring
- Model versioning

## 5. Prompt Library
- Prompt template management
- Prompt enhancement with CoT
- Template categorization
- Version control

## 6. Google Workspace Integration
- OAuth 2.0 authentication
- Gmail, Calendar, Drive integration
- Multi-account support
""",
                "tags": ["features", "capabilities", "modules"],
                "priority": 9,
                "version": "1.0.0"
            },
            {
                "category": "api_endpoints",
                "title": "Master Agent API Endpoints",
                "content": """# Master Agent API Endpoints

## Task Execution
- **POST** `/api/v1/tasks/execute` - Execute a task with Master Agent
  - Request: TaskExecuteRequest with description, agent_type, input_data
  - Response: TaskResponse with result and status

## Knowledge Management
- **GET** `/api/v1/knowledge/system` - Retrieve system knowledge
- **GET** `/api/v1/knowledge/system/search?query=...` - Search knowledge
- **POST** `/api/v1/knowledge/system` - Create knowledge entry (admin)
- **PUT** `/api/v1/knowledge/system/{id}` - Update knowledge entry (admin)
- **DELETE** `/api/v1/knowledge/system/{id}` - Delete knowledge entry (admin)
- **POST** `/api/v1/knowledge/system/context` - Get Master Agent context

## File Upload
- **POST** `/api/v1/knowledge/upload` - Upload files to knowledge base

## Prompt Enhancement
- **POST** `/api/v1/prompts/enhance` - Enhance prompts with CoT

## LLM Training
- **GET** `/api/v1/llm-training/stats` - Get training statistics
- **GET** `/api/v1/llm-training/health` - Check training system health
""",
                "tags": ["api", "endpoints", "reference"],
                "priority": 8,
                "version": "1.0.0"
            },
            {
                "category": "capabilities",
                "title": "Master Agent Capabilities",
                "content": """# Master Agent Capabilities

## Core Capabilities
- **Multi-Agent Coordination**: Orchestrate multiple specialized agents
- **Context Awareness**: Maintain conversation history and context
- **File Processing**: Analyze uploaded documents and files
- **Knowledge Retrieval**: Access system knowledge base
- **Task Execution**: Execute complex multi-step tasks
- **Learning**: Adapt and improve from interactions

## Agent Types
- Research Agent, Coding Agent, Data Analysis Agent
- Content Creation Agent, Task Automation Agent
- Customer Support Agent, Sales Agent, Marketing Agent
- Finance Agent, General Agent

## Performance
- Response Time: < 5 seconds for typical queries
- Concurrent Users: 100+ simultaneous connections
- File Size: Up to 100MB per upload
- Context Window: 8K tokens (configurable)
- Timeout: 60 seconds per task
""",
                "tags": ["capabilities", "master-agent", "features"],
                "priority": 8,
                "version": "1.0.0"
            }
        ]

        try:
            result = await self.collection.insert_many(seed_data)
            logger.info(f"✅ Seeded {len(result.inserted_ids)} knowledge entries")
        except Exception as e:
            logger.error(f"❌ Failed to seed knowledge base: {e}")
            raise

    async def get_master_agent_context(self, max_entries: int = 20) -> str:
        """Get formatted context for Master Agent system prompt"""
        try:
            # Check cache
            cache_key = f"master_context_{max_entries}"
            if cache_key in self.cache:
                if datetime.utcnow() < self.cache_expiry.get(cache_key, datetime.utcnow()):
                    return self.cache[cache_key]

            # Fetch high-priority entries
            cursor = self.collection.find(
                {"priority": {"$gte": 5}}
            ).sort("priority", -1).limit(max_entries)

            context_parts = []
            async for doc in cursor:
                context_parts.append(f"## {doc['title']}\n{doc['content']}\n")

            context = "\n".join(context_parts)

            # Cache result
            self.cache[cache_key] = context
            self.cache_expiry[cache_key] = datetime.utcnow() + timedelta(seconds=self.cache_ttl)

            return context
        except Exception as e:
            logger.error(f"Error getting Master Agent context: {e}")
            return ""

    async def clear_cache(self):
        """Clear knowledge base cache"""
        self.cache.clear()
        self.cache_expiry.clear()
        logger.info("Knowledge base cache cleared")

