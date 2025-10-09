# CPAS4 IMPLEMENTATION GUIDE
## Step-by-Step Build Instructions with Code Examples

**Guide Date**: October 7, 2025  
**Target**: Production-Ready CPAS4 in 4-6 Weeks  
**Approach**: Incremental, testable, agent-first

---

## 🚀 GETTING STARTED

### Prerequisites

1. **Development Environment**
   ```bash
   # Python 3.11+
   python --version  # Should be 3.11 or higher
   
   # MongoDB
   mongod --version  # Should be 6.0 or higher
   
   # Ollama (for local LLM)
   ollama --version
   ```

2. **Install Ollama and Models**
   ```bash
   # macOS
   brew install ollama
   
   # Start Ollama service
   ollama serve &
   
   # Pull models
   ollama pull llama3.2
   ollama pull mistral
   ```

3. **MongoDB Setup**
   ```bash
   # macOS
   brew install mongodb-community
   
   # Start MongoDB
   brew services start mongodb-community
   
   # Verify connection
   mongosh --eval "db.version()"
   ```

---

## 📁 PROJECT STRUCTURE

### Option 1: Separate Repository (RECOMMENDED)

```bash
# Create new CPAS4 repository
mkdir cpas4
cd cpas4
git init

# Create structure
mkdir -p backend/{services,routers,models,utils}
mkdir -p frontend/src/{components,pages,services}
mkdir -p tests/{unit,integration}
mkdir -p docs
```

### Option 2: Subdirectory in GRETA

```bash
cd /Users/macone/Desktop/Greta
mkdir -p cpas4/{backend,frontend,tests,docs}
```

### Final Structure

```
cpas4/
├── backend/
│   ├── main.py                    # FastAPI application
│   ├── config.py                  # Configuration
│   ├── database.py                # MongoDB connection (fixed from GRETA)
│   │
│   ├── services/
│   │   ├── agent_builder.py       # Agent Builder Service
│   │   ├── agent_deployer.py      # Agent Deployer Service
│   │   ├── agent_orchestrator.py  # Agent Orchestrator Service
│   │   ├── agent_monitor.py       # Agent Monitor Service
│   │   │
│   │   ├── pai_intelligence.py    # PAI Intelligence (from GRETA)
│   │   ├── memory_service.py      # MemOS (from GRETA)
│   │   ├── learning_service.py    # Learning (from GRETA)
│   │   │
│   │   ├── crm_service.py         # CRM Business Module
│   │   ├── ats_service.py         # ATS Business Module
│   │   ├── finance_service.py     # Finance Business Module
│   │   ├── project_service.py     # Project Business Module
│   │   └── inventory_service.py   # Inventory Business Module
│   │
│   ├── routers/
│   │   ├── agent_router.py        # Agent management endpoints
│   │   ├── crm_router.py          # CRM endpoints
│   │   ├── ats_router.py          # ATS endpoints
│   │   ├── finance_router.py      # Finance endpoints
│   │   ├── project_router.py      # Project endpoints
│   │   └── inventory_router.py    # Inventory endpoints
│   │
│   ├── models/
│   │   ├── agent_models.py        # Agent data models
│   │   ├── business_models.py     # Business data models
│   │   └── workflow_models.py     # Workflow data models
│   │
│   └── utils/
│       ├── llm_client.py          # Ollama client
│       └── helpers.py             # Utility functions
│
├── frontend/
│   ├── src/
│   │   ├── components/
│   │   │   ├── AgentBuilder.tsx
│   │   │   ├── AgentMonitor.tsx
│   │   │   └── Dashboard.tsx
│   │   │
│   │   ├── pages/
│   │   │   ├── Agents.tsx
│   │   │   ├── CRM.tsx
│   │   │   ├── ATS.tsx
│   │   │   ├── Finance.tsx
│   │   │   ├── Projects.tsx
│   │   │   └── Inventory.tsx
│   │   │
│   │   └── services/
│   │       └── api.ts
│   │
│   ├── package.json
│   └── tsconfig.json
│
├── tests/
│   ├── unit/
│   └── integration/
│
├── docs/
│   ├── API.md
│   ├── USER_GUIDE.md
│   └── DEPLOYMENT.md
│
├── requirements.txt
├── README.md
└── .env.example
```

---

## 🔧 PHASE 1: FOUNDATION (Week 1)

### Day 1-2: Setup & Database Fix

#### Step 1: Create Project

```bash
# Create CPAS4 directory
mkdir cpas4
cd cpas4

# Initialize git
git init

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Create requirements.txt
cat > requirements.txt << EOF
fastapi==0.104.1
uvicorn[standard]==0.24.0
motor==3.3.2
pydantic==2.5.0
python-dotenv==1.0.0
aiohttp==3.9.1
langgraph==0.0.26
langchain==0.1.0
pymongo==4.6.0
pytest==7.4.3
pytest-asyncio==0.21.1
EOF

# Install dependencies
pip install -r requirements.txt
```

#### Step 2: Fix MongoDB Connection (from GRETA bug)

```python
# backend/database.py

from motor.motor_asyncio import AsyncIOMotorClient
from typing import Optional, Dict, Any, List
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

class Database:
    """MongoDB database connection - FIXED from GRETA bug"""
    
    def __init__(self, connection_string: str = "mongodb://localhost:27017"):
        self.connection_string = connection_string
        self.client: Optional[AsyncIOMotorClient] = None
        self.db = None
        self.connected = False
    
    async def connect(self) -> bool:
        """Connect to MongoDB - FIXED coroutine bug"""
        try:
            # FIX: AsyncIOMotorClient is synchronous, not a coroutine
            self.client = AsyncIOMotorClient(self.connection_string)
            
            # Test connection with ping
            await self.client.admin.command('ping')
            
            # Get database
            self.db = self.client.cpas4
            self.connected = True
            
            logger.info("✅ MongoDB connected successfully")
            return True
            
        except Exception as e:
            logger.error(f"❌ MongoDB connection failed: {e}")
            self.connected = False
            return False
    
    async def disconnect(self):
        """Disconnect from MongoDB"""
        if self.client:
            self.client.close()
            self.connected = False
            logger.info("MongoDB disconnected")
    
    async def insert_one(self, collection: str, document: Dict[str, Any]) -> Any:
        """Insert single document"""
        if not self.connected:
            raise Exception("Database not connected")
        
        result = await self.db[collection].insert_one(document)
        return result
    
    async def find_one(self, collection: str, query: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Find single document"""
        if not self.connected:
            raise Exception("Database not connected")
        
        return await self.db[collection].find_one(query)
    
    async def find_many(
        self,
        collection: str,
        query: Dict[str, Any],
        limit: int = 100,
        skip: int = 0,
        sort: Optional[List] = None
    ) -> List[Dict[str, Any]]:
        """Find multiple documents"""
        if not self.connected:
            raise Exception("Database not connected")
        
        cursor = self.db[collection].find(query).skip(skip).limit(limit)
        
        if sort:
            cursor = cursor.sort(sort)
        
        return await cursor.to_list(length=limit)
    
    async def update_one(
        self,
        collection: str,
        query: Dict[str, Any],
        update: Dict[str, Any]
    ) -> Any:
        """Update single document"""
        if not self.connected:
            raise Exception("Database not connected")
        
        result = await self.db[collection].update_one(query, update)
        return result
    
    async def delete_one(self, collection: str, query: Dict[str, Any]) -> Any:
        """Delete single document"""
        if not self.connected:
            raise Exception("Database not connected")
        
        result = await self.db[collection].delete_one(query)
        return result

# Global database instance
db = Database()

async def get_database() -> Database:
    """Dependency for FastAPI routes"""
    if not db.connected:
        await db.connect()
    return db
```

#### Step 3: Test Database Connection

```python
# tests/test_database.py

import pytest
import asyncio
from backend.database import Database

@pytest.mark.asyncio
async def test_database_connection():
    """Test MongoDB connection"""
    db = Database()
    
    # Test connection
    connected = await db.connect()
    assert connected == True
    assert db.connected == True
    
    # Test insert
    result = await db.insert_one("test_collection", {"test": "data"})
    assert result.inserted_id is not None
    
    # Test find
    doc = await db.find_one("test_collection", {"test": "data"})
    assert doc is not None
    assert doc["test"] == "data"
    
    # Cleanup
    await db.delete_one("test_collection", {"test": "data"})
    await db.disconnect()

# Run test
# pytest tests/test_database.py -v
```

#### Step 4: Create FastAPI Application

```python
# backend/main.py

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import logging
from contextlib import asynccontextmanager

from database import db
from routers import agent_router

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown events"""
    # Startup
    logger.info("🚀 Starting CPAS4...")
    
    # Connect to database
    await db.connect()
    
    logger.info("✅ CPAS4 started successfully")
    
    yield
    
    # Shutdown
    logger.info("Shutting down CPAS4...")
    await db.disconnect()

# Create FastAPI app
app = FastAPI(
    title="CPAS4 - Comprehensive Personal Automation System",
    description="Agent-centric business automation platform",
    version="4.0.0",
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:3002"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(agent_router.router, prefix="/api/v1/agents", tags=["Agents"])

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "name": "CPAS4",
        "version": "4.0.0",
        "status": "operational",
        "description": "Agent-centric business automation platform"
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "database": "connected" if db.connected else "disconnected"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8001,  # Different port from GRETA
        reload=True
    )
```

#### Step 5: Test Backend

```bash
# Start backend
cd backend
python main.py

# Test in another terminal
curl http://localhost:8001/
curl http://localhost:8001/health
```

---

### Day 3-5: Agent Builder Implementation

#### Step 1: Create Agent Models

```python
# backend/models/agent_models.py

from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
from datetime import datetime
from enum import Enum

class AgentType(str, Enum):
    """Agent types"""
    AUTONOMOUS = "autonomous"
    REACTIVE = "reactive"
    PROACTIVE = "proactive"
    COLLABORATIVE = "collaborative"
    SPECIALIZED = "specialized"

class AgentStatus(str, Enum):
    """Agent status"""
    CREATED = "created"
    DEPLOYED = "deployed"
    RUNNING = "running"
    STOPPED = "stopped"
    ERROR = "error"

class Tool(BaseModel):
    """Tool definition"""
    name: str
    description: str
    function: str
    parameters: Dict[str, Any] = {}

class AgentTemplate(BaseModel):
    """Agent template"""
    template_id: str
    name: str
    description: str
    agent_type: AgentType
    capabilities: List[str]
    tools: List[Tool]
    default_config: Dict[str, Any] = {}
    prompt_template: str

class AgentConfig(BaseModel):
    """Agent configuration"""
    name: str
    description: str
    template_id: str
    agent_type: AgentType
    tools: List[str] = []
    llm_model: str = "llama3.2"
    temperature: float = 0.7
    max_tokens: int = 500
    custom_config: Dict[str, Any] = {}

class Agent(BaseModel):
    """Agent instance"""
    agent_id: str
    name: str
    description: str
    agent_type: AgentType
    status: AgentStatus
    template_id: str
    config: AgentConfig
    created_at: datetime = Field(default_factory=datetime.now)
    deployed_at: Optional[datetime] = None
    last_active: Optional[datetime] = None
    tasks_completed: int = 0
    tasks_failed: int = 0
    performance_score: float = 0.0
```

#### Step 2: Implement Agent Builder Service

```python
# backend/services/agent_builder.py

from typing import Dict, Any, List, Optional
import uuid
from datetime import datetime
import logging

from models.agent_models import Agent, AgentConfig, AgentTemplate, AgentType, AgentStatus, Tool
from database import Database

logger = logging.getLogger(__name__)

class AgentBuilder:
    """Build agents from templates or custom specifications"""
    
    def __init__(self, db: Database):
        self.db = db
        self.templates = self._load_templates()
    
    def _load_templates(self) -> Dict[str, AgentTemplate]:
        """Load agent templates"""
        return {
            "crm_agent": AgentTemplate(
                template_id="crm_agent",
                name="CRM Agent",
                description="Manages customer relationships and sales pipeline",
                agent_type=AgentType.SPECIALIZED,
                capabilities=[
                    "contact_management",
                    "deal_tracking",
                    "email_automation",
                    "reporting"
                ],
                tools=[
                    Tool(
                        name="create_contact",
                        description="Create new contact in CRM",
                        function="crm_service.create_contact",
                        parameters={"name": "str", "email": "str", "company": "str"}
                    ),
                    Tool(
                        name="update_deal",
                        description="Update deal status",
                        function="crm_service.update_deal",
                        parameters={"deal_id": "str", "status": "str"}
                    )
                ],
                default_config={
                    "llm_model": "llama3.2",
                    "temperature": 0.7,
                    "max_tokens": 500
                },
                prompt_template="""You are a CRM agent responsible for managing customer relationships.
Your capabilities include:
- Managing contacts and companies
- Tracking deals through the sales pipeline
- Automating follow-up emails
- Generating sales reports

Always prioritize customer satisfaction and sales efficiency."""
            ),
            
            "finance_agent": AgentTemplate(
                template_id="finance_agent",
                name="Finance Agent",
                description="Manages financial operations and forecasting",
                agent_type=AgentType.PROACTIVE,
                capabilities=[
                    "transaction_tracking",
                    "budget_management",
                    "cash_flow_forecasting",
                    "financial_reporting"
                ],
                tools=[
                    Tool(
                        name="track_transaction",
                        description="Record financial transaction",
                        function="finance_service.track_transaction",
                        parameters={"amount": "float", "category": "str", "description": "str"}
                    ),
                    Tool(
                        name="forecast_cash_flow",
                        description="Forecast cash flow for next N months",
                        function="finance_service.forecast_cash_flow",
                        parameters={"months": "int"}
                    )
                ],
                default_config={
                    "llm_model": "llama3.2",
                    "temperature": 0.5,
                    "max_tokens": 1000
                },
                prompt_template="""You are a finance agent responsible for financial management.
Your capabilities include:
- Tracking all financial transactions
- Managing budgets and forecasts
- Predicting cash flow
- Generating financial reports

Always ensure accuracy and compliance with financial regulations."""
            ),
            
            # Add more templates...
        }
    
    async def build_from_template(
        self,
        template_id: str,
        config: Dict[str, Any]
    ) -> Agent:
        """Build agent from template"""
        
        # Get template
        template = self.templates.get(template_id)
        if not template:
            raise ValueError(f"Template {template_id} not found")
        
        # Generate agent ID
        agent_id = f"{template_id}_{uuid.uuid4().hex[:8]}"
        
        # Create agent config
        agent_config = AgentConfig(
            name=config.get("name", template.name),
            description=config.get("description", template.description),
            template_id=template_id,
            agent_type=template.agent_type,
            tools=[tool.name for tool in template.tools],
            llm_model=config.get("llm_model", template.default_config.get("llm_model", "llama3.2")),
            temperature=config.get("temperature", template.default_config.get("temperature", 0.7)),
            max_tokens=config.get("max_tokens", template.default_config.get("max_tokens", 500)),
            custom_config=config.get("custom_config", {})
        )
        
        # Create agent instance
        agent = Agent(
            agent_id=agent_id,
            name=agent_config.name,
            description=agent_config.description,
            agent_type=agent_config.agent_type,
            status=AgentStatus.CREATED,
            template_id=template_id,
            config=agent_config,
            created_at=datetime.now()
        )
        
        # Save to database
        await self.db.insert_one("agents", agent.dict())
        
        logger.info(f"✅ Agent created: {agent_id}")
        
        return agent
    
    async def list_templates(self) -> List[AgentTemplate]:
        """List available templates"""
        return list(self.templates.values())
    
    async def get_template(self, template_id: str) -> Optional[AgentTemplate]:
        """Get specific template"""
        return self.templates.get(template_id)
```

#### Step 3: Create Agent Router

```python
# backend/routers/agent_router.py

from fastapi import APIRouter, Depends, HTTPException
from typing import List, Dict, Any
from pydantic import BaseModel

from services.agent_builder import AgentBuilder
from models.agent_models import Agent, AgentTemplate
from database import get_database, Database

router = APIRouter()

class BuildAgentRequest(BaseModel):
    template_id: str
    config: Dict[str, Any] = {}

@router.post("/build", response_model=Agent)
async def build_agent(
    request: BuildAgentRequest,
    db: Database = Depends(get_database)
):
    """Build agent from template"""
    builder = AgentBuilder(db)
    
    try:
        agent = await builder.build_from_template(
            template_id=request.template_id,
            config=request.config
        )
        return agent
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/templates", response_model=List[AgentTemplate])
async def list_templates(db: Database = Depends(get_database)):
    """List available agent templates"""
    builder = AgentBuilder(db)
    return await builder.list_templates()

@router.get("/templates/{template_id}", response_model=AgentTemplate)
async def get_template(
    template_id: str,
    db: Database = Depends(get_database)
):
    """Get specific template"""
    builder = AgentBuilder(db)
    template = await builder.get_template(template_id)
    
    if not template:
        raise HTTPException(status_code=404, detail="Template not found")
    
    return template

@router.get("/{agent_id}", response_model=Agent)
async def get_agent(
    agent_id: str,
    db: Database = Depends(get_database)
):
    """Get agent by ID"""
    agent_doc = await db.find_one("agents", {"agent_id": agent_id})
    
    if not agent_doc:
        raise HTTPException(status_code=404, detail="Agent not found")
    
    return Agent(**agent_doc)

@router.get("/", response_model=List[Agent])
async def list_agents(
    limit: int = 50,
    skip: int = 0,
    db: Database = Depends(get_database)
):
    """List all agents"""
    agent_docs = await db.find_many("agents", {}, limit=limit, skip=skip)
    return [Agent(**doc) for doc in agent_docs]
```

#### Step 4: Test Agent Builder

```bash
# Start backend
python backend/main.py

# Test in another terminal

# List templates
curl http://localhost:8001/api/v1/agents/templates

# Build CRM agent
curl -X POST http://localhost:8001/api/v1/agents/build \
  -H "Content-Type: application/json" \
  -d '{
    "template_id": "crm_agent",
    "config": {
      "name": "Sales CRM Agent",
      "description": "Manages sales pipeline"
    }
  }'

# List agents
curl http://localhost:8001/api/v1/agents/
```

---

## 📊 PROGRESS TRACKING

### Week 1 Checklist
- [x] Project structure created
- [x] MongoDB connection fixed
- [x] FastAPI application running
- [x] Agent models defined
- [x] Agent Builder service implemented
- [x] Agent templates created
- [x] Agent router with endpoints
- [x] Tests passing

### Week 2 Checklist (Next Steps)
- [ ] Agent Deployer service
- [ ] Agent Orchestrator service
- [ ] Agent Monitor service
- [ ] LLM integration (Ollama)
- [ ] Multi-agent workflows

---

**Implementation Guide - Part 1 Complete**  
**Status**: Foundation ready for Week 2  
**Next**: Continue with Agent Deployer and Orchestrator

