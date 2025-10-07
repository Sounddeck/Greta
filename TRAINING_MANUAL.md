# GRETA - Comprehensive Training Manual

## 📚 Table of Contents
1. [Introduction](#introduction)
2. [System Architecture](#system-architecture)
3. [Core Concepts](#core-concepts)
4. [Module Guide](#module-guide)
5. [Agent Development](#agent-development)
6. [Workflow Management](#workflow-management)
7. [Best Practices](#best-practices)

---

## 🎯 Introduction

### What is GRETA?

GRETA (General Reasoning and Execution Training Assistant) is a comprehensive Personal AI (PAI) system that combines:

- **12 Integrated Modules** for complete AI functionality
- **Local LLM Integration** for privacy and control
- **Continuous Learning** from all interactions
- **Multi-Agent System** for complex task execution
- **Knowledge Management** with semantic search
- **Autonomous Operations** with workflow automation

### Key Features

✅ **Privacy-First:** All data stays local  
✅ **Continuous Learning:** LLM improves from every interaction  
✅ **Modular Design:** Use only what you need  
✅ **Extensible:** Build custom agents and workflows  
✅ **Production-Ready:** 160+ API endpoints, full error handling  

---

## 🏗️ System Architecture

### High-Level Overview

```
┌─────────────────────────────────────────────────────────┐
│                    GRETA Frontend                        │
│  (React + TypeScript + TanStack Query)                  │
└────────────────┬────────────────────────────────────────┘
                 │ HTTP/REST API
┌────────────────┴────────────────────────────────────────┐
│                  FastAPI Backend                         │
│  ┌──────────────────────────────────────────────────┐  │
│  │  12 Core Modules (Services + Routers)            │  │
│  ├──────────────────────────────────────────────────┤  │
│  │  Learning Service (Continuous Training)          │  │
│  ├──────────────────────────────────────────────────┤  │
│  │  LLM Service (Local Model Integration)           │  │
│  └──────────────────────────────────────────────────┘  │
└────────────────┬────────────────────────────────────────┘
                 │
┌────────────────┴────────────────────────────────────────┐
│         Data Layer (MongoDB + File Fallback)            │
└─────────────────────────────────────────────────────────┘
```

### Module Architecture

Each module consists of:

1. **Service Layer** (`backend/services/`)
   - Business logic
   - Database operations
   - Learning integration
   - Activity logging

2. **Router Layer** (`backend/routers/`)
   - API endpoints
   - Request validation (Pydantic)
   - Response formatting
   - Error handling

3. **Frontend Component** (`greta-frontend/src/pages/`)
   - User interface
   - API integration
   - State management
   - Real-time updates

---

## 💡 Core Concepts

### 1. Continuous Learning

**How it Works:**
- Every user interaction is recorded
- Data includes: query, response, context, quality metrics
- LLM is fine-tuned periodically on this data
- System improves over time

**Example:**
```python
# Automatic learning integration
await learning_service.record_pai_interaction({
    "user_id": "user123",
    "type": "agent_creation",
    "category": "development",
    "strategic_reasoning": "User wants to build a research agent",
    "tactical_reasoning": "Selected web search and document analysis tools",
    "operational_reasoning": "Agent created successfully",
    "response_quality": 0.95,
    "user_satisfaction": 1.0,
    "context_relevance": 0.98
})
```

### 2. Multi-Agent System

**Agent Types:**
- **Autonomous:** Self-directed, goal-oriented
- **Reactive:** Responds to events
- **Proactive:** Anticipates needs
- **Collaborative:** Works with other agents
- **Specialized:** Domain-specific expertise

**Agent Lifecycle:**
1. Design → 2. Build → 3. Test → 4. Deploy → 5. Monitor → 6. Optimize

### 3. Workflow Automation

**Workflow Types:**
- **Sequential:** Steps execute in order
- **Parallel:** Steps execute simultaneously
- **Conditional:** Steps based on conditions
- **Loop:** Repeat steps until condition met
- **Scheduled:** Time-based execution

### 4. Knowledge Management

**Features:**
- Document storage with auto-categorization
- Semantic search using embeddings
- Knowledge graph for relationships
- Version control for documents
- Tag-based organization

---

## 📦 Module Guide

### Module 1: Enhanced Master Agent

**Purpose:** Central control and monitoring

**Key Features:**
- System health monitoring (CPU, memory, disk)
- Agent orchestration
- Emergency controls
- Performance metrics

**Common Use Cases:**
```bash
# Check system status
GET /api/v1/master-agent/status

# Monitor agents
GET /api/v1/master-agent/agents

# Emergency pause
POST /api/v1/master-agent/emergency/pause
```

### Module 2: PAI Hub Management

**Purpose:** Build and deploy agents

**Key Features:**
- Agent templates
- Build system
- Deployment management
- Version control

**Example Workflow:**
1. Choose template (basic, advanced, ML)
2. Configure agent parameters
3. Build agent
4. Test functionality
5. Deploy to production

### Module 3: Autonomous Operations Center

**Purpose:** Workflow automation

**Key Features:**
- 5 workflow templates
- Execution engine with retry logic
- Workflow control (pause/resume/cancel)
- Activity logging

**Creating a Workflow:**
```python
# Sequential workflow example
workflow = {
    "name": "Data Processing Pipeline",
    "type": "sequential",
    "steps": [
        {"action": "fetch_data", "params": {"source": "api"}},
        {"action": "process_data", "params": {"method": "clean"}},
        {"action": "store_data", "params": {"destination": "db"}}
    ]
}

# Create workflow
POST /api/v1/operations/workflows
Body: workflow
```

### Module 4: Predictive Analytics Engine

**Purpose:** ML-powered predictions

**Key Features:**
- 5 model types (time series, classification, regression, clustering, anomaly)
- Model training
- Real-time predictions
- Model evaluation

**Training a Model:**
```python
# Prepare dataset
dataset = {
    "name": "Sales Forecast",
    "data": [...],  # Your data
    "features": ["date", "product", "region"],
    "target": "sales"
}

# Train model
POST /api/v1/analytics/models/train
Body: {
    "dataset_id": "dataset_123",
    "model_type": "time_series",
    "hyperparameters": {"epochs": 100}
}
```

### Module 5: Agent Management Framework

**Purpose:** Manage agent lifecycle

**Key Features:**
- Agent registry
- Version control
- Performance monitoring
- A/B testing

**Agent Versioning:**
```python
# Create new version
POST /api/v1/agent-management/agents/{agent_id}/versions
Body: {
    "changes": "Improved response accuracy",
    "config": {...}
}

# Rollback to previous version
POST /api/v1/agent-management/agents/{agent_id}/rollback
Body: {"version_id": "v1.2.0"}
```

### Module 6: MCP Hub Management

**Purpose:** Tool integration for agents

**Key Features:**
- 14 pre-built tools (web search, ArXiv, image search, etc.)
- Custom tool integration
- Tool access control
- Usage monitoring

**Available Tools:**
1. Web Search
2. ArXiv Research
3. Image Search
4. Screenshot Capture
5. File Operations
6. Database Query
7. API Calls
8. Code Execution
9. Data Analysis
10. Document Processing
11. Email Integration
12. Calendar Management
13. Task Automation
14. System Commands

### Module 7: Functional Memory System

**Purpose:** Intelligent memory management

**Key Features:**
- 4 memory types (episodic, semantic, procedural, working)
- Vector database with embeddings
- Semantic search
- Memory consolidation

**Memory Types:**
- **Episodic:** Specific events and experiences
- **Semantic:** Facts and knowledge
- **Procedural:** How-to knowledge
- **Working:** Temporary, active information

### Module 8: OSINT Management Platform

**Purpose:** Intelligence gathering

**Key Features:**
- 5 data sources (web, databases, technical, social, geospatial)
- Investigation management
- Data analysis with confidence scoring
- Report generation

**Creating an Investigation:**
```python
POST /api/v1/osint/investigations
Body: {
    "title": "Market Research",
    "description": "Analyze competitor landscape",
    "sources": ["web", "social"],
    "priority": "high"
}
```

### Module 9: Multimodal Processing

**Purpose:** Process multiple data types

**Key Features:**
- Image processing (OCR, object detection)
- Audio processing (transcription)
- Video processing (frame extraction)
- Document processing (text extraction)

**Processing an Image:**
```python
POST /api/v1/multimodal/process/image
Body: {
    "image_url": "https://example.com/image.jpg",
    "operations": ["ocr", "object_detection"]
}
```

### Module 10: Agent Builder Development Environment

**Purpose:** Build agents with code

**Key Features:**
- 4 templates (basic, advanced, ML, LangChain)
- Code editor integration
- Build system
- Testing framework
- Deployment tools

**Project Structure:**
```
project/
├── agent.py          # Main agent code
├── config.json       # Configuration
├── requirements.txt  # Dependencies
├── tests/           # Test files
└── README.md        # Documentation
```

### Module 11: Intelligent Knowledge Base

**Purpose:** Document management

**Key Features:**
- Auto-categorization
- Semantic search
- Knowledge graphs
- Version control
- Related document discovery

**Adding a Document:**
```python
POST /api/v1/knowledge/documents
Body: {
    "title": "API Documentation",
    "content": "...",
    "tags": ["api", "documentation"],
    "category": "technical"
}
```

### Module 12: Training Platform Management

**Purpose:** Course and training management

**Key Features:**
- Course creation
- Enrollment system
- Progress tracking
- Assessments
- Certificate issuance

**Creating a Course:**
```python
POST /api/v1/training/courses
Body: {
    "title": "Introduction to AI Agents",
    "description": "Learn to build AI agents",
    "modules": [
        {"title": "Basics", "duration": 2},
        {"title": "Advanced", "duration": 3}
    ],
    "difficulty": "intermediate"
}
```

---

## 🤖 Agent Development

### Step 1: Design Your Agent

**Questions to Answer:**
1. What problem does it solve?
2. What data does it need?
3. What tools does it require?
4. How will it interact with users?
5. What are success metrics?

### Step 2: Choose a Template

**Basic Agent:** Simple task automation  
**Advanced Agent:** Complex reasoning  
**ML Agent:** Machine learning capabilities  
**LangChain Agent:** LLM-powered reasoning  

### Step 3: Build the Agent

```python
# Using Agent Builder
POST /api/v1/builder/projects
Body: {
    "name": "Research Assistant",
    "template": "advanced",
    "description": "Automated research agent"
}

# Add custom code
PUT /api/v1/builder/projects/{project_id}/files
Body: {
    "filename": "agent.py",
    "content": "# Your agent code here"
}
```

### Step 4: Test the Agent

```python
# Run tests
POST /api/v1/builder/projects/{project_id}/test
Body: {
    "config": {"coverage": true}
}
```

### Step 5: Deploy the Agent

```python
# Deploy to production
POST /api/v1/builder/projects/{project_id}/deploy
Body: {
    "environment": "production"
}
```

---

## 🔄 Workflow Management

### Creating Workflows

**Sequential Workflow:**
```python
{
    "type": "sequential",
    "steps": [
        {"action": "step1"},
        {"action": "step2"},
        {"action": "step3"}
    ]
}
```

**Parallel Workflow:**
```python
{
    "type": "parallel",
    "steps": [
        {"action": "task_a"},
        {"action": "task_b"},
        {"action": "task_c"}
    ]
}
```

**Conditional Workflow:**
```python
{
    "type": "conditional",
    "condition": "result > 100",
    "if_true": [{"action": "process_high"}],
    "if_false": [{"action": "process_low"}]
}
```

### Monitoring Workflows

```python
# Get workflow status
GET /api/v1/operations/workflows/{workflow_id}

# Pause workflow
POST /api/v1/operations/workflows/{workflow_id}/pause

# Resume workflow
POST /api/v1/operations/workflows/{workflow_id}/resume
```

---

## ✅ Best Practices

### 1. Agent Design
- Start simple, iterate
- Define clear objectives
- Use appropriate tools
- Monitor performance
- Version control everything

### 2. Workflow Automation
- Break complex tasks into steps
- Add error handling
- Use retry logic
- Log all activities
- Test thoroughly

### 3. Knowledge Management
- Organize with categories and tags
- Use semantic search for discovery
- Keep documents updated
- Build knowledge graphs
- Regular backups

### 4. Performance Optimization
- Cache frequently used data
- Use async operations
- Batch similar requests
- Monitor resource usage
- Scale horizontally when needed

### 5. Security
- Validate all inputs
- Use authentication
- Encrypt sensitive data
- Regular security audits
- Keep dependencies updated

---

## 📊 Monitoring and Metrics

### Key Metrics to Track

**System Health:**
- CPU usage
- Memory usage
- Disk space
- Network latency

**Agent Performance:**
- Response time
- Success rate
- Error rate
- User satisfaction

**Learning Progress:**
- Training data volume
- Model accuracy
- Inference speed
- Quality improvements

---

## 🎓 Advanced Topics

### Custom Tool Development

```python
# Create custom MCP tool
class CustomTool:
    def __init__(self):
        self.name = "custom_tool"
    
    async def execute(self, params):
        # Your tool logic
        return result
```

### LLM Fine-Tuning

```python
# Prepare training data
training_data = await learning_service.get_training_data()

# Fine-tune model
await llm_service.fine_tune(
    model="llama-3.1-8b",
    data=training_data,
    epochs=3
)
```

### Multi-Agent Coordination

```python
# Create agent team
team = {
    "leader": "coordinator_agent",
    "members": ["research_agent", "analysis_agent", "report_agent"]
}

# Execute coordinated task
await agent_management.execute_team_task(team, task)
```

---

**Continue to [User Guide](USER_GUIDE.md) for detailed usage instructions.**

