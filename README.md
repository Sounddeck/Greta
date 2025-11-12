# GRETA PAI - Personal AI Assistant

GRETA (German-accented Reasoning and Enhanced Thinking Assistant) is a comprehensive Personal AI system with advanced capabilities including voice processing, multi-agent collaboration, and extensive integrations.

## 🚀 Quick Start

### Prerequisites

- Python 3.11+
- Node.js 18+
- MongoDB
- Redis (optional, for caching)

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/Sounddeck/Greta.git
   cd Greta
   ```

2. **Set up environment variables**
   ```bash
   cp .env.example .env
   # Edit .env with your actual values
   ```

3. **Install backend dependencies**
   ```bash
   cd backend
   pip install -r requirements_enhanced.txt
   ```

4. **Install frontend dependencies**
   ```bash
   cd cpas4-ui
   npm install
   ```

5. **Start the services**
   ```bash
   # Backend (Terminal 1)
   cd backend
   python main.py

   # Frontend (Terminal 2)
   cd cpas4-ui
   npm run dev
   ```

## 🔧 Configuration

### Environment Variables

GRETA uses environment variables for configuration. Copy `.env.example` to `.env` and configure the following:

#### Security & Authentication (Required)
```bash
# Generate with: python3 -c "import secrets; print(secrets.token_urlsafe(32))"
SECRET_KEY=your-secret-key-here-change-in-production
JWT_SECRET=your-jwt-secret-here-64-bytes-minimum
JWT_REFRESH_SECRET=your-jwt-refresh-secret-here-64-bytes-minimum
ALGORITHM=HS256
ACCESS_TOKEN_EXPIRE_MINUTES=30
```

#### Database Configuration
```bash
MONGODB_URI=mongodb://localhost:27017
MONGODB_DATABASE=greta_pai
```

#### Application Settings
```bash
ENVIRONMENT=development  # development | production
DEBUG=true
HOST=0.0.0.0
PORT=8000
```

#### CORS Configuration
```bash
CORS_ALLOWED_ORIGINS=http://localhost:5173,http://localhost:5174,http://localhost:3000
```

#### Feature Flags
```bash
ENABLE_VOICE_FEATURES=true
ENABLE_OSINT_FREE_TOOLS=true
ENABLE_JINA_MCP=true
ENABLE_HOOKS=false
```

#### External API Keys (Optional)
```bash
OPENAI_API_KEY=your-openai-api-key-here
ANTHROPIC_API_KEY=your-anthropic-api-key-here
GOOGLE_API_KEY=your-google-api-key-here
JINA_API_KEY=your-jina-api-key-here
```

### Security Best Practices

1. **Never commit `.env` files** - They contain sensitive information
2. **Use strong secret keys** - Generate with `python3 -c "import secrets; print(secrets.token_urlsafe(32))"`
3. **Change default secrets** - All default values must be changed in production
4. **Use HTTPS in production** - Set `ENABLE_HTTPS_REDIRECT=true`

## 🏗️ Architecture

### Backend (FastAPI)
- **Main Application**: `backend/main.py`
- **Configuration**: `backend/config.py`
- **Routers**: `backend/routers/`
- **Services**: `backend/services/`
- **Middleware**: `backend/middleware/`
- **Models**: `backend/models/`

### Frontend (React + TypeScript)
- **Main App**: `cpas4-ui/src/App.tsx`
- **Components**: `cpas4-ui/src/components/`
- **Stores (Zustand)**: `cpas4-ui/src/stores/`
- **API Layer**: `cpas4-ui/src/lib/api.ts`

### Key Features

#### Core Features
- ✅ JWT Authentication with role-based access
- ✅ Rate limiting with configurable limits
- ✅ Voice processing (STT/TTS)
- ✅ Zustand state management with persistence
- ✅ Skeleton loading states
- ✅ Comprehensive error handling

#### Multi-Agent System (Phase 4-8)
- ✅ **Master Agent Control** - Create, deploy, monitor, and route agents
- ✅ **Agent Builder** - Build custom agents with specific capabilities
- ✅ **Tool Registry** - 13 production-ready tools (OSINT, Data, Web, Code, AI/ML)
- ✅ **MCP Integration** - Model Context Protocol for extensibility (Slack, Stripe, +more)
- ✅ **Collaboration Patterns** - Sequential, Parallel, and Hierarchical workflows
- ✅ **Intelligent Routing** - Automatic task routing to best-suited agents
- ✅ **Real-time Monitoring** - Agent performance tracking with health scores
- ✅ **Comprehensive Testing** - 61+ test cases with full coverage
- ✅ **Complete Documentation** - API docs, usage guides, and examples

#### LLM Training Automation (Priority 2)
- ✅ **Dataset Management** - Collect and filter training data from conversations
- ✅ **Fast Training** - Unsloth-powered LoRA training (2-5x faster, 50% less memory)
- ✅ **Model Conversion** - Automatic GGUF conversion with multiple quantization options
- ✅ **Version Control** - Track, deploy, and rollback model versions
- ✅ **Real-time Monitoring** - Live training progress and metrics
- ✅ **Complete UI** - Full-featured React interface for the entire pipeline
- ✅ **Comprehensive Testing** - 50+ backend tests with 80%+ coverage
- ✅ **Complete Documentation** - User guide, API reference, and architecture docs

## 🧪 Testing

### Run All Tests (Automated)

```bash
cd /Users/macone/Desktop/CPAS5
chmod +x run_all_tests.sh
./run_all_tests.sh
```

### Run Individual Test Suites

```bash
# Tool Registry tests (13 tools)
pytest backend/tests/test_tool_registry.py -v --asyncio-mode=auto

# Collaboration Patterns tests (Sequential, Parallel, Hierarchical)
pytest backend/tests/test_collaboration_patterns.py -v --asyncio-mode=auto

# Master Agent Integration tests
pytest backend/tests/test_master_agent_integration.py -v --asyncio-mode=auto

# API Endpoint tests
pytest backend/tests/test_api_endpoints.py -v

# Frontend tests
cd cpas4-ui
npm test
```

### Test Coverage

- **Unit Tests:** 34+ test cases
- **Integration Tests:** 12+ test cases
- **API Endpoint Tests:** 15+ test cases
- **Total:** 61+ test cases

## 📚 API Documentation

### Interactive Documentation

Once the backend is running, visit:
- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

### Comprehensive Guides

#### Multi-Agent System
- **[API Documentation](API_DOCUMENTATION.md)** - Complete API reference with examples
- **[Usage Guide](USAGE_GUIDE.md)** - Usage examples and best practices
- **[Architecture Guide](ARCHITECTURE.md)** - System architecture (coming soon)

#### LLM Training System
- **[User Guide](docs/LLM_TRAINING_USER_GUIDE.md)** - Complete workflow tutorial and best practices
- **[API Reference](docs/LLM_TRAINING_API_REFERENCE.md)** - Training API endpoints documentation
- **[Architecture](docs/LLM_TRAINING_ARCHITECTURE.md)** - System architecture and data flow

### Quick API Examples

#### Create a Custom Agent
```python
import requests

response = requests.post("http://localhost:8000/api/v1/master/agent/build", json={
    "name": "Research Agent",
    "agent_type": "research",
    "capabilities": ["web_search", "data_analysis"],
    "tools": ["osint_web_scraper", "data_json_parser"]
})
```

#### Execute an Agent
```python
response = requests.post("http://localhost:8000/api/v1/master/agent/execute", json={
    "agent_id": "agent_123",
    "task": {"description": "Research AI trends in 2025"}
})
```

#### Create a Workflow
```python
response = requests.post("http://localhost:8000/api/v1/agent-builder/workflows", json={
    "name": "Research Pipeline",
    "workflow_type": "sequential",
    "agents": ["agent1", "agent2", "agent3"]
})
```

## 🔒 Security Features

- JWT-based authentication
- Rate limiting on all endpoints
- CORS protection
- Security headers
- Input validation
- File upload restrictions

## 🚀 Deployment

### Production Checklist

1. Set `ENVIRONMENT=production` in `.env`
2. Generate secure secret keys
3. Configure production database
4. Set up HTTPS
5. Configure proper CORS origins
6. Set up monitoring and logging

## 📖 Documentation

### Implementation Documentation
- **[Project Complete Summary](PROJECT_COMPLETE_SUMMARY.md)** - Complete project overview
- **[Phase 4-7 Complete](PHASE_4_TO_7_COMPLETE.md)** - Master Agent integration summary
- **[Phase 8 Complete](PHASE_8_TESTING_COMPLETE.md)** - Testing & documentation summary
- **[Phase 3 Complete](PHASE_3_COMPLETE.md)** - Rate limiting & state management

### Technical Documentation
- **[API Documentation](API_DOCUMENTATION.md)** - Complete API reference
- **[Usage Guide](USAGE_GUIDE.md)** - Usage examples and best practices
- **[Master Agent Integration Plan](MASTER_AGENT_INTEGRATION_PLAN.md)** - Implementation plan
- **[Implementation Progress](IMPLEMENTATION_PROGRESS.md)** - Progress tracking

### Research & Verification
- **[AutoGen Verification](AUTOGEN_VERIFICATION_REPORT.md)** - AutoGen compatibility report
- **[Additional Tools Research](ADDITIONAL_TOOLS_RESEARCH_REPORT.md)** - Tools and libraries research

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License.

## 🆘 Support

For issues and questions:
- Create an issue on GitHub
- Check the documentation
- Review the logs in `backend/logs/`

---

**GRETA PAI** - The World's Most Advanced Personal AI System
