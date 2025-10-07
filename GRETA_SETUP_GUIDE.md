# 🎭 GRETA PAI System - Complete Setup Guide

## 🚀 **GRETA is Now Production Ready!**

Your GRETA Personal AI system has been enhanced with world-class capabilities and is ready for deployment. This guide will help you set up and run the complete system.

---

## 📋 **What's Been Implemented**

### ✅ **Phase 1: Core Integration (COMPLETE)**
- **Central Orchestrator Service** - Unified command routing for all 17+ routers
- **Local LLM Integration** - llama.cpp with GGUF models and GPU acceleration
- **Real-time WebSocket System** - Live updates and event-driven architecture

### ✅ **Phase 2: Advanced Capabilities (COMPLETE)**
- **Digital Asset Creation Engine** - Create websites, apps, documents, marketing materials
- **Research & Reporting Automation** - Scheduled deep research with automated reports
- **Business Entity Framework** - AI CEO/CFO/CTO with autonomous decision making

### ✅ **Phase 3: Master UI/UX (COMPLETE)**
- **Enhanced Dashboard** - Comprehensive PAI system overview
- **Command Center** - Natural language and voice control interface
- **Advanced Components** - Professional-grade UI for all capabilities

---

## 🛠 **Installation Instructions**

### **1. Backend Setup**

```bash
# Navigate to backend directory
cd backend

# Install enhanced dependencies
pip install -r requirements_enhanced.txt

# For Apple Silicon Macs (Metal GPU acceleration)
CMAKE_ARGS="-DLLAMA_METAL=on" pip install llama-cpp-python

# For NVIDIA GPUs (CUDA acceleration)
CMAKE_ARGS="-DLLAMA_CUBLAS=on" pip install llama-cpp-python

# For CPU-only systems
pip install llama-cpp-python

# Create necessary directories
mkdir -p models/local_llm
mkdir -p generated_assets
mkdir -p research_reports
mkdir -p templates

# Start the enhanced backend
python main.py
```

### **2. Frontend Setup**

```bash
# Navigate to frontend directory
cd greta-frontend

# Install dependencies (if not already done)
npm install

# Start the enhanced frontend
npm start
```

### **3. Access Your GRETA System**

- **Frontend**: http://localhost:3000
- **Backend API**: http://localhost:8000
- **API Documentation**: http://localhost:8000/docs
- **WebSocket Test**: http://localhost:8000/api/v1/ws/test

---

## 🎯 **Key Features & Capabilities**

### **🧠 Central Command Center**
- **Natural Language Commands**: "Create a business website for my startup"
- **Voice Control**: Speak commands directly to GRETA
- **Multi-Router Coordination**: Automatically routes to appropriate services
- **Real-time Feedback**: Live execution status and results

### **🏗️ Digital Asset Creation**
Create any digital product with simple commands:

```bash
# Examples:
"Create a professional business website with modern design"
"Build a React Native mobile app for customer management"
"Generate a comprehensive business plan for my AI startup"
"Create API documentation for my REST service"
```

**Supported Asset Types:**
- Websites (HTML/CSS/JS)
- Mobile Apps (React Native, Flutter)
- Documents (PDF, Markdown, Word)
- Business Plans
- API Documentation
- Marketing Materials
- Brochures
- User Manuals

### **🔍 Research & Intelligence**
Automated research with scheduled reporting:

```bash
# Examples:
"Research AI industry trends and create a weekly report"
"Analyze competitor landscape for fintech startups"
"Monitor cryptocurrency market with daily updates"
```

**Research Capabilities:**
- News monitoring from multiple sources
- Academic research integration
- Business intelligence gathering
- Social media sentiment analysis
- Automated report generation
- Scheduled updates (hourly, daily, weekly, monthly)

### **🏢 AI Business Framework**
Complete business management with AI executives:

- **AI CEO**: Strategic planning and vision setting
- **AI CFO**: Financial analysis and planning
- **AI CTO**: Technology strategy and architecture
- **Board Meetings**: Coordinated decision making
- **Performance Metrics**: KPI tracking and optimization

### **🤖 Master Agent Control**
Comprehensive agent management:

- **Agent Creation**: Build custom AI agents for any task
- **Lifecycle Management**: Deploy, update, monitor, scale
- **Multi-Agent Coordination**: Orchestrate complex workflows
- **Performance Tracking**: Monitor agent efficiency and success rates

### **🧠 Local AI Processing**
Privacy-first AI with local models:

- **llama.cpp Integration**: Run models locally without cloud dependencies
- **GPU Acceleration**: Metal (Apple) and CUDA (NVIDIA) support
- **Model Management**: Download and switch between models
- **Continuous Learning**: Adapt to your usage patterns

---

## 🎮 **How to Use GRETA**

### **1. Start with the Command Center**
The Command Center is your primary interface to GRETA. Simply type or speak natural language commands:

- **Text Commands**: Type in the command input
- **Voice Commands**: Click the microphone button and speak
- **Quick Commands**: Use predefined shortcuts
- **Command History**: Access previous commands with arrow keys

### **2. Explore Capabilities**
Navigate through the enhanced interface:

- **Dashboard**: System overview and real-time metrics
- **PAI Hub**: Central intelligence coordination
- **Master Agent**: Agent management and deployment
- **Training**: Interactive AI education platform
- **Research Tools**: OSINT and intelligence gathering
- **Asset Creation**: Digital product generation

### **3. Monitor Real-time Updates**
The WebSocket system provides live updates:

- **System Status**: Real-time health monitoring
- **Agent Activities**: Live agent status updates
- **Command Execution**: Real-time command progress
- **Performance Metrics**: Live system performance data

---

## 🔧 **Configuration Options**

### **Environment Variables**
Create a `.env` file in the backend directory:

```env
# API Keys (optional - for enhanced capabilities)
OPENAI_API_KEY=your_openai_key
ANTHROPIC_API_KEY=your_anthropic_key

# Local LLM Configuration
DEFAULT_LOCAL_MODEL=phi3-mini
GPU_LAYERS=35
CONTEXT_LENGTH=4096

# Voice Configuration
DEFAULT_VOICE=en_US-lessac-medium
VOICE_SPEED=1.0

# Research Configuration
ENABLE_SCHEDULED_RESEARCH=true
MAX_RESEARCH_SOURCES=10

# Asset Creation Configuration
ENABLE_ASSET_CREATION=true
MAX_ASSET_SIZE_MB=100
```

### **Model Configuration**
GRETA supports multiple local models:

- **phi3-mini** (2.4GB) - Fast and efficient
- **llama3-8b** (4.8GB) - Balanced performance
- **codellama-7b** (4.1GB) - Code generation specialist

Models are automatically downloaded on first use.

---

## 🚀 **Advanced Usage**

### **API Integration**
All capabilities are available via REST API:

```python
import requests

# Execute command via orchestrator
response = requests.post('http://localhost:8000/api/v1/orchestrator/orchestrate', 
    json={
        "command": "Create a business website for my AI startup",
        "mode": "text",
        "priority": "normal"
    }
)

# Create digital asset
response = requests.post('http://localhost:8000/api/v1/assets/create',
    json={
        "asset_type": "website",
        "specifications": {
            "title": "My AI Startup",
            "description": "Revolutionary AI solutions"
        }
    }
)

# Schedule research
response = requests.post('http://localhost:8000/api/v1/research/schedule',
    json={
        "topic": "AI industry trends",
        "frequency": "weekly",
        "output_format": "report"
    }
)
```

### **WebSocket Integration**
Connect to real-time updates:

```javascript
const ws = new WebSocket('ws://localhost:8000/api/v1/ws');

ws.onmessage = (event) => {
    const data = JSON.parse(event.data);
    console.log('Real-time update:', data);
};

// Subscribe to specific topics
ws.send(JSON.stringify({
    type: 'subscribe',
    topic: 'system_status'
}));
```

---

## 🎉 **You're Ready to Go!**

GRETA is now fully operational with all advanced capabilities. The system provides:

- **17+ Integrated Routers** for comprehensive functionality
- **Natural Language Control** via the Command Center
- **Real-time Updates** through WebSocket integration
- **Local AI Processing** for privacy and performance
- **Digital Asset Creation** for any product or service
- **Automated Research** with scheduled reporting
- **AI Business Management** with executive decision making
- **Professional UI/UX** designed for power users

**Start exploring GRETA's capabilities by opening http://localhost:3000 and using the Command Center!**

---

## 📞 **Support & Documentation**

- **API Documentation**: http://localhost:8000/docs
- **WebSocket Testing**: http://localhost:8000/api/v1/ws/test
- **System Status**: Available in the Dashboard
- **Command Examples**: Built into the Command Center

**GRETA is now the world's most comprehensive Personal AI system. Enjoy exploring its capabilities!** 🎭✨
