#!/bin/bash
# GRETA PAI Launch Script
# Personal AI Infrastructure with German precision engineering

echo "🎭 Starting GRETA PAI - The World's Most Advanced Personal AI System"
echo "=================================================================="

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Check if we're in the right directory
if [ ! -f "backend/main.py" ]; then
    echo "❌ Error: Not in Greta project directory"
    echo "Please run this script from the Greta project root"
    exit 1
fi

# Check Python installation
if ! command -v python3 &> /dev/null; then
    echo "❌ Error: Python 3 is required but not installed"
    echo "Please install Python 3 and try again"
    exit 1
fi

# Create virtual environment if it doesn't exist
if [ ! -d "greta_env" ]; then
    echo "🔧 Creating Python virtual environment..."
    python3 -m venv greta_env
    
    if [ $? -ne 0 ]; then
        echo "❌ Error: Failed to create virtual environment"
        exit 1
    fi
fi

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source greta_env/bin/activate

if [ $? -ne 0 ]; then
    echo "❌ Error: Failed to activate virtual environment"
    exit 1
fi

# Install basic requirements if needed
echo "📦 Checking requirements..."

# Install FastAPI if not available
if ! python3 -c "import fastapi" &> /dev/null; then
    echo "📦 Installing FastAPI..."
    pip install fastapi uvicorn python-dotenv loguru
fi

# Install motor if available (optional MongoDB support)
if python3 -c "import motor" &> /dev/null; then
    echo "✅ MongoDB support available"
else
    echo "⚠️ MongoDB support not available - using file-based storage"
fi

# Create .env file if it doesn't exist
if [ ! -f ".env" ]; then
    echo "📝 Creating .env configuration file..."
    cp .env.example .env
    
    if [ $? -ne 0 ]; then
        echo "📝 Creating basic .env file..."
        cat > .env << EOF
# GRETA PAI Configuration
DEBUG=true
LOG_LEVEL=INFO
MONGODB_URL=mongodb://localhost:27017
DATABASE_NAME=greta_pai
HOST=0.0.0.0
PORT=8000
RELOAD=true
DA=Greta
DA_COLOR=blue
ENABLE_EXPERIMENTAL_FEATURES=true
QUANTUM_REASONING_ENABLED=true
EMOTIONAL_INTELLIGENCE_ENABLED=true
NLP_COMMUNICATION_ENABLED=true
ENABLE_AI_TRAINING=true
PERSONALIZED_LEARNING=true
PYTHONPATH=$SCRIPT_DIR
CORS_ORIGINS=["*"]
EOF
    fi
fi

# Create data directory for file-based storage
echo "📁 Creating data directories..."
mkdir -p data/{memory,interactions,agents,context,learning}

# Set PYTHONPATH
export PYTHONPATH="$SCRIPT_DIR:$PYTHONPATH"

echo ""
echo "🚀 Launching GRETA PAI Backend..."
echo "🌐 Web interface will be available at: http://localhost:8000"
echo "📊 Health check: http://localhost:8000/health"
echo "📖 API docs: http://localhost:8000/docs"
echo ""
echo "Press Ctrl+C to stop the server"
echo "=================================================================="

# Start the FastAPI server
cd backend
python3 main.py
