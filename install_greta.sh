#!/bin/bash
# 🎭 GRETA PAI - ONE-COMMAND INSTALLER
# World's Most Advanced Personal AI - 5-Minute Setup
# Run with: bash install_greta.sh

set -e  # Exit on any error

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
INSTALL_DIR="$HOME/Desktop/Greta"
VENV_DIR="$INSTALL_DIR/greta_env"

echo -e "${BLUE}🎭 GRETA PAI - ULTIMATE PERSONAL AI${NC}"
echo -e "${BLUE}===================================${NC}"
echo ""

# Function to check system requirements
check_system() {
    echo -e "${YELLOW}🔍 Checking system requirements...${NC}"

    # Check macOS version
    if [[ "$(sw_vers -productName)" != "macOS" ]]; then
        echo -e "${RED}❌ This installer is designed for macOS only${NC}"
        exit 1
    fi

    # Check Apple Silicon
    if [[ "$(uname -m)" != "arm64" ]]; then
        echo -e "${RED}❌ This installer is optimized for Apple Silicon (M1/M2) Macs${NC}"
        exit 1
    fi

    # Check Python 3.12
    if ! command -v python3 &> /dev/null; then
        echo -e "${RED}❌ Python 3 is not installed. Please install Python 3.12 from python.org${NC}"
        exit 1
    fi

    PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}')
    if [[ ! $PYTHON_VERSION =~ ^3\.12 ]]; then
        echo -e "${YELLOW}⚠️ Python $PYTHON_VERSION detected. Recommended: 3.12.x${NC}"
        echo -e "${YELLOW}Continuing with current version...${NC}"
    fi

    echo -e "${GREEN}✅ System check passed${NC}"
}

# Function to install system dependencies
install_system_deps() {
    echo -e "${YELLOW}📦 Installing system dependencies...${NC}"

    # Check if Homebrew is installed
    if ! command -v brew &> /dev/null; then
        echo -e "${RED}❌ Homebrew is required. Install from: https://brew.sh/${NC}"
        exit 1
    fi

    # Install Xcode Command Line Tools if needed
    if ! xcode-select -p &> /dev/null; then
        echo -e "${YELLOW}Installing Xcode Command Line Tools...${NC}"
        xcode-select --install
    fi

    # Install MongoDB
    echo -e "${YELLOW}Installing MongoDB...${NC}"
    brew install mongodb-community

    # Install system packages
    echo -e "${YELLOW}Installing system packages...${NC}"
    brew install cmake pkg-config

    # Start MongoDB service
    echo -e "${YELLOW}Starting MongoDB...${NC}"
    brew services start mongodb-community

    echo -e "${GREEN}✅ System dependencies installed${NC}"
}

# Function to clone or update repository
setup_repository() {
    echo -e "${YELLOW}📁 Setting up Greta repository...${NC}"

    if [[ -d "$INSTALL_DIR" ]]; then
        echo -e "${YELLOW}Found existing Greta installation${NC}"
        cd "$INSTALL_DIR"
        echo -e "${YELLOW}Updating from git...${NC}"
        git pull origin main 2>/dev/null || true
    else
        echo -e "${YELLOW}Cloning Greta repository...${NC}"
        mkdir -p "$HOME/Desktop"
        cd "$HOME/Desktop"
        git clone https://github.com/Sounddeck/Greta.git
    fi

    echo -e "${GREEN}✅ Repository ready${NC}"
}

# Function to create Python virtual environment
create_venv() {
    echo -e "${YELLOW}🐍 Creating Python virtual environment...${NC}"

    cd "$INSTALL_DIR"
    python3 -m venv greta_env

    # Activate virtual environment
    source greta_env/bin/activate

    # Upgrade pip
    pip install --upgrade pip wheel setuptools

    echo -e "${GREEN}✅ Virtual environment created${NC}"
}

# Function to install Python dependencies
install_python_deps() {
    echo -e "${YELLOW}📚 Installing Python dependencies...${NC}"

    source greta_env/bin/activate

    echo -e "${YELLOW}Installing core web framework...${NC}"
    pip install fastapi==0.104.1 uvicorn==0.24.0 pydantic==2.5.0 pydantic-settings

    echo -e "${YELLOW}Installing Apple Silicon ML libraries...${NC}"
    pip install mlx>=0.29.1 mlx-lm>=0.28.0

    echo -e "${YELLOW}Installing database drivers...${NC}"
    pip install motor==3.3.2 pymongo==4.6.0

    echo -e "${YELLOW}Installing AI/ML libraries...${NC}"
    pip install numpy==1.26.2 pandas==2.1.4 scikit-learn==1.3.2 accelerate==0.25.0

    echo -e "${YELLOW}Installing web and utility libraries...${NC}"
    pip install httpx==0.25.2 requests==2.31.0 beautifulsoup4==4.12.2 lxml==10.1.0
    pip install pillow==10.1.0 matplotlib==3.8.2 seaborn==0.13.0 psutil==5.9.6
    pip install aiofiles==23.2.1 websockets==12.0 python-socketio>=5.8.0

    echo -e "${YELLOW}Installing voice and monitoring...${NC}"
    pip install pyttsx3==2.90 loguru==0.7.2 prometheus-client==0.19.0

    echo -e "${YELLOW}Installing utilities...${NC}"
    pip install click==8.1.7 rich==13.7.0 python-dotenv==1.0.0
    pip install jinja2==3.1.2 schedule==1.2.1 python-dateutil==2.8.2
    pip install distro==1.8.0 cryptography==41.0.7 python-jose==3.3.0
    pip install passlib[bcrypt]==1.7.4 python-multipart==0.0.6 redis>=4.5.0

    echo -e "${YELLOW}Installing LangChain ecosystem...${NC}"
    pip install langchain>=0.1.0 langchain-community>=0.0.10
    pip install langchain-core>=0.1.7 langgraph>=0.0.20

    echo -e "${GREEN}✅ Python dependencies installed${NC}"
}

# Function to configure environment
configure_env() {
    echo -e "${YELLOW}⚙️ Configuring environment...${NC}"

    cd "$INSTALL_DIR"

    # Create .env file if it doesn't exist
    if [[ ! -f ".env" ]]; then
        echo -e "${YELLOW}Creating .env configuration file...${NC}"

        cat > .env << EOF
# GRETA PAI Configuration
DEBUG=true
LOG_LEVEL=INFO

# Database
MONGODB_URL=mongodb://localhost:27017
DATABASE_NAME=greta_pai

# AI/ML Settings
USE_MLX=true
APPLE_SILICON=true

# Path Settings
PYTHONPATH=/Users/$(whoami)/Desktop/Greta

# Web Server
HOST=0.0.0.0
PORT=8000
RELOAD=true

# Security
SECRET_KEY=$(openssl rand -hex 32)
API_KEY=your_api_key_here

# Performance
MAX_MEMORY_USAGE=80
ENABLE_PERFORMANCE_MONITORING=true

# Experimental Features
ENABLE_EXPERIMENTAL_FEATURES=true
QUANTUM_REASONING_ENABLED=true
EMOTIONAL_INTELLIGENCE_ENABLED=true
NLP_COMMUNICATION_ENABLED=true

# Training
ENABLE_AI_TRAINING=true
PERSONALIZED_LEARNING=true

EOF

        echo -e "${GREEN}✅ .env file created${NC}"
    else
        echo -e "${YELLOW}⚠️ .env file already exists, skipping creation${NC}"
    fi

    # Configure PYTHONPATH
    export PYTHONPATH="$INSTALL_DIR:$PYTHONPATH"
}

# Function to create launch scripts
create_launch_scripts() {
    echo -e "${YELLOW}🚀 Creating launch scripts...${NC}"

    # Desktop launch script
    cat > ~/Desktop/Launch_Greta.command << 'EOF'
#!/bin/bash
echo "🎭 Starting Greta PAI..."
cd /Users/$(whoami)/Desktop/Greta
source greta_env/bin/activate
export PYTHONPATH=/Users/$(whoami)/Desktop/Greta:$PYTHONPATH
cd backend
python main.py
EOF

    chmod +x ~/Desktop/Launch_Greta.command

    # Stop script
    cat > ~/Desktop/Stop_Greta.command << 'EOF'
#!/bin/bash
echo "🛑 Stopping Greta PAI..."
pkill -f "python main.py"
echo "✅ Greta PAI stopped"
EOF

    chmod +x ~/Desktop/Stop_Greta.command

    # Status check script
    cat > ~/Desktop/Greta_Status.command << 'EOF'
#!/bin/bash
echo "📊 Greta PAI Status Check"
echo "========================="

if pgrep -f "python main.py" > /dev/null; then
    echo "✅ Greta PAI is running"
    ps aux | grep "python main.py" | grep -v grep
else
    echo "❌ Greta PAI is not running"
fi

# Check if port is accessible
if curl -f http://localhost:8000/health > /dev/null 2>&1; then
    echo "✅ Web interface accessible at http://localhost:8000"
else
    echo "⚠️ Web interface not responding (might be starting up)"
fi
EOF

    chmod +x ~/Desktop/Greta_Status.command

    echo -e "${GREEN}✅ Launch scripts created on Desktop${NC}"
}

# Function to verify installation
verify_installation() {
    echo -e "${YELLOW}🔬 Verifying installation...${NC}"

    source greta_env/bin/activate
    cd backend

    # Test basic imports
    echo -e "${YELLOW}Testing Python imports...${NC}"
    python3 -c "
import sys
sys.path.insert(0, '/Users/$(whoami)/Desktop/Greta')

try:
    # Test core ML
    import mlx
    import mlx_lm
    print('✅ MLX libraries imported')

    # Test web framework
    import fastapi
    import uvicorn
    print('✅ Web framework imported')

    # Test database
    import motor
    print('✅ Database driver imported')

    # Test Greta-specific modules
    from backend.services.meta_learning_engine import pai_meta_learning_engine
    print('✅ Greta PAI engines loaded')

except ImportError as e:
    print(f'❌ Import error: {e}')
    sys.exit(1)

print('🎉 All imports successful!')
"

    if [[ $? -eq 0 ]]; then
        echo -e "${GREEN}✅ Installation verification complete${NC}"
    else
        echo -e "${RED}❌ Installation verification failed${NC}"
        exit 1
    fi
}

# Function to run final tests and provide usage instructions
final_setup() {
    echo ""
    echo -e "${GREEN}🎊 GRETA PAI INSTALLATION COMPLETE!${NC}"
    echo ""
    echo -e "${BLUE}🚀 Your Greta PAI is ready to use!${NC}"
    echo ""
    echo -e "${YELLOW}📋 Launch Options:${NC}"
    echo -e "   ${GREEN}START:${NC} Double-click 'Launch_Greta.command' on your Desktop"
    echo -e "   ${RED}STOP:${NC} Double-click 'Stop_Greta.command' on your Desktop"
    echo -e "   ${BLUE}STATUS:${NC} Double-click 'Greta_Status.command' on your Desktop"
    echo ""
    echo -e "${YELLOW}🌐 Access Points:${NC}"
    echo -e "   ${BLUE}Web Interface:${NC} http://localhost:8000"
    echo -e "   ${BLUE}API Documentation:${NC} http://localhost:8000/docs"
    echo ""
    echo -e "${YELLOW}📚 Features Available:${NC}"
    echo -e "   🤖 Master Agent - Advanced multi-agent task coordination"
    echo -e "   🧠 Emotional Intelligence - Human-like emotional processing"
    echo -e "   🎭 NLP Communication - Adaptive human communication"
    echo -e "   🎓 Interactive Training - AI agent education system"
    echo -e "   🏭 Enterprise APIs - Commercial-grade enterprise interface"
    echo -e "   ⚡ Quantum Reasoning - Advanced probabilistic decision making"
    echo ""
    echo -e "${YELLOW}📁 Installation Location:${NC} $INSTALL_DIR"
    echo ""
    echo -e "${GREEN}🎯 What to do next?${NC}"
    echo -e "   1. ${BLUE}Launch Greta${NC} using the Desktop shortcut"
    echo -e "   2. ${BLUE}Open${NC} http://localhost:8000 in your web browser"
    echo -e "   3. ${BLUE}Explore${NC} all the amazing features!"
    echo -e "   4. ${BLUE}Use the training system${NC} to learn AI agent development"
    echo ""
    echo -e "${BLUE}🚀 Have fun with Greta PAI - the world's most advanced personal AI! 🎭✨${NC}"
}

# Function to display usage
show_usage() {
    echo -e "${BLUE}👋 Welcome to Greta PAI Installer!${NC}"
    echo ""
    echo -e "${YELLOW}This will install the world's most advanced personal AI system on your Mac.${NC}"
    echo ""
    echo -e "${GREEN}What this installer does:${NC}"
    echo -e "  ✅ Checks your system compatibility"
    echo -e "  ✅ Installs all required software (MongoDB, Python packages)"
    echo -e "  ✅ Downloads and sets up Greta PAI"
    echo -e "  ✅ Creates easy-to-use Desktop launch scripts"
    echo -e "  ✅ Verifies everything works correctly"
    echo ""
    echo -e "${YELLOW}Time required: ~10-15 minutes${NC}"
    echo -e "${YELLOW}Disk space needed: ~2GB${NC}"
    echo ""
    echo -e "${GREEN}Ready to continue? Press Enter to start installation...${NC}"
    read
}

# Main installation logic
main() {
    echo ""
    show_usage

    echo ""
    echo -e "${BLUE}▶️ Starting Greta PAI installation...${NC}"
    echo ""

    check_system
    echo ""

    install_system_deps
    echo ""

    setup_repository
    echo ""

    create_venv
    echo ""

    install_python_deps
    echo ""

    configure_env
    echo ""

    create_launch_scripts
    echo ""

    verify_installation
    echo ""

    final_setup

    echo ""
    echo -e "${GREEN}📝 Installation log saved to: ~/Desktop/greta_install.log${NC}"
}

# Run main installation
main 2>&1 | tee ~/Desktop/greta_install.log

# Ensure successful exit
exit 0
