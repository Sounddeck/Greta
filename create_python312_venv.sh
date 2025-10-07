#!/bin/bash
# Greta PAI - Python 3.12 Virtual Environment Setup
# This script creates a Python 3.12 environment to avoid Python 3.13 compatibility issues

echo "🐍 GRETA PAI - Python 3.12 Environment Setup"
echo "==========================================="

# Check if Python 3.12 is available
if ! command -v python3.12 &> /dev/null; then
    echo "❌ Python 3.12 not found."
    echo ""
    echo "📦 Please install Python 3.12 using:"
    echo "  brew install python@3.12"
    echo ""
    echo "Or use conda:"
    echo "  conda create -n greta_pai python=3.12 -y"
    echo "  conda activate greta_pai"
    exit 1
fi

echo "✅ Python 3.12 found: $(python3.12 --version)"

# Remove any existing virtual environment
if [ -d "greta_env" ]; then
    echo "🗑️  Removing existing virtual environment..."
    rm -rf greta_env
fi

# Create new Python 3.12 virtual environment
echo "🏗️  Creating Python 3.12 virtual environment..."
python3.12 -m venv greta_env --clear

if [ $? -ne 0 ]; then
    echo "❌ Failed to create virtual environment"
    exit 1
fi

echo "✅ Virtual environment created successfully!"

# Activate and verify
echo "🔐 Activating virtual environment..."
source greta_env/bin/activate

if [ $? -ne 0 ]; then
    echo "❌ Failed to activate virtual environment"
    exit 1
fi

echo "✅ Virtual environment activated!"
echo "📍 Location: $(which python)"
echo "📍 Python version: $(python --version)"

echo ""
echo "🎉 SETUP COMPLETE!"
echo ""
echo "📝 Next steps:"
echo "1. Run: source greta_env/bin/activate"
echo "2. Run: pip install -r requirements.txt"
echo "3. Run: python main.py (from backend directory)"
echo ""
echo "💡 This environment uses Python 3.12 and should be fully compatible with all Greta PAI dependencies."

# Provide deployment command
cat << 'EOF'

🚀 DEPLOYMENT COMMANDS:

# 1. Activate environment
source greta_env/bin/activate

# 2. Install dependencies (should work now!)
pip install -r requirements.txt

# 3. Start MongoDB
brew services start mongodb-community

# 4. Launch Greta PAI
cd backend
python main.py

