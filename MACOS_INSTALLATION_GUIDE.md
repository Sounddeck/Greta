# 🍎 GRETA/CPAS4 - macOS Installation Guide

## Complete Setup Guide for macOS

**Last Updated**: 2025-10-08  
**macOS Version**: 10.15+ (Catalina or later)  
**Estimated Setup Time**: 30-45 minutes

---

## 📋 Table of Contents

1. [Prerequisites](#prerequisites)
2. [System Requirements](#system-requirements)
3. [Installation Steps](#installation-steps)
4. [Configuration](#configuration)
5. [Running the Application](#running-the-application)
6. [Troubleshooting](#troubleshooting)
7. [Uninstallation](#uninstallation)

---

## 🔧 Prerequisites

### Required Software

#### 1. Homebrew (Package Manager)
```bash
# Install Homebrew if not already installed
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Verify installation
brew --version
```

#### 2. Python 3.11+
```bash
# Install Python 3.11
brew install python@3.11

# Verify installation
python3 --version  # Should show 3.11.x or higher

# Add to PATH (add to ~/.zshrc or ~/.bash_profile)
echo 'export PATH="/opt/homebrew/bin:$PATH"' >> ~/.zshrc
source ~/.zshrc
```

#### 3. Node.js 18+ and npm
```bash
# Install Node.js
brew install node

# Verify installation
node --version  # Should show v18.x or higher
npm --version   # Should show 9.x or higher
```

#### 4. MongoDB 8.2+
```bash
# Install MongoDB
brew tap mongodb/brew
brew install mongodb-community@8.2

# Start MongoDB service
brew services start mongodb-community@8.2

# Verify MongoDB is running
mongosh --eval "db.version()"  # Should show 8.2.x
```

#### 5. Ollama (Local LLM Runtime)
```bash
# Download and install Ollama
brew install ollama

# Start Ollama service
brew services start ollama

# Pull Llama 3 model (8B parameters, ~4.7GB)
ollama pull llama3:latest

# Verify Ollama is running
curl http://localhost:11434/api/tags
```

---

## 💻 System Requirements

### Minimum Requirements
- **macOS**: 10.15 (Catalina) or later
- **RAM**: 8GB (16GB recommended for Llama 3)
- **Storage**: 20GB free space
- **CPU**: Intel Core i5 or Apple Silicon M1/M2/M3

### Recommended Requirements
- **macOS**: 13.0 (Ventura) or later
- **RAM**: 16GB or more
- **Storage**: 50GB free space
- **CPU**: Apple Silicon M1/M2/M3 (better performance)

---

## 📥 Installation Steps

### Step 1: Clone the Repository

```bash
# Navigate to your preferred directory
cd ~/Desktop

# Clone the repository
git clone https://github.com/Sounddeck/Greta.git

# Navigate into the project
cd Greta
```

### Step 2: Backend Setup

```bash
# Create Python virtual environment
python3 -m venv greta_env

# Activate virtual environment
source greta_env/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install backend dependencies
pip install -r requirements.txt

# Install additional dependencies for Priority 2 modules
pip install pydantic[email]

# Verify installation
pip list | grep -E "fastapi|motor|pydantic|ollama"
```

### Step 3: Frontend Setup

```bash
# Navigate to frontend directory
cd greta-frontend

# Install dependencies
npm install

# Verify installation
npm list react react-dom typescript
```

### Step 4: Database Setup

```bash
# Ensure MongoDB is running
brew services list | grep mongodb

# If not running, start it
brew services start mongodb-community@8.2

# Create database (optional - will be created automatically)
mongosh --eval "use greta_db"
```

### Step 5: Environment Configuration

```bash
# Navigate back to project root
cd ~/Desktop/Greta

# Create backend .env file (if needed)
cat > backend/.env << 'EOF'
# MongoDB Configuration
MONGODB_URL=mongodb://localhost:27017
MONGODB_DB_NAME=greta_db

# Ollama Configuration
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=llama3:latest

# API Configuration
API_HOST=0.0.0.0
API_PORT=8000

# CORS Configuration
CORS_ORIGINS=http://localhost:3000,http://localhost:3001

# Environment
ENVIRONMENT=development
EOF

# Create frontend .env file
cat > greta-frontend/.env << 'EOF'
REACT_APP_API_URL=http://localhost:8000
REACT_APP_WS_URL=ws://localhost:8000
EOF
```

---

## ⚙️ Configuration

### MongoDB Configuration

```bash
# Check MongoDB configuration
cat /opt/homebrew/etc/mongod.conf

# Default settings should work, but you can customize:
# - Port: 27017 (default)
# - Data directory: /opt/homebrew/var/mongodb
# - Log directory: /opt/homebrew/var/log/mongodb
```

### Ollama Configuration

```bash
# Check available models
ollama list

# Pull additional models if needed
ollama pull llama3:8b      # 8B parameter model (default)
ollama pull llama3:70b     # 70B parameter model (requires 64GB+ RAM)

# Test Ollama
curl http://localhost:11434/api/generate -d '{
  "model": "llama3",
  "prompt": "Hello, how are you?",
  "stream": false
}'
```

---

## 🚀 Running the Application

### Option 1: Manual Start (Development)

#### Terminal 1: Start Backend
```bash
cd ~/Desktop/Greta
source greta_env/bin/activate
cd backend
python main.py
```

**Expected Output**:
```
INFO:     Started server process [12345]
INFO:     Waiting for application startup.
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:8000
✅ MongoDB connected successfully
✅ CRM Router integrated
✅ ATS Router integrated
✅ Agent Builder Router integrated
```

#### Terminal 2: Start Frontend
```bash
cd ~/Desktop/Greta/greta-frontend
npm start
```

**Expected Output**:
```
Compiled successfully!

You can now view greta-frontend in the browser.

  Local:            http://localhost:3000
  On Your Network:  http://192.168.1.x:3000
```

#### Terminal 3: Verify Services
```bash
# Check backend health
curl http://localhost:8000/

# Check MongoDB
mongosh --eval "db.adminCommand('ping')"

# Check Ollama
curl http://localhost:11434/api/tags
```

### Option 2: Automated Start (Production)

Create a startup script:

```bash
# Create startup script
cat > ~/Desktop/Greta/start.sh << 'EOF'
#!/bin/bash

# Start MongoDB
brew services start mongodb-community@8.2

# Start Ollama
brew services start ollama

# Wait for services to start
sleep 5

# Start Backend
cd ~/Desktop/Greta
source greta_env/bin/activate
cd backend
python main.py &
BACKEND_PID=$!

# Wait for backend to start
sleep 10

# Start Frontend
cd ~/Desktop/Greta/greta-frontend
npm start &
FRONTEND_PID=$!

echo "✅ GRETA/CPAS4 is starting..."
echo "Backend PID: $BACKEND_PID"
echo "Frontend PID: $FRONTEND_PID"
echo ""
echo "Access the application at: http://localhost:3000"
echo "API documentation at: http://localhost:8000/docs"
echo ""
echo "Press Ctrl+C to stop all services"

# Wait for user interrupt
wait
EOF

# Make executable
chmod +x ~/Desktop/Greta/start.sh

# Run the script
~/Desktop/Greta/start.sh
```

### Option 3: Background Services (Always Running)

```bash
# Create LaunchAgent for backend
cat > ~/Library/LaunchAgents/com.greta.backend.plist << 'EOF'
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>com.greta.backend</string>
    <key>ProgramArguments</key>
    <array>
        <string>/Users/YOUR_USERNAME/Desktop/Greta/greta_env/bin/python</string>
        <string>/Users/YOUR_USERNAME/Desktop/Greta/backend/main.py</string>
    </array>
    <key>RunAtLoad</key>
    <true/>
    <key>KeepAlive</key>
    <true/>
    <key>StandardOutPath</key>
    <string>/tmp/greta-backend.log</string>
    <key>StandardErrorPath</key>
    <string>/tmp/greta-backend-error.log</string>
</dict>
</plist>
EOF

# Replace YOUR_USERNAME with your actual username
sed -i '' "s/YOUR_USERNAME/$USER/g" ~/Library/LaunchAgents/com.greta.backend.plist

# Load the service
launchctl load ~/Library/LaunchAgents/com.greta.backend.plist

# Check status
launchctl list | grep greta
```

---

## 🔍 Verification

### 1. Check All Services

```bash
# MongoDB
brew services list | grep mongodb
# Should show: mongodb-community@8.2 started

# Ollama
brew services list | grep ollama
# Should show: ollama started

# Backend
curl http://localhost:8000/
# Should return: {"message": "GRETA Backend API", ...}

# Frontend
curl http://localhost:3000/
# Should return HTML content
```

### 2. Test API Endpoints

```bash
# Test CRM health
curl http://localhost:8000/api/v1/crm/health | python3 -m json.tool

# Test ATS health
curl http://localhost:8000/api/v1/ats/health | python3 -m json.tool

# Test Agent Builder health
curl http://localhost:8000/api/v1/agent-builder/health | python3 -m json.tool
```

### 3. Access the Application

1. Open browser: http://localhost:3000
2. You should see the GRETA dashboard
3. Navigate to different modules:
   - CRM: http://localhost:3000/business
   - ATS: http://localhost:3000/business
   - Agent Builder: http://localhost:3000/agent-builder

---

## 🐛 Troubleshooting

### Issue 1: MongoDB Won't Start

**Symptoms**: `brew services start mongodb-community@8.2` fails

**Solutions**:
```bash
# Check if port 27017 is in use
lsof -i :27017

# Kill existing process if needed
kill -9 <PID>

# Remove lock file
rm /opt/homebrew/var/mongodb/mongod.lock

# Repair database
mongod --repair --dbpath /opt/homebrew/var/mongodb

# Restart service
brew services restart mongodb-community@8.2
```

### Issue 2: Ollama Not Responding

**Symptoms**: `curl http://localhost:11434` fails

**Solutions**:
```bash
# Check if Ollama is running
ps aux | grep ollama

# Restart Ollama
brew services restart ollama

# Check logs
tail -f /opt/homebrew/var/log/ollama.log

# Reinstall if needed
brew reinstall ollama
ollama pull llama3:latest
```

### Issue 3: Backend Won't Start

**Symptoms**: `python main.py` shows errors

**Solutions**:
```bash
# Check Python version
python3 --version  # Must be 3.11+

# Reinstall dependencies
pip install --upgrade -r requirements.txt

# Check for port conflicts
lsof -i :8000

# Try different port
cd backend
uvicorn main:app --host 0.0.0.0 --port 8001
```

### Issue 4: Frontend Won't Start

**Symptoms**: `npm start` fails

**Solutions**:
```bash
# Clear npm cache
npm cache clean --force

# Remove node_modules and reinstall
rm -rf node_modules package-lock.json
npm install

# Check for port conflicts
lsof -i :3000

# Try different port
PORT=3001 npm start
```

### Issue 5: Database Connection Errors

**Symptoms**: Backend logs show MongoDB connection errors

**Solutions**:
```bash
# Verify MongoDB is running
mongosh --eval "db.adminCommand('ping')"

# Check connection string in backend/.env
cat backend/.env | grep MONGODB

# Test connection manually
mongosh mongodb://localhost:27017/greta_db
```

---

## 🗑️ Uninstallation

### Complete Removal

```bash
# Stop all services
brew services stop mongodb-community@8.2
brew services stop ollama
launchctl unload ~/Library/LaunchAgents/com.greta.backend.plist

# Remove application
rm -rf ~/Desktop/Greta

# Remove LaunchAgent
rm ~/Library/LaunchAgents/com.greta.backend.plist

# Optional: Remove dependencies
brew uninstall mongodb-community@8.2
brew uninstall ollama
brew uninstall node
brew uninstall python@3.11

# Optional: Remove data
rm -rf /opt/homebrew/var/mongodb
rm -rf ~/.ollama
```

---

## 📚 Additional Resources

- **API Documentation**: http://localhost:8000/docs
- **MongoDB Compass**: https://www.mongodb.com/products/compass
- **Ollama Documentation**: https://ollama.ai/docs
- **React Documentation**: https://react.dev

---

**Installation Guide Version**: 1.0  
**Last Updated**: 2025-10-08  
**Support**: Check GitHub issues or documentation

