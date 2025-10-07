# GRETA - Installation Manual

## 📋 Table of Contents
1. [System Requirements](#system-requirements)
2. [Installation Steps](#installation-steps)
3. [Configuration](#configuration)
4. [Running GRETA](#running-greta)
5. [Troubleshooting](#troubleshooting)

---

## 🖥️ System Requirements

### Minimum Requirements:
- **OS:** macOS 10.15+, Linux (Ubuntu 20.04+), Windows 10+
- **CPU:** 4 cores (8 recommended)
- **RAM:** 8GB (16GB recommended)
- **Storage:** 20GB free space
- **Node.js:** v18.0.0 or higher
- **Python:** 3.11 or higher
- **MongoDB:** 6.0+ (optional, file-based fallback available)

### Recommended for LLM:
- **RAM:** 32GB+
- **GPU:** NVIDIA GPU with 8GB+ VRAM (for local LLM)
- **Storage:** 50GB+ for models

---

## 📦 Installation Steps

### 1. Clone the Repository

```bash
git clone https://github.com/Sounddeck/Greta.git
cd Greta
```

### 2. Backend Setup

#### Install Python Dependencies

```bash
# Create virtual environment
python3 -m venv greta_env

# Activate virtual environment
# On macOS/Linux:
source greta_env/bin/activate
# On Windows:
greta_env\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

#### Install Additional Dependencies

```bash
# For Apple Silicon Macs
pip install --upgrade pip
pip install numpy scipy scikit-learn
pip install fastapi uvicorn motor pydantic loguru
pip install python-multipart aiofiles
pip install psutil requests httpx
```

### 3. Frontend Setup

```bash
cd greta-frontend

# Install Node.js dependencies
npm install

# Or using yarn
yarn install
```

### 4. Database Setup (Optional)

#### MongoDB Installation

**macOS (using Homebrew):**
```bash
brew tap mongodb/brew
brew install mongodb-community@6.0
brew services start mongodb-community@6.0
```

**Linux (Ubuntu):**
```bash
wget -qO - https://www.mongodb.org/static/pgp/server-6.0.asc | sudo apt-key add -
echo "deb [ arch=amd64,arm64 ] https://repo.mongodb.org/apt/ubuntu focal/mongodb-org/6.0 multiverse" | sudo tee /etc/apt/sources.list.d/mongodb-org-6.0.list
sudo apt-get update
sudo apt-get install -y mongodb-org
sudo systemctl start mongod
```

**Windows:**
Download and install from: https://www.mongodb.com/try/download/community

**Note:** GRETA will automatically fall back to file-based storage if MongoDB is not available.

### 5. LLM Setup (Optional)

#### Install llama.cpp

```bash
# Clone llama.cpp
git clone https://github.com/ggerganov/llama.cpp
cd llama.cpp

# Build
make

# Download a model (example: Llama 3.1 8B)
# Place models in: backend/models/
```

#### Supported Models:
- Llama 3.1 (8B, 70B)
- Mistral 7B
- Mixtral 8x7B
- Any GGUF format model

---

## ⚙️ Configuration

### 1. Environment Variables

Create `.env` file in the root directory:

```bash
# Backend Configuration
BACKEND_PORT=8000
BACKEND_HOST=0.0.0.0

# Database Configuration
MONGODB_URI=mongodb://localhost:27017
DATABASE_NAME=greta_pai

# LLM Configuration
LLM_MODEL_PATH=backend/models/llama-3.1-8b.gguf
LLM_CONTEXT_SIZE=4096
LLM_THREADS=8

# Frontend Configuration
REACT_APP_API_URL=http://localhost:8000

# Security
SECRET_KEY=your-secret-key-here
CORS_ORIGINS=http://localhost:3000,http://localhost:3001
```

### 2. Backend Configuration

Edit `backend/config.py` if needed:

```python
# API Configuration
API_VERSION = "v1"
API_PREFIX = "/api/v1"

# Database
DB_NAME = "greta_pai"
DB_FALLBACK_DIR = "data/storage"

# LLM
LLM_MODEL = "llama-3.1-8b"
LLM_MAX_TOKENS = 2048
```

### 3. Frontend Configuration

Edit `greta-frontend/src/config.ts`:

```typescript
export const API_BASE_URL = process.env.REACT_APP_API_URL || 'http://localhost:8000';
export const API_VERSION = 'v1';
```

---

## 🚀 Running GRETA

### Development Mode

#### 1. Start Backend

```bash
# From project root
cd backend
python main.py
```

Backend will start on: `http://localhost:8000`

#### 2. Start Frontend

```bash
# In a new terminal
cd greta-frontend
npm start
```

Frontend will start on: `http://localhost:3000`

### Production Mode

#### 1. Build Frontend

```bash
cd greta-frontend
npm run build
```

#### 2. Run Backend with Production Settings

```bash
cd backend
uvicorn main:app --host 0.0.0.0 --port 8000 --workers 4
```

#### 3. Serve Frontend

Use a web server like Nginx or serve the build folder:

```bash
npm install -g serve
serve -s build -p 3000
```

---

## 🔧 Troubleshooting

### Common Issues

#### 1. Backend Won't Start

**Error:** `Address already in use`
```bash
# Kill process on port 8000
lsof -ti:8000 | xargs kill -9
```

**Error:** `Module not found`
```bash
# Reinstall dependencies
pip install -r requirements.txt --force-reinstall
```

#### 2. Frontend Won't Start

**Error:** `npm ERR! code ELIFECYCLE`
```bash
# Clear cache and reinstall
rm -rf node_modules package-lock.json
npm install
```

**Error:** `Cannot connect to backend`
- Check backend is running on port 8000
- Verify CORS settings in backend
- Check firewall settings

#### 3. Database Connection Issues

**MongoDB not connecting:**
```bash
# Check MongoDB status
# macOS:
brew services list | grep mongodb

# Linux:
sudo systemctl status mongod

# Restart if needed
brew services restart mongodb-community@6.0  # macOS
sudo systemctl restart mongod  # Linux
```

**File-based fallback:**
- GRETA automatically uses file-based storage if MongoDB fails
- Data stored in: `backend/data/storage/`

#### 4. LLM Issues

**Model not loading:**
- Verify model path in `.env`
- Check model format (must be GGUF)
- Ensure sufficient RAM

**Slow inference:**
- Reduce context size
- Use smaller model
- Enable GPU acceleration if available

### Performance Optimization

#### Backend:
```bash
# Use multiple workers
uvicorn main:app --workers 4

# Enable caching
# Add to backend/config.py:
ENABLE_CACHE = True
CACHE_TTL = 3600
```

#### Frontend:
```bash
# Build with optimization
npm run build -- --optimization

# Enable compression
# Install compression middleware
```

---

## 📊 Verification

### Check Installation

```bash
# Backend health check
curl http://localhost:8000/api/v1/health

# Expected response:
# {"status": "healthy", "service": "greta-pai", "timestamp": "..."}
```

### Test API Endpoints

```bash
# Test system stats
curl http://localhost:8000/api/v1/system/stats

# Test LLM
curl -X POST http://localhost:8000/api/v1/llm/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "Hello, GRETA!"}'
```

---

## 🎯 Next Steps

After successful installation:

1. Read the [Training Manual](TRAINING_MANUAL.md)
2. Review the [User Guide](USER_GUIDE.md)
3. Explore the [API Documentation](API_DOCUMENTATION.md)
4. Start building your first agent!

---

## 📞 Support

- **GitHub Issues:** https://github.com/Sounddeck/Greta/issues
- **Documentation:** https://github.com/Sounddeck/Greta/wiki
- **Email:** support@greta-ai.com

---

**Installation complete! Welcome to GRETA! 🎉**

