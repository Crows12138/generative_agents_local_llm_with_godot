# Installation Guide

Complete installation guide for Generative Agents with Local LLM and Godot.

## 📋 Prerequisites

### System Requirements

**Minimum:**
- Python 3.8+
- 4GB RAM
- 2GB free disk space
- Internet connection (for initial setup)

**Recommended:**
- Python 3.10+
- 8GB+ RAM
- 5GB free disk space
- SSD storage
- Dedicated GPU (optional, improves performance)

### Supported Platforms
- ✅ Windows 10/11
- ✅ Linux (Ubuntu 20.04+, Debian 11+)
- ✅ macOS 10.15+

## 🚀 Quick Installation

### Option 1: One-Click Setup (Recommended)

1. **Clone the repository:**
```bash
git clone https://github.com/your-repo/generative_agents_local_llm_with_godot.git
cd generative_agents_local_llm_with_godot
```

2. **Run the installer:**
```bash
# Windows
run_optimized_demo.bat

# Linux/macOS
python run_optimized_demo.py --performance medium
```

The installer will automatically:
- Check system requirements
- Install Python dependencies
- Download AI models
- Configure the system
- Launch the demo

### Option 2: Manual Installation

#### Step 1: Python Environment Setup

**Create virtual environment:**
```bash
# Create virtual environment
python -m venv .venv

# Activate virtual environment
# Windows:
.venv\Scripts\activate
# Linux/macOS:
source .venv/bin/activate
```

#### Step 2: Install Dependencies

**Install core dependencies:**
```bash
pip install -r requirements.txt
```

**Core packages included:**
- `gpt4all>=2.5.0` - Local LLM support
- `fastapi>=0.100.0` - API framework
- `uvicorn>=0.23.0` - ASGI server
- `pydantic>=2.0.0` - Data validation
- `pyyaml>=6.0` - Configuration management
- `psutil>=5.9.0` - System monitoring
- `requests>=2.31.0` - HTTP client
- `aiohttp>=3.8.0` - Async HTTP
- `redis>=4.6.0` - Cache (optional)

#### Step 3: AI Model Setup

**Automatic download (recommended):**
```bash
python ai_service/ai_service.py
```

**Manual download:**
1. Create models directory:
```bash
mkdir -p models/gpt4all
```

2. Download Qwen3-30B model:
```bash
# Download from Hugging Face
# Place in models/gpt4all/Qwen3-30B-A3B-Instruct-2507-UD-Q4_K_XL.gguf
```

#### Step 4: Configuration

**Create configuration file:**
```bash
cp ai_config.json.example ai_config.json
```

**Edit configuration:**
```json
{
  "model": {
    "active_model": "qwen",
    "models_dir": "./models/gpt4all",
    "max_tokens": 800
  },
  "service": {
    "host": "localhost",
    "port": 8080
  },
  "performance": {
    "enable_cache": true,
    "memory_limit_gb": 4
  }
}
```

#### Step 5: Verify Installation

**Run system test:**
```bash
python simple_test.py
```

**Expected output:**
```
=== Simple Integration Test ===
1. Testing agent import...
   ✓ Agents imported
2. Creating demo agents...
   ✓ Created 5 agents
3. Testing basic functionality...
   ✓ Memory working: 1 memories
   ✓ Decision: greet
   ✓ Serialization: 7 fields

=== ALL BASIC TESTS PASSED ===
```

## 🎮 Godot Setup (Optional)

### Install Godot Engine

**Download Godot 4.x:**
- Visit [godotengine.org](https://godotengine.org/download)
- Download Godot 4.2+ (stable)
- Extract to your preferred location

**Add to PATH (optional):**
```bash
# Windows
# Add Godot installation folder to system PATH

# Linux/macOS
export PATH=$PATH:/path/to/godot
```

### Setup Godot Project

1. **Open Godot**
2. **Import project:**
   - Click "Import"
   - Navigate to `godot/live-with-ai/`
   - Select `project.godot`
   - Click "Import & Edit"

3. **Configure performance:**
   - Copy `godot_performance_config.gd` to project
   - Add as autoload scene

## 🐳 Docker Installation (Alternative)

### Using Docker Compose

**Create docker-compose.yml:**
```yaml
version: '3.8'

services:
  ai-service:
    build: .
    ports:
      - "8080:8080"
    volumes:
      - ./models:/app/models
      - ./logs:/app/logs
    environment:
      - PYTHONPATH=/app
    command: python ai_service/ai_service.py

  demo:
    build: .
    depends_on:
      - ai-service
    volumes:
      - ./logs:/app/logs
    command: python demo_performance_suite.py
```

**Run with Docker:**
```bash
docker-compose up -d
```

## ⚙️ Advanced Configuration

### Performance Optimization

**Memory settings:**
```json
{
  "performance": {
    "memory_limit_gb": 8,
    "cache_size": 1000,
    "enable_memory_optimization": true,
    "gc_threshold": 0.8
  }
}
```

**AI model settings:**
```json
{
  "model": {
    "max_tokens": 1200,
    "temperature": 0.7,
    "batch_size": 4,
    "context_length": 4096
  }
}
```

### Network Configuration

**API settings:**
```json
{
  "service": {
    "host": "0.0.0.0",
    "port": 8080,
    "workers": 4,
    "timeout": 30
  }
}
```

**Cache configuration:**
```json
{
  "cache": {
    "type": "redis",
    "host": "localhost",
    "port": 6379,
    "ttl_hours": 24
  }
}
```

## 🧪 Testing Installation

### Quick Tests

**Basic functionality:**
```bash
python simple_test.py
```

**Full integration test:**
```bash
python integration_test.py
```

**Performance test:**
```bash
python demo_performance_suite.py --test-mode --duration 2
```

### Troubleshooting Tests

**Check Python environment:**
```bash
python --version
pip list | grep -E "(gpt4all|fastapi|pydantic)"
```

**Check model availability:**
```bash
python -c "from ai_service.ai_service import get_health_status; print(get_health_status())"
```

**Check system resources:**
```bash
python -c "import psutil; print(f'Memory: {psutil.virtual_memory().available/1024**3:.1f}GB')"
```

## 🔧 Environment Variables

**Optional environment variables:**

```bash
# AI Service Configuration
export AI_SERVICE_PORT=8080
export AI_SERVICE_HOST=localhost
export AI_MODEL_PATH=./models/gpt4all

# Performance Settings
export ENABLE_CACHE=true
export MEMORY_LIMIT_GB=4
export LOG_LEVEL=INFO

# Development Settings
export DEBUG_MODE=false
export PERFORMANCE_MONITORING=true
```

## 🚨 Common Issues

### Installation Fails

**Python version error:**
```bash
# Check Python version
python --version

# Upgrade Python if needed
# Windows: Download from python.org
# Linux: sudo apt update && sudo apt install python3.10
# macOS: brew install python@3.10
```

**Memory error during model download:**
```bash
# Increase virtual memory or use smaller model
# Edit ai_config.json to use lighter model
```

**Permission errors:**
```bash
# Windows: Run as administrator
# Linux/macOS: Check file permissions
chmod +x run_optimized_demo.py
```

### Performance Issues

**Slow AI responses:**
- Reduce `max_tokens` in configuration
- Enable caching
- Use performance mode "low" or "medium"

**High memory usage:**
- Reduce `memory_limit_gb`
- Enable memory optimization
- Use fewer agents

**Network timeouts:**
- Increase timeout settings
- Check firewall settings
- Verify network connectivity

## 📞 Support

If you encounter issues:

1. Check [Troubleshooting Guide](TROUBLESHOOTING.md)
2. Run diagnostic: `python integration_test.py`
3. Check logs in `logs/` directory
4. Create GitHub issue with system info and logs

## ✅ Installation Checklist

- [ ] Python 3.8+ installed
- [ ] Virtual environment created and activated
- [ ] Dependencies installed (`pip install -r requirements.txt`)
- [ ] AI models downloaded (automatic or manual)
- [ ] Configuration file created (`ai_config.json`)
- [ ] Basic test passed (`python simple_test.py`)
- [ ] Integration test passed (`python integration_test.py`)
- [ ] Demo launcher works (`python run_optimized_demo.py`)
- [ ] (Optional) Godot project imported and configured

**Installation complete! 🎉**

Continue to [Quick Start Guide](QUICKSTART.md) for usage instructions.