# Troubleshooting Guide

Solutions for common issues when using Generative Agents with Local LLM and Godot.

## 🚨 Quick Diagnosis

### Run System Check
```bash
# Quick system health check
python simple_test.py

# Full diagnostic
python integration_test.py

# Performance check
python demo_performance_suite.py --test-mode --duration 1
```

### Check System Resources
```bash
# Memory availability
python -c "import psutil; print(f'Available RAM: {psutil.virtual_memory().available/1024**3:.1f}GB')"

# Python version
python --version

# Installed packages
pip list | grep -E "(gpt4all|fastapi|pydantic)"
```

## 🛠️ Installation Issues

### Python Version Problems

**Problem:** `Python version not supported`
```
ERROR: Python 3.7 is not supported
```

**Solution:**
```bash
# Check Python version
python --version

# Install Python 3.8+ from python.org
# Or use package manager:
# Windows: winget install Python.Python.3.11
# macOS: brew install python@3.11
# Linux: sudo apt install python3.11
```

### Dependency Installation Failures

**Problem:** `pip install fails with permission error`
```
ERROR: Could not install packages due to an EnvironmentError
```

**Solutions:**
```bash
# Option 1: Use virtual environment (recommended)
python -m venv .venv
# Windows:
.venv\Scripts\activate
# Linux/macOS:
source .venv/bin/activate
pip install -r requirements.txt

# Option 2: User installation
pip install --user -r requirements.txt

# Option 3: Administrator (Windows)
# Run terminal as administrator
pip install -r requirements.txt
```

**Problem:** `Package conflict or version incompatibility`
```
ERROR: pip's dependency resolver does not currently consider all the packages
```

**Solution:**
```bash
# Clean installation
pip uninstall -y gpt4all fastapi pydantic
pip install --no-cache-dir -r requirements.txt

# Or use specific versions
pip install gpt4all==2.5.0 fastapi==0.100.0 pydantic==2.0.0
```

### Model Download Issues

**Problem:** `Model download fails or is very slow`
```
ERROR: Failed to download model file
```

**Solutions:**
```bash
# Check internet connection
ping google.com

# Manual model download
mkdir -p models/gpt4all
cd models/gpt4all

# Download Qwen model manually from Hugging Face
# Place file as: Qwen3-30B-A3B-Instruct-2507-UD-Q4_K_XL.gguf

# Verify model file
ls -lh *.gguf
```

**Problem:** `Model file corrupted or invalid`
```
ERROR: Failed to load model
```

**Solution:**
```bash
# Remove corrupted model
rm models/gpt4all/*.gguf

# Re-download
python ai_service/ai_service.py

# Check file integrity
python -c "
from gpt4all import GPT4All
model = GPT4All('models/gpt4all/Qwen3-30B-A3B-Instruct-2507-UD-Q4_K_XL.gguf')
print('Model loaded successfully')
"
```

## ⚡ Performance Issues

### Slow AI Responses

**Problem:** AI responses take more than 5 seconds
```
⚠️ Performance Warning: High AI response time 6.2s
```

**Solutions:**
```bash
# 1. Use lighter performance mode
python run_optimized_demo.py --performance low

# 2. Enable caching
# Edit ai_config.json:
{
  "performance": {
    "enable_cache": true,
    "cache_size": 1000
  }
}

# 3. Reduce token limit
{
  "model": {
    "max_tokens": 400
  }
}

# 4. Check system load
# Close other applications
# Ensure sufficient RAM available
```

### High Memory Usage

**Problem:** Memory usage exceeds 8GB
```
⚠️ Performance Warning: High memory usage 8.2GB
```

**Solutions:**
```bash
# 1. Lower memory limit
python run_optimized_demo.py --performance low

# 2. Enable memory optimization
# Edit ai_config.json:
{
  "performance": {
    "enable_memory_optimization": true,
    "memory_limit_gb": 4
  }
}

# 3. Reduce agent count
{
  "demo_config": {
    "max_agents": 2
  }
}

# 4. Force cleanup
python -c "
from memory_optimizer import get_memory_optimizer
optimizer = get_memory_optimizer()
optimizer.force_cleanup()
"
```

### Low Frame Rate (Godot)

**Problem:** Game FPS drops below 20
```
⚠️ Performance Warning: Low FPS 18
```

**Solutions:**
```gdscript
# In godot_performance_config.gd
func optimize_for_low_end():
    current_quality_level = 0  # Lowest quality
    max_visible_characters = 5
    character_lod_distance = 200.0
    get_viewport().set_render_scale(0.7)
```

**Or via Python:**
```bash
# Use mobile optimization
python run_optimized_demo.py --performance low --ignore-ai-failure
```

## 🤖 AI Service Issues

### AI Service Won't Start

**Problem:** `AI service failed to start`
```
❌ AI service startup failed: Connection refused
```

**Solutions:**
```bash
# 1. Check port availability
netstat -an | grep 8080
# If port is busy, kill the process or change port

# 2. Start service manually
python ai_service/ai_service.py

# 3. Check for error messages
python ai_service/ai_service.py 2>&1 | tee ai_service.log

# 4. Use different port
# Edit ai_config.json:
{
  "service": {
    "port": 8081
  }
}
```

### AI Service Crashes

**Problem:** `AI service stops responding`
```
ERROR: AI service crashed unexpectedly
```

**Solutions:**
```bash
# 1. Check logs
cat ai_service/ai_service.log

# 2. Restart with verbose logging
python ai_service/ai_service.py --log-level DEBUG

# 3. Use fallback mode
python run_optimized_demo.py --ignore-ai-failure

# 4. Check system resources
# Ensure sufficient memory
# Close other applications
```

### Model Loading Errors

**Problem:** `Failed to load AI model`
```
ERROR: Could not load model file
```

**Solutions:**
```bash
# 1. Verify model file exists
ls -la models/gpt4all/

# 2. Check file permissions
chmod 644 models/gpt4all/*.gguf

# 3. Test model loading
python -c "
from ai_service.ai_service import _get_model_instance
model = _get_model_instance('qwen')
print('Model loaded successfully')
"

# 4. Download model again
rm models/gpt4all/*.gguf
python ai_service/ai_service.py
```

## 🎮 Godot Integration Issues

### Godot Project Won't Open

**Problem:** `Godot project fails to import`
```
ERROR: Failed to import project
```

**Solutions:**
```bash
# 1. Check Godot version
godot --version  # Should be 4.2+

# 2. Update Godot
# Download latest from godotengine.org

# 3. Open project manually
godot --path godot/live-with-ai/

# 4. Check project.godot file
cat godot/live-with-ai/project.godot
```

### Godot-Python Communication Fails

**Problem:** `Godot can't connect to AI service`
```
ERROR: HTTP request failed
```

**Solutions:**
```gdscript
# In Godot script, check connection
func test_ai_connection():
    var http_request = HTTPRequest.new()
    add_child(http_request)
    var error = http_request.request("http://localhost:8080/ai/health")
    if error != OK:
        print("Connection failed: ", error)
```

**Python side:**
```bash
# Ensure AI service is running
curl http://localhost:8080/ai/health

# Check firewall settings
# Ensure localhost:8080 is accessible
```

### Performance Issues in Godot

**Problem:** Game stutters or freezes
```
⚠️ Game performance degraded
```

**Solutions:**
```gdscript
# Optimize Godot settings
func optimize_for_performance():
    # Reduce visual quality
    get_viewport().set_render_scale(0.8)
    
    # Limit characters
    max_visible_characters = 3
    
    # Disable expensive effects
    enable_shadows = false
    enable_particles = false
```

## 🔧 Configuration Issues

### Invalid Configuration File

**Problem:** `Configuration parsing error`
```
ERROR: Invalid JSON in ai_config.json
```

**Solutions:**
```bash
# 1. Validate JSON
python -m json.tool ai_config.json

# 2. Reset to default
cp ai_config.json.example ai_config.json

# 3. Use minimal config
cat > ai_config.json << EOF
{
  "model": {
    "active_model": "qwen"
  },
  "service": {
    "port": 8080
  }
}
EOF
```

### Environment Variables Issues

**Problem:** `Environment variables not recognized`

**Solutions:**
```bash
# Windows
set AI_SERVICE_PORT=8080
set PYTHONPATH=%CD%

# Linux/macOS
export AI_SERVICE_PORT=8080
export PYTHONPATH=$PWD

# Or create .env file
cat > .env << EOF
AI_SERVICE_PORT=8080
PYTHONPATH=.
LOG_LEVEL=INFO
EOF
```

## 🌐 Network Issues

### Port Already in Use

**Problem:** `Port 8080 is already in use`
```
ERROR: [Errno 98] Address already in use
```

**Solutions:**
```bash
# 1. Find process using port
# Windows:
netstat -ano | findstr :8080
# Linux/macOS:
lsof -i :8080

# 2. Kill process
# Windows:
taskkill /PID <process_id> /F
# Linux/macOS:
kill -9 <process_id>

# 3. Use different port
python run_optimized_demo.py --port 8081
```

### Firewall Blocking Connections

**Problem:** `Connection timeout`

**Solutions:**
```bash
# Windows Firewall
# Add exception for Python.exe
# Or temporarily disable firewall for testing

# Linux iptables
sudo iptables -A INPUT -p tcp --dport 8080 -j ACCEPT

# macOS
# System Preferences > Security & Privacy > Firewall
# Allow incoming connections for Python
```

## 🧪 Testing & Debugging

### Tests Fail

**Problem:** `Integration tests fail`
```
❌ Some integration tests FAILED!
```

**Solutions:**
```bash
# 1. Run specific test
python integration_test.py --test ai_service

# 2. Check detailed output
python integration_test.py --verbose

# 3. Run with fallbacks
python integration_test.py --ignore-failures

# 4. Check test environment
python -c "
import sys
print('Python:', sys.version)
print('Path:', sys.path)
"
```

### Debug Mode

**Enable detailed logging:**
```bash
# Set debug mode
export DEBUG_MODE=true

# Run with verbose output
python run_optimized_demo.py --performance medium 2>&1 | tee debug.log

# Check specific component
python -c "
from ai_service.ai_service import get_ai_logger
logger = get_ai_logger()
logger.setLevel('DEBUG')
"
```

## 🚨 Emergency Recovery

### Complete System Reset

**When everything fails:**
```bash
# 1. Stop all processes
pkill -f python
pkill -f godot

# 2. Clean virtual environment
rm -rf .venv
python -m venv .venv
source .venv/bin/activate  # Linux/macOS
# .venv\Scripts\activate    # Windows

# 3. Fresh installation
pip install --upgrade pip
pip install -r requirements.txt

# 4. Reset configuration
rm -f ai_config.json
cp ai_config.json.example ai_config.json

# 5. Clear caches
rm -rf __pycache__
rm -rf .pytest_cache
rm -rf logs/*

# 6. Download models fresh
rm -rf models/gpt4all/*.gguf
python ai_service/ai_service.py

# 7. Test basic functionality
python simple_test.py
```

### Backup and Restore

**Create backup before major changes:**
```bash
# Backup current state
tar -czf backup_$(date +%Y%m%d).tar.gz \
  ai_config.json \
  models/ \
  logs/ \
  .venv/

# Restore from backup
tar -xzf backup_20240115.tar.gz
```

## 📞 Getting Help

### Diagnostic Information

**When reporting issues, include:**
```bash
# System information
python --version
pip --version
uname -a  # Linux/macOS
systeminfo  # Windows

# Package versions
pip list

# Configuration
cat ai_config.json

# Recent logs
tail -50 logs/ai_service.log

# Performance metrics
python -c "
from demo_performance_suite import DemoPerformanceSuite
suite = DemoPerformanceSuite()
print(suite.get_performance_report())
"
```

### Log Collection

**Gather logs for troubleshooting:**
```bash
# Collect all logs
mkdir troubleshooting_$(date +%Y%m%d)
cp logs/* troubleshooting_$(date +%Y%m%d)/
cp ai_config.json troubleshooting_$(date +%Y%m%d)/
pip list > troubleshooting_$(date +%Y%m%d)/packages.txt
python --version > troubleshooting_$(date +%Y%m%d)/python_version.txt
```

### Community Support

- **GitHub Issues:** Report bugs and feature requests
- **Discussions:** Community help and Q&A
- **Documentation:** Check latest docs for updates
- **Stack Overflow:** Tag questions with `generative-agents` and `godot`

## ✅ Troubleshooting Checklist

**Before reporting issues:**

- [ ] Checked system requirements (Python 3.8+, 4GB RAM)
- [ ] Ran `python simple_test.py` successfully
- [ ] Verified dependencies with `pip list`
- [ ] Checked available disk space (2GB+ required)
- [ ] Tested with minimal configuration
- [ ] Reviewed error logs in `logs/` directory
- [ ] Tried lower performance mode
- [ ] Restarted system/terminal
- [ ] Checked for conflicting processes on port 8080
- [ ] Verified model files exist and aren't corrupted

**Common quick fixes:**
- Restart terminal/system
- Use lower performance mode
- Clear caches and restart
- Check available memory
- Update Python packages

---

Most issues can be resolved with the solutions above. If problems persist, please create a GitHub issue with diagnostic information.