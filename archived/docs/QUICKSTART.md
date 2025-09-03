# Quick Start Guide

Get up and running with AI-powered game characters in minutes!

## 🚀 30-Second Quick Start

### One-Click Launch

**Windows:**
```bash
# Double-click to run
run_optimized_demo.bat
```

**Python (Cross-platform):**
```bash
python run_optimized_demo.py --performance medium --start-ai
```

That's it! The system will automatically:
- ✅ Check your system
- ✅ Install dependencies  
- ✅ Download AI models
- ✅ Start optimized demo
- ✅ Launch performance monitoring

## 🎯 Performance Modes

Choose the right mode for your system:

### 🔋 Low Performance Mode
**Best for:** Older hardware, laptops
```bash
python run_optimized_demo.py --performance low
```
- 2GB RAM usage
- Basic AI features
- 20+ FPS
- 2-3 agents

### ⚡ Medium Performance Mode (Recommended)
**Best for:** Most desktop systems
```bash
python run_optimized_demo.py --performance medium
```
- 4GB RAM usage
- Standard AI features
- 30+ FPS
- 3-5 agents

### 🚄 High Performance Mode
**Best for:** Gaming PCs, workstations
```bash
python run_optimized_demo.py --performance high
```
- 6GB RAM usage
- Advanced AI features
- 45+ FPS
- 5+ agents

### 🚀 Ultra Performance Mode
**Best for:** High-end systems
```bash
python run_optimized_demo.py --performance ultra
```
- 8GB+ RAM usage
- All features enabled
- 60+ FPS
- Full agent ecosystem

## 🎮 Demo Scenarios

### 1. Basic Agent Interaction
```bash
# Test AI character dialogue
python run_optimized_demo.py --test-mode --duration 2
```
**What you'll see:**
- AI characters responding to messages
- Memory system working
- Emotions changing
- Performance metrics

### 2. Multi-Agent Social Simulation
```bash
# Watch agents interact with each other
python run_optimized_demo.py --performance medium --start-ai
```
**Features demonstrated:**
- Agent-to-agent conversations
- Relationship building
- Group dynamics
- Contextual responses

### 3. Godot Game Integration
```bash
# Launch with Godot game
python run_optimized_demo.py --performance high --start-godot
```
**Experience:**
- Real-time character AI
- Visual character movement
- Interactive dialogue system
- Performance optimization

## 📊 Monitoring Your Demo

### Real-Time Performance Dashboard

During demo execution, you'll see:

```
📊 REAL-TIME DEMO STATUS
====================================
🕐 Runtime: 0:02:45
🤖 AI Response: 1.2s (target: <2.0s)
💾 Memory: 2.1GB (target: <4.0GB)
🎮 FPS: 42 (target: >30)
👥 Active Agents: 3
📦 Cache Hit Rate: 67%
⚙️ Optimization Level: MEDIUM
✅ All targets met!
====================================
```

### Performance Metrics

**Key indicators to watch:**
- **AI Response Time:** Should be under 2 seconds
- **Memory Usage:** Should stay within configured limits
- **FPS:** Should maintain 30+ for smooth experience
- **Cache Hit Rate:** Higher is better (30%+ good)

## 🎪 Interactive Demo Features

### Chat with AI Characters

While demo is running, you can interact directly:

```python
# In another terminal
python -c "
from agents.simple_agents import create_demo_characters
agents = create_demo_characters()
alice = agents[0]
response = alice.respond_to('User', 'Hello Alice!')
print(f'Alice: {response}')
"
```

### Test Agent Decision Making

```python
# Test agent choices
python -c "
from agents.simple_agents import create_demo_characters
agents = create_demo_characters()
bob = agents[1]
action, reason = bob.decide_action(['work', 'rest', 'socialize'])
print(f'Bob decides to: {action} because {reason}')
"
```

### Monitor Performance Live

```python
# Get real-time performance stats
python -c "
from performance_optimizer import get_performance_optimizer
optimizer = get_performance_optimizer()
report = optimizer.get_performance_report()
print('Performance Status:', report['status'])
"
```

## 🛠️ Customization Options

### Adjust Agent Behavior

**Modify agent personalities:**
```python
# Edit agent characteristics
from agents.simple_agents import SimpleAgent

custom_agent = SimpleAgent(
    name="CustomBot",
    personality="Curious and helpful assistant",
    background="Loves to explore new topics"
)
```

### Configure Performance Settings

**Create custom config:**
```json
{
  "demo_config": {
    "max_ai_response_time": 1.5,
    "max_memory_usage_gb": 3.0,
    "min_fps": 45.0,
    "max_agents": 4
  }
}
```

### Enable Advanced Features

**Unlock more capabilities:**
```bash
# Enable all optimizations
python run_optimized_demo.py \
  --performance ultra \
  --start-ai \
  --start-godot \
  --duration 10
```

## 🧪 Testing & Validation

### Quick System Check

**Verify everything works:**
```bash
# 30-second system test
python simple_test.py
```

**Expected output:**
```
=== Simple Integration Test ===
✓ Agents imported
✓ Created 5 agents
✓ Memory working
✓ Decision making working
✓ Serialization working
=== ALL BASIC TESTS PASSED ===
```

### Full Integration Test

**Comprehensive testing:**
```bash
# 2-minute full test
python integration_test.py
```

**What it tests:**
- AI service startup
- Agent creation and behavior
- Memory systems
- Performance optimization
- Error recovery
- Godot integration

### Performance Benchmark

**Measure your system:**
```bash
# 5-minute performance test
python demo_performance_suite.py --test-mode --duration 5
```

**Results you'll get:**
- Average response times
- Memory efficiency metrics
- FPS stability analysis
- Optimization effectiveness

## 🎯 Next Steps

### After Quick Start Success

1. **Explore the Codebase:**
   - Look at `agents/simple_agents.py` for AI character logic
   - Check `performance_optimizer.py` for optimization techniques
   - Review `godot/live-with-ai/` for game integration

2. **Customize Your Experience:**
   - Modify agent personalities
   - Adjust performance targets
   - Create new demo scenarios

3. **Integrate into Your Project:**
   - Copy agent system to your game
   - Adapt performance optimizer
   - Implement Godot scripts

### Learn More

- **[Architecture Overview](ARCHITECTURE.md)** - Understand system design
- **[Installation Guide](INSTALLATION.md)** - Detailed setup instructions
- **[Troubleshooting](TROUBLESHOOTING.md)** - Fix common issues

## 🚨 Troubleshooting Quick Start

### Demo Won't Start

**Check Python:**
```bash
python --version  # Should be 3.8+
```

**Check dependencies:**
```bash
pip install -r requirements.txt
```

**Check system resources:**
```bash
# Ensure 4GB+ RAM available
python -c "import psutil; print(f'Available RAM: {psutil.virtual_memory().available/1024**3:.1f}GB')"
```

### Slow Performance

**Reduce performance mode:**
```bash
python run_optimized_demo.py --performance low
```

**Enable optimization:**
```bash
python run_optimized_demo.py --performance medium
```

**Check system load:**
- Close other applications
- Ensure sufficient RAM
- Check CPU usage

### AI Not Responding

**Check AI service:**
```bash
python -c "from ai_service.ai_service import get_health_status; print(get_health_status())"
```

**Restart with fallback:**
```bash
python run_optimized_demo.py --ignore-ai-failure
```

**Use test mode:**
```bash
python run_optimized_demo.py --test-mode --duration 1
```

## ⭐ Success Indicators

**Your demo is working correctly when you see:**

✅ **System Check Passed:** No requirement warnings  
✅ **AI Service Started:** Health check returns OK  
✅ **Agents Created:** 3-5 demo characters active  
✅ **Performance Stable:** Metrics within targets  
✅ **Interactions Working:** Agents respond to inputs  
✅ **Optimization Active:** Memory and cache working  

**Congratulations! You're ready to build AI-powered games! 🎉**

---

**Need help?** Check [Troubleshooting Guide](TROUBLESHOOTING.md) or create an issue on GitHub.