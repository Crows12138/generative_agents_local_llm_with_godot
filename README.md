# Generative Agents with Local LLM and Godot

Create intelligent game characters with local AI models and Godot engine. Features optimized performance, smart memory management, and seamless Godot integration for immersive AI-driven gameplay.

## ✨ Key Features

- 🤖 **Smart AI Agents** - Memory, emotions, decision-making
- 🎮 **Godot Integration** - Real-time game character AI
- 🏠 **Local Models** - Fully offline with GPT4All/Qwen
- ⚡ **Performance Optimized** - Caching, memory management
- 🛠️ **Easy Setup** - One-click demo launcher
- 📊 **Real-time Monitoring** - Performance metrics and optimization

## 🚀 Quick Start

### One-Click Demo Launch
```bash
# Windows
run_optimized_demo.bat

# Python
python run_optimized_demo.py --performance medium --start-ai
```

### Manual Setup
```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Download AI model (automatic on first run)
python ai_service/ai_service.py

# 3. Test the system
python integration_test.py
```

## 🏗️ Architecture

```
Project/
├── 🤖 ai_service/          # AI service core with local LLM
├── 👥 agents/              # Simple agent system
├── 🎮 godot/              # Godot game project
├── 🧠 models/             # AI models (auto-downloaded)
├── 🌐 api/                # API bridge layer
├── ⚡ performance_optimizer.py    # Performance suite
├── 🎬 demo_performance_suite.py  # Demo orchestration
└── 🚀 run_optimized_demo.py     # One-click launcher
```

### Core Components

1. **AI Service** - Local LLM processing with model management
2. **Simple Agents** - Lightweight characters with memory & emotions  
3. **Godot Integration** - Real-time game character AI
4. **Performance Suite** - Caching, memory optimization, monitoring
5. **Demo Launcher** - Automated setup and performance tuning

## 🎯 Performance Modes

Choose the optimal mode for your system:

| Mode | Memory | Features | Best For |
|------|--------|----------|----------|
| **Low** | 2GB | Basic AI | Older hardware |
| **Medium** | 4GB | Standard features | Most systems |
| **High** | 6GB | Advanced AI | Gaming PCs |
| **Ultra** | 8GB+ | All features | High-end systems |

```bash
# Set performance mode
python run_optimized_demo.py --performance high
```

## 📱 Usage Examples

### Python Agent API
```python
from agents.simple_agents import create_demo_characters
from performance_optimizer import get_performance_optimizer

# Create optimized agents
agents = create_demo_characters()
alice = agents[0]

# Cached AI interaction
optimizer = get_performance_optimizer()
response = optimizer.cached_ai_generate("Hello there!")
print(response)
```

### Godot Integration
```gdscript
# godot_performance_config.gd
extends Node

func _ready():
    setup_performance_settings()
    start_performance_monitoring()

func optimize_for_demo():
    set_quality_level(2)  # High quality
    max_visible_characters = 10
```

## 🧪 Testing & Monitoring

```bash
# Quick system test
python simple_test.py

# Full integration test
python integration_test.py

# Performance monitoring
python demo_performance_suite.py
```

## ⚙️ Configuration

Key settings in `ai_config.json`:
```json
{
  "model": {
    "active_model": "qwen",
    "max_tokens": 800
  },
  "performance": {
    "enable_cache": true,
    "memory_limit_gb": 4
  }
}
```

## 📋 System Requirements

**Minimum:**
- Python 3.8+
- 4GB RAM
- 2GB free disk space
- Windows/Linux/macOS

**Recommended:**
- Python 3.10+
- 8GB+ RAM  
- 5GB free disk space
- Dedicated GPU (optional)

## 🤝 Contributing

1. Fork the repository
2. Create feature branch: `git checkout -b feature/amazing-feature`
3. Run tests: `python integration_test.py`
4. Commit changes: `git commit -m 'Add amazing feature'`
5. Push to branch: `git push origin feature/amazing-feature`
6. Open a Pull Request

## 📚 Documentation

- [Installation Guide](docs/INSTALLATION.md)
- [Quick Start Guide](docs/QUICKSTART.md)
- [Architecture Overview](docs/ARCHITECTURE.md)
- [Troubleshooting](docs/TROUBLESHOOTING.md)

## 📄 License

Apache License 2.0 - see [LICENSE](LICENSE) file for details

## 🙏 Acknowledgments

- [GPT4All](https://github.com/nomic-ai/gpt4all) for local LLM support
- [Godot Engine](https://godotengine.org/) for game development framework
- [Qwen Models](https://huggingface.co/Qwen) for AI capabilities

---

*Made with ❤️ for AI-powered game development*
