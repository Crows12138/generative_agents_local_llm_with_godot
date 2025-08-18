# Generative Agents with Local LLM and Godot

AI-powered game characters using local language models and Godot engine.

## Quick Start

### 1. Install Dependencies
```bash
pip install fastapi uvicorn gpt4all pyyaml watchdog requests
```

### 2. Download Models
Place GGUF models in `models/gpt4all/`:
- [Qwen2.5-Coder-7B GGUF](https://huggingface.co/Qwen/Qwen2.5-Coder-7B-Instruct-GGUF)

### 3. Run API Server
```bash
python -m api.godot_bridge
```

### 4. Run Demo
```bash
cd cozy_bar_demo
python main.py
```

## Architecture

```
Project/
├── ai_service/         # AI service core
├── config/            # Configuration files  
├── godot/            # Godot game project
├── models/           # AI models
├── api/                  # API server layer
│   └── godot_bridge.py   # Godot-AI bridge server
└── cozy_bar_demo/        # Demo applications
```

## Key Features

- **Local AI**: Fully offline with GPT4All
- **Godot Integration**: Real-time game AI
- **Simple Agents**: Memory, emotions, decisions
- **FastAPI Bridge**: High-performance API
- **Auto-config**: Automatic model detection

## API Usage

```python
# Chat endpoint
POST /ai/chat
{
  "character_name": "Alice",
  "message": "Hello!",
  "context": {}
}

# Decision endpoint  
POST /ai/decide
{
  "character_name": "Bob",
  "situation": "Where to go?",
  "options": ["shop", "home"]
}
```

## Godot Integration

```gdscript
# In Godot script
@onready var ai_manager = preload("res://scripts/ai_manager.gd").new()

func request_ai_response():
    ai_manager.chat("NPC", "Hello player!", {})
    
func _on_ai_response(data):
    print(data.response)
```

## Testing

```bash
# Run all tests
python integration_test.py

# Test specific component
python integration_test.py --test ai
```

## Configuration

Edit `config/ai_service.yaml`:

```yaml
model:
  active_model: "auto"
  max_tokens: 800
  
service:
  port: 8080
```

## License

Apache License 2.0
