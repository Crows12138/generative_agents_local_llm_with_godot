# GPT4All NPC Server

## Overview
This server provides AI-powered NPC dialogue using GPT4All with Llama 3.2 model.

## Requirements
- Python 3.8+
- GPT4All with CUDA support: `pip install gpt4all[cuda]`
- Llama 3.2 model file in `models/llms/`

## Server Features
- Continuous conversation with memory
- Character-specific personalities
- Response times: 3-7 seconds
- GPU acceleration support

## Starting the Server

### Windows
```bash
start_gpt4all_server.bat
```

### Manual Start
```bash
cd finalbuild/server
python gpt4all_server.py
```

## Configuration
Edit `finalbuild/server/gpt4all_config.json`:
- `model_file`: Model filename
- `device`: "gpu" or "cpu"
- `max_tokens`: Response length limit
- `temperature`: Creativity level (0.0-1.0)

## Memory Storage
- Conversations and memories: `finalbuild/npc_memories/`

## Godot Integration
The server listens on port 9999 and is compatible with the existing Godot client.

## Testing
```bash
python test_gpt4all.py
```
