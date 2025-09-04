# Quick Start Guide

## Start Server

```bash
cd finalbuild/server
../../.venv/Scripts/python.exe gpt4all_server.py
```

The server will start on port 9999, waiting for Godot connections.

## View Memories

View all NPC memories:
```bash
cd finalbuild
../.venv/Scripts/python.exe tools/view_memories.py
```

View Bob's memories:
```bash
cd finalbuild
../.venv/Scripts/python.exe tools/view_memories.py Bob
```

View last 5 memories with statistics:
```bash
cd finalbuild
../.venv/Scripts/python.exe tools/view_memories.py Bob --last 5 --stats
```

## Configuration

Edit `server/gpt4all_config.json` to adjust parameters:

- `model_file`: LLM model file name (default: Llama-3.2-3B-Instruct-Q4_0.gguf)
- `device`: "gpu" or "cpu" (default: gpu)
- `max_tokens`: Maximum response length (default: 150)
- `temperature`: Creativity level 0.0-1.0 (default: 0.7)
- `max_conversation_entries`: Maximum conversation history (default: 20)

## System Requirements

- Python 3.8+
- CUDA GPU (recommended 4GB+ VRAM)
- Virtual environment installed (.venv)
- GPT4All with CUDA support

## Directory Structure

```
finalbuild/
├── server/          # Server files
│   ├── gpt4all_server.py
│   └── gpt4all_config.json
├── client/          # Client files
│   └── llm_client_cognitive.py
├── tools/           # Utility scripts
│   └── view_memories.py
├── godot_project/   # Godot game files
│   └── scripts/
└── npc_memories/    # NPC memory storage (unified)
    ├── Bob.json
    ├── Alice.json
    └── Sam.json
```

## Features

- **WebSocket Streaming**: Real-time token streaming for better UX
- **Unified Memory System**: Single storage format, 50% less I/O
- **GPU Acceleration**: Fast response times (0.5-2 seconds)
- **Conversation Context**: NPCs remember previous interactions

Note: Memories are stored only under `finalbuild/npc_memories/`.
