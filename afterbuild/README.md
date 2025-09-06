# Afterbuild - Simplified Dialogue System

A clean, unified dialogue system for NPCs using GPT4All, designed for progressive development.

## Architecture

### Simplified Design
```
Godot Client ←→ WebSocket ←→ Unified Server ←→ GPT4All Model
```

**Key Improvement**: Combined `llm_client_cognitive.py` and `gpt4all_server.py` into a single unified server, eliminating the intermediate socket layer.

## Project Structure

```
afterbuild/
├── server/
│   ├── dialogue_server.py          # Single server handling everything
│   └── config.json                 # Configuration
├── godot_project/
│   └── scripts/
│       └── dialogue_client.gd      # Godot WebSocket client
├── npc_memories/                   # NPC conversation histories
├── docs/                           # Development documentation
└── README.md
```

## Core Features

### 1. Dialogue Server (`dialogue_server.py`)
- **Single Process**: No need for separate client/server processes
- **Direct WebSocket**: Godot connects directly, no intermediate layers
- **Memory Management**: Built-in conversation history
- **NPC Name Canonicalization**: Prevents memory mixing (Bob/bob/BOB → Bob)
- **Streaming Support**: Real-time token streaming for natural dialogue

### 2. Configuration (`config.json`)
```json
{
  "model_file": "Llama-3.2-3B-Instruct-Q4_0.gguf",
  "device": "gpu",
  "max_tokens": 150,
  "websocket_port": 9999
}
```

### 3. Godot Client (`dialogue_client.gd`)
- WebSocket connection to unified server
- Speech bubble UI system
- Memory viewer interface
- Support for Bob, Alice, and Sam NPCs

## Quick Start

### 1. Start the Server
```bash
cd afterbuild/server
python dialogue_server.py
```

The server will:
- Load the GPT4All model
- Start WebSocket server on port 9999
- Load existing NPC memories
- Wait for Godot connections

### 2. Connect from Godot
The Godot client automatically connects to `ws://127.0.0.1:9999`

### 3. Interact with NPCs
- Left-click NPC: Send default greeting
- Right-click NPC: Custom message dialog

## Memory System

Conversations are automatically saved to `npc_memories/`:
- `Bob.json` - Bob's conversation history
- `Alice.json` - Alice's conversation history
- `Sam.json` - Sam's conversation history

Each memory entry contains:
- Timestamp
- User input
- NPC response
- Response time
- Importance score (for future use)

## Development Roadmap

### Phase 1: Core Dialogue ✅
- Basic conversation system
- Memory persistence
- Multi-NPC support

### Phase 2: Enhanced Memory (Future)
- Importance-based memory retention
- Long-term vs short-term memory
- Memory summarization

### Phase 3: Context Awareness (Future)
- Environment state integration
- Time-of-day awareness
- Mood and relationship tracking

### Phase 4: Advanced Features (Future)
- Emotion detection
- Dynamic personality adjustment
- Inter-NPC conversations

## Key Advantages

1. **Simplicity**: Single Python process, no complex networking
2. **Performance**: Direct communication, lower latency
3. **Maintainability**: All logic in one place
4. **Debugging**: Single log output, easier to trace issues
5. **Deployment**: Just run one script

## Requirements

- Python 3.8+
- GPT4All with CUDA support (optional)
- Godot 4.0+
- WebSocket support in Godot

## Installation

```bash
# Install dependencies
pip install gpt4all websockets

# Download model
# Place Llama-3.2-3B-Instruct-Q4_0.gguf in models/llms/
```

## Notes

- The unified server approach reduces complexity significantly
- NPC name canonicalization prevents memory fragmentation
- WebSocket streaming provides smooth, real-time responses
- System is designed for easy extension and modification