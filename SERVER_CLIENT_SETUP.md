# LLM Server-Client Setup for Godot

## Quick Start

### 1. Start LLM Server
```bash
# Windows
server_client\start_llm_server.bat

# Or manually
.venv\Scripts\python.exe server_client\llm_server_nocache.py
```

### 2. Run Godot Scene
- Open `godot/live-with-ai/` project
- Open `scenes/cozy_bar.tscn`
- Script already attached: `bar_server_client_fixed.gd`
- Press F6 to run

### 3. Interact
- Click on Bob (blue), Alice (red), or Sam (green)
- Wait 2-3 seconds for AI response
- Each response is freshly generated (no cache)

## Files

### Core Server Files (in `server_client/` folder)
- `server_client/llm_server_nocache.py` - Main LLM server (keeps model loaded)
- `server_client/llm_client.py` - Python client for testing
- `server_client/start_llm_server.bat` - One-click server starter

### Godot Integration
- `godot/live-with-ai/scripts/bar_server_client_fixed.gd` - Working integration script
- `godot/live-with-ai/scenes/cozy_bar.tscn` - Main scene with NPCs

## Performance
- Server startup: 15-20 seconds (model loading)
- Response time: 2-3 seconds per interaction
- Memory: ~4-6GB with model loaded

## Troubleshooting

### No response when clicking
1. Check server is running (see console output)
2. Ensure you started `server_client\start_llm_server.bat` first

### Server won't start
- Check port 9999 is free
- Verify model exists in `models/llms/`

### Test server manually
```bash
.venv\Scripts\python.exe server_client\llm_client.py "Hello test"
```

## Architecture
```
Godot (Click NPC)
  ↓
bar_server_client_fixed.gd
  ↓
OS.execute → llm_client.py
  ↓
Socket → llm_server_nocache.py (port 9999)
  ↓
Local LLM Model (always loaded)
  ↓
Fresh response (2-3 seconds)
```