# Setup Guide for Generative Agents with Local LLM and Godot

## System Requirements

### Minimum Requirements
- **OS**: Windows 10/11, Ubuntu 20.04+, or macOS 12+
- **CPU**: 8-core processor (Intel i7-8700 or AMD Ryzen 7 2700 or better)
- **RAM**: 16GB minimum, 32GB recommended
- **GPU**: NVIDIA GPU with 4GB+ VRAM (GTX 1060 or better)
  - CUDA 11.8+ support required
  - Tested on: RTX 4080, RTX 3070, RTX 2060
- **Storage**: 20GB free space
- **Python**: 3.8 - 3.11
- **Godot**: 4.0 or higher

### Recommended Specifications
- **CPU**: 12+ cores
- **RAM**: 32GB
- **GPU**: RTX 3070 or better with 8GB+ VRAM
- **Storage**: SSD with 50GB free space

## Installation Steps

### 1. Clone the Repository
```bash
git clone https://github.com/yourusername/generative_agents_local_llm_with_godot.git
cd generative_agents_local_llm_with_godot
```

### 2. Set Up Python Environment

#### Windows:
```bash
# Create virtual environment
python -m venv .venv

# Activate environment
.venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

#### Linux/macOS:
```bash
# Create virtual environment
python3 -m venv .venv

# Activate environment
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Download Required Models

You need to download the following models and place them in the `models/llms/` directory:

1. **Llama 3.2 3B Instruct (Q4_0)**
   - Download: [Llama-3.2-3B-Instruct.Q4_0.gguf](https://huggingface.co/Meta-Llama/Llama-3.2-3B-Instruct)
   - Size: ~2GB
   - Purpose: Main dialogue generation

2. **Phi-3 Mini 4K Instruct (Q4_0)**
   - Download: [Phi-3-mini-4k-instruct.Q4_0.gguf](https://huggingface.co/microsoft/Phi-3-mini-4k-instruct)
   - Size: ~2.3GB
   - Purpose: Decision making engine

#### Model Installation:
```bash
# Create models directory
mkdir -p models/llms

# Download models (example with wget)
cd models/llms
wget [model_url]

# Or use the provided script (if available)
python scripts/download_models.py
```

### 4. CUDA Setup (for NVIDIA GPUs)

#### Windows:
1. Install [CUDA Toolkit 11.8](https://developer.nvidia.com/cuda-11-8-0-download-archive)
2. Install [cuDNN 8.9](https://developer.nvidia.com/cudnn)
3. Verify installation:
```bash
nvidia-smi
nvcc --version
```

#### Linux:
```bash
# Install CUDA
sudo apt-get update
sudo apt-get install -y cuda-toolkit-11-8

# Verify
nvidia-smi
```

### 5. Godot Engine Setup

1. Download [Godot 4.x](https://godotengine.org/download)
2. Extract to your preferred location
3. Open the project:
```bash
# Navigate to project
cd finalbuild/godot_project

# Open with Godot
godot --path . --editor
```

### 6. Configuration

Create a `.env` file in the project root:
```env
# Server Configuration
DIALOGUE_PORT=9999
DECISION_PORT=9998
HTTP_PORT=8020

# Model Settings
MODEL_PATH=./models/llms
DIALOGUE_MODEL=Llama-3.2-3B-Instruct.Q4_0.gguf
DECISION_MODEL=Phi-3-mini-4k-instruct.Q4_0.gguf

# GPU Settings
DEVICE=gpu
GPU_LAYERS=35

# Memory Settings
MAX_CONTEXT_LENGTH=2048
CACHE_SIZE=100
```

## Running the System

### 1. Start the NPC Dialogue Server
```bash
cd finalbuild/server
python gpt4all_server.py
```
You should see:
```
Loading model: Llama-3.2-3B-Instruct-Q4_0.gguf
Model loaded successfully on gpu
Starting WebSocket server on ws://127.0.0.1:9999
```

### 2. Start the Decision Server (Optional)
```bash
# In a new terminal
cd finalbuild/server
python decision_server.py
```

### 3. Launch Godot Game
1. Open Godot Engine
2. Load the project from `finalbuild/godot_project`
3. Press F6 or click "Play Scene"

## Verification

### Test WebSocket Connection
```python
# test_connection.py
import asyncio
import websockets
import json

async def test():
    async with websockets.connect("ws://localhost:9999") as ws:
        await ws.send(json.dumps({
            "npc": "Bob",
            "message": "Bob|Hello!"
        }))
        response = await ws.recv()
        print(f"Response: {response}")

asyncio.run(test())
```

### Run Performance Tests
```bash
cd finalbuild/performance_tests
python "Token Generation Performance/gpu_monitor.py" 30
```

## Troubleshooting

### Common Issues

#### 1. "Model file not found"
- Ensure models are in `models/llms/` directory
- Check file names match exactly (case-sensitive)

#### 2. "CUDA out of memory"
- Reduce `GPU_LAYERS` in `.env`
- Close other GPU-intensive applications
- Try smaller model variants

#### 3. "WebSocket connection failed"
- Check if servers are running
- Verify firewall settings
- Ensure ports 9999 and 9998 are free

#### 4. "ImportError: No module named 'gpt4all'"
- Activate virtual environment: `.venv\Scripts\activate`
- Reinstall: `pip install gpt4all --upgrade`

#### 5. Low FPS in Godot
- Enable GPU acceleration in Project Settings
- Reduce scene complexity
- Check VSync settings

### Performance Optimization

#### For faster response times:
```python
# In gpt4all_server.py, adjust:
n_ctx=1024  # Reduce context size
n_batch=8   # Reduce batch size
```

#### For lower memory usage:
```python
# Use smaller models or more aggressive quantization
MODEL = "Llama-3.2-1B-Instruct.Q4_0.gguf"  # Smaller variant
```

## Testing

### Run all tests:
```bash
# Unit tests
pytest tests/

# Performance benchmarks
python finalbuild/performance_tests/run_all_tests.py

# NPC consistency evaluation
cd "finalbuild/performance_tests/Automated Character Consistency Evaluation"
python collect_npc_responses.py
```

## Development

### Project Structure
```
finalbuild/
├── server/                 # Python servers
│   ├── gpt4all_server.py  # Main NPC server
│   └── decision_server.py # Decision engine
├── godot_project/         # Godot game files
├── npc_memories/          # Persistent storage
└── performance_tests/     # Test suites
```

### Adding New NPCs
1. Define personality in `server/gpt4all_server.py`
2. Create memory file in `npc_memories/`
3. Add character scene in Godot
4. Register in WebSocket handler

## Support

For issues or questions:
- Open an issue on [GitHub](https://github.com/yourusername/repo/issues)
- Check existing [documentation](./docs/)
- Review [FAQ](./FAQ.md)

## Next Steps

1. ✅ Verify installation with test scripts
2. ✅ Run performance benchmarks
3. ✅ Launch demo scene in Godot
4. 📖 Read [API documentation](./docs/API.md)
5. 🎮 Start building your own scenes!