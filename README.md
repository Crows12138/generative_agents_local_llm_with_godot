# Generative Agents with Local LLM and Godot

A real-time interactive simulation of autonomous NPCs (Non-Player Characters) powered by local Large Language Models, integrated with the Godot game engine.

## Overview

This project implements a generative agent system where NPCs exhibit autonomous behavior, personality consistency, and contextual memory. The system runs entirely locally without requiring cloud services, making it suitable for game development and research applications.

## Key Features

### 🤖 Autonomous NPC System
- **Personality-driven NPCs**: Each character maintains consistent personality traits and behaviors
- **Dynamic decision making**: NPCs make contextual decisions based on environment state
- **Memory system**: Short-term and long-term memory with importance-based retention
- **Natural dialogue generation**: Streaming responses with <300ms first-token latency

### 🎮 Game Integration
- **Godot 4.x integration**: Seamless WebSocket communication with game engine
- **Real-time interaction**: Players can interact with NPCs through natural language
- **State synchronization**: Bi-directional state updates between game and AI system

### ⚡ Performance Optimizations
- **Local LLM deployment**: Uses quantized models (Llama 3.2 3B Q4_0) for efficient inference
- **Streaming responses**: WebSocket-based streaming for immediate feedback
- **Intelligent caching**: Reduces repetitive computations by 40%
- **GPU acceleration**: Optimized for consumer GPUs (tested on RTX 4080)

## Architecture

```
finalbuild/
├── server/                  # Core server implementations
│   ├── gpt4all_server.py   # Main NPC dialogue server
│   ├── decision_server.py  # Decision-making engine
│   └── bar_state.py        # Environment state management
├── godot_project/          # Godot game files
│   ├── scenes/             # Game scenes
│   └── scripts/            # GDScript implementations
├── npc_memories/           # Persistent NPC memory storage
└── performance_tests/      # Comprehensive test suite
```

## NPCs

The system includes three distinct NPCs:

- **Bob**: Professional bartender - warm, experienced, customer-focused
- **Alice**: Contemplative regular - philosophical, observant, thoughtful
- **Sam**: Energetic musician - creative, passionate, social

## Technical Specifications

### Models
- **Dialogue**: Llama 3.2 3B Instruct (Q4_0 quantization)
- **Decision**: Phi-3 Mini 4K Instruct (Q4_0 quantization)

### Performance Metrics
- **First token latency**: ~290ms
- **GPU utilization**: 30-35% average (RTX 4080)
- **Memory usage**: ~3.4GB VRAM
- **Concurrent NPCs**: Supports 3+ simultaneous agents

### Communication Protocol
- **WebSocket**: Port 9999 for real-time streaming
- **Message format**: JSON with token streaming support
- **State compression**: Bit flags for efficient state transfer

## Installation

### Prerequisites
- Python 3.8+
- Godot 4.x
- NVIDIA GPU with 4GB+ VRAM (recommended)
- 16GB+ system RAM

### Setup

1. Clone the repository:
```bash
git clone https://github.com/yourusername/generative_agents_local_llm_with_godot.git
cd generative_agents_local_llm_with_godot
```

2. Install Python dependencies:
```bash
pip install -r requirements.txt
```

3. Download required models:
```bash
# Models should be placed in models/llms/
# - Llama-3.2-3B-Instruct.Q4_0.gguf
# - Phi-3-mini-4k-instruct.Q4_0.gguf
```

4. Start the NPC server:
```bash
cd finalbuild/server
python gpt4all_server.py
```

5. Launch Godot project:
```bash
# Open godot_project/ in Godot Engine
# Run the main scene
```

## Usage

### Basic Interaction
```python
import asyncio
import websockets
import json

async def chat_with_npc():
    async with websockets.connect("ws://localhost:9999") as ws:
        # Send message to Bob
        await ws.send(json.dumps({
            "npc": "Bob",
            "message": "Bob|Hello, how are you today?"
        }))
        
        # Receive streaming response
        while True:
            response = await ws.recv()
            data = json.loads(response)
            if data["type"] == "token":
                print(data["content"], end="")
            elif data["type"] == "complete":
                break
```

### Running Tests

Performance tests:
```bash
cd finalbuild/performance_tests
python "Token Generation Performance/gpu_benchmark_suite.py"
```

Character consistency evaluation:
```bash
cd "Automated Character Consistency Evaluation"
python collect_npc_responses.py
```

## Research Applications

This system has been evaluated for:
- **Character consistency**: Automated evaluation using LLM-as-judge methodology
- **Response latency**: Comprehensive benchmarking of streaming performance
- **GPU utilization**: Detailed profiling of resource usage patterns
- **Memory persistence**: Analysis of context retention strategies

## Limitations

- NPCs may occasionally break character under edge cases
- Memory system limited to 15-20 recent interactions
- Requires dedicated GPU for optimal performance
- Currently supports English language only

## Future Improvements

- [ ] Fine-tuning models for better character adherence
- [ ] Implementing more sophisticated memory retrieval
- [ ] Adding emotional state tracking
- [ ] Expanding to support more NPCs simultaneously
- [ ] Multi-language support

## Contributing

Contributions are welcome! Please feel free to submit issues or pull requests.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- GPT4All team for the local LLM framework
- Meta for Llama models
- Microsoft for Phi-3 models
- Godot Engine community

## Citation

If you use this work in your research, please cite:
```bibtex
@software{generative_agents_godot,
  title = {Generative Agents with Local LLM and Godot},
  year = {2024},
  author = {Your Name},
  url = {https://github.com/yourusername/generative_agents_local_llm_with_godot}
}
```

## Contact

For questions or collaborations, please open an issue on GitHub.