# Model Directory

This directory contains the Large Language Models (LLMs) required for the Generative Agents system.

## Required Models

### 1. Llama 3.2 3B Instruct (Quantized)
- **File**: `Llama-3.2-3B-Instruct.Q4_0.gguf`
- **Size**: ~2GB
- **Purpose**: Main dialogue generation for NPCs
- **Download Options**:
  - [Hugging Face - Official Meta Release](https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct-GGUF)
  - [TheBloke Quantized Versions](https://huggingface.co/TheBloke)

### 2. Phi-3 Mini 4K Instruct (Quantized)
- **File**: `Phi-3-mini-4k-instruct.Q4_0.gguf`
- **Size**: ~2.3GB
- **Purpose**: Fast decision-making engine
- **Download Options**:
  - [Microsoft Official](https://huggingface.co/microsoft/Phi-3-mini-4k-instruct-gguf)

## Directory Structure
```
models/
├── README.md (this file)
└── llms/
    ├── Llama-3.2-3B-Instruct.Q4_0.gguf
    └── Phi-3-mini-4k-instruct.Q4_0.gguf
```

## Installation Instructions

### Option 1: Manual Download
1. Create the `llms` subdirectory if it doesn't exist:
   ```bash
   mkdir -p models/llms
   ```

2. Download the models from the links above
3. Place the `.gguf` files in the `models/llms/` directory

### Option 2: Using wget (Linux/macOS)
```bash
cd models/llms

# Download Llama 3.2 (replace with actual URL)
wget https://huggingface.co/[path-to-model]/Llama-3.2-3B-Instruct.Q4_0.gguf

# Download Phi-3 (replace with actual URL)
wget https://huggingface.co/[path-to-model]/Phi-3-mini-4k-instruct.Q4_0.gguf
```

### Option 3: Using Hugging Face CLI
```bash
# Install huggingface-hub
pip install huggingface-hub

# Download models
huggingface-cli download meta-llama/Llama-3.2-3B-Instruct-GGUF --local-dir ./llms
huggingface-cli download microsoft/Phi-3-mini-4k-instruct-gguf --local-dir ./llms
```

## Model Information

### Quantization Levels
- **Q4_0**: 4-bit quantization, best balance of size and quality
- **Q5_K_M**: 5-bit quantization, slightly better quality, larger size
- **Q8_0**: 8-bit quantization, near-original quality, much larger

### Performance Expectations
| Model | RAM Usage | VRAM Usage | Tokens/sec | First Token |
|-------|-----------|------------|------------|-------------|
| Llama 3.2 3B Q4_0 | ~4GB | ~3.4GB | 15-25 | ~290ms |
| Phi-3 Mini Q4_0 | ~3GB | ~2.5GB | 30-40 | ~150ms |

## Alternative Models

If you want to experiment with different models:

### Smaller Models (Less RAM)
- `TinyLlama-1.1B-Chat.Q4_0.gguf` (~700MB)
- `Qwen2-0.5B-Instruct.Q4_0.gguf` (~350MB)

### Larger Models (Better Quality)
- `Llama-3.2-7B-Instruct.Q4_0.gguf` (~4GB)
- `Mistral-7B-Instruct.Q4_0.gguf` (~4GB)

### Updating Model Configuration
To use different models, update the server configuration:

```python
# In finalbuild/server/gpt4all_server.py
MODEL_NAME = "your-new-model.gguf"

# In finalbuild/server/decision_server.py
MODEL_NAME = "your-decision-model.gguf"
```

## Verification

After downloading, verify the models:
```python
from pathlib import Path

models_dir = Path("models/llms")
required_models = [
    "Llama-3.2-3B-Instruct.Q4_0.gguf",
    "Phi-3-mini-4k-instruct.Q4_0.gguf"
]

for model in required_models:
    model_path = models_dir / model
    if model_path.exists():
        size_gb = model_path.stat().st_size / (1024**3)
        print(f"✓ {model}: {size_gb:.2f}GB")
    else:
        print(f"✗ {model}: NOT FOUND")
```

## Troubleshooting

### "Model file not found"
- Check exact filename (case-sensitive)
- Ensure models are in `models/llms/` not just `models/`

### "Not enough VRAM"
- Use smaller quantization (Q3_K_S)
- Try smaller models (TinyLlama)
- Reduce context size in server config

### "Slow inference"
- Ensure GPU is being used: check `device="gpu"` in server
- Close other GPU applications
- Consider using Phi-3 for faster responses

## License Notice

These models have their own licenses:
- **Llama 3.2**: [Meta Llama License](https://ai.meta.com/llama/license/)
- **Phi-3**: [MIT License](https://huggingface.co/microsoft/Phi-3-mini-4k-instruct/blob/main/LICENSE)

Please review and comply with model licenses for your use case.