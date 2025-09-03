"""
Utility configuration for Reverie backend.

Defines paths and runtime flags consumed by reverie/backend_server/reverie.py
via `from utils import *`.
"""

from pathlib import Path

# Resolve project root based on this file location to avoid CWD issues
_PROJECT_ROOT = Path(__file__).resolve().parents[2]

# AI / API keys (leave empty for local-only)
openai_api_key = ""
key_owner = "LocalUser"

# Local LLM configuration
use_local_llm = True

# Model paths
_MODELS_DIR = _PROJECT_ROOT / "models"
models_dir = str(_MODELS_DIR)
embedding_model_path = str(_MODELS_DIR / "all-MiniLM-L6-v2")
llm_model_path = str(_MODELS_DIR / "gpt4all")

# Environment assets
_ASSETS_DIR = _PROJECT_ROOT / "environment" / "frontend_server" / "static_dirs" / "assets"
maze_assets_loc = str(_ASSETS_DIR)
env_matrix = f"{maze_assets_loc}/the_ville/matrix"
env_visuals = f"{maze_assets_loc}/the_ville/visuals"

# File-system storage
fs_storage = str(_PROJECT_ROOT / "environment" / "frontend_server" / "storage")
fs_temp_storage = str(_PROJECT_ROOT / "environment" / "frontend_server" / "temp_storage")

# Misc
collision_block_id = "32125"
debug = True


