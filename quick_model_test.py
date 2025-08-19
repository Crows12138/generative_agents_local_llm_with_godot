#!/usr/bin/env python
# -*- coding: utf-8 -*-
import sys
import time
import json
from pathlib import Path

sys.path.append(str(Path(__file__).parent / "ai_service"))

def test_model_performance():
    """Simple model performance test"""
    try:
        from ai_service.config import get_config
        config = get_config()
        models = config.config.model.supported_models
        active = config.config.model.active_model
        
        print("="*50)
        print("Model Configuration Status")
        print("="*50)
        print(f"Active Model: {active}")
        print(f"Available Models: {len(models)}")
        
        for name, file in models.items():
            status = "[ACTIVE]" if name == active else ""
            print(f"  - {name}: {file} {status}")
        
        # Check model files
        models_dir = Path(config.config.model.models_dir)
        print(f"\nModel Directory: {models_dir}")
        
        for name, file in models.items():
            model_path = models_dir / file
            if model_path.exists():
                size_mb = model_path.stat().st_size / (1024 * 1024)
                print(f"  {name}: {size_mb:.1f}MB [OK]")
            else:
                print(f"  {name}: [MISSING]")
        
        print("="*50)
        return True
        
    except Exception as e:
        print(f"Error: {e}")
        return False

def switch_model(model_name):
    """Switch active model"""
    try:
        from ai_service.config import get_config
        config = get_config()
        
        if model_name in config.config.model.supported_models:
            config.config.model.active_model = model_name
            config.save_config()
            print(f"Success: Switched to {model_name}")
            return True
        else:
            print(f"Error: Model {model_name} not found")
            return False
    except Exception as e:
        print(f"Error: {e}")
        return False

if __name__ == "__main__":
    if len(sys.argv) > 1:
        # Switch model
        switch_model(sys.argv[1])
    else:
        # Show status
        test_model_performance()