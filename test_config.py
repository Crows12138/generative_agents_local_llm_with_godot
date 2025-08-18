#!/usr/bin/env python3
"""
测试配置管理器脚本
"""

if __name__ == "__main__":
    print("=== Testing AI Service Configuration ===")
    
    try:
        from ai_service.config import get_config
        
        # 获取配置管理器
        config = get_config()
        
        # 打印配置
        config.print_config()
        
        # 验证配置
        print("\n=== Validation ===")
        is_valid = config.validate_config()
        print(f"Configuration is {'VALID' if is_valid else 'INVALID'}")
        
        # 检查活动模型
        print(f"\nActive model: {config.config.model.active_model}")
        print(f"Supported models: {list(config.config.model.supported_models.keys())}")
        
        # 检查模型文件存在性
        import os
        models_dir = config.config.model.models_dir
        active_model = config.config.model.active_model
        model_file = config.config.model.supported_models.get(active_model)
        
        if model_file:
            model_path = os.path.join(models_dir, model_file)
            exists = os.path.exists(model_path)
            print(f"Model file exists: {exists}")
            if exists:
                size_mb = os.path.getsize(model_path) / (1024 * 1024)
                print(f"Model size: {size_mb:.1f} MB")
            else:
                print(f"Model path: {model_path}")
        
        print("\n=== Test completed successfully ===")
        
    except Exception as e:
        print(f"Error during configuration test: {e}")
        import traceback
        traceback.print_exc()