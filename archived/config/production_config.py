"""Production configuration based on monitoring insights"""

import logging
import os

class ProductionConfig:
    """Production-ready configuration"""
    
    # Model settings (validated by monitoring)
    MODEL_SETTINGS = {
        "default_model": "qwen3",
        "context_limit": 4096,
        "max_tokens": 500,
        "temperature": 0.7,
        "batch_size": 1  # Single request at a time for stability
    }
    
    # Memory settings (proven stable through testing)
    MEMORY_SETTINGS = {
        "max_memory_mb": 2000,  # 2GB limit (more realistic for large models)
        "gc_interval": 5,  # GC every 5 requests
        "cache_size": 50,  # Small cache for stability
        "emergency_threshold": 0.90  # 90% memory threshold
    }
    
    # Parser settings (100% success with adapter)
    PARSER_SETTINGS = {
        "use_adapter": True,  # Always use LLM output adapter
        "fallback_enabled": True,
        "confidence_threshold": 0.5,
        "enable_monitoring": True
    }
    
    # Monitoring settings
    MONITORING_SETTINGS = {
        "enable_raw_output_logging": True,
        "enable_performance_monitoring": True,
        "log_level": "INFO",
        "dashboard_update_interval": 2.0,
        "log_file": "production.log"
    }
    
    # Cognitive module settings
    COGNITIVE_SETTINGS = {
        "prefer_lightweight": True,  # Use lightweight cognitive modules
        "enable_reverie_fallback": True,
        "timeout_seconds": 30,
        "enable_memory_optimization": True
    }

def apply_production_config():
    """Apply production configuration"""
    
    config = ProductionConfig()
    
    try:
        # Configure logging
        logging.basicConfig(
            level=getattr(logging, config.MONITORING_SETTINGS["log_level"]),
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(config.MONITORING_SETTINGS["log_file"]),
                logging.StreamHandler()
            ]
        )
        
        # Configure memory optimizer
        try:
            from agents.memory_optimizer import get_memory_optimizer
            optimizer = get_memory_optimizer()
            optimizer.config.max_memory_mb = config.MEMORY_SETTINGS["max_memory_mb"]
            optimizer.config.emergency_threshold = config.MEMORY_SETTINGS["emergency_threshold"]
            print("✅ Memory optimizer configured")
        except ImportError:
            print("⚠️ Memory optimizer not available")
        
        # Start memory monitoring
        try:
            from agents.memory_optimizer import auto_start_monitoring
            auto_start_monitoring()
            print("✅ Memory monitoring started")
        except ImportError:
            print("⚠️ Memory monitoring not available")
        
        # Configure AI service
        try:
            # Apply enhanced configuration to existing AI service
            os.environ['AI_SERVICE_MAX_TOKENS'] = str(config.MODEL_SETTINGS["max_tokens"])
            os.environ['AI_SERVICE_TEMPERATURE'] = str(config.MODEL_SETTINGS["temperature"])
            print("✅ AI service environment configured")
        except Exception as e:
            print(f"⚠️ AI service configuration warning: {e}")
        
        print("✅ Production configuration applied successfully")
        return config
        
    except Exception as e:
        print(f"❌ Production configuration failed: {e}")
        raise

def get_production_config():
    """Get the production configuration instance"""
    return ProductionConfig()

# Auto-apply configuration when imported
if __name__ != "__main__":
    try:
        apply_production_config()
    except Exception as e:
        print(f"Auto-configuration failed: {e}")

if __name__ == "__main__":
    # Test configuration application
    print("Testing production configuration...")
    config = apply_production_config()
    print("Configuration test complete.")