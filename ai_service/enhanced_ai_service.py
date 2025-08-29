#!/usr/bin/env python
"""
Enhanced AI Service with Dual Model Support
Integrates dual model scheduler for optimized performance
"""

import logging
import time
import asyncio
from typing import Dict, Any, Optional, List
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

# Import dual model scheduler
from .dual_model_scheduler import (
    DualModelScheduler, 
    TaskType, 
    get_scheduler
)

# Import CPU optimizer
from .cpu_optimizer import get_cpu_optimizer

# Import GPU detector
from .gpu_detector import get_gpu_detector

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import Ollama service for model inference
from .ollama_ai_service import (
    OllamaService,
    OllamaModelType,
    get_ollama_service
)


class EnhancedAIService:
    """Enhanced AI Service with dual model support"""
    
    def __init__(self):
        # Initialize scheduler
        self.scheduler = get_scheduler()
        
        # Initialize CPU optimizer
        self.cpu_optimizer = get_cpu_optimizer()
        
        # Initialize GPU detector
        self.gpu_detector = get_gpu_detector()
        
        # Initialize Ollama service
        self.ollama_service = get_ollama_service()
        
        # Model mapping
        self.model_mapping = {
            "fast": OllamaModelType.FAST,  # 4B on CPU
            "deep": OllamaModelType.DEEP   # 30B with GPU
        }
        
        # Thread pool for async operations
        self.executor = ThreadPoolExecutor(max_workers=2)
        
        # Cache for recent responses
        self.response_cache = {}
        self.cache_max_size = 100
        
        # Configure Ollama for CPU/GPU separation
        self._configure_ollama_models()
        
        logger.info("Enhanced AI Service initialized with Ollama backend and CPU/GPU separation")
    
    def _configure_ollama_models(self):
        """Configure Ollama models for CPU/GPU separation"""
        try:
            # Get GPU configuration
            gpu_info = self.gpu_detector.get_gpu_info()
            gpu_config = self.gpu_detector.get_recommended_config()
            
            logger.info(f"GPU detected: {gpu_info.gpu_name if gpu_info.available else 'None'}")
            if gpu_info.available:
                logger.info(f"GPU memory: {gpu_info.memory_total_mb}MB")
            
            # Configure models in Ollama
            # Note: Ollama automatically manages CPU/GPU allocation
            # The 4B model will primarily use CPU due to its small size
            # The 30B model will automatically use GPU when available
            
            logger.info("Configuring Ollama models:")
            logger.info("  - qwen3:4b (Fast): Optimized for CPU execution")
            logger.info("  - qwen3:30b (Deep): GPU acceleration enabled")
            
            # Set environment variables for Ollama GPU configuration
            import os
            if gpu_info.available:
                # Enable GPU for larger models
                os.environ["OLLAMA_GPU_LAYERS"] = "999"  # Use maximum GPU layers
                os.environ["OLLAMA_NUM_GPU"] = "1"  # Use one GPU
                logger.info("Ollama GPU acceleration enabled")
            else:
                os.environ["OLLAMA_GPU_LAYERS"] = "0"  # CPU only
                logger.info("Ollama running in CPU-only mode")
                
        except Exception as e:
            logger.error(f"Failed to configure Ollama models: {e}")
    
    def generate(self,
                prompt: str,
                task_type: Optional[TaskType] = None,
                context: Dict[str, Any] = None,
                max_tokens: Optional[int] = None,
                temperature: Optional[float] = None,
                use_cache: bool = True) -> str:
        """Generate response using appropriate model
        
        Args:
            prompt: Input prompt
            task_type: Optional task type for model selection
            context: Optional context for task inference
            max_tokens: Optional max tokens override
            temperature: Optional temperature override
            use_cache: Whether to use response cache
            
        Returns:
            Generated text response
        """
        # Check cache first
        if use_cache:
            cache_key = self._get_cache_key(prompt, task_type)
            if cache_key in self.response_cache:
                logger.debug("Using cached response")
                return self.response_cache[cache_key]
        
        # Infer task type if not provided
        if task_type is None:
            task_type = self.scheduler.infer_task_type(prompt, context)
        
        # Select model based on task type
        model_key = self.scheduler.select_model(task_type)
        model_config = self.scheduler.get_model_config(task_type)
        
        # Use config values if not overridden
        if max_tokens is None:
            max_tokens = model_config.max_tokens
        if temperature is None:
            temperature = model_config.temperature
        
        logger.info(f"Generating with {model_config.name} for {task_type.value}")
        
        # Apply CPU optimization for model
        self.cpu_optimizer.optimize_for_model(model_key)
        
        # Generate response
        start_time = time.time()
        
        # Use Ollama service for generation
        ollama_model = self.model_mapping.get(model_key)
        if ollama_model:
            # Use streaming for 4B model, regular for 14B optimized
            if model_key == "fast":
                # For fast model, use streaming for immediate response
                response = ""
                for chunk in self.ollama_service.generate_stream(
                    prompt=prompt,
                    model=ollama_model,
                    temperature=temperature
                ):
                    response += chunk
                    if len(response) >= (max_tokens or 100):
                        break
            else:
                # For deep model, use optimized version with regular generation
                raw_response = self.ollama_service.generate(
                    prompt=prompt,
                    model=ollama_model,
                    temperature=temperature,
                    max_tokens=max_tokens
                )
                # Clean thinking tags if present
                response = self._clean_thinking(raw_response)
        else:
            # Fallback to mock response
            response = self._generate_mock_response(
                model_config.name,
                prompt,
                task_type
            )
        
        elapsed = time.time() - start_time
        logger.info(f"Generation took {elapsed:.2f}s")
        
        # Restore CPU defaults after generation
        self.cpu_optimizer.restore_defaults()
        
        # Update cache
        if use_cache:
            self._update_cache(cache_key, response)
        
        # Track performance
        self.scheduler._update_performance_stats(model_key, elapsed)
        
        return response
    
    def _clean_thinking(self, text: str) -> str:
        """Remove thinking tags from response
        
        Args:
            text: Raw text with potential thinking tags
            
        Returns:
            Cleaned text without thinking content
        """
        if "<think>" in text and "</think>" in text:
            # Extract content after thinking
            idx = text.find("</think>") + 8
            return text[idx:].strip()
        elif "<think>" in text:
            # Thinking not properly closed, try to extract after newlines
            parts = text.split('\n\n')
            if len(parts) > 1:
                return '\n\n'.join(parts[1:]).strip()
        return text.strip()
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about current model configuration
        
        Returns:
            Model configuration info
        """
        gpu_info = self.gpu_detector.get_gpu_info()
        
        return {
            "backend": "Ollama",
            "models": {
                "fast": {
                    "name": "qwen3:4b",
                    "device": "CPU",
                    "purpose": "Fast responses, greetings, simple queries"
                },
                "deep": {
                    "name": "qwen3:30b",
                    "device": "GPU" if gpu_info.available else "CPU",
                    "purpose": "Deep thinking, philosophy, complex reasoning"
                }
            },
            "gpu_available": gpu_info.available,
            "gpu_name": gpu_info.gpu_name if gpu_info.available else None,
            "performance_stats": self.get_performance_stats()
        }
    
    def _generate_mock_response(self,
                              model_name: str,
                              prompt: str,
                              task_type: TaskType) -> str:
        """Generate mock response for testing
        
        Args:
            model_name: Name of model
            prompt: Input prompt
            task_type: Task type
            
        Returns:
            Mock response
        """
        responses = {
            TaskType.GREETING: "Hello! How can I help you today?",
            TaskType.QUICK_RESPONSE: "Sure, I can help with that!",
            TaskType.THOUGHTFUL: "That's an interesting question. Let me think...",
            TaskType.REFLECTION: "Upon reflection, I believe...",
            TaskType.PHILOSOPHY: "From a philosophical perspective...",
            TaskType.REASONING: "Based on the evidence, I conclude...",
            TaskType.PLANNING: "Here's my plan..."
        }
        
        base_response = responses.get(task_type, "Let me consider that...")
        return f"[{model_name}] {base_response}"
    
    async def generate_async(self,
                            prompt: str,
                            task_type: Optional[TaskType] = None,
                            context: Dict[str, Any] = None) -> str:
        """Async generation for non-blocking operations
        
        Args:
            prompt: Input prompt
            task_type: Optional task type
            context: Optional context
            
        Returns:
            Generated text
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self.executor,
            self.generate,
            prompt,
            task_type,
            context
        )
    
    def generate_with_reflection(self,
                                prompt: str,
                                initial_context: Dict[str, Any] = None) -> Dict[str, str]:
        """Generate response with both fast and deep models
        
        Args:
            prompt: Input prompt
            initial_context: Initial context
            
        Returns:
            Dictionary with both fast and deep responses
        """
        results = {}
        
        # Fast response first
        fast_response = self.generate(
            prompt,
            TaskType.QUICK_RESPONSE,
            initial_context
        )
        results["fast_response"] = fast_response
        
        # Then deep reflection
        reflection_prompt = f"""
        Original question: {prompt}
        Initial response: {fast_response}
        
        Please provide a more thoughtful and nuanced response:
        """
        
        deep_response = self.generate(
            reflection_prompt,
            TaskType.REFLECTION,
            initial_context
        )
        results["deep_response"] = deep_response
        
        return results
    
    def _get_cache_key(self, prompt: str, task_type: Optional[TaskType]) -> str:
        """Generate cache key for response
        
        Args:
            prompt: Input prompt
            task_type: Task type
            
        Returns:
            Cache key
        """
        task_str = task_type.value if task_type else "auto"
        # Simple hash of prompt and task type
        return f"{hash(prompt)}_{task_str}"
    
    def _update_cache(self, key: str, value: str):
        """Update response cache
        
        Args:
            key: Cache key
            value: Response to cache
        """
        self.response_cache[key] = value
        
        # Limit cache size
        if len(self.response_cache) > self.cache_max_size:
            # Remove oldest entry (simple FIFO)
            oldest_key = next(iter(self.response_cache))
            del self.response_cache[oldest_key]
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics
        
        Returns:
            Performance report from scheduler
        """
        return self.scheduler.get_performance_report()
    
    def optimize_for_npc(self, npc_state: Dict[str, Any]) -> TaskType:
        """Optimize task type for NPC state
        
        Args:
            npc_state: NPC state dictionary
            
        Returns:
            Recommended task type
        """
        return self.scheduler.optimize_for_context(npc_state)


# Singleton instance
_service_instance = None


def get_enhanced_service() -> EnhancedAIService:
    """Get or create singleton service instance"""
    global _service_instance
    if _service_instance is None:
        _service_instance = EnhancedAIService()
    return _service_instance


def test_enhanced_service():
    """Test the enhanced AI service"""
    
    print("\n" + "="*50)
    print("Testing Enhanced AI Service")
    print("="*50)
    
    service = get_enhanced_service()
    
    # Test 1: Quick response
    print("\n[Test 1: Quick Response]")
    response = service.generate(
        "Hello! How are you?",
        TaskType.GREETING
    )
    print(f"Response: {response}")
    
    # Test 2: Thoughtful response
    print("\n[Test 2: Thoughtful Response]")
    response = service.generate(
        "What makes a good bartender?",
        TaskType.THOUGHTFUL
    )
    print(f"Response: {response}")
    
    # Test 3: Auto task detection
    print("\n[Test 3: Auto Task Detection]")
    response = service.generate(
        "Tell me about the meaning of life"
    )
    print(f"Response: {response}")
    
    # Test 4: Dual response
    print("\n[Test 4: Dual Response (Fast + Deep)]")
    responses = service.generate_with_reflection(
        "What should I order tonight?"
    )
    print(f"Fast: {responses['fast_response']}")
    print(f"Deep: {responses['deep_response']}")
    
    # Show performance stats
    print("\n" + "="*50)
    print("Performance Statistics")
    print("="*50)
    
    stats = service.get_performance_stats()
    for model_type, data in stats.items():
        print(f"\n{model_type}:")
        print(f"  Model: {data['name']}")
        print(f"  Calls: {data['stats']['count']}")
        if data['stats']['count'] > 0:
            print(f"  Avg Time: {data['stats']['avg_time']:.3f}s")


if __name__ == "__main__":
    test_enhanced_service()