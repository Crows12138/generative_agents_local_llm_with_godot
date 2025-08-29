#!/usr/bin/env python
"""
Dual Model Scheduler for Qwen3 Models
Intelligently selects models based on task type:
- Qwen3-4B: Fast response (reactive dialogue)
- Qwen3-14B: Deep processing (thoughtful + reflective)
"""

import time
import logging
from typing import Dict, Any, Optional, Literal
from pathlib import Path
from dataclasses import dataclass
from enum import Enum

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TaskType(Enum):
    """Task types for model selection"""
    
    # Fast response tasks (4B model)
    QUICK_RESPONSE = "quick_response"  # Quick dialogue
    GREETING = "greeting"               # Greetings
    SIMPLE_QUERY = "simple_query"       # Simple queries
    REACTIVE = "reactive"               # Reactive responses
    
    # Deep processing tasks (14B model)  
    THOUGHTFUL = "thoughtful"           # Thoughtful processing
    REFLECTION = "reflection"           # Reflection and memory consolidation
    REASONING = "reasoning"             # Reasoning
    PLANNING = "planning"               # Planning
    PHILOSOPHY = "philosophy"           # Philosophical thinking
    COMPLEX_DIALOGUE = "complex"        # Complex dialogue


@dataclass
class ModelConfig:
    """Model configuration"""
    name: str
    path: str
    context_size: int
    max_tokens: int
    temperature: float
    response_time_target: float  # Target response time in seconds


class DualModelScheduler:
    """Dual model scheduler for optimized LLM usage"""
    
    def __init__(self):
        # Model configurations
        self.models = {
            "fast": ModelConfig(
                name="qwen3-4b",
                path="models/llms/Qwen3-4B-Instruct-2507-Q4_0.gguf",
                context_size=4096,
                max_tokens=100,
                temperature=0.7,
                response_time_target=1.0  # Target: <1 second
            ),
            "deep": ModelConfig(
                name="qwen3-14b",
                path="models/llms/Qwen3-14B-Instruct.gguf",
                context_size=8192,
                max_tokens=500,
                temperature=0.8,
                response_time_target=5.0  # Acceptable: 3-5 seconds with GPU
            )
        }
        
        # Task mapping to models
        self.task_model_map = {
            # Fast tasks -> 4B model
            TaskType.QUICK_RESPONSE: "fast",
            TaskType.GREETING: "fast",
            TaskType.SIMPLE_QUERY: "fast",
            TaskType.REACTIVE: "fast",
            
            # Deep tasks -> 30B model
            TaskType.THOUGHTFUL: "deep",
            TaskType.REFLECTION: "deep",
            TaskType.REASONING: "deep",
            TaskType.PLANNING: "deep",
            TaskType.PHILOSOPHY: "deep",
            TaskType.COMPLEX_DIALOGUE: "deep"
        }
        
        # Performance tracking
        self.performance_stats = {
            "fast": {"count": 0, "total_time": 0, "avg_time": 0},
            "deep": {"count": 0, "total_time": 0, "avg_time": 0}
        }
        
        # Model instances (will be loaded on demand)
        self._model_instances = {}
        
        logger.info("DualModelScheduler initialized with Qwen3-4B (fast) and Qwen3-14B (deep)")
    
    def select_model(self, task_type: TaskType) -> str:
        """Select appropriate model based on task type
        
        Args:
            task_type: Type of task to perform
            
        Returns:
            Model key ("fast" or "deep")
        """
        model_key = self.task_model_map.get(task_type, "fast")
        logger.debug(f"Selected {model_key} model for task {task_type.value}")
        return model_key
    
    def infer_task_type(self, prompt: str, context: Dict[str, Any] = None) -> TaskType:
        """Infer task type from prompt and context
        
        Args:
            prompt: Input prompt
            context: Optional context (e.g., thinking_depth, mood)
            
        Returns:
            Inferred task type
        """
        prompt_lower = prompt.lower()
        
        # Check context first (if provided)
        if context:
            thinking_depth = context.get("thinking_depth", 0)
            if thinking_depth == 0:
                return TaskType.REACTIVE
            elif thinking_depth == 1:
                return TaskType.THOUGHTFUL
            elif thinking_depth >= 2:
                return TaskType.REFLECTION
        
        # Analyze prompt for keywords
        # Greetings and simple queries
        if any(word in prompt_lower for word in ["hello", "hi", "hey", "goodbye"]):
            return TaskType.GREETING
        
        if any(word in prompt_lower for word in ["what time", "where is", "how much"]):
            return TaskType.SIMPLE_QUERY
        
        # Complex thinking tasks
        if any(word in prompt_lower for word in ["philosophy", "meaning", "existential"]):
            return TaskType.PHILOSOPHY
        
        if any(word in prompt_lower for word in ["plan", "strategy", "organize"]):
            return TaskType.PLANNING
        
        if any(word in prompt_lower for word in ["think about", "consider", "reflect"]):
            return TaskType.REFLECTION
        
        if any(word in prompt_lower for word in ["because", "therefore", "if...then"]):
            return TaskType.REASONING
        
        # Check prompt length as a heuristic
        if len(prompt) < 50:
            return TaskType.QUICK_RESPONSE
        elif len(prompt) > 200:
            return TaskType.COMPLEX_DIALOGUE
        
        # Default to thoughtful for medium complexity
        return TaskType.THOUGHTFUL
    
    def get_model_config(self, task_type: TaskType) -> ModelConfig:
        """Get model configuration for task type
        
        Args:
            task_type: Type of task
            
        Returns:
            Model configuration
        """
        model_key = self.select_model(task_type)
        return self.models[model_key]
    
    def generate(self, 
                 prompt: str,
                 task_type: Optional[TaskType] = None,
                 context: Dict[str, Any] = None,
                 **kwargs) -> Dict[str, Any]:
        """Generate response using appropriate model
        
        Args:
            prompt: Input prompt
            task_type: Optional explicit task type
            context: Optional context for task inference
            **kwargs: Additional generation parameters
            
        Returns:
            Dictionary with response and metadata
        """
        start_time = time.time()
        
        # Determine task type if not provided
        if task_type is None:
            task_type = self.infer_task_type(prompt, context)
        
        # Get model configuration
        model_config = self.get_model_config(task_type)
        model_key = self.select_model(task_type)
        
        logger.info(f"Using {model_config.name} for {task_type.value}")
        
        # Here we would actually call the model
        # For now, returning mock response with metadata
        response = {
            "text": f"[{model_config.name}] Response to: {prompt[:50]}...",
            "model_used": model_config.name,
            "task_type": task_type.value,
            "model_config": {
                "max_tokens": model_config.max_tokens,
                "temperature": model_config.temperature,
                "context_size": model_config.context_size
            },
            "generation_time": 0
        }
        
        # Track performance
        elapsed = time.time() - start_time
        response["generation_time"] = elapsed
        self._update_performance_stats(model_key, elapsed)
        
        # Log if exceeding target time
        if elapsed > model_config.response_time_target:
            logger.warning(f"{model_config.name} took {elapsed:.2f}s (target: {model_config.response_time_target}s)")
        
        return response
    
    def _update_performance_stats(self, model_key: str, elapsed: float):
        """Update performance statistics"""
        stats = self.performance_stats[model_key]
        stats["count"] += 1
        stats["total_time"] += elapsed
        stats["avg_time"] = stats["total_time"] / stats["count"]
    
    def get_performance_report(self) -> Dict[str, Any]:
        """Get performance report
        
        Returns:
            Performance statistics for both models
        """
        report = {
            "fast_model": {
                "name": self.models["fast"].name,
                "stats": self.performance_stats["fast"],
                "target_time": self.models["fast"].response_time_target
            },
            "deep_model": {
                "name": self.models["deep"].name,
                "stats": self.performance_stats["deep"],
                "target_time": self.models["deep"].response_time_target
            }
        }
        return report
    
    def optimize_for_context(self, 
                            npc_state: Dict[str, Any]) -> TaskType:
        """Optimize task type selection based on NPC state
        
        Args:
            npc_state: NPC state including mood, energy, etc.
            
        Returns:
            Recommended task type
        """
        # Low energy -> use fast model
        if npc_state.get("energy", 100) < 30:
            return TaskType.QUICK_RESPONSE
        
        # Philosophical mood -> use deep model
        if npc_state.get("mood") in ["philosophical", "contemplative"]:
            return TaskType.PHILOSOPHY
        
        # High social interaction -> balance between models
        if npc_state.get("social_interaction_count", 0) > 5:
            # Alternate between fast and deep for variety
            if npc_state.get("last_model") == "deep":
                return TaskType.QUICK_RESPONSE
            else:
                return TaskType.THOUGHTFUL
        
        # Default based on time of day
        hour = npc_state.get("hour", 12)
        if 20 <= hour <= 23:  # Peak evening hours
            return TaskType.THOUGHTFUL  # More thoughtful in evening
        else:
            return TaskType.REACTIVE  # Quick during day
    
    def preload_models(self):
        """Preload both models for faster first response"""
        logger.info("Preloading models...")
        
        # Here we would actually load the models
        # For now, just simulate loading time
        import time
        
        logger.info(f"Loading {self.models['fast'].name}...")
        time.sleep(0.5)  # Simulate load time
        self._model_instances["fast"] = f"MockModel({self.models['fast'].name})"
        
        logger.info(f"Loading {self.models['deep'].name}...")
        time.sleep(1.0)  # Simulate load time  
        self._model_instances["deep"] = f"MockModel({self.models['deep'].name})"
        
        logger.info("Models preloaded successfully!")


# Singleton instance
_scheduler_instance = None


def get_scheduler() -> DualModelScheduler:
    """Get or create the singleton scheduler instance"""
    global _scheduler_instance
    if _scheduler_instance is None:
        _scheduler_instance = DualModelScheduler()
    return _scheduler_instance


def test_scheduler():
    """Test the dual model scheduler"""
    
    print("\n" + "="*50)
    print("Testing Dual Model Scheduler")
    print("="*50)
    
    scheduler = get_scheduler()
    
    # Test different prompts
    test_cases = [
        ("Hello!", None, None),  # Should use fast model
        ("What is the meaning of life?", None, {"thinking_depth": 2}),  # Should use deep model
        ("Can I get a beer?", TaskType.QUICK_RESPONSE, None),  # Explicit fast
        ("Let me reflect on our conversation...", TaskType.REFLECTION, None),  # Explicit deep
    ]
    
    for prompt, task_type, context in test_cases:
        print(f"\nPrompt: {prompt[:50]}...")
        response = scheduler.generate(prompt, task_type, context)
        print(f"  Model: {response['model_used']}")
        print(f"  Task Type: {response['task_type']}")
        print(f"  Response: {response['text'][:80]}...")
    
    # Show performance report
    print("\n" + "="*50)
    print("Performance Report")
    print("="*50)
    
    report = scheduler.get_performance_report()
    for model_type, data in report.items():
        print(f"\n{model_type}:")
        print(f"  Model: {data['name']}")
        print(f"  Calls: {data['stats']['count']}")
        if data['stats']['count'] > 0:
            print(f"  Avg Time: {data['stats']['avg_time']:.3f}s")
        print(f"  Target: <{data['target_time']}s")


if __name__ == "__main__":
    test_scheduler()