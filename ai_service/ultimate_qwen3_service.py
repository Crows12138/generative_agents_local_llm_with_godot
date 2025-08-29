#!/usr/bin/env python
"""
Ultimate Qwen3 Dual Model Service
1.7B for ultra-fast responses + 4B for standard dialogue
Perfect balance between speed and quality
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Optional, Dict, Any, Tuple
import threading
import queue
import time
from dataclasses import dataclass
from enum import Enum
import logging
import psutil
import warnings
warnings.filterwarnings('ignore')

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TaskType(Enum):
    """Task types for smart routing"""
    GREETING = "greeting"      # Simple greetings
    QUICK = "quick"            # Quick facts
    DIALOGUE = "dialogue"      # Normal conversation  
    REASONING = "reasoning"    # Complex reasoning
    REFLECTION = "reflection"  # Deep thinking


@dataclass
class ModelConfig:
    """Model configuration"""
    name: str
    device: str
    dtype: torch.dtype
    max_memory: Optional[Dict[str, str]] = None


class UltimateQwen3Service:
    """Ultimate dual model service: 1.7B ultra-fast + 4B standard"""
    
    def __init__(self, load_both: bool = True):
        """Initialize service
        
        Args:
            load_both: Whether to load both models (set False for testing with just 1.7B)
        """
        logger.info("Initializing Ultimate Qwen3 Service...")
        
        # Check system resources
        self._check_resources()
        
        # Model configurations - Optimized dual model setup
        self.configs = {
            "ultra_fast": ModelConfig(
                name="Qwen/Qwen3-1.7B",
                device="cpu",  # CPU for maximum stability and speed
                dtype=torch.float32  # CPU works better with float32
            ),
            "standard": ModelConfig(
                name="Qwen/Qwen3-4B", 
                device="cuda:0" if self.gpu_available else "cpu",
                dtype=torch.float16 if self.gpu_available else torch.float32
            )
        }
        
        # Load models
        self.models = {}
        self.tokenizers = {}
        self._load_ultra_fast_model()  # Always load 1.7B
        
        if load_both:
            self._load_standard_model()  # Load 4B if requested
        
        # Background processing queue for batch operations
        self.processing_queue = queue.Queue()
        self.processing_results = {}
        self.processing_thread = None
        
        if load_both:
            self._start_processing_worker()
        
        logger.info("Service initialized successfully!")
    
    def _check_resources(self):
        """Check system resources"""
        # CPU
        self.cpu_count = psutil.cpu_count()
        self.ram_gb = psutil.virtual_memory().total / (1024**3)
        
        # GPU
        self.gpu_available = torch.cuda.is_available()
        if self.gpu_available:
            self.gpu_name = torch.cuda.get_device_name(0)
            self.gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        else:
            self.gpu_name = "None"
            self.gpu_memory_gb = 0
        
        logger.info(f"System: {self.cpu_count} CPU cores, {self.ram_gb:.1f}GB RAM")
        logger.info(f"GPU: {self.gpu_name}, {self.gpu_memory_gb:.1f}GB VRAM")
    
    def _load_ultra_fast_model(self):
        """Load 1.7B model for ultra-fast responses (CPU)"""
        config = self.configs["ultra_fast"]
        logger.info(f"Loading Qwen3-1.7B on {config.device} for ultra-fast responses...")
        
        try:
            # Load tokenizer
            self.tokenizers["ultra_fast"] = AutoTokenizer.from_pretrained(
                config.name,
                trust_remote_code=True
            )
            
            # Load model on CPU for consistent fast performance
            self.models["ultra_fast"] = AutoModelForCausalLM.from_pretrained(
                config.name,
                torch_dtype=config.dtype,
                device_map=config.device,
                trust_remote_code=True,
                low_cpu_mem_usage=True
            )
            
            logger.info(f"✓ Qwen3-1.7B loaded on {config.device.upper()}")
            
        except Exception as e:
            logger.error(f"Failed to load ultra-fast model: {e}")
            raise
    
    def _load_standard_model(self):
        """Load 4B model for standard dialogue (GPU if available)"""
        config = self.configs["standard"]
        logger.info(f"Loading Qwen3-4B on {config.device} for standard dialogue...")
        
        try:
            # Load tokenizer
            self.tokenizers["standard"] = AutoTokenizer.from_pretrained(
                config.name,
                trust_remote_code=True
            )
            
            # Load model on configured device
            self.models["standard"] = AutoModelForCausalLM.from_pretrained(
                config.name,
                torch_dtype=config.dtype,
                device_map=config.device,
                trust_remote_code=True,
                low_cpu_mem_usage=True
            )
            
            logger.info(f"✓ Qwen3-4B loaded on {config.device.upper()}")
            
        except Exception as e:
            logger.error(f"Failed to load standard model: {e}")
            logger.info("Will use 1.7B model for all tasks")
    
    def ultra_fast_response(self, 
                           prompt: str, 
                           max_tokens: int = 50) -> Dict[str, Any]:
        """Ultra-fast response using 1.7B on CPU
        
        Args:
            prompt: User prompt
            max_tokens: Maximum tokens to generate
            
        Returns:
            Response dict with metadata
        """
        start_time = time.time()
        
        # Use ultra-fast model
        tokenizer = self.tokenizers["ultra_fast"]
        model = self.models["ultra_fast"]
        
        # Build messages
        messages = [{"role": "user", "content": prompt}]
        
        # Apply chat template with thinking disabled for speed
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False  # Disable thinking for 3-4x speedup!
        )
        
        # Generate on CPU
        inputs = tokenizer(text, return_tensors="pt").to("cpu")
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=0.7,
                do_sample=True,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id
            )
        
        response = tokenizer.decode(
            outputs[0][len(inputs.input_ids[0]):], 
            skip_special_tokens=True
        )
        
        elapsed = time.time() - start_time
        
        return {
            "response": response,
            "model": "Qwen3-1.7B",
            "device": "CPU",
            "time": elapsed,
            "tokens_per_sec": max_tokens / elapsed if elapsed > 0 else 0
        }
    
    def standard_response(self, 
                         prompt: str, 
                         enable_thinking: bool = False,
                         max_tokens: int = 200) -> Dict[str, Any]:
        """Standard response using 4B (or fallback to 1.7B)
        
        Args:
            prompt: User prompt
            enable_thinking: Whether to enable thinking mode (if supported)
            max_tokens: Maximum tokens to generate
            
        Returns:
            Response dict with metadata
        """
        start_time = time.time()
        
        # Use standard model if available, otherwise fallback to ultra-fast
        if "standard" in self.models:
            tokenizer = self.tokenizers["standard"]
            model = self.models["standard"]
            model_name = "Qwen3-4B"
            device = self.configs["standard"].device
        else:
            logger.info("4B not available, using 1.7B")
            tokenizer = self.tokenizers["ultra_fast"]
            model = self.models["ultra_fast"]
            model_name = "Qwen3-1.7B"
            device = "cpu"
        
        messages = [{"role": "user", "content": prompt}]
        
        # Apply chat template with optional thinking mode
        if hasattr(tokenizer, 'apply_chat_template'):
            # Try with enable_thinking parameter (Qwen3 specific)
            try:
                text = tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=enable_thinking
                )
            except TypeError:
                # Fallback without enable_thinking for models that don't support it
                text = tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True
                )
        else:
            # Manual format if apply_chat_template not available
            text = f"User: {prompt}\nAssistant:"
        
        # Move to appropriate device
        inputs = tokenizer(text, return_tensors="pt").to(device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=0.7,
                do_sample=True,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id
            )
        
        response_full = tokenizer.decode(
            outputs[0][len(inputs.input_ids[0]):], 
            skip_special_tokens=True
        )
        
        # Parse thinking content if present
        response = response_full
        thinking_content = None
        
        if enable_thinking and "<think>" in response_full and "</think>" in response_full:
            think_start = response_full.find("<think>")
            think_end = response_full.find("</think>")
            thinking_content = response_full[think_start+7:think_end]
            response = response_full[think_end+8:].strip()
        
        elapsed = time.time() - start_time
        
        return {
            "response": response,
            "thinking": thinking_content,
            "model": model_name,
            "device": device.upper(),
            "enable_thinking": enable_thinking,
            "time": elapsed,
            "tokens_per_sec": max_tokens / elapsed if elapsed > 0 else 0
        }
    
    def smart_route(self, prompt: str) -> Tuple[str, Dict[str, Any]]:
        """Smart routing to select best model
        
        Args:
            prompt: User prompt
            
        Returns:
            Tuple of (response, metadata)
        """
        prompt_lower = prompt.lower()
        prompt_length = len(prompt.split())
        
        # Simple greetings/quick responses -> 1.7B ultra-fast
        if any(word in prompt_lower for word in ["hello", "hi", "hey", "how are", "good morning", "goodbye"]):
            result = self.ultra_fast_response(prompt, max_tokens=30)
            return result["response"], result
        
        # Quick facts/simple questions -> 1.7B ultra-fast
        elif "?" in prompt and prompt_length < 8:
            result = self.ultra_fast_response(prompt, max_tokens=50)
            return result["response"], result
        
        # Medium complexity -> 4B standard
        elif prompt_length < 20:
            result = self.standard_response(prompt, enable_thinking=False, max_tokens=150)
            return result["response"], result
        
        # Complex questions -> 4B with thinking
        else:
            result = self.standard_response(prompt, enable_thinking=True, max_tokens=300)
            return result["response"], result
    
    def async_process(self, task_id: str, prompt: str):
        """Submit async processing task
        
        Args:
            task_id: Unique task identifier
            prompt: Prompt to process
        """
        self.processing_queue.put((task_id, prompt))
        logger.info(f"Queued processing task: {task_id}")
    
    def get_processing_result(self, task_id: str) -> Optional[Dict[str, Any]]:
        """Get result of async processing
        
        Args:
            task_id: Task identifier
            
        Returns:
            Result dict or None if not ready
        """
        return self.processing_results.get(task_id)
    
    def _processing_worker(self):
        """Background processing worker thread"""
        logger.info("Started background processing worker")
        
        while True:
            try:
                task_id, prompt = self.processing_queue.get(timeout=1)
                logger.info(f"Processing task: {task_id}")
                
                result = self.standard_response(prompt, enable_thinking=True)
                self.processing_results[task_id] = result
                
                logger.info(f"Completed task: {task_id}")
                self.processing_queue.task_done()
                
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Error in processing worker: {e}")
    
    def _start_processing_worker(self):
        """Start background processing thread"""
        self.processing_thread = threading.Thread(
            target=self._processing_worker, 
            daemon=True
        )
        self.processing_thread.start()
        logger.info("Background processing worker started")
    
    def benchmark(self) -> Dict[str, Any]:
        """Benchmark all configurations"""
        results = {}
        test_prompt = "What is the meaning of life?"
        
        # Test 1.7B ultra-fast
        logger.info("Benchmarking 1.7B ultra-fast...")
        result = self.ultra_fast_response(test_prompt, max_tokens=50)
        results["1.7B_ultra_fast"] = {
            "time": result["time"],
            "tokens_per_sec": result["tokens_per_sec"]
        }
        
        # Test 4B standard if available
        if "standard" in self.models:
            logger.info("Benchmarking 4B standard...")
            result = self.standard_response(test_prompt, enable_thinking=False, max_tokens=50)
            results["4B_standard"] = {
                "time": result["time"],
                "tokens_per_sec": result["tokens_per_sec"]
            }
            
            logger.info("Benchmarking 4B with thinking...")
            result = self.standard_response(test_prompt, enable_thinking=True, max_tokens=50)
            results["4B_thinking"] = {
                "time": result["time"],
                "tokens_per_sec": result["tokens_per_sec"]
            }
        
        return results
    
    def get_status(self) -> Dict[str, Any]:
        """Get service status"""
        return {
            "models_loaded": list(self.models.keys()),
            "cpu_usage": psutil.cpu_percent(),
            "ram_usage": psutil.virtual_memory().percent,
            "gpu_memory_used": torch.cuda.memory_allocated() / (1024**3) if self.gpu_available else 0,
            "processing_queue_size": self.processing_queue.qsize() if self.processing_queue else 0,
            "completed_tasks": len(self.processing_results)
        }


def test_ultimate_service():
    """Test the ultimate service"""
    print("="*60)
    print("TESTING ULTIMATE QWEN3 SERVICE")
    print("="*60)
    
    # Initialize with both models
    service = UltimateQwen3Service(load_both=True)
    
    # Test cases
    test_cases = [
        ("Hello!", "Greeting"),
        ("What is 2+2?", "Simple math"),
        ("Explain quantum computing", "Complex topic"),
        ("Tell me a joke", "Creative")
    ]
    
    for prompt, category in test_cases:
        print(f"\n[{category}]: {prompt}")
        print("-"*40)
        
        # Use smart routing
        response, metadata = service.smart_route(prompt)
        
        print(f"Model: {metadata['model']}")
        print(f"Device: {metadata['device']}")
        print(f"Time: {metadata['time']:.2f}s")
        print(f"Speed: {metadata['tokens_per_sec']:.1f} tokens/s")
        print(f"Response: {response[:100]}...")
    
    # Benchmark
    print("\n" + "="*60)
    print("BENCHMARK")
    print("="*60)
    
    results = service.benchmark()
    for config, metrics in results.items():
        print(f"{config}: {metrics['time']:.2f}s ({metrics['tokens_per_sec']:.1f} tok/s)")
    
    # Status
    print("\n" + "="*60)
    print("SERVICE STATUS")
    print("="*60)
    
    status = service.get_status()
    for key, value in status.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    test_ultimate_service()
    
    print("\n" + "="*60)
    print("DEPLOYMENT GUIDE")
    print("="*60)
    print("\n1. For production (both models):")
    print("   service = UltimateQwen3Service(load_both=True)")
    print("\n2. For testing (1.7B only):")
    print("   service = UltimateQwen3Service(load_both=False)")
    print("\n3. Smart routing:")
    print("   response, _ = service.smart_route(prompt)")
    print("\n4. Direct model access:")
    print("   result = service.ultra_fast_response(prompt)  # 1.7B")
    print("   result = service.standard_response(prompt)    # 4B")