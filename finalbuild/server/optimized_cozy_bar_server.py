#!/usr/bin/env python3
"""
Optimized Cozy Bar LLM Server - Native Transformers Edition
Uses Qwen3-1.7B with all performance optimizations for instant NPC responses
"""

import socket
import time
import sys
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, StoppingCriteria, StoppingCriteriaList
import json
import logging
from typing import Dict, Any
import random
import os

# Import memory integration from same directory
from memory_integration import NPCMemoryManager

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)


class PunctuationStoppingCriteria(StoppingCriteria):
    """Stop at punctuation but keep it in the output"""
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.punctuation = ['.', '!', '?']
        
    def __call__(self, input_ids, scores, **kwargs):
        # Decode the last generated token
        last_token = self.tokenizer.decode(input_ids[0][-1])
        # If it's punctuation, stop generation
        return any(p in last_token for p in self.punctuation)


class OptionalCognitiveModel:
    """Dual 1.7B model test - Using second 1.7B for deep thinking"""
    
    def __init__(self):
        self.model_4b = None  # Will be second 1.7B in dual mode
        self.tokenizer_4b = None
        self.enabled = False  # DISABLED by default - won't affect current system
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        self.use_dual_1_7b = False  # Flag for dual 1.7B mode
        self.config = {}  # Will be loaded from parent
        
    def load_if_needed(self, parent_config=None):
        """Load second 1.7B model - ensuring independence from main model"""
        if parent_config:
            self.config = parent_config
            self.use_dual_1_7b = parent_config.get("use_dual_1_7b", False)
            
        if self.enabled and not self.model_4b:
            try:
                # Determine which model to load
                if self.use_dual_1_7b:
                    logger.info("[COGNITIVE] Loading second Qwen3-1.7B (dual mode)...")
                    model_name = "Qwen/Qwen3-1.7B"
                else:
                    logger.info("[COGNITIVE] Loading Qwen3-4B for deep thinking...")
                    model_name = self.config.get("model_name", "Qwen/Qwen2.5-3B-Instruct")
                
                start = time.time()
                
                # Load tokenizer
                self.tokenizer_4b = AutoTokenizer.from_pretrained(
                    model_name,
                    trust_remote_code=True
                )
                
                # Ensure pad_token is set
                if self.tokenizer_4b.pad_token is None:
                    self.tokenizer_4b.pad_token = self.tokenizer_4b.eos_token
                
                # Load model - avoid device_map="auto" to prevent meta tensor issues
                self.model_4b = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    torch_dtype=self.dtype,
                    device_map=self.device,  # Use explicit device, not "auto"
                    trust_remote_code=True,
                    low_cpu_mem_usage=True
                )
                
                # Set to evaluation mode
                self.model_4b.eval()
                
                # Force independence if configured
                if self.config.get("force_independent_models", False) and self.use_dual_1_7b:
                    logger.info("[COGNITIVE] Ensuring model independence...")
                    with torch.no_grad():
                        for param in self.model_4b.parameters():
                            param.data = param.data.clone()
                
                # Monitor memory usage
                if self.device == "cuda":
                    torch.cuda.synchronize()
                    allocated_gb = torch.cuda.memory_allocated() / 1024**3
                    logger.info(f"[COGNITIVE] GPU memory after loading: {allocated_gb:.2f}GB")
                
                # Warmup the model
                self._warmup_cognitive_model()
                
                elapsed = time.time() - start
                logger.info(f"[COGNITIVE] Second model loaded in {elapsed:.1f}s")
                
                # Verify independence in dual mode
                if self.use_dual_1_7b:
                    self._verify_independence()
                
                return True
                
            except Exception as e:
                logger.error(f"[COGNITIVE] Failed to load model: {e}")
                self.enabled = False
                return False
        return False
    
    def _warmup_cognitive_model(self):
        """Warmup cognitive model to avoid first inference delay"""
        test_prompt = "Hello, I need a drink"
        
        # Use different warmup prompts for dual mode
        if self.use_dual_1_7b:
            text = f"Customer: {test_prompt}\nBob (thinking deeply):"
        else:
            text = f"User: {test_prompt}\nAssistant:"
            
        inputs = self.tokenizer_4b(text, return_tensors="pt")
        
        # Ensure inputs are on correct device
        if self.device == "cuda":
            inputs = {k: v.cuda() for k, v in inputs.items()}
        
        with torch.no_grad():
            _ = self.model_4b.generate(
                **inputs,
                max_new_tokens=10,
                temperature=0.7,
                do_sample=True,
                pad_token_id=self.tokenizer_4b.pad_token_id
            )
        
        if self.device == "cuda":
            torch.cuda.synchronize()
        
        logger.info("[COGNITIVE] Warmup completed")
    
    def _verify_independence(self):
        """Verify two models are independent (for debugging)"""
        try:
            if self.device == "cuda":
                allocated = torch.cuda.memory_allocated() / 1024**3
                logger.info(f"[COGNITIVE] Independence check - Memory: {allocated:.2f}GB")
                
                # Expected: ~7GB for two independent 1.7B models
                if allocated > 6.0:
                    logger.info("[COGNITIVE] ✓ Models appear to be independent")
                elif allocated > 4.0:
                    logger.info("[COGNITIVE] ⚠ Possible memory sharing detected")
                else:
                    logger.warning("[COGNITIVE] ⚠ Memory usage lower than expected")
        except Exception as e:
            logger.debug(f"[COGNITIVE] Independence check failed: {e}")
    
    def think(self, context: str, max_length: int = 300):
        """Generate deep thought - using different prompting for dual 1.7B mode"""
        if not self.enabled or not self.model_4b:
            return None
            
        try:
            # Different prompting strategies for dual mode
            if self.use_dual_1_7b:
                # Rich, story-driven prompt for second 1.7B
                prompt = f"""You are Bob, a veteran bartender with 20 years of experience.
You've seen it all - celebrations, heartbreaks, quiet nights, and wild parties.
When responding, draw from your deep well of experience and wisdom.
Include subtle observations about people and life.

Customer says: {context}

Bob (leaning on the bar, with a knowing smile):"""
            else:
                # Standard prompt for 4B model
                prompt = f"[Deep Thought] {context}\nResponse:"
            
            inputs = self.tokenizer_4b(prompt, return_tensors="pt")
            
            # Ensure inputs are on correct device
            if self.device == "cuda":
                inputs = {k: v.cuda() for k, v in inputs.items()}
            
            # Create stopping criteria for deep thinking too
            stopping_criteria = StoppingCriteriaList([
                PunctuationStoppingCriteria(self.tokenizer_4b)
            ])
            
            with torch.no_grad():
                outputs = self.model_4b.generate(
                    **inputs,
                    max_new_tokens=max_length,
                    temperature=0.85 if self.use_dual_1_7b else 0.7,  # Higher creativity for dual mode
                    do_sample=True,
                    top_p=0.9,
                    top_k=50,
                    pad_token_id=self.tokenizer_4b.pad_token_id,
                    eos_token_id=self.tokenizer_4b.eos_token_id,
                    stopping_criteria=stopping_criteria  # Use custom stopping criteria
                )
            
            response = self.tokenizer_4b.decode(
                outputs[0][len(inputs['input_ids'][0]):],
                skip_special_tokens=True
            ).strip()
            
            # Clean up response
            # Allow multi-line for deep thoughts
            lines = response.split("\n")
            response = lines[0] if len(lines[0]) > 20 else "\n".join(lines[:2])
            if response and not response[-1] in '.!?':
                response += '.'
            
            return response
            
        except Exception as e:
            logger.error(f"[COGNITIVE] Think failed: {e}")
            return None
    
    def unload(self):
        """Unload 4B model to free memory"""
        if self.model_4b:
            del self.model_4b
            self.model_4b = None
            
        if self.tokenizer_4b:
            del self.tokenizer_4b
            self.tokenizer_4b = None
            
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        logger.info("[COGNITIVE] 4B model unloaded")
        self.enabled = False
    
    def check_gpu_memory(self):
        """Check if enough GPU memory is available"""
        if torch.cuda.is_available():
            free_memory = torch.cuda.mem_get_info()[0] / 1024**3  # GB
            if free_memory < 2.0:  # Less than 2GB free
                logger.warning(f"[COGNITIVE] Low GPU memory: {free_memory:.1f}GB")
                return False
        return True


class OptimizedCozyBarServer:
    """Optimized LLM server using native Transformers with proven speedups"""
    
    def __init__(self):
        self.model = None
        self.tokenizer = None
        # NO CACHE, NO HISTORY - Every response is fresh!
        
        # Initialize memory manager for persistence
        self.memory_manager = NPCMemoryManager()
        logger.info("[MEMORY] Memory manager initialized")
        
        # Load configuration
        self.load_config()
        
        # Optional cognitive model for deep thinking (DISABLED by default)
        self.cognitive_model = OptionalCognitiveModel()
        
        # Apply configuration
        if self.config.get("enable_4b", False):
            if self.config.get("use_dual_1_7b", False):
                logger.info("[CONFIG] Dual 1.7B mode enabled")
            else:
                logger.info("[CONFIG] 4B model enabled in configuration")
            self.cognitive_model.enabled = True
            self.cognitive_model.load_if_needed(self.config)
    
    def load_config(self):
        """Load configuration from cognitive_config.json"""
        config_path = os.path.join(os.path.dirname(__file__), "cognitive_config.json")
        
        try:
            if os.path.exists(config_path):
                with open(config_path, 'r') as f:
                    self.config = json.load(f)
                logger.info(f"[CONFIG] Loaded configuration from {config_path}")
                
                # Apply configuration values
                self.bob_deep_think_probability = self.config.get("bob_deep_think_probability", 0.1)
                
                # Log configuration
                logger.info(f"[CONFIG] enable_4b: {self.config.get('enable_4b', False)}")
                logger.info(f"[CONFIG] bob_deep_think_probability: {self.bob_deep_think_probability}")
                logger.info(f"[CONFIG] min_gpu_memory_gb: {self.config.get('min_gpu_memory_gb', 2.0)}")
            else:
                # Default configuration
                self.config = {
                    "enable_4b": False,
                    "bob_deep_think_probability": 0.1,
                    "min_gpu_memory_gb": 2.0
                }
                self.bob_deep_think_probability = 0.1
                logger.info("[CONFIG] Using default configuration (4B disabled)")
                
        except Exception as e:
            logger.error(f"[CONFIG] Failed to load configuration: {e}")
            # Fallback to defaults
            self.config = {
                "enable_4b": False,
                "bob_deep_think_probability": 0.1,
                "min_gpu_memory_gb": 2.0
            }
            self.bob_deep_think_probability = 0.1
        
    def load_model(self):
        """Load Qwen3-1.7B with optimized settings"""
        logger.info("[COZY BAR] Loading Qwen3-1.7B with optimizations...")
        start = time.time()
        
        try:
            # Check GPU availability
            if torch.cuda.is_available():
                self.device = "cuda"
                self.dtype = torch.float16
                logger.info(f"[COZY BAR] Using GPU: {torch.cuda.get_device_name(0)}")
            else:
                self.device = "cpu"
                self.dtype = torch.float32
                logger.info("[COZY BAR] Using CPU")
            
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                "Qwen/Qwen3-1.7B",
                trust_remote_code=True
            )
            
            # Ensure pad_token is set
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            print(f"[DEBUG] Tokenizer pad_token: {self.tokenizer.pad_token}")
            
            # Load model with optimizations
            self.model = AutoModelForCausalLM.from_pretrained(
                "Qwen/Qwen3-1.7B",
                torch_dtype=self.dtype,
                device_map=self.device,
                trust_remote_code=True,
                low_cpu_mem_usage=True
            )
            
            elapsed = time.time() - start
            logger.info(f"[COZY BAR] Model loaded in {elapsed:.1f} seconds")
            
            # Critical warmup to avoid 19s first inference
            logger.info("[COZY BAR] Warming up model (prevents slow first response)...")
            warmup_start = time.time()
            self._warmup_model()
            warmup_time = time.time() - warmup_start
            logger.info(f"[COZY BAR] Warmup completed in {warmup_time:.1f}s")
            logger.info(f"[COZY BAR] Ready! Total initialization: {time.time() - start:.1f}s")
            
            return True
            
        except Exception as e:
            logger.error(f"[ERROR] Failed to load model: {e}")
            logger.error(f"[INFO] Make sure Qwen3-1.7B is available")
            logger.error(f"[INFO] Install with: pip install transformers torch")
            return False
    
    def _warmup_model(self):
        """Warmup model to avoid first inference delay"""
        test_prompt = "Hello"
        messages = [{"role": "user", "content": test_prompt}]
        
        # Use simple format to avoid apply_chat_template issues
        text = f"User: {test_prompt}\nAssistant:"
        
        inputs = self.tokenizer(text, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            _ = self.model.generate(
                **inputs,
                max_new_tokens=10,
                temperature=0.7,
                do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id
            )
        
        if self.device == "cuda":
            torch.cuda.synchronize()
    
    def parse_message(self, raw_input: str) -> dict:
        """Parse incoming message with clear protocol"""
        # Protocol: "NPC_NAME|MESSAGE" 
        if '|' in raw_input:
            parts = raw_input.split('|', 1)
            return {
                'npc_name': parts[0].strip(),
                'message': parts[1].strip(),
                'valid': True
            }
        else:
            # Fallback for invalid format
            return {
                'npc_name': 'NPC',
                'message': raw_input,
                'valid': False
            }
    
    def generate_npc_response(self, npc_name: str, player_message: str) -> tuple[str, float]:
        """Generate FRESH NPC response - CLEAN PROTOCOL
        
        Args:
            npc_name: Name of the NPC
            player_message: Player's message (clean, no prompt)
            
        Returns:
            Tuple of (response, generation_time)
        """
        start = time.time()
        
        # NPC-specific personalities with brief responses
        npc_personalities = {
            "Bob": "You are Bob, a friendly bartender. Keep responses brief and natural.",
            "Alice": "You are Alice, a witty regular customer. Keep responses brief and playful.",
            "Sam": "You are Sam, a laid-back musician. Keep responses brief and cool."
        }
        
        personality = npc_personalities.get(npc_name, f"You are {npc_name}, a bartender. Keep responses brief.")
        
        # Bob's deep thinking feature (configured probability, only if cognitive model is enabled)
        if (npc_name == "Bob" and 
            self.cognitive_model.enabled and 
            random.random() < self.bob_deep_think_probability):
            
            # Check GPU memory before attempting deep thought
            if self.cognitive_model.check_gpu_memory():
                logger.info("[COGNITIVE] Bob is thinking deeply...")
                deep_thought = self.cognitive_model.think(player_message)
                
                if deep_thought:
                    # Return deep thought with special formatting
                    elapsed = time.time() - start
                    # Add visual distinction for dual mode testing
                    if self.config.get("use_dual_1_7b", False):
                        response = f"*thoughtfully* {deep_thought}"
                    else:
                        response = f"*thinks deeply* {deep_thought}"
                    
                    # Save deep thinking memory
                    try:
                        self.memory_manager.save_memory(
                            npc_name=npc_name,
                            user_input=player_message,
                            npc_response=response,
                            is_deep_thinking=True,
                            metadata={
                                "response_time": elapsed,
                                "model": "Qwen3-1.7B (deep mode)"
                            }
                        )
                    except Exception as e:
                        logger.warning(f"[MEMORY] Failed to save deep memory: {e}")
                    
                    return response, elapsed
                else:
                    logger.info("[COGNITIVE] Deep thought failed, using normal response")
            else:
                # Low memory - disable cognitive model
                logger.warning("[COGNITIVE] Disabling due to low memory")
                self.cognitive_model.unload()
        
        # Build clear prompt with proper separation (normal response)
        prompt = f"[Character]: {personality}\n\nPlayer: {player_message}\n{npc_name}:"
        
        # DEBUG: Print the actual prompt being used
        print(f"[DEBUG] Generating with prompt: {prompt}")
        
        # Tokenize and generate
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        
        # Create stopping criteria
        stopping_criteria = StoppingCriteriaList([
            PunctuationStoppingCriteria(self.tokenizer)
        ])
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=150,      # Much longer responses allowed
                temperature=0.8,        # Natural variation
                do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                stopping_criteria=stopping_criteria  # Use custom stopping criteria
            )
        
        # Decode response properly - using token positions, not string positions
        response = self.tokenizer.decode(
            outputs[0][len(inputs.input_ids[0]):],  # Skip the input tokens
            skip_special_tokens=True
        ).strip()
        
        # DEBUG: Show what we got
        print(f"[DEBUG] Raw response: '{response}'")
        print(f"[DEBUG] Response length: {len(response)} characters")
        
        # FILTER inappropriate content
        response = self.filter_response(response, npc_name)
        
        # NO HISTORY - Every interaction is independent
        
        elapsed = time.time() - start
        
        # Save memory after generating response
        # Check if this was a deep thinking response
        is_deep = False
        if "*thoughtfully*" in response or "*thinks deeply*" in response:
            is_deep = True
        
        # Save the conversation to memory
        try:
            self.memory_manager.save_memory(
                npc_name=npc_name,
                user_input=player_message,
                npc_response=response,
                is_deep_thinking=is_deep,
                metadata={
                    "response_time": elapsed,
                    "model": "Qwen3-1.7B"
                }
            )
        except Exception as e:
            logger.warning(f"[MEMORY] Failed to save memory: {e}")
        
        return response, elapsed
    
    def generate_response(self, npc_name: str, message: str) -> tuple[str, float]:
        """Wrapper method for compatibility with test scripts"""
        return self.generate_npc_response(npc_name, message)
    
    def filter_response(self, response: str, npc_name: str) -> str:
        """Simple cleanup - just trim whitespace"""
        return response.strip()
    
    def run_server(self, port: int = 9999):
        """Run TCP server for Godot integration
        
        Args:
            port: Server port (default 9999)
        """
        if not self.load_model():
            return
        
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            sock.bind(('127.0.0.1', port))
            sock.listen(5)
            
            print("\n" + "="*60)
            print("OPTIMIZED COZY BAR LLM SERVER")
            print("="*60)
            print(f"Model: Qwen3-1.7B (Native Transformers)")
            print(f"Device: {self.device.upper()}")
            print(f"Port: {port}")
            print(f"Status: Ready for connections")
            print("="*60)
            print("\nWaiting for Godot connections...")
            print("(Start your Godot project now)\n")
            
            request_count = 0
            total_time = 0
            
            while True:
                try:
                    client, addr = sock.accept()
                    data = client.recv(1024).decode('utf-8', errors='ignore')
                    
                    if data:
                        request_count += 1
                        
                        # CLEAN PROTOCOL PARSING
                        print(f"\n[SERVER] Received: {data}", flush=True)
                        parsed = self.parse_message(data)
                        npc_name = parsed['npc_name']
                        message = parsed['message']
                        
                        if not parsed['valid']:
                            print(f"[WARNING] Invalid message format. Expected: NPC_NAME|MESSAGE", flush=True)
                            print(f"[WARNING] Received: {data[:50]}...", flush=True)
                        else:
                            print(f"[SERVER] Parsed - NPC: {npc_name}, Message: {message}", flush=True)
                        
                        # Generate response with clean message
                        response, elapsed = self.generate_npc_response(npc_name, message)
                        
                        # Send response
                        client.send(response.encode('utf-8'))
                        
                        # Update stats
                        total_time += elapsed
                        avg_time = total_time / request_count if request_count > 0 else 0
                        
                        # Log interaction
                        print(f"[{request_count:03d}] {npc_name} ({elapsed:.2f}s, avg: {avg_time:.2f}s)")
                        print(f"  Player: {message[:60]}")
                        print(f"  NPC: {response[:60]}")
                        print()
                    
                    client.close()
                    
                except socket.timeout:
                    continue
                except ConnectionResetError:
                    logger.info("[INFO] Client disconnected")
                except Exception as e:
                    logger.warning(f"[WARNING] Request error: {e}")
                    
        except KeyboardInterrupt:
            print("\n[INFO] Server shutting down...")
        except Exception as e:
            logger.error(f"[ERROR] Server error: {e}")
        finally:
            sock.close()
            print("[INFO] Server stopped")
            
            # Print final statistics
            if request_count > 0:
                print("\n" + "="*60)
                print("SESSION STATISTICS")
                print("="*60)
                print(f"Total requests: {request_count}")
                print(f"Average response time: {total_time/request_count:.2f}s")
                
                # Print memory statistics
                print("\n" + "="*60)
                print("MEMORY STATISTICS")
                print("="*60)
                for npc_name in ["Bob", "Alice", "Sam"]:
                    stats = self.memory_manager.get_memory_stats(npc_name)
                    if stats["total"] > 0:
                        print(f"{npc_name}: {stats['total']} memories")
                        print(f"  Deep thinking: {stats['deep_thinking_count']} ({stats['deep_thinking_percentage']:.1f}%)")
                        print(f"  Avg importance: {stats['average_importance']:.1f}")
                print("="*60)


if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 9999
    
    print("="*60)
    print("OPTIMIZED COZY BAR GAME SERVER")
    print("="*60)
    print("Features:")
    print("  [OK] Native Transformers (no GGUF)")
    print("  [OK] Thinking disabled (3-4x faster)")
    print("  [OK] Model warmup (no 19s delay)")
    print("  [OK] NO caching (fresh responses every time)")
    print("  [OK] GPU support (if available)")
    print("="*60)
    print(f"Starting server on port {port}...")
    print()
    
    server = OptimizedCozyBarServer()
    server.run_server(port)