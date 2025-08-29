#!/usr/bin/env python3
"""
Optimized Cozy Bar LLM Server - Native Transformers Edition
Uses Qwen3-1.7B with all performance optimizations for instant NPC responses
"""

import socket
import time
import sys
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import json
import logging
from typing import Dict, Any

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)


class OptimizedCozyBarServer:
    """Optimized LLM server using native Transformers with proven speedups"""
    
    def __init__(self):
        self.model = None
        self.tokenizer = None
        # NO CACHE, NO HISTORY - Every response is fresh!
        
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
        
        # Build clear prompt with proper separation
        prompt = f"[Character]: {personality}\n\nPlayer: {player_message}\n{npc_name}:"
        
        # DEBUG: Print the actual prompt being used
        print(f"[DEBUG] Generating with prompt: {prompt}")
        
        # Tokenize and generate
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=60,      # Longer responses allowed
                temperature=0.8,        # Natural variation
                do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                stop_strings=[".", "!", "?", "\n"],  # Stop at sentence end
                tokenizer=self.tokenizer  # Needed for stop_strings
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
        
        return response, elapsed
    
    def filter_response(self, response: str, npc_name: str) -> str:
        """Simple cleanup - ensure proper ending"""
        response = response.strip()
        # Add period if missing (due to stop_strings)
        if response and not response[-1] in '.!?':
            response += '.'
        return response
    
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