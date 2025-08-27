#!/usr/bin/env python3
"""Start LLM server for Cozy Bar game - Fast 4B model"""

import socket
import time
import sys
from llama_cpp import Llama

class CozyBarLLMServer:
    def __init__(self):
        self.model = None
        self.conversation_history = {}
        
    def load_model(self):
        """Load 4B model optimized for game AI"""
        print("[COZY BAR] Loading Qwen3-4B model...")
        start = time.time()
        
        try:
            self.model = Llama(
                model_path="models/llms/Qwen3-4B-Instruct-2507-Q4_0.gguf",
                n_ctx=512,       # Small context for speed
                n_batch=64,      # Optimal batch size
                n_threads=4,     # 4 threads for best CPU performance
                n_gpu_layers=0,  # CPU mode (faster for 4B)
                verbose=False,
                seed=42          # Consistent responses
            )
            
            elapsed = time.time() - start
            print(f"[COZY BAR] Model loaded in {elapsed:.1f} seconds")
            
            # Warmup
            print("[COZY BAR] Warming up model...")
            warmup_start = time.time()
            self.model("Hello", max_tokens=5, echo=False)
            warmup_time = time.time() - warmup_start
            print(f"[COZY BAR] Ready! Warmup: {warmup_time:.2f}s")
            
            return True
            
        except Exception as e:
            print(f"[ERROR] Failed to load model: {e}")
            print(f"[INFO] Make sure the model file exists:")
            print(f"       models/llms/Qwen3-4B-Instruct-2507-Q4_0.gguf")
            return False
    
    def generate_npc_response(self, npc_name, player_message):
        """Generate contextual NPC response"""
        start = time.time()
        
        # Get conversation history for this NPC
        history = self.conversation_history.get(npc_name, [])
        
        # Build context-aware prompt
        context = ""
        if history:
            # Include last 2 exchanges for context
            recent = history[-4:] if len(history) >= 4 else history
            for msg in recent:
                context += f"{msg}\n"
        
        # Create NPC-specific prompt
        npc_prompts = {
            "Isabella": "You are Isabella, a friendly bartender. Keep responses brief and natural.",
            "John": "You are John, a regular customer who loves sports. Keep responses brief and casual.",
            "Maria": "You are Maria, a local artist. Keep responses brief and creative."
        }
        
        system = npc_prompts.get(npc_name, f"You are {npc_name}, an NPC in a cozy bar. Keep responses brief.")
        
        # Build full prompt
        prompt = f"{system}\n{context}Player: {player_message}\n{npc_name}:"
        
        # Generate response (optimized for speed)
        result = self.model(
            prompt,
            max_tokens=30,      # Short responses for natural conversation
            temperature=0.8,    # Natural variation
            stop=["\n", "Player:"],  # Stop at natural breaks
            echo=False
        )
        
        response = result['choices'][0]['text'].strip()
        
        # Update history
        history.append(f"Player: {player_message}")
        history.append(f"{npc_name}: {response}")
        # Keep only last 10 exchanges
        self.conversation_history[npc_name] = history[-10:]
        
        elapsed = time.time() - start
        
        return response, elapsed
    
    def run_server(self, port=9997):
        """Run TCP server for Godot integration"""
        if not self.load_model():
            return
        
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            sock.bind(('127.0.0.1', port))
            sock.listen(5)
            
            print("\n" + "="*60)
            print("COZY BAR LLM SERVER")
            print("="*60)
            print(f"Port: {port}")
            print(f"Model: Qwen3-4B (Fast Mode)")
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
                        
                        # Parse NPC name from message (format: "NPC_NAME: message")
                        if ':' in data:
                            npc_name, message = data.split(':', 1)
                            npc_name = npc_name.strip()
                            message = message.strip()
                        else:
                            npc_name = "NPC"
                            message = data
                        
                        # Generate response
                        response, elapsed = self.generate_npc_response(npc_name, message)
                        
                        # Send response
                        client.send(response.encode('utf-8'))
                        
                        # Update stats
                        total_time += elapsed
                        avg_time = total_time / request_count
                        
                        # Log interaction
                        print(f"[{request_count:03d}] {npc_name} ({elapsed:.2f}s, avg: {avg_time:.2f}s)")
                        print(f"  Player: {message[:50]}")
                        print(f"  NPC: {response[:50]}")
                        print()
                    
                    client.close()
                    
                except socket.timeout:
                    continue
                except ConnectionResetError:
                    print("[INFO] Client disconnected")
                except Exception as e:
                    print(f"[WARNING] Request error: {e}")
                    
        except KeyboardInterrupt:
            print("\n[INFO] Server shutting down...")
        except Exception as e:
            print(f"[ERROR] Server error: {e}")
        finally:
            sock.close()
            print("[INFO] Server stopped")

if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 9997
    
    print("="*60)
    print("COZY BAR GAME SERVER LAUNCHER")
    print("="*60)
    print(f"Starting server on port {port}...")
    print()
    
    server = CozyBarLLMServer()
    server.run_server(port)