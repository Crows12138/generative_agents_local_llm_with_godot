#!/usr/bin/env python
"""
Standalone LLM Server - No dependencies on other modules
Works directly with llama-cpp-python
"""

import sys
import socket
import time
from llama_cpp import Llama

class StandaloneLLMServer:
    def __init__(self, port=9999):
        self.port = port
        self.model = None
        self.model_path = "models/llms/Qwen3-4B-Instruct-2507-Q4_0.gguf"
        
    def load_model(self):
        """Load the LLM model"""
        try:
            print(f"[Server] Loading model from {self.model_path}...")
            start_time = time.time()
            
            self.model = Llama(
                model_path=self.model_path,
                n_ctx=512,       # Small context for speed
                n_batch=64,      # Optimal batch size
                n_threads=4,     # 4 threads for CPU
                n_gpu_layers=0,  # CPU mode (faster for 4B)
                verbose=False
            )
            
            elapsed = time.time() - start_time
            print(f"[Server] Model loaded in {elapsed:.1f} seconds")
            
            # Warmup
            print("[Server] Warming up...")
            self.model("test", max_tokens=1, echo=False)
            print("[Server] Ready!")
            
            return True
            
        except Exception as e:
            print(f"[Error] Failed to load model: {e}")
            print(f"[Info] Make sure the model file exists at: {self.model_path}")
            return False
    
    def generate_response(self, prompt):
        """Generate response using the model"""
        try:
            start_time = time.time()
            
            # Generate response
            result = self.model(
                prompt,
                max_tokens=50,      # Reasonable response length
                temperature=0.7,
                stop=["\n", "User:", "Human:"],
                echo=False
            )
            
            response = result['choices'][0]['text'].strip()
            elapsed = time.time() - start_time
            
            return response, elapsed
            
        except Exception as e:
            print(f"[Error] Generation failed: {e}")
            return f"Error: {str(e)}", 0
    
    def start_server(self):
        """Start the TCP server"""
        if not self.load_model():
            print("[Error] Cannot start server without model")
            return
        
        # Create socket
        server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        
        try:
            server_socket.bind(('127.0.0.1', self.port))
            server_socket.listen(5)
            
            print(f"\n{'='*60}")
            print(f"LLM Server Running")
            print(f"{'='*60}")
            print(f"Port: {self.port}")
            print(f"Model: Qwen3-4B (Fast)")
            print(f"Status: Waiting for connections...")
            print(f"{'='*60}\n")
            
            request_count = 0
            total_time = 0
            
            while True:
                try:
                    # Accept connection
                    client_socket, client_address = server_socket.accept()
                    
                    # Receive message
                    data = client_socket.recv(4096).decode('utf-8')
                    
                    if data:
                        request_count += 1
                        print(f"\n[{request_count}] Received: {data[:100]}")
                        
                        # Generate response
                        response, elapsed = self.generate_response(data)
                        
                        # Send response
                        client_socket.send(response.encode('utf-8'))
                        
                        # Stats
                        total_time += elapsed
                        avg_time = total_time / request_count
                        
                        print(f"    Response ({elapsed:.2f}s, avg {avg_time:.2f}s): {response[:100]}")
                    
                    client_socket.close()
                    
                except socket.timeout:
                    continue
                except ConnectionResetError:
                    print("[Info] Client disconnected")
                except Exception as e:
                    print(f"[Warning] Connection error: {e}")
                    
        except KeyboardInterrupt:
            print("\n[Server] Shutting down...")
        except Exception as e:
            print(f"[Error] Server error: {e}")
        finally:
            server_socket.close()
            print("[Server] Stopped")

def main():
    # Get port from command line or use default
    port = 9999
    if len(sys.argv) > 1:
        try:
            port = int(sys.argv[1])
        except:
            print(f"Invalid port, using default {port}")
    
    # Start server
    server = StandaloneLLMServer(port)
    server.start_server()

if __name__ == "__main__":
    main()