#!/usr/bin/env python
"""
LLM Server No Cache - Keeps model loaded but generates fresh responses
Run this in background: python llm_server_nocache.py
Then call: python llm_client.py "Your message"
"""

import sys
import os
import time
import json
import socket
import threading
from pathlib import Path

# Add project to path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

class LLMServerNocache:
    """LLM server that keeps model loaded but doesn't cache responses"""
    
    def __init__(self, port=9999):
        self.port = port
        self.model = None
        self.running = False
        self.socket = None
        
    def load_model(self):
        """Load the model once"""
        print("[Server] Loading LLM model...")
        from ai_service.direct_llm_service import DirectLLMService
        
        self.llm = DirectLLMService()
        if self.llm.ensure_model_loaded():
            print("[Server] Model loaded successfully!")
            return True
        else:
            print("[Server] Failed to load model")
            return False
    
    def generate_response(self, message):
        """Generate fresh response every time"""
        try:
            # Always generate new response
            prompt = f"User: {message}\nAssistant:"
            
            response = self.llm.generate_complete(
                prompt,
                max_tokens=50,
                expected_type="conversation"
            )
            
            # Clean response
            import re
            response = re.sub(r'[^\x00-\x7F]+', '', response)
            
            if "\n" in response:
                response = response.split("\n")[0]
            
            return response.strip()
            
        except Exception as e:
            return f"Error: {str(e)}"
    
    def start_server(self):
        """Start the server"""
        # Load model first
        if not self.load_model():
            return
        
        # Create socket
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.socket.bind(('127.0.0.1', self.port))
        self.socket.listen(5)
        
        self.running = True
        print(f"[Server] Listening on port {self.port}")
        print("[Server] Ready for real-time LLM requests (no caching)")
        
        while self.running:
            try:
                client, addr = self.socket.accept()
                threading.Thread(target=self.handle_client, args=(client,)).start()
            except:
                break
    
    def handle_client(self, client):
        """Handle client request"""
        try:
            # Receive message
            data = client.recv(1024).decode('utf-8')
            
            if data:
                start_time = time.time()
                
                # Generate fresh response
                response = self.generate_response(data)
                
                elapsed = time.time() - start_time
                print(f"[Server] Generated in {elapsed:.2f}s: {response[:50]}...")
                
                # Send response
                client.send(response.encode('utf-8'))
            
        except Exception as e:
            print(f"[Server] Error: {e}")
        finally:
            client.close()
    
    def stop(self):
        """Stop the server"""
        self.running = False
        if self.socket:
            self.socket.close()

if __name__ == "__main__":
    server = LLMServerNocache()
    
    try:
        server.start_server()
    except KeyboardInterrupt:
        print("\n[Server] Shutting down...")
        server.stop()