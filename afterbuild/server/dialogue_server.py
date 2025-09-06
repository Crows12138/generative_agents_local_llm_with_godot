#!/usr/bin/env python3
"""
Dialogue Server - Direct WebSocket server for NPC dialogue system
Handles all dialogue functionality in a single process
"""

import json
import logging
import asyncio
import websockets
import time
from pathlib import Path
from typing import Dict, Optional
from gpt4all import GPT4All

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)


class DialogueServer:
    """Single server handling all dialogue functionality"""
    
    def __init__(self, config_path: str = "config.json"):
        """Initialize with configuration"""
        self.config = self.load_config(config_path)
        self.model = None
        self.npc_sessions = {}  # Store chat sessions for each NPC
        
        # Memory directory
        self.memory_dir = Path("../npc_memories")
        self.memory_dir.mkdir(exist_ok=True)
        
        # Memory cache for faster access
        self.memory_cache = {}
        self.cache_dirty = {}
        self.load_all_memories()
        
    def canonicalize_npc_name(self, name: str) -> str:
        """Prevent NPC memory mixing by standardizing names"""
        if not name:
            return ""
        n = name.strip()
        low = n.lower()
        if low == "bob":
            return "Bob"
        if low == "alice":
            return "Alice"
        if low == "sam":
            return "Sam"
        return n.title()
    
    def load_config(self, config_path: str) -> dict:
        """Load or create default configuration"""
        config_file = Path(config_path)
        if config_file.exists():
            with open(config_file, 'r') as f:
                return json.load(f)
        else:
            # Default configuration
            default_config = {
                "model_file": "Llama-3.2-3B-Instruct-Q4_0.gguf",
                "model_path": "../../models/llms",
                "device": "gpu",
                "max_tokens": 150,
                "temperature": 0.7,
                "top_k": 40,
                "top_p": 0.9,
                "repeat_penalty": 1.18,
                "repeat_last_n": 64,
                "max_memory_entries": 20,
                "websocket_port": 9999
            }
            with open(config_path, 'w') as f:
                json.dump(default_config, f, indent=2)
            logger.info(f"Created default config at {config_path}")
            return default_config
    
    def load_all_memories(self):
        """Load all NPC memories into cache at startup"""
        logger.info("Loading NPC memories...")
        for npc_name in ["Bob", "Alice", "Sam"]:
            memory_file = self.memory_dir / f"{npc_name}.json"
            if memory_file.exists():
                try:
                    with open(memory_file, 'r', encoding='utf-8') as f:
                        self.memory_cache[npc_name] = json.load(f)
                        logger.info(f"Loaded {len(self.memory_cache[npc_name])} memories for {npc_name}")
                except Exception as e:
                    logger.error(f"Failed to load memories for {npc_name}: {e}")
                    self.memory_cache[npc_name] = []
            else:
                self.memory_cache[npc_name] = []
            self.cache_dirty[npc_name] = False
    
    def save_memory(self, npc_name: str, user_input: str, response: str, elapsed_time: float):
        """Save conversation to memory"""
        from datetime import datetime
        
        npc_name = self.canonicalize_npc_name(npc_name)
        
        if npc_name not in self.memory_cache:
            self.memory_cache[npc_name] = []
        
        # Create memory entry
        memory_entry = {
            "timestamp": time.time(),
            "datetime": datetime.now().isoformat(),
            "user_input": user_input,
            "npc_response": response,
            "importance": 3.0,  # Simple fixed importance for now
            "metadata": {
                "response_time": elapsed_time,
                "model": self.config['model_file']
            }
        }
        
        self.memory_cache[npc_name].append(memory_entry)
        
        # Keep only recent memories
        max_entries = self.config.get("max_memory_entries", 20)
        if len(self.memory_cache[npc_name]) > max_entries:
            self.memory_cache[npc_name] = self.memory_cache[npc_name][-max_entries:]
        
        # Save to disk
        memory_file = self.memory_dir / f"{npc_name}.json"
        try:
            with open(memory_file, 'w', encoding='utf-8') as f:
                json.dump(self.memory_cache[npc_name], f, indent=2, ensure_ascii=False)
            logger.debug(f"Saved memories for {npc_name}")
        except Exception as e:
            logger.error(f"Failed to save memories for {npc_name}: {e}")
    
    def load_model(self):
        """Load GPT4All model"""
        try:
            model_path = Path(self.config["model_path"]).resolve()
            model_file = self.config["model_file"]
            
            logger.info(f"Loading model: {model_file} from {model_path}")
            
            self.model = GPT4All(
                model_name=model_file,
                model_path=str(model_path),
                device=self.config["device"],
                verbose=False
            )
            
            logger.info(f"Model loaded successfully on {self.config['device']}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            return False
    
    def get_or_create_session(self, npc_name: str):
        """Get or create a chat session for an NPC"""
        npc_name = self.canonicalize_npc_name(npc_name)
        
        if npc_name not in self.npc_sessions:
            # Create appropriate system prompt
            prompts = {
                "Bob": "You are Bob, a friendly bartender. Keep responses short and natural.",
                "Alice": "You are Alice, a thoughtful bar regular. Keep responses brief and genuine.",
                "Sam": "You are Sam, a cool musician. Keep responses short and casual."
            }
            system_prompt = prompts.get(npc_name, f"You are {npc_name}. Keep responses brief.")
            
            # Create chat session
            session_context = self.model.chat_session(system_prompt=system_prompt)
            session = session_context.__enter__()
            
            self.npc_sessions[npc_name] = {
                "session": session,
                "session_context": session_context,
                "system_prompt": system_prompt
            }
            
            logger.info(f"Created session for {npc_name}")
        
        return self.npc_sessions[npc_name]
    
    async def handle_websocket(self, websocket):
        """Handle WebSocket connections from Godot"""
        logger.info("WebSocket client connected")
        
        try:
            async for message in websocket:
                try:
                    # Parse message
                    data = json.loads(message)
                    npc_name = data.get("npc", "")
                    user_message = data.get("message", "")
                    
                    # Handle pipe protocol
                    if '|' in user_message:
                        parts = user_message.split('|', 1)
                        npc_name = parts[0]
                        user_message = parts[1]
                    
                    # Canonicalize name
                    npc_name = self.canonicalize_npc_name(npc_name)
                    logger.info(f"[{npc_name}] Received: {user_message}")
                    
                    # Get session
                    npc_data = self.get_or_create_session(npc_name)
                    session = npc_data["session"]
                    
                    # Generate response with streaming
                    start_time = time.time()
                    full_response = ""
                    
                    for token in session.generate(
                        user_message,
                        max_tokens=self.config["max_tokens"],
                        temp=self.config["temperature"],
                        top_k=self.config["top_k"],
                        top_p=self.config["top_p"],
                        repeat_penalty=self.config["repeat_penalty"],
                        repeat_last_n=self.config["repeat_last_n"],
                        streaming=True
                    ):
                        # Send each token
                        await websocket.send(json.dumps({
                            "type": "token",
                            "content": token,
                            "npc": npc_name
                        }))
                        full_response += token
                        await asyncio.sleep(0.02)  # Small delay for smooth streaming
                    
                    # Send completion signal
                    await websocket.send(json.dumps({
                        "type": "complete",
                        "content": full_response.strip(),
                        "npc": npc_name
                    }))
                    
                    # Save to memory
                    elapsed_time = time.time() - start_time
                    self.save_memory(npc_name, user_message, full_response.strip(), elapsed_time)
                    
                    logger.info(f"[{npc_name}] Response in {elapsed_time:.2f}s")
                    
                except json.JSONDecodeError as e:
                    logger.error(f"Invalid JSON: {e}")
                    await websocket.send(json.dumps({
                        "type": "error",
                        "content": "Invalid JSON format"
                    }))
                except Exception as e:
                    logger.error(f"Error processing message: {e}")
                    await websocket.send(json.dumps({
                        "type": "error",
                        "content": str(e)
                    }))
        
        except websockets.exceptions.ConnectionClosed:
            logger.info("WebSocket client disconnected")
        except Exception as e:
            logger.error(f"WebSocket error: {e}")
    
    def cleanup(self):
        """Clean up resources"""
        # Close all chat sessions
        for npc_name, npc_data in self.npc_sessions.items():
            try:
                session_context = npc_data.get("session_context")
                if session_context:
                    session_context.__exit__(None, None, None)
                logger.info(f"Closed session for {npc_name}")
            except Exception as e:
                logger.error(f"Error closing session for {npc_name}: {e}")
    
    async def start_server(self):
        """Start WebSocket server"""
        if not self.load_model():
            return
        
        port = self.config.get("websocket_port", 9999)
        
        print("\n" + "="*60)
        print("DIALOGUE SERVER")
        print("="*60)
        print(f"Model: {self.config['model_file']}")
        print(f"Device: {self.config['device'].upper()}")
        print(f"WebSocket Port: {port}")
        print("="*60)
        print("Waiting for connections...\n")
        
        async with websockets.serve(self.handle_websocket, "127.0.0.1", port):
            await asyncio.Future()  # Run forever
    
    def run(self):
        """Main entry point"""
        try:
            asyncio.run(self.start_server())
        except KeyboardInterrupt:
            print("\nShutting down...")
        except Exception as e:
            logger.error(f"Server error: {e}")
        finally:
            self.cleanup()


if __name__ == "__main__":
    server = DialogueServer()
    server.run()