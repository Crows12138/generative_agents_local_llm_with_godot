#!/usr/bin/env python3
"""
GPT4All NPC Server using Llama 3.2 model
Supports continuous conversation with chat sessions
"""

import socket
import time
import json
import logging
from pathlib import Path
from typing import Dict, Optional
from gpt4all import GPT4All

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)


class GPT4AllNPCServer:
    """NPC server using GPT4All with Llama 3.2"""
    
    def __init__(self, config_path: str = "gpt4all_config.json"):
        """Initialize server with configuration"""
        self.config = self.load_config(config_path)
        self.model = None
        self.npc_sessions = {}  # Store chat sessions for each NPC
        self.conversation_dir = Path("npc_gpt4all_conversations")
        self.conversation_dir.mkdir(exist_ok=True)
        
        # Also save to standard Bob.json for compatibility
        self.standard_memory_dir = Path("../npc_memories")
        self.standard_memory_dir.mkdir(exist_ok=True)
        
        # Memory cache for faster access
        self.memory_cache = {}  # Cache all NPC memories in RAM
        self.cache_dirty = {}  # Track which caches need saving
        self.load_all_memories()  # Load memories at startup
        
    def load_config(self, config_path: str) -> dict:
        """Load configuration from JSON file"""
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
                "max_conversation_entries": 20
            }
            # Save default config
            with open(config_path, 'w') as f:
                json.dump(default_config, f, indent=2)
            logger.info(f"Created default config at {config_path}")
            return default_config
    
    def load_all_memories(self):
        """Load all NPC memories into cache at startup"""
        logger.info("Loading all NPC memories into cache...")
        npc_names = ["Bob", "Alice", "Sam"]
        
        for npc_name in npc_names:
            # Load GPT4All conversation format
            conv_file = self.conversation_dir / f"{npc_name}.json"
            if conv_file.exists():
                try:
                    with open(conv_file, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        self.memory_cache[f"{npc_name}_conv"] = data
                        logger.info(f"Loaded {len(data.get('conversation', []))} conversations for {npc_name}")
                except Exception as e:
                    logger.error(f"Failed to load conversations for {npc_name}: {e}")
                    self.memory_cache[f"{npc_name}_conv"] = {"npc": npc_name, "conversation": []}
            else:
                self.memory_cache[f"{npc_name}_conv"] = {"npc": npc_name, "conversation": []}
            
            # Load standard memory format  
            std_file = self.standard_memory_dir / f"{npc_name}.json"
            if std_file.exists():
                try:
                    with open(std_file, 'r', encoding='utf-8') as f:
                        memories = json.load(f)
                        self.memory_cache[f"{npc_name}_std"] = memories
                        logger.info(f"Loaded {len(memories)} standard memories for {npc_name}")
                except Exception as e:
                    logger.error(f"Failed to load standard memories for {npc_name}: {e}")
                    self.memory_cache[f"{npc_name}_std"] = []
            else:
                self.memory_cache[f"{npc_name}_std"] = []
            
            self.cache_dirty[npc_name] = False
        
        logger.info("Memory cache loaded successfully")
    
    def get_cached_conversation(self, npc_name: str) -> list:
        """Get conversation from cache (fast)"""
        cache_key = f"{npc_name}_conv"
        if cache_key in self.memory_cache:
            return self.memory_cache[cache_key].get("conversation", [])
        return []
    
    def update_cache_and_save_async(self, npc_name: str, user_input: str, response: str, elapsed_time: float):
        """Update cache immediately and mark for saving"""
        # Update conversation cache
        cache_key_conv = f"{npc_name}_conv"
        if cache_key_conv not in self.memory_cache:
            self.memory_cache[cache_key_conv] = {"npc": npc_name, "conversation": []}
        
        entry = {
            "timestamp": time.time(),
            "user": user_input,
            "assistant": response,
            "response_time": elapsed_time
        }
        
        self.memory_cache[cache_key_conv]["conversation"].append(entry)
        
        # Keep only recent conversations in cache
        max_entries = self.config.get("max_conversation_entries", 20) * 2
        if len(self.memory_cache[cache_key_conv]["conversation"]) > max_entries:
            self.memory_cache[cache_key_conv]["conversation"] = \
                self.memory_cache[cache_key_conv]["conversation"][-max_entries:]
        
        # Update standard memory cache
        from datetime import datetime
        cache_key_std = f"{npc_name}_std"
        if cache_key_std not in self.memory_cache:
            self.memory_cache[cache_key_std] = []
        
        memory_entry = {
            "timestamp": time.time(),
            "datetime": datetime.now().isoformat(),
            "user_input": user_input,
            "npc_response": response,
            "is_deep_thinking": False,
            "importance": 3.0,
            "metadata": {
                "response_time": elapsed_time,
                "model": f"{self.config['model_file']} (GPT4All)"
            }
        }
        
        self.memory_cache[cache_key_std].append(memory_entry)
        
        # Mark cache as dirty (needs saving)
        self.cache_dirty[npc_name] = True
        
        # Save to disk (can be done in background thread in production)
        self.save_dirty_caches()
    
    def save_dirty_caches(self):
        """Save all dirty caches to disk"""
        for npc_name, is_dirty in self.cache_dirty.items():
            if is_dirty:
                try:
                    # Save conversation format
                    conv_file = self.conversation_dir / f"{npc_name}.json"
                    cache_key_conv = f"{npc_name}_conv"
                    if cache_key_conv in self.memory_cache:
                        with open(conv_file, 'w', encoding='utf-8') as f:
                            json.dump(self.memory_cache[cache_key_conv], f, indent=2, ensure_ascii=False)
                    
                    # Save standard format
                    std_file = self.standard_memory_dir / f"{npc_name}.json"
                    cache_key_std = f"{npc_name}_std"
                    if cache_key_std in self.memory_cache:
                        with open(std_file, 'w', encoding='utf-8') as f:
                            json.dump(self.memory_cache[cache_key_std], f, indent=2, ensure_ascii=False)
                    
                    self.cache_dirty[npc_name] = False
                    logger.debug(f"Saved memories for {npc_name}")
                    
                except Exception as e:
                    logger.error(f"Failed to save memories for {npc_name}: {e}")
    
    def load_model(self):
        """Load GPT4All model"""
        try:
            model_path = Path(self.config["model_path"]).resolve()
            model_file = self.config["model_file"]
            
            logger.info(f"Loading model: {model_file} from {model_path}")
            
            # Initialize GPT4All with the model
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
        if npc_name not in self.npc_sessions:
            # Get appropriate system prompt
            if npc_name.lower() == "bob":
                system_prompt = "You are Bob, a friendly bartender. Reply with ONE short response as Bob only."
            elif npc_name.lower() == "alice":
                system_prompt = "You are Alice, a thoughtful bar regular. Reply with ONE short response as Alice only."
            elif npc_name.lower() == "sam":
                system_prompt = "You are Sam, a cool musician. Reply with ONE short response as Sam only."
            else:
                system_prompt = f"You are {npc_name}."
            
            # Load conversation history
            history = self.load_conversation_history(npc_name)
            
            # Create new chat session as context manager
            session_context = self.model.chat_session(system_prompt=system_prompt)
            
            # Enter the context and get the actual session
            session = session_context.__enter__()
            
            # Store session data
            self.npc_sessions[npc_name] = {
                "session": session,  # This is the actual session object
                "session_context": session_context,  # Keep the context manager for cleanup
                "system_prompt": system_prompt,
                "history": history
            }
            
            # If we have history, feed it to the session
            if history:
                # Use configurable context size (default 7 for balance)
                context_size = self.config.get("active_context_size", 7)
                for entry in history[-context_size:]:
                    try:
                        # Feed history to establish context
                        # We generate with the historical prompt but don't use the response
                        list(session.generate(entry["user"], max_tokens=1, streaming=False))
                    except:
                        pass
            
            logger.info(f"Created new session for {npc_name} with system prompt")
        
        return self.npc_sessions[npc_name]
    
    def create_system_prompt(self, npc_name: str) -> str:
        """Create system prompt for NPC"""
        if npc_name.lower() == "bob":
            return (
                "You are Bob, a friendly bartender in a cozy bar. "
                "You are warm, welcoming, and enjoy chatting with customers. "
                "Keep responses conversational and natural. "
                "Remember details from our conversation. "
            )
        else:
            return f"You are {npc_name}, a helpful assistant."
    
    def load_conversation_history(self, npc_name: str) -> list:
        """Load conversation history from cache (fast)"""
        return self.get_cached_conversation(npc_name)
    
    def save_conversation(self, npc_name: str, user_input: str, response: str, elapsed_time: float):
        """Save conversation using cache system"""
        self.update_cache_and_save_async(npc_name, user_input, response, elapsed_time)
    
    
    def generate_response(self, npc_name: str, user_input: str) -> tuple:
        """Generate response using GPT4All with chat session"""
        start_time = time.time()
        
        try:
            # Debug: Log exact NPC name received
            logger.info(f"generate_response called with npc_name='{npc_name}' (lower='{npc_name.lower()}')")
            
            # Get or create session for this NPC
            npc_data = self.get_or_create_session(npc_name)
            session = npc_data["session"]
            
            # Generate response using the session (maintains context)
            response_tokens = []
            for token in session.generate(
                user_input,  # Just the user input, session handles context
                max_tokens=self.config["max_tokens"],
                temp=self.config["temperature"],
                top_k=self.config["top_k"],
                top_p=self.config["top_p"],
                repeat_penalty=self.config["repeat_penalty"],
                repeat_last_n=self.config["repeat_last_n"],
                streaming=True
            ):
                response_tokens.append(token)
            
            response = ''.join(response_tokens).strip()
            
            # Clean up response
            response = self.clean_response(response)
            
            elapsed_time = time.time() - start_time
            
            # Save conversation
            self.save_conversation(npc_name, user_input, response, elapsed_time)
            
            logger.info(f"[{npc_name}] Generated response in {elapsed_time:.2f}s")
            
            return response, elapsed_time
            
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return "I'm having trouble understanding. Could you rephrase that?", 0.0
    
    def clean_response(self, response: str) -> str:
        """Clean up the response"""
        # Remove any role markers
        for marker in ["User:", "Assistant:", "Human:", "AI:", "Bob:", "Customer:"]:
            if marker in response:
                response = response.split(marker)[0]
        
        # Remove incomplete sentences at the end
        if response and not response[-1] in '.!?':
            # Try to find the last complete sentence
            for punct in ['.', '!', '?']:
                if punct in response:
                    parts = response.rsplit(punct, 1)
                    if len(parts) > 1 and len(parts[0]) > 20:
                        response = parts[0] + punct
                        break
        
        return response.strip()
    
    def handle_client(self, client_socket):
        """Handle client connection"""
        try:
            # Receive message
            data = client_socket.recv(4096).decode('utf-8')
            
            if not data:
                return
            
            # Parse message (format: "NPC_NAME|MESSAGE")
            parts = data.split('|', 1)
            if len(parts) != 2:
                client_socket.send(b"Error: Invalid format")
                return
            
            npc_name, message = parts
            
            logger.info(f"[{npc_name}] Received: {message}")
            
            # Generate response
            response, elapsed_time = self.generate_response(npc_name, message)
            
            logger.info(f"[{npc_name}] Response: {response}")
            
            # Send response
            client_socket.send(response.encode('utf-8'))
            
        except Exception as e:
            logger.error(f"Error handling client: {e}")
            client_socket.send(f"Error: {str(e)}".encode('utf-8'))
        finally:
            client_socket.close()
    
    def cleanup(self):
        """Clean up resources"""
        # Save all dirty caches before shutdown
        logger.info("Saving all memory caches...")
        self.save_dirty_caches()
        
        # Close all chat sessions
        for npc_name, npc_data in self.npc_sessions.items():
            try:
                session_context = npc_data.get("session_context")
                # Properly exit the context manager
                if session_context:
                    session_context.__exit__(None, None, None)
                logger.info(f"Closed session for {npc_name}")
            except Exception as e:
                logger.error(f"Error closing session for {npc_name}: {e}")
    
    def run_server(self, port: int = 9999):
        """Run the server"""
        if not self.load_model():
            return
        
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            sock.bind(('127.0.0.1', port))
            sock.listen(5)
            
            print("\n" + "="*60)
            print("GPT4ALL NPC SERVER")
            print("="*60)
            print(f"Model: {self.config['model_file']}")
            print(f"Device: {self.config['device'].upper()}")
            print(f"Max Tokens: {self.config['max_tokens']}")
            print("="*60)
            print(f"Listening on port {port}...")
            print("Waiting for connections...\n")
            
            while True:
                client, addr = sock.accept()
                logger.info(f"Connection from {addr}")
                self.handle_client(client)
                
        except KeyboardInterrupt:
            print("\nShutting down...")
        except Exception as e:
            logger.error(f"Server error: {e}")
        finally:
            self.cleanup()
            sock.close()


if __name__ == "__main__":
    server = GPT4AllNPCServer()
    server.run_server()