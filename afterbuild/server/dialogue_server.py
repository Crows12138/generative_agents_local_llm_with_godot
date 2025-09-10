#!/usr/bin/env python3
"""
Dialogue Server - Direct WebSocket server for NPC dialogue system
Handles all dialogue functionality in a single process
Now with Level 1 decision making system
"""

import json
import logging
import asyncio
import websockets
import time
import string
from pathlib import Path
from typing import Dict, Optional, List
from datetime import datetime
from gpt4all import GPT4All

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)


class DialogueServer:
    """Single server handling all dialogue functionality"""
    
    def __init__(self, config_path: str = "config.json", decision_config_path: str = "decision_config.json"):
        """Initialize with configuration"""
        self.config = self.load_config(config_path)
        self.decision_config = self.load_decision_config(decision_config_path)
        self.model = None
        self.npc_sessions = {}  # Store chat sessions for each NPC
        self.decision_sessions = {}  # Store decision sessions (separate from dialogue)
        
        # Step 1: Basic lock for model access
        self.model_lock = asyncio.Lock()
        
        # Memory directory
        self.memory_dir = Path("../npc_memories")
        self.memory_dir.mkdir(exist_ok=True)
        
        # Decision log directory
        self.decision_log_dir = Path("../decision_logs")
        self.decision_log_dir.mkdir(exist_ok=True)
        
        # Memory cache for faster access
        self.memory_cache = {}
        self.cache_dirty = {}
        self.load_all_memories()
        
        # No more levels, just direct config
        
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
    
    def load_decision_config(self, config_path: str) -> dict:
        """Load decision configuration from JSON file"""
        config_file = Path(config_path)
        if config_file.exists():
            with open(config_file, 'r') as f:
                decision_config = json.load(f)
                logger.info(f"Loaded decision config from {config_path}")
                return decision_config
        else:
            logger.error(f"Decision config not found at {config_path}, using defaults")
            # Return minimal default config
            return {
                "valid_actions": ["idle", "walk", "serve", "chat", "clean"],
                "system_prompts": {
                    "strict": "You are a decision bot. Reply with EXACTLY the format: action|target. No explanations. No spaces."
                },
                "npc_prompts": {
                    "default": {
                        "template": "You are {npc}. Choose an action and target.\nActions: {actions}\nTargets: {targets}\nContext: {context}\nReply ONLY with format: action|target"
                    }
                },
                "generation_params": {
                    "max_tokens": 15,
                    "temperature": 0.3,
                    "top_k": 10,
                    "top_p": 0.5
                }
            }
    
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
    
    def get_or_create_session(self, npc_name: str, from_speaker: str = "user"):
        """Get or create a chat session for an NPC dialogue
        Sessions are unique per speaker-NPC pair to maintain separate contexts
        """
        npc_name = self.canonicalize_npc_name(npc_name)
        
        # Special case: system requests are for NPC to generate initial dialogue
        if from_speaker == "system":
            # This is a request for the NPC to generate their own greeting
            # The message contains the prompt for generation
            session_key = f"{npc_name}_generation"
            logger.info(f"Generation request for {npc_name}")
        else:
            # Create unique session key with consistent ordering
            # Always use alphabetical ordering to ensure consistency
            # This ensures Alice->Bob and Bob->Alice use the same session
            # Also ensures user->Bob and Bob->user use the same session
            participants = sorted([from_speaker, npc_name])
            session_key = f"{participants[0]}<->{participants[1]}"
            
            logger.info(f"Session key: {session_key} (speaker: {from_speaker}, target: {npc_name})")
        
        if session_key not in self.npc_sessions:
            # Create appropriate system prompt based on who is speaking
            if from_speaker == "system":
                # For generation requests, use the message as the system prompt
                system_prompt = ""  # Will be set from the message
            elif from_speaker == "user":
                # User talking to NPC - standard prompts
                prompts = {
                    "Bob": "You are Bob, a professional bartender talking to a customer.\nBe friendly.\nReply with ONE short sentence only.",
                    "Alice": "You are Alice, a thoughtful bar regular talking to someone.\nBe creative.\nReply with ONE short sentence only.",
                    "Sam": "You are Sam, a cool musician at the bar.\nBe cool.\nReply with ONE short sentence only."
                }
            else:
                # NPC talking to another NPC - relationship-aware prompts
                # Define relationships
                relationships = {
                    ("Alice", "Bob"): "Alice, a regular customer",
                    ("Bob", "Alice"): "Bob, the bartender",
                    ("Sam", "Bob"): "Sam, the bar musician",
                    ("Bob", "Sam"): "Bob, the bartender",
                    ("Alice", "Sam"): "Alice, a fellow bar regular",
                    ("Sam", "Alice"): "Sam, the bar musician"
                }
                
                # Get appropriate description
                speaker_desc = relationships.get((from_speaker, npc_name), from_speaker)
                
                prompts = {
                    "Bob": f"You are Bob, a professional bartender. {speaker_desc} is talking to you.\nBe friendly.\nReply with ONE short sentence only.",
                    "Alice": f"You are Alice, a bar regular. {speaker_desc} is talking to you.\nBe creative.\nReply with ONE short sentence only.",
                    "Sam": f"You are Sam, the bar musician. {speaker_desc} is talking to you.\nBe cool.\nReply with ONE short sentence only."
                }
            
            system_prompt = prompts.get(npc_name, f"You are {npc_name}. {from_speaker} is talking to you. Reply with ONE short sentences only.")
            
            # Add memory context to system prompt
            if npc_name in self.memory_cache and len(self.memory_cache[npc_name]) > 0:
                memory_text = "\n\nRecent conversations you remember:\n"
                for memory in self.memory_cache[npc_name][-7:]:  # Use last 5 memories
                    memory_text += f"- User said: {memory['user_input']}\n"
                    memory_text += f"  You replied: {memory['npc_response'][:100]}...\n"
                system_prompt += memory_text
            
            
            # Create chat session
            session_context = self.model.chat_session(system_prompt=system_prompt)
            session = session_context.__enter__()
            
            self.npc_sessions[session_key] = {
                "session": session,
                "session_context": session_context,
                "system_prompt": system_prompt
            }
            
            logger.info(f"Created session for {session_key}")
        
        return self.npc_sessions[session_key]
    
    def get_or_create_decision_session(self, npc_name: str):
        """Get or create a decision-only session (separate from dialogue)"""
        session_key = f"{npc_name}_decision"
        
        if session_key not in self.decision_sessions:
            # Get system prompt from config
            system_prompt = self.decision_config["system_prompts"]["strict"]
            
            # Create new independent session
            session_context = self.model.chat_session(system_prompt=system_prompt)
            session = session_context.__enter__()
            
            self.decision_sessions[session_key] = {
                "session": session,
                "session_context": session_context,
                "system_prompt": system_prompt
            }
            
            logger.info(f"Created decision session for {npc_name}")
        
        return self.decision_sessions[session_key]["session"]
    
    def process_decision_output(self, raw_response: str, valid_actions: List[str]) -> str:
        """Process LLM output to extract valid action"""
        # Clean the output
        clean = raw_response.strip().lower()
        
        # Take only the first word
        first_word = clean.split()[0] if clean else "idle"
        
        # Remove punctuation
        first_word = first_word.translate(str.maketrans('', '', string.punctuation))
        
        # Check if valid for this NPC
        if first_word in valid_actions:
            return first_word
        
        # Check action mappings from config
        level_config = self.decision_config.get(self.current_level, {})
        action_mappings = level_config.get("action_mappings", {})
        
        if first_word in action_mappings:
            mapped_action = action_mappings[first_word]
            if mapped_action in valid_actions:
                logger.info(f"Mapped '{first_word}' to '{mapped_action}'")
                return mapped_action
        
        # Try fuzzy matching
        for action in valid_actions:
            if action in first_word or first_word in action:
                logger.info(f"Fuzzy matched '{first_word}' to '{action}'")
                return action
        
        # Default fallback
        logger.warning(f"Invalid action '{raw_response}' -> defaulting to 'idle'")
        return "idle"
    
    def save_decision_log(self, npc_name: str, log_entry: Dict):
        """Save decision log for debugging"""
        # Organize logs by date
        today = datetime.now().strftime("%Y-%m-%d")
        log_file = self.decision_log_dir / f"{npc_name}_{today}.json"
        
        # Load existing logs
        logs = []
        if log_file.exists():
            try:
                with open(log_file, 'r') as f:
                    logs = json.load(f)
            except:
                logs = []
        
        # Append new log
        logs.append(log_entry)
        
        # Save
        with open(log_file, 'w') as f:
            json.dump(logs, f, indent=2, ensure_ascii=False)
        
        # Console output for real-time debugging
        logger.info(f"[DECISION] {npc_name}: '{log_entry['raw_response']}' -> {log_entry['processed_action']} (valid: {log_entry['valid']})")
    
    def make_simple_decision(self, npc_name: str, context: str) -> Dict:
        """Action with target decision (stateless)"""
        
        # Create fresh session for each decision (stateless)
        system_prompt = self.decision_config.get("system_prompts", {}).get("strict", 
            "You are a decision bot. Reply with EXACTLY the format: action|target. No explanations. No spaces.")
        
        # Get config directly from decision_config
        level_config = self.decision_config
        
        # Get NPC-specific targets
        npc_targets = level_config.get("npc_specific_targets", {})
        if npc_name in npc_targets:
            valid_targets = npc_targets[npc_name]
        else:
            valid_targets = npc_targets.get("default", ["bar", "table", "player"])
        
        # Get NPC-specific actions
        npc_actions = level_config.get("npc_specific_actions", {})
        if npc_name in npc_actions:
            valid_actions = npc_actions[npc_name]
        else:
            valid_actions = npc_actions.get("default", level_config["valid_actions"])
        
        # Get NPC-specific prompt template
        npc_prompts = level_config["npc_prompts"]
        if npc_name in npc_prompts:
            prompt_template = npc_prompts[npc_name]["template"]
            examples = npc_prompts[npc_name].get("examples", "")
        else:
            prompt_template = npc_prompts["default"]["template"]
            examples = ""
        
        # Build prompt
        actions_str = ", ".join(valid_actions)
        targets_str = ", ".join(valid_targets)
        prompt = prompt_template.format(
            npc=npc_name,
            actions=actions_str,
            targets=targets_str,
            context=context
        )
        
        # Add examples if available
        if examples:
            prompt = f"{prompt}\n{examples}"
        
        # Get generation parameters
        gen_params = level_config["generation_params"]
        
        # Record start time
        start_time = time.time()
        
        # Generate response using stateless chat session
        # Each request creates a new session context
        with self.model.chat_session(system_prompt=system_prompt) as session:
            raw_response = session.generate(
                prompt,
                max_tokens=gen_params["max_tokens"],
                temp=gen_params["temperature"],
                top_k=gen_params["top_k"],
                top_p=gen_params["top_p"],
                streaming=False
            )
        
        elapsed_time = time.time() - start_time
        
        # Process output
        processed_result = self.process_decision_output_with_target(raw_response, valid_actions, valid_targets, level_config)
        
        # Create detailed log
        decision_log = {
            "timestamp": time.time(),
            "datetime": datetime.now().isoformat(),
            "npc": npc_name,
            "context": context,
            "prompt": prompt,
            "raw_response": raw_response,
            "processed_action": processed_result["action"],
            "processed_target": processed_result["target"],
            "response_time": elapsed_time,
            "valid": processed_result["valid"],
            "valid_actions": valid_actions,
            "valid_targets": valid_targets
        }
        
        # Save log
        self.save_decision_log(npc_name, decision_log)
        
        return {
            "action": processed_result["action"],
            "target": processed_result["target"],
            "raw": raw_response,
            "time": elapsed_time,
            "valid": processed_result["valid"]
        }
    
    def process_decision_output_with_target(self, raw_response: str, valid_actions: List[str], valid_targets: List[str], level_config: Dict) -> Dict:
        """Process decision output (action|target format)"""
        
        # Clean the output
        clean = raw_response.strip().lower()
        
        # Get separator
        separator = level_config.get("separator", "|")
        
        # Default result
        result = {
            "action": "idle",
            "target": "self",
            "valid": False
        }
        
        # Try to split by separator
        if separator in clean:
            parts = clean.split(separator)
            if len(parts) >= 2:
                action_part = parts[0].strip()
                target_part = parts[1].strip()
                
                # Remove punctuation
                action_part = action_part.translate(str.maketrans('', '', string.punctuation))
                target_part = target_part.translate(str.maketrans('', '', string.punctuation))
                
                # Process action
                action_mappings = level_config.get("action_mappings", {})
                if action_part in valid_actions:
                    result["action"] = action_part
                elif action_part in action_mappings and action_mappings[action_part] in valid_actions:
                    result["action"] = action_mappings[action_part]
                    logger.info(f"Mapped action '{action_part}' to '{result['action']}'")
                
                # Process target (case-insensitive check for names)
                target_mappings = level_config.get("target_mappings", {})
                
                # First check exact match
                if target_part in valid_targets:
                    result["target"] = target_part
                # Then check case-insensitive match for proper names
                elif target_part.capitalize() in valid_targets:
                    result["target"] = target_part.capitalize()
                    logger.info(f"Capitalized target '{target_part}' to '{result['target']}'")
                # Check mappings
                elif target_part in target_mappings and target_mappings[target_part] in valid_targets:
                    result["target"] = target_mappings[target_part]
                    logger.info(f"Mapped target '{target_part}' to '{result['target']}'")
                
                # Check if both are valid
                if result["action"] != "idle" and result["target"] != "self":
                    result["valid"] = True
        else:
            # Fallback: try to extract first word as action
            words = clean.split()
            if words:
                first_word = words[0].translate(str.maketrans('', '', string.punctuation))
                action_mappings = level_config.get("action_mappings", {})
                
                if first_word in valid_actions:
                    result["action"] = first_word
                elif first_word in action_mappings and action_mappings[first_word] in valid_actions:
                    result["action"] = action_mappings[first_word]
                
                # Try to find a target in remaining words
                if len(words) > 1:
                    for word in words[1:]:
                        word_clean = word.translate(str.maketrans('', '', string.punctuation))
                        if word_clean in valid_targets:
                            result["target"] = word_clean
                            result["valid"] = (result["action"] != "idle")
                            break
        
        if not result["valid"]:
            logger.warning(f"Invalid Level 2 output: '{raw_response}' -> {result['action']}|{result['target']}")
        
        return result
    
    async def handle_websocket(self, websocket):
        """Handle WebSocket connections from Godot"""
        logger.info("WebSocket client connected")
        
        try:
            async for message in websocket:
                try:
                    # Parse message
                    data = json.loads(message)
                    
                    # Check message type
                    message_type = data.get("type", "dialogue")
                    
                    # Handle decision requests
                    if message_type == "decision":
                        npc_name = self.canonicalize_npc_name(data.get("npc", "Bob"))
                        context = data.get("context", "")
                        
                        logger.info(f"[DECISION REQUEST] {npc_name}: {context}")
                        
                        # Use lock to prevent concurrent model access
                        async with self.model_lock:
                            logger.info(f"[LOCK] {npc_name} decision acquired lock")
                            try:
                                # Make decision - this uses the model
                                result = self.make_simple_decision(npc_name, context)
                                logger.info(f"[LOCK] {npc_name} decision completed")
                            except Exception as e:
                                logger.error(f"[LOCK] {npc_name} decision failed: {e}")
                                result = {"action": "idle", "target": "self", "raw": "error", "time": 0, "valid": False}
                            # Lock automatically released here after try/except
                        
                        logger.info(f"[LOCK] {npc_name} decision lock released")
                        
                        # Send response (outside lock - doesn't use model)
                        await websocket.send(json.dumps({
                            "type": "decision_result",
                            "action": result["action"],
                            "target": result.get("target", "self"),
                            "npc": npc_name,
                            "valid": result.get("valid", False),
                            "debug": {
                                "raw": result["raw"],
                                "time": result["time"]
                            }
                        }))
                        continue
                    
                    # Handle dialogue requests
                    npc_name = data.get("npc", "")
                    user_message = data.get("message", "")
                    from_speaker = data.get("from", "user")  # Who is speaking - default to user
                    
                    # Handle pipe protocol (legacy support)
                    if '|' in user_message:
                        parts = user_message.split('|', 1)
                        npc_name = parts[0]
                        user_message = parts[1]
                    
                    # Canonicalize names
                    npc_name = self.canonicalize_npc_name(npc_name)
                    if from_speaker != "user" and from_speaker != "system":
                        from_speaker = self.canonicalize_npc_name(from_speaker)
                    
                    logger.info(f"[{npc_name}] Received from {from_speaker}: {user_message}")
                    
                    # Use lock for entire dialogue generation
                    start_time = time.time()  # Move start_time outside try block
                    async with self.model_lock:
                        logger.info(f"[LOCK] {npc_name} dialogue acquired lock")
                        try:
                            # Special handling for system requests (NPC generating their own greeting)
                            if from_speaker == "system":
                                # Use the message as the prompt directly
                                system_prompt = user_message
                                # Create a temporary session for generation
                                session_context = self.model.chat_session(system_prompt=system_prompt)
                                session = session_context.__enter__()
                                prompt = "Start a conversation."  # Clear prompt to trigger generation
                            else:
                                # Normal dialogue handling
                                # For NPC-to-NPC dialogue, always create a fresh session to maintain proper prompts
                                if from_speaker != "user":
                                    # Create temporary session for NPC-to-NPC dialogue
                                    relationships = {
                                        ("Alice", "Bob"): "Alice, a regular customer",
                                        ("Bob", "Alice"): "Bob, the bartender",
                                        ("Sam", "Bob"): "Sam, the bar musician",
                                        ("Bob", "Sam"): "Bob, the bartender",
                                        ("Alice", "Sam"): "Alice, a fellow bar regular",
                                        ("Sam", "Alice"): "Sam, the bar musician"
                                    }
                                    speaker_desc = relationships.get((from_speaker, npc_name), from_speaker)
                                    
                                    prompts = {
                                        "Bob": f"You are Bob, a professional bartender. {speaker_desc} is talking to you.\nBe friendly.\nReply with ONE short sentence only.",
                                        "Alice": f"You are Alice, a bar regular. {speaker_desc} is talking to you.\nBe creative.\nReply with ONE short sentence only.",
                                        "Sam": f"You are Sam, the bar musician. {speaker_desc} is talking to you.\nBe cool.\nReply with ONE short sentence only."
                                    }
                                    system_prompt = prompts.get(npc_name, f"You are {npc_name}. Reply with ONE short sentence only.")
                                    
                                    session_context = self.model.chat_session(system_prompt=system_prompt)
                                    session = session_context.__enter__()
                                    prompt = user_message
                                else:
                                    # User dialogue - use persistent session
                                    npc_data = self.get_or_create_session(npc_name, from_speaker)
                                    session = npc_data["session"]
                                    prompt = user_message
                            
                            # Generate response with streaming
                            full_response = ""
                            
                            # Debug: Log the prompt being sent
                            logger.info(f"[DEBUG] Sending prompt to {npc_name}: '{prompt[:100]}...' (length: {len(prompt)})")
                            
                            for token in session.generate(
                                prompt,
                                max_tokens=self.config["max_tokens"],
                                temp=self.config["temperature"],
                                top_k=self.config["top_k"],
                                top_p=self.config["top_p"],
                                repeat_penalty=self.config["repeat_penalty"],
                                repeat_last_n=self.config["repeat_last_n"],
                                streaming=True
                            ):
                                # Send each token (still inside lock to ensure generation completes)
                                await websocket.send(json.dumps({
                                    "type": "token",
                                    "content": token,
                                    "npc": npc_name
                                }))
                                full_response += token
                                await asyncio.sleep(0.02)  # Small delay for smooth streaming
                            
                            # Debug: Log the generated response
                            logger.info(f"[DEBUG] {npc_name} generated: '{full_response[:100]}...' (length: {len(full_response)})")
                            logger.info(f"[LOCK] {npc_name} dialogue completed")
                            
                            # Clean up temporary sessions
                            if 'session_context' in locals() and (from_speaker == "system" or (from_speaker != "user" and from_speaker != "system")):
                                session_context.__exit__(None, None, None)
                                
                        except Exception as e:
                            logger.error(f"[LOCK] {npc_name} dialogue failed: {e}")
                            full_response = "Sorry, I'm having trouble responding right now."
                            # Clean up temporary session if it exists
                            if from_speaker == "system" and 'session_context' in locals():
                                session_context.__exit__(None, None, None)
                        # Lock automatically released here
                    
                    logger.info(f"[LOCK] {npc_name} dialogue lock released")
                    
                    # Send completion signal (outside lock - doesn't use model)
                    await websocket.send(json.dumps({
                        "type": "complete",
                        "content": full_response.strip(),
                        "npc": npc_name,
                        "from": from_speaker  # Include who initiated the dialogue
                    }))
                    
                    # Save to memory (but not for system-generated greetings)
                    elapsed_time = time.time() - start_time
                    if from_speaker != "system":
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
                logger.info(f"Closed dialogue session for {npc_name}")
            except Exception as e:
                logger.error(f"Error closing dialogue session for {npc_name}: {e}")
        
        # Close all decision sessions
        for session_key, session_data in self.decision_sessions.items():
            try:
                session_context = session_data.get("session_context")
                if session_context:
                    session_context.__exit__(None, None, None)
                logger.info(f"Closed decision session for {session_key}")
            except Exception as e:
                logger.error(f"Error closing decision session for {session_key}: {e}")
    
    async def start_server(self):
        """Start WebSocket server"""
        if not self.load_model():
            return
        
        port = self.config.get("websocket_port", 9999)
        
        print("\n" + "="*60)
        print("DIALOGUE & DECISION SERVER")
        print("="*60)
        print(f"Model: {self.config['model_file']}")
        print(f"Device: {self.config['device'].upper()}")
        print(f"WebSocket Port: {port}")
        print(f"Decision Format: action|target")
        if "valid_actions" in self.decision_config:
            print(f"Valid Actions: {', '.join(self.decision_config['valid_actions'])}")
        print("="*60)
        print("Waiting for connections...")
        print("Message types: 'dialogue' (default) or 'decision'\n")
        
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