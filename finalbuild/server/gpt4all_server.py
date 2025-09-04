#!/usr/bin/env python3
"""
GPT4All NPC Server using Llama 3.2 model
Supports continuous conversation with chat sessions
Now with WebSocket support for streaming
"""

import socket
import time
import json
import logging
import asyncio
import websockets
import threading
import re
from pathlib import Path
from typing import Dict, Optional, Any, List
from collections import Counter
from gpt4all import GPT4All

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)


class MemoryImportanceScorer:
    """Multi-dimensional memory importance scoring system"""
    
    def __init__(self):
        """Initialize the scorer with emotion keywords and patterns"""
        self.emotion_keywords = {
            'positive': ['happy', 'love', 'great', 'wonderful', 'amazing', 'excellent', 
                        'fantastic', 'beautiful', 'awesome', 'perfect', 'joy', 'excited'],
            'negative': ['sad', 'angry', 'hate', 'terrible', 'awful', 'horrible', 
                        'disgusting', 'disappointed', 'frustrated', 'upset', 'fear', 'worried'],
            'surprise': ['wow', 'amazing', 'unbelievable', 'incredible', 'shocking', 
                        'surprising', 'unexpected', 'astonishing']
        }
        
        self.relationship_indicators = {
            'positive': ['friend', 'trust', 'help', 'together', 'care', 'support', 
                        'appreciate', 'thank', 'love', 'like'],
            'negative': ['enemy', 'distrust', 'hate', 'avoid', 'dislike', 'angry', 
                        'disappointed', 'betray']
        }
        
        self.depth_indicators = ['because', 'why', 'think', 'feel', 'believe', 
                                 'remember', 'understand', 'realize', 'mean', 'important']
        
        # Common English stop words to filter out
        self.stop_words = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
            'of', 'with', 'by', 'from', 'up', 'about', 'into', 'through', 'during',
            'before', 'after', 'above', 'below', 'between', 'under', 'again',
            'further', 'then', 'once', 'is', 'are', 'was', 'were', 'be', 'have',
            'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should',
            'may', 'might', 'must', 'shall', 'can', 'need', 'dare', 'ought', 'i',
            'you', 'he', 'she', 'it', 'we', 'they', 'them', 'their', 'this', 'that',
            'these', 'those', 'there', 'here', 'where', 'when', 'what', 'which', 'who'
        }
        
        # Cache for historical words to avoid recomputation
        self.historical_words_cache = {}  # Key: (npc_name, history_length), Value: word set
    
    def calculate_importance(self, 
                            user_input: str, 
                            npc_response: str,
                            conversation_history: List[Dict],
                            timestamp: float,
                            npc_name: str = None) -> float:
        """
        Calculate importance score (0-10) based on multiple dimensions
        
        Args:
            user_input: User's message
            npc_response: NPC's response
            conversation_history: Previous conversation entries
            timestamp: Current timestamp
            npc_name: Optional NPC name for caching
            
        Returns:
            importance_score: Float between 0 and 10
        """
        score = 0.0
        
        # 1. New Information Detection (0-3 points)
        score += self._score_new_information(user_input, npc_response, conversation_history, npc_name)
        
        # 2. Emotional Intensity (0-2 points)
        score += self._score_emotion_intensity(user_input + " " + npc_response)
        
        # 3. Conversation Depth (0-2 points)
        score += self._score_conversation_depth(user_input, npc_response)
        
        # 4. Relationship Changes (0-2 points)
        score += self._score_relationship_change(user_input + " " + npc_response, conversation_history)
        
        # 5. Recency Bonus (0-1 point)
        score += self._calculate_recency_bonus(timestamp)
        
        # 6. Apply repetition penalty (check recent 5 messages)
        repetition_penalty = self._calculate_repetition_penalty(user_input, conversation_history[-5:] if len(conversation_history) > 0 else [])
        score *= repetition_penalty
        
        # 7. Apply brevity penalty for very short messages
        if len(user_input.strip()) <= 3:  # "OK", "Yes", "No" etc.
            score *= 0.7  # Reduce score by 30%
        
        # 8. Additional penalty for single-word responses (but not if emotionally significant)
        single_word_politeness = ['thanks', 'thank', 'sorry', 'please', 'ok', 'yes', 'no', 'bye', 'nice', 'good']
        emotional_exceptions = ['love', 'want you', 'need you', 'miss', 'care']
        
        # Check if it's a short message that should be penalized
        is_emotional_exception = any(phrase in user_input.lower() for phrase in emotional_exceptions)
        
        if not is_emotional_exception:
            # Stricter penalty for single words
            if len(user_input.split()) == 1 and user_input.lower() in single_word_politeness:
                score *= 0.5  # 50% reduction for single polite words
            elif len(user_input.split()) <= 2 and any(word in user_input.lower() for word in single_word_politeness):
                score *= 0.6  # 40% reduction for 2-word politeness
        
        # 9. Boost for deep relationship expressions and emotional peaks
        deep_relationship_phrases = ['real friend', 'true friend', 'mean so much', 'care about you', 
                                     'love you', 'miss you', 'changed my life', 'saved me',
                                     'i want you', 'i need you', 'in love with']
        if any(phrase in (user_input + " " + npc_response).lower() for phrase in deep_relationship_phrases):
            score += 2.0  # Significant bonus for deep relationship moments
            # Ensure emotional peaks don't score too low
            score = max(score, 6.0)  # Minimum score of 6 for important emotional moments
        
        # 10. Boost for philosophical/existential topics
        philosophical_keywords = ['meaning', 'purpose', 'death', 'existence', 'consciousness', 
                                  'soul', 'destiny', 'universe', 'philosophy', 'wisdom']
        if sum(1 for word in philosophical_keywords if word in (user_input + " " + npc_response).lower()) >= 2:
            score += 1.5  # Bonus for philosophical depth
        
        return min(10.0, score)  # Cap at 10
    
    def _score_new_information(self, user_input: str, npc_response: str, history: List[Dict], npc_name: str = None) -> float:
        """Score based on information novelty using TF-IDF-like approach with caching and stop word filtering"""
        if not history:
            return 3.0  # First conversation is always important
        
        # Create cache key based on NPC name and history length
        cache_key = (npc_name, len(history)) if npc_name else ('default', len(history))
        
        # Check cache or compute historical words
        if cache_key in self.historical_words_cache:
            historical_words = self.historical_words_cache[cache_key]
        else:
            # Extract all previous text
            historical_text = " ".join([
                entry.get('user_input', '') + " " + entry.get('npc_response', '')
                for entry in history[-10:]  # Check last 10 entries
            ])
            
            # Tokenize and filter stop words
            historical_words = set(re.findall(r'\w+', historical_text.lower()))
            historical_words = historical_words - self.stop_words
            
            # Cache the result
            self.historical_words_cache[cache_key] = historical_words
            
            # Limit cache size to prevent memory issues
            if len(self.historical_words_cache) > 50:
                # Remove oldest cache entries
                oldest_key = next(iter(self.historical_words_cache))
                del self.historical_words_cache[oldest_key]
        
        # Tokenize current exchange and filter stop words
        current_words = set(re.findall(r'\w+', (user_input + " " + npc_response).lower()))
        current_words = current_words - self.stop_words
        
        # Calculate novelty ratio
        if not current_words:
            return 0.0
            
        new_words = current_words - historical_words
        novelty_ratio = len(new_words) / len(current_words)
        
        # Score based on novelty (more meaningful with stop words filtered)
        if novelty_ratio > 0.6:
            return 3.0  # Mostly new meaningful information
        elif novelty_ratio > 0.3:
            return 2.0  # Some new meaningful information
        elif novelty_ratio > 0.15:
            return 1.0  # Little new meaningful information
        return 0.5  # Mostly repetitive
    
    def _score_emotion_intensity(self, text: str) -> float:
        """Score based on emotional content"""
        text_lower = text.lower()
        emotion_count = 0
        
        for emotion_type, keywords in self.emotion_keywords.items():
            for keyword in keywords:
                if keyword in text_lower:
                    emotion_count += 1
        
        # Score based on emotion density
        if emotion_count >= 3:
            return 2.0  # High emotion
        elif emotion_count >= 1:
            return 1.0  # Some emotion
        return 0.0  # No significant emotion
    
    def _score_conversation_depth(self, user_input: str, npc_response: str) -> float:
        """Score based on conversation depth and complexity"""
        combined_text = (user_input + " " + npc_response).lower()
        
        # Check for depth indicators
        depth_score = 0
        for indicator in self.depth_indicators:
            if indicator in combined_text:
                depth_score += 0.3
        
        # Check message length (longer usually means deeper)
        if len(combined_text) > 200:
            depth_score += 0.5
        elif len(combined_text) > 100:
            depth_score += 0.3
        
        # Check for questions (indicates engagement)
        if '?' in combined_text:
            depth_score += 0.5
        
        return min(2.0, depth_score)
    
    def _score_relationship_change(self, text: str, history: List[Dict]) -> float:
        """Score based on relationship status changes"""
        text_lower = text.lower()
        
        # Check for routine politeness (should not score high)
        routine_phrases = ['thanks', 'thank you', 'please', 'sorry', 'excuse me']
        is_routine = any(phrase in text_lower for phrase in routine_phrases) and len(text_lower) < 100
        
        positive_count = sum(1 for word in self.relationship_indicators['positive'] 
                           if word in text_lower)
        negative_count = sum(1 for word in self.relationship_indicators['negative'] 
                           if word in text_lower)
        
        # Apply reduction for routine politeness
        if is_routine:
            positive_count *= 0.5  # Reduce weight of routine positive indicators
        
        # Strong relationship indicator
        if positive_count >= 2 or negative_count >= 2:
            return 2.0 if not is_routine else 1.0
        elif positive_count >= 1 or negative_count >= 1:
            return 1.0 if not is_routine else 0.5
        return 0.0
    
    def _calculate_recency_bonus(self, timestamp: float) -> float:
        """Calculate bonus for recent memories"""
        current_time = time.time()
        hours_ago = (current_time - timestamp) / 3600
        
        if hours_ago < 1:
            return 1.0  # Very recent
        elif hours_ago < 24:
            return 0.5  # Same day
        elif hours_ago < 168:  # One week
            return 0.2
        return 0.0  # Older memories
    
    def _calculate_repetition_penalty(self, user_input: str, recent_history: List[Dict]) -> float:
        """Calculate penalty for repetitive messages"""
        if not recent_history:
            return 1.0  # No penalty for first messages
        
        # Normalize input for comparison
        normalized_input = user_input.lower().strip()
        
        # Check for exact or very similar repetitions
        repetition_count = 0
        for entry in recent_history:
            recent_input = entry.get('user_input', '').lower().strip()
            
            # Exact match
            if normalized_input == recent_input:
                repetition_count += 1
            # Very short and similar (like greetings)
            elif len(normalized_input) < 20 and normalized_input in recent_input or recent_input in normalized_input:
                repetition_count += 0.5
        
        # Apply progressive penalty
        if repetition_count >= 2:
            return 0.5  # Heavy penalty for multiple repetitions
        elif repetition_count >= 1:
            return 0.7  # Moderate penalty for one repetition
        elif repetition_count >= 0.5:
            return 0.85  # Light penalty for similar content
        
        return 1.0  # No penalty


class GPT4AllNPCServer:
    """NPC server using GPT4All with Llama 3.2"""
    
    def __init__(self, config_path: str = "gpt4all_config.json"):
        """Initialize server with configuration"""
        self.config = self.load_config(config_path)
        self.model = None
        self.npc_sessions = {}  # Store chat sessions for each NPC
        
        # Single memory directory - unified storage
        self.memory_dir = Path("../npc_memories")
        self.memory_dir.mkdir(exist_ok=True)
        
        # Memory cache for faster access
        self.memory_cache = {}  # Cache all NPC memories in RAM
        self.cache_dirty = {}  # Track which caches need saving
        self.load_all_memories()  # Load memories at startup
        
        # Initialize memory importance scorer
        self.importance_scorer = MemoryImportanceScorer()
        
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
            # Load unified memory format (detailed format as primary)
            memory_file = self.memory_dir / f"{npc_name}.json"
            if memory_file.exists():
                try:
                    with open(memory_file, 'r', encoding='utf-8') as f:
                        memories = json.load(f)
                        self.memory_cache[npc_name] = memories
                        logger.info(f"Loaded {len(memories)} memories for {npc_name}")
                except Exception as e:
                    logger.error(f"Failed to load memories for {npc_name}: {e}")
                    self.memory_cache[npc_name] = []
            else:
                self.memory_cache[npc_name] = []
            
            self.cache_dirty[npc_name] = False
        
        logger.info("Memory cache loaded successfully")
    
    def get_cached_conversation(self, npc_name: str) -> list:
        """Get conversation from cache in GPT4All format"""
        if npc_name in self.memory_cache:
            # Convert detailed format to GPT4All conversation format
            return self.convert_to_gpt4all_format(self.memory_cache[npc_name])
        return []
    
    def convert_to_gpt4all_format(self, detailed_memories: list) -> list:
        """Convert detailed memory format to GPT4All conversation format"""
        conversation = []
        for entry in detailed_memories:
            conversation.append({
                "user": entry.get("user_input", ""),
                "assistant": entry.get("npc_response", ""),
                "timestamp": entry.get("timestamp", time.time()),
                "response_time": entry.get("metadata", {}).get("response_time", 0)
            })
        return conversation
    
    def get_memories(self, npc_name: str, format_type: str = "detailed") -> Any:
        """Get memories in specified format
        
        Args:
            npc_name: Name of the NPC
            format_type: "detailed" (default) or "gpt4all"
        
        Returns:
            Memories in requested format
        """
        if npc_name not in self.memory_cache:
            return [] if format_type == "detailed" else {"npc": npc_name, "conversation": []}
        
        if format_type == "detailed":
            return self.memory_cache[npc_name]
        elif format_type == "gpt4all":
            return {
                "npc": npc_name,
                "conversation": self.convert_to_gpt4all_format(self.memory_cache[npc_name])
            }
        else:
            logger.warning(f"Unknown format type: {format_type}")
            return self.memory_cache[npc_name]
    
    def update_cache_and_save_async(self, npc_name: str, user_input: str, response: str, elapsed_time: float):
        """Update cache immediately and mark for saving with dynamic importance scoring"""
        from datetime import datetime
        
        # Create unified memory entry (detailed format)
        if npc_name not in self.memory_cache:
            self.memory_cache[npc_name] = []
        
        # Calculate dynamic importance score
        timestamp = time.time()
        importance_score = self.importance_scorer.calculate_importance(
            user_input=user_input,
            npc_response=response,
            conversation_history=self.memory_cache[npc_name],
            timestamp=timestamp,
            npc_name=npc_name  # Pass NPC name for caching
        )
        
        # Log importance score for monitoring
        logger.info(f"[MEMORY] {npc_name}: Importance={importance_score:.2f} for '{user_input[:30]}...'")
        
        memory_entry = {
            "timestamp": timestamp,
            "datetime": datetime.now().isoformat(),
            "user_input": user_input,
            "npc_response": response,
            "is_deep_thinking": False,
            "importance": importance_score,  # Now using dynamic score (0-10)
            "metadata": {
                "response_time": elapsed_time,
                "model": f"{self.config['model_file']} (GPT4All)",
                "importance_details": {
                    "calculated_score": importance_score,
                    "scoring_method": "multi_dimensional"
                }
            }
        }
        
        self.memory_cache[npc_name].append(memory_entry)
        
        # Keep only recent memories in cache
        max_entries = self.config.get("max_conversation_entries", 20)
        if len(self.memory_cache[npc_name]) > max_entries:
            self.memory_cache[npc_name] = self.memory_cache[npc_name][-max_entries:]
        
        # Mark cache as dirty (needs saving)
        self.cache_dirty[npc_name] = True
        
        # Save to disk (can be done in background thread in production)
        self.save_dirty_caches()
    
    def save_dirty_caches(self):
        """Save all dirty caches to disk"""
        for npc_name, is_dirty in self.cache_dirty.items():
            if is_dirty:
                try:
                    # Save unified format (detailed format only)
                    memory_file = self.memory_dir / f"{npc_name}.json"
                    if npc_name in self.memory_cache:
                        with open(memory_file, 'w', encoding='utf-8') as f:
                            json.dump(self.memory_cache[npc_name], f, indent=2, ensure_ascii=False)
                    
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
    
    
    async def handle_websocket(self, websocket):
        """Handle WebSocket client connection for streaming"""
        logger.info(f"WebSocket client connected")
        
        try:
            async for message in websocket:
                try:
                    # Parse JSON message
                    data = json.loads(message)
                    npc_name = data.get("npc", "")
                    user_message = data.get("message", "")
                    
                    # Extract actual message if in NPC|MESSAGE format
                    if '|' in user_message:
                        parts = user_message.split('|', 1)
                        npc_name = parts[0]
                        user_message = parts[1]
                    
                    logger.info(f"[WS][{npc_name}] Received: {user_message}")
                    
                    # Get or create session
                    npc_data = self.get_or_create_session(npc_name)
                    session = npc_data["session"]
                    
                    # Start timing
                    start_time = time.time()
                    full_response = ""
                    
                    # Stream tokens
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
                        # Send each token with small delay for Godot to process
                        await websocket.send(json.dumps({
                            "type": "token",
                            "content": token,
                            "npc": npc_name
                        }))
                        full_response += token
                        # Small delay to ensure Godot can process each update
                        await asyncio.sleep(0.02)  # 20ms delay between tokens
                    
                    # Send completion signal
                    await websocket.send(json.dumps({
                        "type": "complete",
                        "content": full_response.strip(),
                        "npc": npc_name
                    }))
                    
                    # Save conversation
                    elapsed_time = time.time() - start_time
                    self.save_conversation(npc_name, user_message, full_response.strip(), elapsed_time)
                    
                    logger.info(f"[WS][{npc_name}] Streamed response in {elapsed_time:.2f}s")
                    
                except json.JSONDecodeError as e:
                    logger.error(f"Invalid JSON: {e}")
                    await websocket.send(json.dumps({
                        "type": "error",
                        "content": "Invalid JSON format"
                    }))
                except Exception as e:
                    logger.error(f"Error processing WebSocket message: {e}")
                    await websocket.send(json.dumps({
                        "type": "error",
                        "content": str(e)
                    }))
        
        except websockets.exceptions.ConnectionClosed:
            logger.info(f"WebSocket client disconnected")
        except Exception as e:
            logger.error(f"WebSocket connection error: {e}")
    
    async def start_websocket_server(self, host="127.0.0.1", port=9999):
        """Start WebSocket server for streaming"""
        logger.info(f"Starting WebSocket server on ws://{host}:{port}")
        async with websockets.serve(self.handle_websocket, host, port):
            await asyncio.Future()  # Run forever
    
    
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
        """Run WebSocket server only"""
        if not self.load_model():
            return
        
        print("\n" + "="*60)
        print("GPT4ALL WEBSOCKET SERVER")
        print("="*60)
        print(f"Model: {self.config['model_file']}")
        print(f"Device: {self.config['device'].upper()}")
        print(f"Max Tokens: {self.config['max_tokens']}")
        print("="*60)
        print(f"WebSocket Port: {port}")
        print("Waiting for connections...\n")
        
        # Run WebSocket server directly
        try:
            asyncio.run(self.start_websocket_server(port=port))
                
        except KeyboardInterrupt:
            print("\nShutting down...")
        except Exception as e:
            logger.error(f"Server error: {e}")
        finally:
            self.cleanup()


if __name__ == "__main__":
    server = GPT4AllNPCServer()
    server.run_server()