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
import math
import re
from pathlib import Path
from typing import Dict, Optional, List
from datetime import datetime
from gpt4all import GPT4All
import nltk

nltk.download('stopwords')
from nltk.corpus import stopwords

stop_words = set(stopwords.words('english'))

CLEAN_TEXT_PATTERN = re.compile(r'[^\w\s]')

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

        # Stop word configuration
        self.stop_words = stop_words.copy()
        self.stop_words.update({
            'oh', 'ah', 'uh', 'um', 'hmm', 'well', 'yeah', 'ok', 'okay',
            'really', 'just', 'very', 'quite', 'actually', 'basically',
            'hey', 'hello', 'hi', 'bye', 'thanks', 'please', 'sorry',
            'user', 'system'
        })

        # Character-specific known entities for better personalization
        self.character_entities = {
            "Leonardo": {
                'art', 'painting', 'invention', 'anatomy', 'engineering',
                'canvas', 'brush', 'sketch', 'design', 'renaissance',
                'einstein', 'shakespeare', 'socrates', 'customer', 'bar'
            },
            "Einstein": {
                'physics', 'relativity', 'quantum', 'universe', 'time',
                'space', 'equation', 'theory', 'light', 'energy',
                'leonardo', 'shakespeare', 'socrates', 'customer', 'bar'
            },
            "Shakespeare": {
                'play', 'sonnet', 'drama', 'poetry', 'stage',
                'guitar', 'music', 'tragedy', 'comedy', 'verse',
                'leonardo', 'einstein', 'socrates', 'customer', 'bar'
            },
            "Socrates": {
                'philosophy', 'truth', 'wisdom', 'question', 'knowledge',
                'virtue', 'soul', 'justice', 'ethics', 'dialectic',
                'leonardo', 'einstein', 'shakespeare', 'customer', 'bowl'
            }
        }

        # Shared entities everyone knows about
        self.shared_entities = {
            'bar', 'table', 'customer', 'player', 'dog'
        }

        # For backward compatibility
        self.known_entities = self.shared_entities

        # Keyword frequency tracking
        self.word_frequency = {}
        self.total_word_count = 0

        # No more levels, just direct config
        
    def canonicalize_npc_name(self, name: str) -> str:
        """Prevent NPC memory mixing by standardizing names"""
        if not name:
            return ""
        n = name.strip()
        low = n.lower()
        if low in ["bob", "leonardo", "davinci", "leonardo da vinci", "da vinci"]:
            return "Leonardo"
        if low in ["alice", "einstein", "albert", "albert einstein"]:
            return "Einstein"
        if low in ["sam", "shakespeare", "william", "william shakespeare"]:
            return "Shakespeare"
        if low in ["dog", "socrates"]:
            return "Socrates"
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

    def extract_keywords(self, text: str, speaker: str = None) -> List[str]:
        """Dynamically extract keywords with entity prioritization."""
        if not text:
            return []

        keywords: List[str] = []
        seen = set()
        text_lower = text.lower()

        text_clean = CLEAN_TEXT_PATTERN.sub(' ', text_lower)
        words = text_clean.split()

        for word in words:
            if word in self.known_entities and word not in seen:
                keywords.append(word)
                seen.add(word)

        if speaker:
            speaker_lower = speaker.lower()
            if speaker_lower not in seen and speaker_lower not in ('user', 'system'):
                keywords.append(speaker_lower)
                seen.add(speaker_lower)
                if speaker_lower not in self.known_entities:
                    self.known_entities.add(speaker_lower)
                    logger.debug(f"Learned new entity: {speaker_lower}")

        word_occurrences: Dict[str, int] = {}
        for word in words:
            if word not in self.stop_words and not word.isdigit():
                word_occurrences[word] = word_occurrences.get(word, 0) + 1

        meaningful_words = []
        for word, local_count in word_occurrences.items():
            if word in seen or not 2 < len(word) < 15:
                continue

            global_freq = self.word_frequency.get(word, 0)
            if global_freq == 0:
                rarity = 1.0
            else:
                rarity = 1.0 / (1.0 + math.log(global_freq + 1))

            local_importance = min(local_count * 0.1, 0.3)
            score = rarity + local_importance
            meaningful_words.append((word, score))

        meaningful_words.sort(key=lambda item: item[1], reverse=True)
        remaining_slots = max(0, 10 - len(keywords))
        for word, _ in meaningful_words[:remaining_slots]:
            keywords.append(word)
            seen.add(word)

        self._update_word_frequency(word_occurrences)

        return keywords[:10]

    def _update_word_frequency(self, word_occurrences: Dict[str, int]) -> None:
        """Update global word frequency statistics."""
        for word, count in word_occurrences.items():
            if word not in self.known_entities:
                self.word_frequency[word] = self.word_frequency.get(word, 0) + count
                self.total_word_count += count

        if self.total_word_count > 5000:
            filtered = {word: cnt for word, cnt in self.word_frequency.items() if cnt > 2}
            self.total_word_count = sum(filtered.values())
            self.word_frequency = filtered
            logger.debug(f"Cleaned word frequency: {len(filtered)} words kept")

    def calculate_keyword_relevance(self, query_keywords: List[str], memory_keywords: List[str]) -> float:
        """Compute keyword relevance with entity and rarity bonuses."""
        if not query_keywords or not memory_keywords:
            return 0.0

        query_set = set(query_keywords)
        memory_set = set(memory_keywords)
        intersection = query_set & memory_set

        if not intersection:
            return 0.0

        base_score = len(intersection) / len(query_set)

        bonus = 0.0
        for word in intersection:
            if word in self.known_entities:
                bonus += 0.15

            global_freq = self.word_frequency.get(word, 0)
            if global_freq < 10:
                bonus += 0.1 / (1.0 + global_freq * 0.1)

            try:
                if query_keywords.index(word) < 3 and memory_keywords.index(word) < 3:
                    bonus += 0.05
            except ValueError:
                continue

        return min(base_score + bonus, 1.0)

    def retrieve_relevant_memories(self, npc_name: str, query: str, speaker: str = None, k: int = 5) -> List[dict]:
        """Retrieve memories using keyword, importance, and recency scoring."""
        npc_name = self.canonicalize_npc_name(npc_name)

        if npc_name not in self.memory_cache or not self.memory_cache[npc_name]:
            return []

        memories = self.memory_cache[npc_name]
        query_keywords = self.extract_keywords(query, speaker)

        candidate_memories = []
        for idx, memory in enumerate(memories):
            memory_keywords = memory.get('keywords', [])
            if not memory_keywords:
                composed = f"{memory.get('user_input', '')} {memory.get('npc_response', '')}"
                memory_keywords = self.extract_keywords(composed, memory.get('speaker'))
                memory['keywords'] = memory_keywords

            relevance = self.calculate_keyword_relevance(query_keywords, memory_keywords)
            if relevance > 0:
                candidate_memories.append((idx, memory, relevance))

        if not candidate_memories:
            return memories[-k:] if len(memories) > k else list(memories)

        scored_memories = []
        recency_decay = 0.99
        total_memories = len(memories)

        for idx, memory, keyword_relevance in candidate_memories:
            position = idx + 1
            recency = recency_decay ** (total_memories - position)
            importance = float(memory.get('importance', 3.0)) / 10.0
            relevance = keyword_relevance
            score = relevance * 3 + importance * 2 + recency * 0.5
            scored_memories.append((score, memory))

        scored_memories.sort(key=lambda item: item[0], reverse=True)
        return [memory for score, memory in scored_memories[:k]]

    def load_all_memories(self):
        """Load all NPC memories into cache at startup"""
        logger.info("Loading NPC memories...")
        # Load memories for all historical characters
        for npc_name in ["Leonardo", "Einstein", "Shakespeare", "Socrates", "Bob", "Alice", "Sam"]:
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
    
    def save_memory(
        self,
        npc_name: str,
        user_input: str,
        response: str,
        elapsed_time: float,
        from_speaker: str = "user",
    ) -> None:
        """Save conversation memory with keywords and importance"""
        from datetime import datetime

        npc_name = self.canonicalize_npc_name(npc_name)

        if npc_name not in self.memory_cache:
            self.memory_cache[npc_name] = []

        combined_text = f"{user_input or ''} {response or ''}"
        keywords = self.extract_keywords(combined_text, from_speaker)

        importance = 3.0
        if user_input and '?' in user_input:
            importance += 2.0
        lowered_text = combined_text.lower()
        if any(trigger in lowered_text for trigger in ['important', 'urgent', 'help', 'love', 'hate']):
            importance += 2.0
        importance = min(10.0, importance)

        speaker_name = from_speaker or 'user'
        if speaker_name.lower() not in ('user', ''):
            speaker_name = self.canonicalize_npc_name(speaker_name)

        memory_entry = {
            'timestamp': time.time(),
            'datetime': datetime.now().isoformat(),
            'speaker': speaker_name,
            'listener': npc_name,
            'user_input': user_input,
            'npc_response': response,
            'keywords': keywords,
            'importance': importance,
            'interaction_type': 'user_to_npc' if (from_speaker or '').lower() == 'user' else 'npc_to_npc',
            'metadata': {
                'response_time': elapsed_time,
                'model': self.config['model_file']
            }
        }

        self.memory_cache[npc_name].append(memory_entry)

        max_entries = self.config.get('max_memory_entries', 20)
        if len(self.memory_cache[npc_name]) > max_entries:
            self.memory_cache[npc_name] = self.memory_cache[npc_name][-max_entries:]

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
            current_dir = Path(__file__).parent
            model_path = (current_dir / self.config["model_path"]).resolve()
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
                    "Leonardo": "You are Leonardo da Vinci, Renaissance genius bartender.\nSpeak with curiosity about art, science, and inventions.\nReply with ONE short sentence only.",
                    "Einstein": "You are Albert Einstein, brilliant physicist contemplating in a bar.\nSpeak with gentle humor about relativity and the universe.\nReply with ONE short sentence only.",
                    "Shakespeare": "You are William Shakespeare, the great playwright.\nSpeak dramatically and poetically, with theatrical flair.\nReply with ONE short sentence only.",
                    "Socrates": "You are Socrates, the ancient philosopher as a wise dog.\nAsk thought-provoking questions with barks of wisdom.\nReply with ONE short sentence only."
                }
            else:
                # NPC talking to another NPC - relationship-aware prompts
                # Define relationships
                relationships = {
                    ("Einstein", "Leonardo"): "Leonardo, the Renaissance master",
                    ("Leonardo", "Einstein"): "Einstein, the modern genius",
                    ("Shakespeare", "Leonardo"): "Leonardo, the artistic soul",
                    ("Leonardo", "Shakespeare"): "Shakespeare, the wordsmith",
                    ("Einstein", "Shakespeare"): "Shakespeare, the dramatic poet",
                    ("Shakespeare", "Einstein"): "Einstein, the cosmic thinker",
                    ("Socrates", "Leonardo"): "Leonardo, the polymath",
                    ("Leonardo", "Socrates"): "Socrates, the wise hound",
                    ("Socrates", "Einstein"): "Einstein, the truth seeker",
                    ("Einstein", "Socrates"): "Socrates, the philosopher dog",
                    ("Socrates", "Shakespeare"): "Shakespeare, the bard",
                    ("Shakespeare", "Socrates"): "Socrates, the questioning canine"
                }
                
                # Get appropriate description
                speaker_desc = relationships.get((from_speaker, npc_name), from_speaker)
                
                prompts = {
                    "Leonardo": f"You are Leonardo da Vinci. {speaker_desc} is talking to you.\nRespond with Renaissance curiosity and artistic insight.\nReply with ONE short sentence only.",
                    "Einstein": f"You are Albert Einstein. {speaker_desc} is talking to you.\nRespond with scientific wonder and gentle humor.\nReply with ONE short sentence only.",
                    "Shakespeare": f"You are William Shakespeare. {speaker_desc} is talking to you.\nRespond dramatically with poetic flair.\nReply with ONE short sentence only.",
                    "Socrates": f"You are Socrates, a philosopher in dog form. {speaker_desc} is talking to you.\nRespond with wisdom or a thought-provoking question.\nReply with ONE short sentence only."
                }
            
            system_prompt = prompts.get(npc_name, f"You are {npc_name}. {from_speaker} is talking to you. Reply with ONE short sentences only.")
            
            # Add memory context to system prompt
            if npc_name in self.memory_cache and len(self.memory_cache[npc_name]) > 0:
                relevant_memories = self.retrieve_relevant_memories(
                    npc_name,
                    from_speaker,
                    speaker=from_speaker,
                    k=5
                )

                if relevant_memories:
                    memory_text = "\n\nMost relevant memories:\n"
                    for memory in relevant_memories:
                        speaker_name = memory.get('speaker', 'User')
                        keywords = memory.get('keywords', [])
                        memory_text += (
                            f"- [{', '.join(keywords[:3])}] {speaker_name}: "
                            f"{memory.get('user_input', '')[:50]}...\n"
                        )
                        memory_text += (
                            f"  You: {memory.get('npc_response', '')[:50]}...\n"
                        )
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
                                        ("Einstein", "Leonardo"): "Einstein, the modern genius",
                                        ("Leonardo", "Einstein"): "Leonardo, the Renaissance master",
                                        ("Shakespeare", "Leonardo"): "Shakespeare, the wordsmith",
                                        ("Leonardo", "Shakespeare"): "Leonardo, the artistic soul",
                                        ("Einstein", "Shakespeare"): "Einstein, the cosmic thinker",
                                        ("Shakespeare", "Einstein"): "Shakespeare, the dramatic poet",
                                        ("Socrates", "Leonardo"): "Socrates, the philosopher dog",
                                        ("Leonardo", "Socrates"): "Leonardo, the polymath",
                                        ("Socrates", "Einstein"): "Socrates, the wise hound",
                                        ("Einstein", "Socrates"): "Einstein, the truth seeker",
                                        ("Socrates", "Shakespeare"): "Socrates, the questioning canine",
                                        ("Shakespeare", "Socrates"): "Shakespeare, the bard"
                                    }
                                    speaker_desc = relationships.get((from_speaker, npc_name), from_speaker)
                                    
                                    prompts = {
                                        "Leonardo": f"You are Leonardo da Vinci. {speaker_desc} is talking to you.\nRespond with Renaissance curiosity and artistic insight.\nReply with ONE short sentence only.",
                                        "Einstein": f"You are Albert Einstein. {speaker_desc} is talking to you.\nRespond with scientific wonder and gentle humor.\nReply with ONE short sentence only.",
                                        "Shakespeare": f"You are William Shakespeare. {speaker_desc} is talking to you.\nRespond dramatically with poetic flair.\nReply with ONE short sentence only.",
                                        "Socrates": f"You are Socrates, a philosopher in dog form. {speaker_desc} is talking to you.\nRespond with wisdom or a thought-provoking question.\nReply with ONE short sentence only."
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
                        self.save_memory(npc_name, user_message, full_response.strip(), elapsed_time, from_speaker=from_speaker)
                    
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