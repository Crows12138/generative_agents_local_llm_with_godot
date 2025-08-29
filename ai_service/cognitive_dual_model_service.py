"""
Cognitive Dual-Model Service
- Qwen3-1.7B: ALL dialogue responses (fast, for all NPCs)
- Qwen3-4B: Cognitive processing (planning, reflection, decision-making)
"""

import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import logging
from typing import List, Dict, Optional, Tuple, Any
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)


class CognitiveDualModelService:
    """
    Dual model service with clear separation:
    - 1.7B for ALL dialogue/conversation (fast responses)
    - 4B for cognitive operations (thinking, planning, reflection)
    """
    
    def __init__(self):
        self.dialogue_model = None  # Qwen3-1.7B for ALL dialogues
        self.cognitive_model = None  # Qwen3-4B for cognitive processing
        self.dialogue_tokenizer = None
        self.cognitive_tokenizer = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.dtype = torch.float16 if self.device == "cuda" else torch.float32
        
        # Model loading flags
        self.dialogue_model_loaded = False
        self.cognitive_model_loaded = False
        
    def load_dialogue_model(self):
        """Load 1.7B model for dialogue responses"""
        if self.dialogue_model_loaded:
            return True
            
        logger.info("[CognitiveDual] Loading Qwen3-1.7B for dialogue...")
        try:
            self.dialogue_tokenizer = AutoTokenizer.from_pretrained(
                "Qwen/Qwen3-1.7B",
                trust_remote_code=True
            )
            if self.dialogue_tokenizer.pad_token is None:
                self.dialogue_tokenizer.pad_token = self.dialogue_tokenizer.eos_token
            
            self.dialogue_model = AutoModelForCausalLM.from_pretrained(
                "Qwen/Qwen3-1.7B",
                torch_dtype=self.dtype,
                device_map=self.device,
                trust_remote_code=True,
                low_cpu_mem_usage=True
            )
            
            # Warmup
            self._warmup_model(self.dialogue_model, self.dialogue_tokenizer)
            self.dialogue_model_loaded = True
            logger.info("[CognitiveDual] Dialogue model (1.7B) ready!")
            return True
        except Exception as e:
            logger.error(f"[CognitiveDual] Failed to load dialogue model: {e}")
            return False
    
    def load_cognitive_model(self):
        """Load 4B model for cognitive processing"""
        if self.cognitive_model_loaded:
            return True
            
        logger.info("[CognitiveDual] Loading Qwen3-4B for cognitive processing...")
        try:
            self.cognitive_tokenizer = AutoTokenizer.from_pretrained(
                "Qwen/Qwen3-4B",
                trust_remote_code=True
            )
            if self.cognitive_tokenizer.pad_token is None:
                self.cognitive_tokenizer.pad_token = self.cognitive_tokenizer.eos_token
            
            self.cognitive_model = AutoModelForCausalLM.from_pretrained(
                "Qwen/Qwen3-4B",
                torch_dtype=self.dtype,
                device_map=self.device,
                trust_remote_code=True,
                low_cpu_mem_usage=True
            )
            
            # Warmup
            self._warmup_model(self.cognitive_model, self.cognitive_tokenizer)
            self.cognitive_model_loaded = True
            logger.info("[CognitiveDual] Cognitive model (4B) ready!")
            return True
        except Exception as e:
            logger.error(f"[CognitiveDual] Failed to load cognitive model: {e}")
            return False
    
    def _warmup_model(self, model, tokenizer):
        """Warmup model to avoid first inference delay"""
        test_prompt = "Hello"
        text = f"User: {test_prompt}\nAssistant:"
        inputs = tokenizer(text, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            _ = model.generate(
                **inputs,
                max_new_tokens=10,
                temperature=0.7,
                do_sample=True,
                pad_token_id=tokenizer.pad_token_id
            )
        
        if self.device == "cuda":
            torch.cuda.synchronize()
    
    # ========== DIALOGUE FUNCTIONS (1.7B) ==========
    
    def generate_dialogue(self, 
                         npc_name: str, 
                         personality: str, 
                         message: str, 
                         conversation_history: List[Dict] = None) -> str:
        """
        Generate dialogue response using 1.7B model (FAST)
        Used for ALL NPC conversations
        """
        if not self.dialogue_model_loaded:
            self.load_dialogue_model()
        
        if not self.dialogue_model:
            raise RuntimeError(f"Dialogue model not loaded for {npc_name}")
        
        # Build prompt
        prompt = f"[Character]: You are {npc_name}. {personality} Keep responses brief and natural.\n\n"
        
        # Add conversation history if available
        if conversation_history:
            for entry in conversation_history[-3:]:  # Last 3 exchanges
                user_msg = entry.get("user", "")
                response = entry.get("response", "")
                prompt += f"User: {user_msg}\n{npc_name}: {response}\n"
        
        prompt += f"User: {message}\n{npc_name}:"
        
        # Generate with 1.7B model
        try:
            inputs = self.dialogue_tokenizer(prompt, return_tensors="pt").to(self.device)
            
            with torch.no_grad():
                outputs = self.dialogue_model.generate(
                    **inputs,
                    max_new_tokens=60,  # Brief responses
                    temperature=0.7,
                    do_sample=True,
                    pad_token_id=self.dialogue_tokenizer.pad_token_id,
                    eos_token_id=self.dialogue_tokenizer.eos_token_id
                )
            
            response = self.dialogue_tokenizer.decode(
                outputs[0][len(inputs.input_ids[0]):],
                skip_special_tokens=True
            ).strip()
            
            # Clean up
            if response and not response[-1] in '.!?':
                response += '.'
            
            return response
            
        except Exception as e:
            logger.error(f"[CognitiveDual] Dialogue generation error: {e}")
            raise RuntimeError(f"Dialogue generation failed for {npc_name}: {e}")
    
    # ========== COGNITIVE FUNCTIONS (4B) ==========
    
    def generate_json_response(self, prompt: str, max_tokens: int = 100) -> str:
        """
        Generate a JSON response using 4B model for structured outputs.
        Used for decision making, action selection, etc.
        """
        if not self.cognitive_model_loaded:
            self.load_cognitive_model()
        
        if not self.cognitive_model:
            raise RuntimeError("Cognitive model not loaded for JSON generation")
        
        try:
            inputs = self.cognitive_tokenizer(prompt, return_tensors="pt").to(self.device)
            
            with torch.no_grad():
                outputs = self.cognitive_model.generate(
                    **inputs,
                    max_new_tokens=max_tokens,
                    temperature=0.5,  # Lower temperature for more consistent JSON
                    do_sample=True,
                    pad_token_id=self.cognitive_tokenizer.pad_token_id
                )
            
            response = self.cognitive_tokenizer.decode(
                outputs[0][len(inputs.input_ids[0]):],
                skip_special_tokens=True
            ).strip()
            
            return response
            
        except Exception as e:
            logger.error(f"[CognitiveDual] JSON generation error: {e}")
            raise RuntimeError(f"JSON generation failed: {e}")
    
    def cognitive_planning(self, 
                          npc_name: str,
                          current_state: Dict[str, Any],
                          goals: List[str],
                          memories: List[str]) -> str:
        """
        Generate a plan using 4B model (DEEP THINKING)
        Used for NPC planning and decision-making
        """
        if not self.cognitive_model_loaded:
            self.load_cognitive_model()
        
        if not self.cognitive_model:
            raise RuntimeError("Cognitive model not loaded for planning")
        
        # Build cognitive prompt
        prompt = f"""[Cognitive Planning for {npc_name}]
Current State: {current_state}
Goals: {'; '.join(goals)}
Recent Memories: {'; '.join(memories[-5:])}

Generate a detailed action plan:"""
        
        try:
            inputs = self.cognitive_tokenizer(prompt, return_tensors="pt").to(self.device)
            
            with torch.no_grad():
                outputs = self.cognitive_model.generate(
                    **inputs,
                    max_new_tokens=150,  # Longer for planning
                    temperature=0.8,
                    do_sample=True,
                    pad_token_id=self.cognitive_tokenizer.pad_token_id
                )
            
            plan = self.cognitive_tokenizer.decode(
                outputs[0][len(inputs.input_ids[0]):],
                skip_special_tokens=True
            ).strip()
            
            return plan
            
        except Exception as e:
            logger.error(f"[CognitiveDual] Planning error: {e}")
            raise RuntimeError(f"Planning generation failed: {e}")
    
    def cognitive_reflection(self,
                           npc_name: str,
                           experiences: List[str],
                           emotions: Dict[str, float]) -> str:
        """
        Generate reflection/insights using 4B model
        Used for NPC self-reflection and learning
        """
        if not self.cognitive_model_loaded:
            self.load_cognitive_model()
        
        if not self.cognitive_model:
            raise RuntimeError("Cognitive model not loaded for reflection")
        
        # Build reflection prompt
        prompt = f"""[Reflection for {npc_name}]
Recent Experiences: {'; '.join(experiences[-5:])}
Emotional State: {emotions}

Reflect on these experiences and generate insights:"""
        
        try:
            inputs = self.cognitive_tokenizer(prompt, return_tensors="pt").to(self.device)
            
            with torch.no_grad():
                outputs = self.cognitive_model.generate(
                    **inputs,
                    max_new_tokens=100,
                    temperature=0.9,
                    do_sample=True,
                    pad_token_id=self.cognitive_tokenizer.pad_token_id
                )
            
            reflection = self.cognitive_tokenizer.decode(
                outputs[0][len(inputs.input_ids[0]):],
                skip_special_tokens=True
            ).strip()
            
            return reflection
            
        except Exception as e:
            logger.error(f"[CognitiveDual] Reflection error: {e}")
            raise RuntimeError(f"Reflection generation failed: {e}")
    
    def cognitive_decision(self,
                         npc_name: str,
                         situation: str,
                         options: List[str],
                         priorities: List[str]) -> int:
        """
        Make a complex decision using 4B model
        Returns index of chosen option
        """
        if not self.cognitive_model_loaded:
            self.load_cognitive_model()
        
        if not self.cognitive_model:
            raise RuntimeError("Cognitive model not loaded for decision making")
        
        # Build decision prompt
        options_text = "\n".join([f"{i+1}. {opt}" for i, opt in enumerate(options)])
        prompt = f"""[Decision Making for {npc_name}]
Situation: {situation}
Options:
{options_text}
Priorities: {'; '.join(priorities)}

Choose the best option (respond with number only):"""
        
        try:
            inputs = self.cognitive_tokenizer(prompt, return_tensors="pt").to(self.device)
            
            with torch.no_grad():
                outputs = self.cognitive_model.generate(
                    **inputs,
                    max_new_tokens=10,
                    temperature=0.5,  # Lower temperature for decisions
                    do_sample=True,
                    pad_token_id=self.cognitive_tokenizer.pad_token_id
                )
            
            response = self.cognitive_tokenizer.decode(
                outputs[0][len(inputs.input_ids[0]):],
                skip_special_tokens=True
            ).strip()
            
            # Extract number from response
            for i, char in enumerate(response):
                if char.isdigit():
                    choice = int(char) - 1
                    if 0 <= choice < len(options):
                        return choice
            
            raise RuntimeError(f"Decision parsing failed: could not extract choice from '{response}'")
            
        except Exception as e:
            logger.error(f"[CognitiveDual] Decision error: {e}")
            raise RuntimeError(f"Decision making failed: {e}")


# Global service instance
_cognitive_service = None

def get_cognitive_service() -> CognitiveDualModelService:
    """Get or create the global cognitive service instance"""
    global _cognitive_service
    if _cognitive_service is None:
        _cognitive_service = CognitiveDualModelService()
        # Load dialogue model by default (most common use)
        _cognitive_service.load_dialogue_model()
    return _cognitive_service