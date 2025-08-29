"""
Dual-Model Dialogue Service for Generative Agents
Integrates Qwen3-1.7B (fast responses) and Qwen3-4B (complex dialogues)
"""

import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import logging
from typing import List, Dict, Optional, Tuple
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)


class DualModelDialogueService:
    """Service that uses two models: 1.7B for simple NPCs, 4B for complex NPCs"""
    
    def __init__(self):
        self.simple_model = None  # Qwen3-1.7B for simple NPCs
        self.complex_model = None  # Qwen3-4B for complex NPCs
        self.simple_tokenizer = None
        self.complex_tokenizer = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.dtype = torch.float16 if self.device == "cuda" else torch.float32
        
        # NPC complexity mapping
        self.complex_npcs = {
            # Important NPCs that need deeper dialogue
            "quest_giver", "boss", "merchant", "storyteller",
            "Isabella Rodriguez", "Klaus Mueller", "Maria Lopez"
        }
        
        # Model loading flags
        self.simple_model_loaded = False
        self.complex_model_loaded = False
        
    def load_models(self, load_simple=True, load_complex=False):
        """Load models based on requirements
        
        Args:
            load_simple: Load the 1.7B model for simple NPCs
            load_complex: Load the 4B model for complex NPCs
        """
        if load_simple and not self.simple_model_loaded:
            logger.info("[DualModel] Loading Qwen3-1.7B for simple NPCs...")
            try:
                self.simple_tokenizer = AutoTokenizer.from_pretrained(
                    "Qwen/Qwen3-1.7B",
                    trust_remote_code=True
                )
                if self.simple_tokenizer.pad_token is None:
                    self.simple_tokenizer.pad_token = self.simple_tokenizer.eos_token
                
                self.simple_model = AutoModelForCausalLM.from_pretrained(
                    "Qwen/Qwen3-1.7B",
                    torch_dtype=self.dtype,
                    device_map=self.device,
                    trust_remote_code=True,
                    low_cpu_mem_usage=True
                )
                
                # Warmup
                self._warmup_model(self.simple_model, self.simple_tokenizer)
                self.simple_model_loaded = True
                logger.info("[DualModel] Qwen3-1.7B loaded successfully")
            except Exception as e:
                logger.error(f"[DualModel] Failed to load 1.7B model: {e}")
                
        if load_complex and not self.complex_model_loaded:
            logger.info("[DualModel] Loading Qwen3-4B for complex NPCs...")
            try:
                self.complex_tokenizer = AutoTokenizer.from_pretrained(
                    "Qwen/Qwen3-4B",
                    trust_remote_code=True
                )
                if self.complex_tokenizer.pad_token is None:
                    self.complex_tokenizer.pad_token = self.complex_tokenizer.eos_token
                
                self.complex_model = AutoModelForCausalLM.from_pretrained(
                    "Qwen/Qwen3-4B",
                    torch_dtype=self.dtype,
                    device_map=self.device,
                    trust_remote_code=True,
                    low_cpu_mem_usage=True
                )
                
                # Warmup
                self._warmup_model(self.complex_model, self.complex_tokenizer)
                self.complex_model_loaded = True
                logger.info("[DualModel] Qwen3-4B loaded successfully")
            except Exception as e:
                logger.error(f"[DualModel] Failed to load 4B model: {e}")
    
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
    
    def is_complex_npc(self, npc_name: str) -> bool:
        """Determine if an NPC should use the complex model"""
        # Check if NPC name contains any complex NPC identifier
        npc_lower = npc_name.lower()
        for complex_name in self.complex_npcs:
            if complex_name.lower() in npc_lower:
                return True
        return False
    
    def generate_dialogue(self, 
                         speaker_name: str, 
                         speaker_description: str, 
                         message: str, 
                         conversation_history: List[Dict] = None) -> str:
        """
        Generate a dialogue response for a speaker.
        Automatically selects the appropriate model based on NPC complexity.
        
        Args:
            speaker_name: Name of the speaker
            speaker_description: Description of the speaker
            message: The message to respond to
            conversation_history: Previous conversation entries
            
        Returns:
            Generated response
        """
        # Determine which model to use
        use_complex = self.is_complex_npc(speaker_name)
        
        # Load model if needed
        if use_complex and not self.complex_model_loaded:
            self.load_models(load_complex=True)
        elif not use_complex and not self.simple_model_loaded:
            self.load_models(load_simple=True)
        
        # Select model and tokenizer
        if use_complex and self.complex_model_loaded:
            model = self.complex_model
            tokenizer = self.complex_tokenizer
            max_tokens = 100  # Longer responses for complex NPCs
            logger.info(f"[DualModel] Using 4B model for {speaker_name}")
        else:
            model = self.simple_model
            tokenizer = self.simple_tokenizer
            max_tokens = 60  # Shorter responses for simple NPCs
            logger.info(f"[DualModel] Using 1.7B model for {speaker_name}")
        
        if not model:
            return f"[{speaker_name}] Sorry, I'm not available right now."
        
        # Build prompt
        personality = f"You are {speaker_name}. {speaker_description} Keep responses brief and natural."
        
        # Add conversation history if available
        context = ""
        if conversation_history:
            for entry in conversation_history[-3:]:  # Last 3 exchanges
                user_msg = entry.get("user", "")
                response = entry.get("response", "")
                context += f"User: {user_msg}\n{speaker_name}: {response}\n"
        
        prompt = f"[Character]: {personality}\n\n{context}User: {message}\n{speaker_name}:"
        
        # Generate response
        try:
            inputs = tokenizer(prompt, return_tensors="pt").to(self.device)
            
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=max_tokens,
                    temperature=0.8 if use_complex else 0.7,
                    do_sample=True,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id
                )
            
            response = tokenizer.decode(
                outputs[0][len(inputs.input_ids[0]):],
                skip_special_tokens=True
            ).strip()
            
            # Clean up response
            if response and not response[-1] in '.!?':
                response += '.'
            
            return response
            
        except Exception as e:
            logger.error(f"[DualModel] Error generating response: {e}")
            return f"[{speaker_name}] I'm having trouble responding right now."
    
    def batch_dialogue(self, dialogues: List[Tuple[str, str, str, List[Dict]]]) -> List[str]:
        """
        Process multiple dialogues, grouping by model type for efficiency.
        
        Args:
            dialogues: List of (speaker_name, speaker_description, message, conversation_history)
            
        Returns:
            List of responses
        """
        responses = []
        
        # Group dialogues by model type
        simple_dialogues = []
        complex_dialogues = []
        
        for i, (speaker_name, desc, msg, hist) in enumerate(dialogues):
            if self.is_complex_npc(speaker_name):
                complex_dialogues.append((i, speaker_name, desc, msg, hist))
            else:
                simple_dialogues.append((i, speaker_name, desc, msg, hist))
        
        # Process each group
        results = [None] * len(dialogues)
        
        # Process simple NPCs with 1.7B model
        if simple_dialogues:
            logger.info(f"[DualModel] Processing {len(simple_dialogues)} simple NPCs with 1.7B")
            for idx, name, desc, msg, hist in simple_dialogues:
                response = self.generate_dialogue(name, desc, msg, hist)
                results[idx] = response
        
        # Process complex NPCs with 4B model
        if complex_dialogues:
            logger.info(f"[DualModel] Processing {len(complex_dialogues)} complex NPCs with 4B")
            for idx, name, desc, msg, hist in complex_dialogues:
                response = self.generate_dialogue(name, desc, msg, hist)
                results[idx] = response
        
        return results


# Global service instance
_dialogue_service = None

def get_dialogue_service() -> DualModelDialogueService:
    """Get or create the global dialogue service instance"""
    global _dialogue_service
    if _dialogue_service is None:
        _dialogue_service = DualModelDialogueService()
        # Load simple model by default
        _dialogue_service.load_models(load_simple=True, load_complex=False)
    return _dialogue_service


# Compatibility functions for existing code
def DialogueService():
    """Compatibility wrapper for existing code"""
    return get_dialogue_service()