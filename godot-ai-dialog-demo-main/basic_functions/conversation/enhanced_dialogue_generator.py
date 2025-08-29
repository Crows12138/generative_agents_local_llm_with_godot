"""
Enhanced dialogue generation system with relationship awareness and contextual understanding.
This module provides sophisticated dialogue generation based on relationships, memories, and context.
"""

import time
import random
from typing import Dict, List, Optional, Tuple, Any
from basic_functions.memory.memory import MemoryEntry, MemoryType
from basic_functions.conversation.relationship_tracker import relationship_tracker, InteractionType
from basic_functions.perception.embedding import get_single_embedding

try:
    from ai_service.ai_service import local_llm_generate
except ImportError:
    def local_llm_generate(prompt: str) -> str:
        return "I understand. Thank you for sharing that with me."

class DialogueContext:
    """Represents the context for a dialogue generation."""
    
    def __init__(
        self,
        speaker: str,
        listener: str,
        speaker_personality: str,
        listener_personality: str,
        relationship_context: str,
        recent_memories: List[MemoryEntry],
        current_location: Tuple[float, float, float],
        current_time: str,
        conversation_history: List[Dict],
        emotional_state: str = "neutral"
    ):
        self.speaker = speaker
        self.listener = listener
        self.speaker_personality = speaker_personality
        self.listener_personality = listener_personality
        self.relationship_context = relationship_context
        self.recent_memories = recent_memories
        self.current_location = current_location
        self.current_time = current_time
        self.conversation_history = conversation_history
        self.emotional_state = emotional_state

class EnhancedDialogueGenerator:
    """
    Enhanced dialogue generator that creates contextually appropriate conversations.
    """
    
    def __init__(self):
        self.conversation_templates = self._load_conversation_templates()
        self.emotional_responses = self._load_emotional_responses()
    
    def _load_conversation_templates(self) -> Dict[str, List[str]]:
        """Load conversation templates for different contexts."""
        return {
            "greeting": [
                "Hello {listener}! How are you doing today?",
                "Hi {listener}! It's great to see you.",
                "Good to see you, {listener}! How have you been?",
                "Hey {listener}! What's new with you?",
                "Greetings, {listener}! How is your day going?"
            ],
            "farewell": [
                "Goodbye, {listener}! Take care!",
                "See you later, {listener}!",
                "Until next time, {listener}!",
                "Farewell, {listener}! Have a great day!",
                "Bye, {listener}! It was nice talking to you!"
            ],
            "question": [
                "What do you think about {topic}?",
                "Have you ever experienced {topic}?",
                "What's your opinion on {topic}?",
                "I'm curious about {topic}. What are your thoughts?",
                "Do you have any insights about {topic}?"
            ],
            "sharing": [
                "I wanted to share something with you: {content}",
                "Listen to this, {listener}: {content}",
                "I thought you might find this interesting: {content}",
                "I've been thinking about {content}. What do you think?",
                "I'd like to tell you about {content}."
            ],
            "reaction": [
                "That's fascinating!",
                "Wow, that's really interesting!",
                "I see what you mean.",
                "That makes a lot of sense.",
                "I appreciate you sharing that with me."
            ]
        }
    
    def _load_emotional_responses(self) -> Dict[str, List[str]]:
        """Load emotional response patterns."""
        return {
            "excited": [
                "That's amazing! I'm so excited to hear that!",
                "Wow! That's fantastic news!",
                "I'm thrilled about this!",
                "This is incredible! I can't wait to hear more!",
                "That's wonderful! I'm so happy for you!"
            ],
            "concerned": [
                "I'm a bit worried about that.",
                "That sounds concerning. Are you okay?",
                "I hope everything works out for you.",
                "That's troubling to hear.",
                "I'm concerned about this situation."
            ],
            "curious": [
                "That's really interesting! Tell me more.",
                "I'd love to hear more about that.",
                "That sounds fascinating. Can you elaborate?",
                "I'm curious to learn more about this.",
                "That's intriguing! What happened next?"
            ],
            "supportive": [
                "I'm here for you.",
                "You can count on my support.",
                "I believe in you.",
                "You're doing great!",
                "I'm proud of you."
            ],
            "neutral": [
                "I understand.",
                "That makes sense.",
                "I see.",
                "Interesting.",
                "I hear you."
            ]
        }
    
    def generate_dialogue(
        self,
        context: DialogueContext,
        message_type: str = "contextual",
        custom_message: str = ""
    ) -> str:
        """
        Generate a dialogue response based on context.
        
        Args:
            context: Dialogue context containing all relevant information
            message_type: Type of message to generate (greeting, question, sharing, etc.)
            custom_message: Custom message content if provided
            
        Returns:
            Generated dialogue response
        """
        
        # Determine the best approach based on context
        if message_type == "greeting":
            return self._generate_greeting(context)
        elif message_type == "question":
            return self._generate_question(context, custom_message)
        elif message_type == "sharing":
            return self._generate_sharing(context, custom_message)
        elif message_type == "reaction":
            return self._generate_reaction(context)
        else:
            return self._generate_contextual_dialogue(context, custom_message)
    
    def _generate_greeting(self, context: DialogueContext) -> str:
        """Generate an appropriate greeting based on relationship."""
        rel = relationship_tracker.get_relationship(context.speaker, context.listener)
        
        if not rel:
            # First meeting
            templates = [
                f"Hello! I'm {context.speaker}. Nice to meet you, {context.listener}!",
                f"Hi there! I don't think we've met. I'm {context.speaker}.",
                f"Greetings! I'm {context.speaker}. It's a pleasure to meet you, {context.listener}."
            ]
        elif rel.relationship_strength > 0.7:
            # Close relationship
            templates = [
                f"Hey {context.listener}! Great to see you!",
                f"Hi {context.listener}! I've missed you!",
                f"Hello my friend! How are you doing, {context.listener}?"
            ]
        elif rel.relationship_strength > 0.4:
            # Friendly relationship
            templates = [
                f"Hi {context.listener}! How are you today?",
                f"Hello {context.listener}! Good to see you again.",
                f"Hey {context.listener}! How have you been?"
            ]
        else:
            # Acquaintance
            templates = [
                f"Hello {context.listener}.",
                f"Hi {context.listener}. How are you?",
                f"Greetings, {context.listener}."
            ]
        
        return random.choice(templates)
    
    def _generate_question(self, context: DialogueContext, topic: str = "") -> str:
        """Generate a question based on context and memories."""
        
        # If no specific topic, generate one from recent memories
        if not topic and context.recent_memories:
            memory = random.choice(context.recent_memories)
            topic = memory.text.split()[:5]  # First 5 words
            topic = " ".join(topic)
        
        if not topic:
            topic = "your day"
        
        question_templates = [
            f"What do you think about {topic}?",
            f"Have you experienced anything similar to {topic}?",
            f"What's your opinion on {topic}?",
            f"I'm curious about {topic}. What are your thoughts?",
            f"Do you have any insights about {topic}?"
        ]
        
        return random.choice(question_templates)
    
    def _generate_sharing(self, context: DialogueContext, content: str = "") -> str:
        """Generate a sharing statement based on recent memories."""
        
        if not content and context.recent_memories:
            memory = random.choice(context.recent_memories)
            content = memory.text
        
        if not content:
            content = "how my day has been going"
        
        sharing_templates = [
            f"I wanted to share something with you: {content}",
            f"Listen to this, {context.listener}: {content}",
            f"I thought you might find this interesting: {content}",
            f"I've been thinking about {content}. What do you think?",
            f"I'd like to tell you about {content}."
        ]
        
        return random.choice(sharing_templates)
    
    def _generate_reaction(self, context: DialogueContext) -> str:
        """Generate an emotional reaction based on context."""
        
        # Determine emotional tone based on relationship and context
        rel = relationship_tracker.get_relationship(context.speaker, context.listener)
        
        if rel and rel.emotional_bond > 0.5:
            emotion = "excited"
        elif rel and rel.emotional_bond < -0.3:
            emotion = "concerned"
        elif context.emotional_state != "neutral":
            emotion = context.emotional_state
        else:
            emotion = "neutral"
        
        responses = self.emotional_responses.get(emotion, self.emotional_responses["neutral"])
        return random.choice(responses)
    
    def _generate_contextual_dialogue(self, context: DialogueContext, custom_message: str = "") -> str:
        """Generate contextual dialogue using AI."""
        
        # Build comprehensive prompt
        prompt = self._build_dialogue_prompt(context, custom_message)
        
        try:
            response = local_llm_generate(prompt)
            return response.strip()
        except Exception as e:
            print(f"Error generating AI dialogue: {e}")
            # Fallback to template-based generation
            return self._generate_fallback_dialogue(context, custom_message)
    
    def _build_dialogue_prompt(self, context: DialogueContext, custom_message: str = "") -> str:
        """Build a comprehensive prompt for AI dialogue generation."""
        
        # Get relationship information
        rel = relationship_tracker.get_relationship(context.speaker, context.listener)
        relationship_info = context.relationship_context
        
        # Get recent memories as context
        memory_context = ""
        if context.recent_memories:
            memory_texts = [mem.text for mem in context.recent_memories[:3]]
            memory_context = f"Recent experiences: {'; '.join(memory_texts)}"
        
        # Get conversation history
        history_context = ""
        if context.conversation_history:
            recent_exchanges = context.conversation_history[-3:]
            history_parts = []
            for exchange in recent_exchanges:
                user_msg = exchange.get("user", "")
                response = exchange.get("response", "")
                if user_msg and response:
                    history_parts.append(f"{context.listener}: {user_msg}\n{context.speaker}: {response}")
            if history_parts:
                history_context = f"Recent conversation:\n{chr(10).join(history_parts)}"
        
        # Build the prompt
        prompt = f"""
You are {context.speaker} talking to {context.listener}.

Your personality: {context.speaker_personality}
{context.listener}'s personality: {context.listener_personality}

Relationship context: {relationship_info}

Current situation:
- Location: {context.current_location}
- Time: {context.current_time}
- Your emotional state: {context.emotional_state}

{memory_context}

{history_context}

{custom_message if custom_message else "Generate a natural, contextual response that fits your personality and relationship with " + context.listener + "."}

Respond as {context.speaker} in a natural, conversational way. Keep your response concise (1-2 sentences) and appropriate for your relationship with {context.listener}.
"""
        
        return prompt
    
    def _generate_fallback_dialogue(self, context: DialogueContext, custom_message: str = "") -> str:
        """Generate fallback dialogue using templates when AI fails."""
        
        if custom_message:
            return custom_message
        
        # Choose a random template based on context
        if not context.conversation_history:
            return self._generate_greeting(context)
        else:
            # Randomly choose a dialogue type
            dialogue_types = ["question", "sharing", "reaction"]
            dialogue_type = random.choice(dialogue_types)
            
            if dialogue_type == "question":
                return self._generate_question(context)
            elif dialogue_type == "sharing":
                return self._generate_sharing(context)
            else:
                return self._generate_reaction(context)
    
    def analyze_conversation_sentiment(self, message: str) -> str:
        """Analyze the sentiment of a conversation message."""
        positive_words = ["good", "great", "wonderful", "amazing", "excellent", "happy", "joy", "love", "excited"]
        negative_words = ["bad", "terrible", "awful", "sad", "angry", "frustrated", "worried", "scared", "hate"]
        
        message_lower = message.lower()
        
        positive_count = sum(1 for word in positive_words if word in message_lower)
        negative_count = sum(1 for word in negative_words if word in message_lower)
        
        if positive_count > negative_count:
            return "positive"
        elif negative_count > positive_count:
            return "negative"
        else:
            return "neutral"
    
    def generate_conversation_summary(self, conversation_history: List[Dict]) -> str:
        """Generate a summary of a conversation."""
        if not conversation_history:
            return "No conversation to summarize."
        
        # Extract key points from conversation
        messages = []
        for exchange in conversation_history:
            user_msg = exchange.get("user", "")
            response = exchange.get("response", "")
            if user_msg and response:
                messages.append(f"User: {user_msg}")
                messages.append(f"Response: {response}")
        
        conversation_text = "\n".join(messages)
        
        try:
            prompt = f"""
Summarize this conversation in 1-2 sentences, focusing on the main topics and emotional tone:

{conversation_text}

Summary:
"""
            summary = local_llm_generate(prompt)
            return summary.strip()
        except Exception as e:
            print(f"Error generating conversation summary: {e}")
            return f"Conversation with {len(conversation_history)} exchanges."

# Global dialogue generator instance
dialogue_generator = EnhancedDialogueGenerator() 