"""
Relationship tracking system for AI personas.
This module tracks and manages relationships between personas and objects.
"""

import time
import json
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from enum import Enum

class RelationshipType(Enum):
    FRIEND = "friend"
    ACQUAINTANCE = "acquaintance"
    STRANGER = "stranger"
    ENEMY = "enemy"
    LOVER = "lover"
    FAMILY = "family"
    COLLEAGUE = "colleague"
    MENTOR = "mentor"
    STUDENT = "student"

class InteractionType(Enum):
    CONVERSATION = "conversation"
    HELP = "help"
    CONFLICT = "conflict"
    GIFT = "gift"
    WORK_TOGETHER = "work_together"
    SHARE_MEAL = "share_meal"
    TRAVEL_TOGETHER = "travel_together"
    CELEBRATION = "celebration"

@dataclass
class Interaction:
    """Represents a single interaction between two entities."""
    timestamp: float
    interaction_type: InteractionType
    initiator: str
    target: str
    description: str
    emotional_tone: str  # positive, negative, neutral
    duration_minutes: int = 0
    location: Optional[Tuple[float, float, float]] = None
    metadata: Dict[str, Any] = None

@dataclass
class Relationship:
    """Represents a relationship between two entities."""
    entity_a: str
    entity_b: str
    relationship_type: RelationshipType
    trust_level: float  # 0.0 to 1.0
    familiarity: float  # 0.0 to 1.0
    emotional_bond: float  # -1.0 to 1.0 (negative = dislike, positive = like)
    last_interaction: float
    interaction_count: int
    shared_experiences: List[str]
    metadata: Dict[str, Any] = None
    relationship_strength: float = 0.0  # Calculated from other factors
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}
        if self.shared_experiences is None:
            self.shared_experiences = []
        # Calculate relationship strength
        self.relationship_strength = self._calculate_strength()
    
    def _calculate_strength(self) -> float:
        """Calculate overall relationship strength."""
        # Weighted combination of factors
        weights = {
            'trust': 0.3,
            'familiarity': 0.25,
            'emotional_bond': 0.25,
            'interaction_frequency': 0.2
        }
        
        # Normalize interaction count (assume 100+ interactions = max familiarity)
        interaction_factor = min(self.interaction_count / 100.0, 1.0)
        
        strength = (
            weights['trust'] * self.trust_level +
            weights['familiarity'] * self.familiarity +
            weights['emotional_bond'] * (self.emotional_bond + 1.0) / 2.0 +  # Normalize to 0-1
            weights['interaction_frequency'] * interaction_factor
        )
        
        return max(0.0, min(1.0, strength))

class RelationshipTracker:
    """
    Tracks relationships between personas and objects in the simulation.
    """
    
    def __init__(self):
        self.relationships: Dict[str, Relationship] = {}  # key: "entity_a:entity_b"
        self.interactions: List[Interaction] = []
        self.entity_personalities: Dict[str, str] = {}  # Store personality descriptions
        
    def _get_relationship_key(self, entity_a: str, entity_b: str) -> str:
        """Get consistent key for relationship between two entities."""
        # Sort names to ensure consistent key regardless of order
        sorted_names = sorted([entity_a, entity_b])
        return f"{sorted_names[0]}:{sorted_names[1]}"
    
    def add_interaction(
        self,
        initiator: str,
        target: str,
        interaction_type: InteractionType,
        description: str,
        emotional_tone: str = "neutral",
        duration_minutes: int = 0,
        location: Optional[Tuple[float, float, float]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """Record a new interaction between two entities."""
        
        interaction = Interaction(
            timestamp=time.time(),
            interaction_type=interaction_type,
            initiator=initiator,
            target=target,
            description=description,
            emotional_tone=emotional_tone,
            duration_minutes=duration_minutes,
            location=location,
            metadata=metadata or {}
        )
        
        self.interactions.append(interaction)
        self._update_relationship(interaction)
    
    def _update_relationship(self, interaction: Interaction) -> None:
        """Update relationship based on new interaction."""
        key = self._get_relationship_key(interaction.initiator, interaction.target)
        
        if key not in self.relationships:
            # Create new relationship
            self.relationships[key] = Relationship(
                entity_a=interaction.initiator,
                entity_b=interaction.target,
                relationship_type=RelationshipType.STRANGER,
                trust_level=0.1,
                familiarity=0.1,
                emotional_bond=0.0,
                last_interaction=interaction.timestamp,
                interaction_count=1,
                shared_experiences=[interaction.description],
                metadata=interaction.metadata or {}
            )
        else:
            # Update existing relationship
            rel = self.relationships[key]
            rel.interaction_count += 1
            rel.last_interaction = interaction.timestamp
            rel.shared_experiences.append(interaction.description)
            
            # Update relationship factors based on interaction
            self._adjust_relationship_factors(rel, interaction)
            
            # Recalculate strength
            rel.relationship_strength = rel._calculate_strength()
    
    def _adjust_relationship_factors(self, relationship: Relationship, interaction: Interaction) -> None:
        """Adjust relationship factors based on interaction type and tone."""
        
        # Base adjustments for different interaction types
        adjustments = {
            InteractionType.CONVERSATION: {
                'familiarity': 0.05,
                'emotional_bond': 0.02 if interaction.emotional_tone == "positive" else -0.02
            },
            InteractionType.HELP: {
                'trust_level': 0.1,
                'emotional_bond': 0.15,
                'familiarity': 0.08
            },
            InteractionType.CONFLICT: {
                'trust_level': -0.15,
                'emotional_bond': -0.2,
                'familiarity': 0.05
            },
            InteractionType.GIFT: {
                'trust_level': 0.08,
                'emotional_bond': 0.12,
                'familiarity': 0.05
            },
            InteractionType.WORK_TOGETHER: {
                'trust_level': 0.12,
                'familiarity': 0.1,
                'emotional_bond': 0.08
            },
            InteractionType.SHARE_MEAL: {
                'familiarity': 0.08,
                'emotional_bond': 0.1
            },
            InteractionType.TRAVEL_TOGETHER: {
                'trust_level': 0.15,
                'familiarity': 0.12,
                'emotional_bond': 0.15
            },
            InteractionType.CELEBRATION: {
                'emotional_bond': 0.2,
                'familiarity': 0.1
            }
        }
        
        # Apply adjustments
        if interaction.interaction_type in adjustments:
            adj = adjustments[interaction.interaction_type]
            for factor, change in adj.items():
                if factor == 'trust_level':
                    relationship.trust_level = max(0.0, min(1.0, relationship.trust_level + change))
                elif factor == 'familiarity':
                    relationship.familiarity = max(0.0, min(1.0, relationship.familiarity + change))
                elif factor == 'emotional_bond':
                    relationship.emotional_bond = max(-1.0, min(1.0, relationship.emotional_bond + change))
        
        # Emotional tone adjustments
        if interaction.emotional_tone == "positive":
            relationship.emotional_bond = min(1.0, relationship.emotional_bond + 0.05)
        elif interaction.emotional_tone == "negative":
            relationship.emotional_bond = max(-1.0, relationship.emotional_bond - 0.05)
    
    def get_relationship(self, entity_a: str, entity_b: str) -> Optional[Relationship]:
        """Get relationship between two entities."""
        key = self._get_relationship_key(entity_a, entity_b)
        return self.relationships.get(key)
    
    def get_relationship_summary(self, entity_a: str, entity_b: str) -> str:
        """Get a human-readable summary of the relationship."""
        rel = self.get_relationship(entity_a, entity_b)
        if not rel:
            return f"{entity_a} and {entity_b} have no recorded relationship."
        
        # Determine relationship description
        if rel.relationship_strength > 0.8:
            status = "very close"
        elif rel.relationship_strength > 0.6:
            status = "close"
        elif rel.relationship_strength > 0.4:
            status = "friendly"
        elif rel.relationship_strength > 0.2:
            status = "acquainted"
        else:
            status = "distant"
        
        # Emotional description
        if rel.emotional_bond > 0.5:
            emotion = "positive"
        elif rel.emotional_bond < -0.5:
            emotion = "negative"
        else:
            emotion = "neutral"
        
        return (
            f"{entity_a} and {entity_b} have a {status} relationship "
            f"({rel.relationship_type.value}). They have interacted {rel.interaction_count} times. "
            f"Their emotional bond is {emotion} (trust: {rel.trust_level:.2f}, "
            f"familiarity: {rel.familiarity:.2f})."
        )
    
    def get_recent_interactions(self, entity_a: str, entity_b: str, hours: int = 24) -> List[Interaction]:
        """Get recent interactions between two entities."""
        cutoff_time = time.time() - (hours * 3600)
        key = self._get_relationship_key(entity_a, entity_b)
        
        recent = []
        for interaction in self.interactions:
            if (interaction.initiator == entity_a and interaction.target == entity_b) or \
               (interaction.initiator == entity_b and interaction.target == entity_a):
                if interaction.timestamp >= cutoff_time:
                    recent.append(interaction)
        
        return sorted(recent, key=lambda x: x.timestamp, reverse=True)
    
    def get_entity_relationships(self, entity: str) -> List[Relationship]:
        """Get all relationships for a specific entity."""
        relationships = []
        for rel in self.relationships.values():
            if rel.entity_a == entity or rel.entity_b == entity:
                relationships.append(rel)
        return sorted(relationships, key=lambda x: x.relationship_strength, reverse=True)
    
    def set_entity_personality(self, entity: str, personality: str) -> None:
        """Set personality description for an entity."""
        self.entity_personalities[entity] = personality
    
    def get_entity_personality(self, entity: str) -> str:
        """Get personality description for an entity."""
        return self.entity_personalities.get(entity, f"{entity} is an entity in this world.")
    
    def get_conversation_context(self, speaker: str, listener: str) -> str:
        """Get conversation context based on relationship."""
        rel = self.get_relationship(speaker, listener)
        if not rel:
            return f"You are talking to {listener} for the first time."
        
        # Get recent interactions
        recent_interactions = self.get_recent_interactions(speaker, listener, hours=24)
        
        context_parts = []
        
        # Relationship status
        context_parts.append(f"You have a {rel.relationship_type.value} relationship with {listener}.")
        
        # Trust and familiarity
        if rel.trust_level > 0.7:
            context_parts.append(f"You trust {listener} deeply.")
        elif rel.trust_level > 0.4:
            context_parts.append(f"You have some trust in {listener}.")
        else:
            context_parts.append(f"You don't know {listener} well yet.")
        
        # Emotional bond
        if rel.emotional_bond > 0.5:
            context_parts.append(f"You have positive feelings toward {listener}.")
        elif rel.emotional_bond < -0.5:
            context_parts.append(f"You have negative feelings toward {listener}.")
        
        # Recent interactions
        if recent_interactions:
            context_parts.append(f"Recently, you have interacted {len(recent_interactions)} times.")
            if recent_interactions:
                last_interaction = recent_interactions[0]
                context_parts.append(f"Last interaction: {last_interaction.description}")
        
        return " ".join(context_parts)
    
    def save_to_file(self, filename: str) -> None:
        """Save relationships and interactions to file."""
        data = {
            'relationships': [asdict(rel) for rel in self.relationships.values()],
            'interactions': [asdict(interaction) for interaction in self.interactions],
            'entity_personalities': self.entity_personalities
        }
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    
    def load_from_file(self, filename: str) -> None:
        """Load relationships and interactions from file."""
        try:
            with open(filename, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Load relationships
            self.relationships.clear()
            for rel_data in data.get('relationships', []):
                rel = Relationship(**rel_data)
                key = self._get_relationship_key(rel.entity_a, rel.entity_b)
                self.relationships[key] = rel
            
            # Load interactions
            self.interactions.clear()
            for interaction_data in data.get('interactions', []):
                interaction = Interaction(**interaction_data)
                self.interactions.append(interaction)
            
            # Load personalities
            self.entity_personalities = data.get('entity_personalities', {})
            
        except FileNotFoundError:
            print(f"Relationship file {filename} not found. Starting with empty relationships.")
        except Exception as e:
            print(f"Error loading relationships from {filename}: {e}")

# Global relationship tracker instance
relationship_tracker = RelationshipTracker() 