"""
Agents Package
Simple AI character system for Godot integration
"""

from .simple_agents import SimpleAgent, EmotionalState, ActivityType, Location, create_demo_characters

__all__ = [
    'SimpleAgent',
    'EmotionalState', 
    'ActivityType',
    'Location',
    'create_demo_characters'
]