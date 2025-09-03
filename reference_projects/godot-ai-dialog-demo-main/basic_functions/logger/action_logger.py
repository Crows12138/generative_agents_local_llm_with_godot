import os
import json
from datetime import datetime
from typing import Dict, Any, Optional

class ActionLogger:
    """
Action logger for recording important character actions to files
"""
    
    def __init__(self, log_dir: str = "logs"):
        self.log_dir = log_dir
        self.ensure_log_directory()
        
        # Action types that need detailed logging
        self.detailed_actions = {
            "talk", "think", "read", "write", "observe", 
            "sleep", "work", "play", "search"
        }
    
    def ensure_log_directory(self):
        """Ensure log directory exists"""
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)
    
    def log_action(self, persona_name: str, action_type: str, content: str, 
                   metadata: Optional[Dict[str, Any]] = None, location: tuple = None):
        """
        Log action to log file
        
        Args:
            persona_name: Character name
            action_type: Action type
            content: Action content
            metadata: Additional metadata
            location: Character location
        """
        if action_type not in self.detailed_actions:
            return
        
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Create log entry
        log_entry = {
            "timestamp": timestamp,
            "persona": persona_name,
            "action": action_type,
            "content": content,
            "location": location,
            "metadata": metadata or {}
        }
        
        # Write to general log file
        self._write_to_general_log(log_entry)
        
        # Write to character-specific log file
        self._write_to_persona_log(persona_name, log_entry)
        
        # Write to action type-specific log file
        self._write_to_action_log(action_type, log_entry)
    
    def _write_to_general_log(self, log_entry: Dict[str, Any]):
        """Write to general log file"""
        log_file = os.path.join(self.log_dir, "all_actions.log")
        self._append_to_file(log_file, log_entry)
    
    def _write_to_persona_log(self, persona_name: str, log_entry: Dict[str, Any]):
        """Write to character-specific log file"""
        safe_name = persona_name.replace(" ", "_").replace("/", "_")
        log_file = os.path.join(self.log_dir, f"{safe_name}_actions.log")
        self._append_to_file(log_file, log_entry)
    
    def _write_to_action_log(self, action_type: str, log_entry: Dict[str, Any]):
        """Write to action type-specific log file"""
        log_file = os.path.join(self.log_dir, f"{action_type}_actions.log")
        self._append_to_file(log_file, log_entry)
    
    def _append_to_file(self, file_path: str, log_entry: Dict[str, Any]):
        """Append log entry to file"""
        try:
            with open(file_path, 'a', encoding='utf-8') as f:
                # Write formatted log entry
                f.write(f"\n{'='*80}\n")
                f.write(f"Time: {log_entry['timestamp']}\n")
                f.write(f"Character: {log_entry['persona']}\n")
                f.write(f"Action: {log_entry['action']}\n")
                f.write(f"Location: {log_entry['location']}\n")
                f.write(f"Content: {log_entry['content']}\n")
                
                if log_entry['metadata']:
                    f.write(f"Metadata: {json.dumps(log_entry['metadata'], ensure_ascii=False, indent=2)}\n")
                
                f.write(f"{'='*80}\n")
        except Exception as e:
            print(f"Failed to write log file: {e}")
    
    def get_conversation_history(self, persona_name: str, limit: int = 10) -> list:
        """Get character's conversation history"""
        safe_name = persona_name.replace(" ", "_").replace("/", "_")
        log_file = os.path.join(self.log_dir, f"{safe_name}_actions.log")
        
        conversations = []
        
        if not os.path.exists(log_file):
            return conversations
        
        try:
            with open(log_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            entries = content.split('='*80)
            
            for entry in entries:
                if "Action: talk" in entry and entry.strip():
                    # Extract dialogue content
                    lines = entry.strip().split('\n')
                    conversation_data = {}
                    
                    for line in lines:
                        if line.startswith("Time: "):
                            conversation_data['timestamp'] = line[6:]
                        elif line.startswith("Content: "):
                            conversation_data['content'] = line[9:]
                        elif line.startswith("Location: "):
                            conversation_data['location'] = line[10:]
                    
                    if conversation_data:
                        conversations.append(conversation_data)
            
            # Return recent conversation records
            return conversations[-limit:] if len(conversations) > limit else conversations
            
        except Exception as e:
            print(f"Failed to get conversation history: {e}")
            return []


# Global logger instance
_action_logger = ActionLogger()

def get_action_logger() -> ActionLogger:
    """Get global action logger"""
    return _action_logger

def log_action(persona_name: str, action_type: str, content: str, 
               metadata: Optional[Dict[str, Any]] = None, location: tuple = None):
    """Convenient logging function"""
    _action_logger.log_action(persona_name, action_type, content, metadata, location)
