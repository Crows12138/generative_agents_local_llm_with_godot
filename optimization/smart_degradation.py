"""Smart Degradation System for Dynamic Performance Scaling

Automatically adjusts AI complexity based on system load, ensuring consistent
performance even under high demand while maintaining acceptable quality.
"""

import psutil
import time
from typing import Dict, Any, List, Tuple
from enum import Enum
from datetime import datetime, timedelta

class LoadLevel(Enum):
    LOW = "low"
    MEDIUM = "medium" 
    HIGH = "high"
    CRITICAL = "critical"

class SmartDegradation:
    """Intelligent system degradation based on load and performance"""
    
    def __init__(self):
        self.load_configs = {
            LoadLevel.LOW: {
                "max_tokens": 800,
                "modules": "all",
                "cognitive_depth": "full",
                "reflection_frequency": "normal",
                "memory_retrieval": "comprehensive",
                "conversation_length": "natural"
            },
            LoadLevel.MEDIUM: {
                "max_tokens": 400,
                "modules": "essential",
                "cognitive_depth": "standard", 
                "reflection_frequency": "reduced",
                "memory_retrieval": "relevant",
                "conversation_length": "concise"
            },
            LoadLevel.HIGH: {
                "max_tokens": 200,
                "modules": "minimal",
                "cognitive_depth": "basic",
                "reflection_frequency": "rare",
                "memory_retrieval": "recent",
                "conversation_length": "brief"
            },
            LoadLevel.CRITICAL: {
                "max_tokens": 100,
                "modules": "essential_only",
                "cognitive_depth": "reactive",
                "reflection_frequency": "none",
                "memory_retrieval": "none",
                "conversation_length": "minimal"
            }
        }
        
        # Performance tracking
        self.load_history = []
        self.response_times = []
        self.current_load = LoadLevel.LOW
        self.degradation_active = False
        
        # Thresholds
        self.cpu_thresholds = {
            LoadLevel.MEDIUM: 30,
            LoadLevel.HIGH: 60,
            LoadLevel.CRITICAL: 80
        }
        
        self.memory_thresholds = {
            LoadLevel.MEDIUM: 70,  # 70% memory usage
            LoadLevel.HIGH: 85,
            LoadLevel.CRITICAL: 95
        }
        
        self.response_time_thresholds = {
            LoadLevel.MEDIUM: 15.0,  # seconds
            LoadLevel.HIGH: 30.0,
            LoadLevel.CRITICAL: 60.0
        }
    
    def assess_system_load(self) -> LoadLevel:
        """Assess current system load and determine appropriate level"""
        
        # Get system metrics
        cpu_percent = psutil.cpu_percent(interval=1)
        memory_percent = psutil.virtual_memory().percent
        
        # Get recent response time average
        avg_response_time = self._get_avg_response_time()
        
        # Determine load level based on multiple factors
        load_scores = {
            LoadLevel.LOW: 0,
            LoadLevel.MEDIUM: 0,
            LoadLevel.HIGH: 0,
            LoadLevel.CRITICAL: 0
        }
        
        # CPU-based scoring
        if cpu_percent >= self.cpu_thresholds[LoadLevel.CRITICAL]:
            load_scores[LoadLevel.CRITICAL] += 3
        elif cpu_percent >= self.cpu_thresholds[LoadLevel.HIGH]:
            load_scores[LoadLevel.HIGH] += 2
        elif cpu_percent >= self.cpu_thresholds[LoadLevel.MEDIUM]:
            load_scores[LoadLevel.MEDIUM] += 1
        else:
            load_scores[LoadLevel.LOW] += 1
        
        # Memory-based scoring
        if memory_percent >= self.memory_thresholds[LoadLevel.CRITICAL]:
            load_scores[LoadLevel.CRITICAL] += 3
        elif memory_percent >= self.memory_thresholds[LoadLevel.HIGH]:
            load_scores[LoadLevel.HIGH] += 2
        elif memory_percent >= self.memory_thresholds[LoadLevel.MEDIUM]:
            load_scores[LoadLevel.MEDIUM] += 1
        else:
            load_scores[LoadLevel.LOW] += 1
        
        # Response time-based scoring
        if avg_response_time >= self.response_time_thresholds[LoadLevel.CRITICAL]:
            load_scores[LoadLevel.CRITICAL] += 2
        elif avg_response_time >= self.response_time_thresholds[LoadLevel.HIGH]:
            load_scores[LoadLevel.HIGH] += 2
        elif avg_response_time >= self.response_time_thresholds[LoadLevel.MEDIUM]:
            load_scores[LoadLevel.MEDIUM] += 1
        
        # Determine final load level
        max_score = max(load_scores.values())
        for level, score in load_scores.items():
            if score == max_score:
                detected_load = level
                break
        else:
            detected_load = LoadLevel.LOW
        
        # Update tracking
        self._update_load_tracking(detected_load, cpu_percent, memory_percent, avg_response_time)
        
        return detected_load
    
    def get_optimal_config(self, current_load: LoadLevel = None) -> Dict[str, Any]:
        """Get optimal configuration for current load"""
        
        if current_load is None:
            current_load = self.assess_system_load()
        
        self.current_load = current_load
        config = self.load_configs[current_load].copy()
        
        # Add dynamic adjustments
        config.update({
            "load_level": current_load.value,
            "degradation_active": current_load != LoadLevel.LOW,
            "timestamp": datetime.now().isoformat(),
            "system_metrics": self._get_current_metrics()
        })
        
        return config
    
    def should_skip_module(self, module_name: str, current_config: Dict[str, Any] = None) -> bool:
        """Determine if specific cognitive module should be skipped"""
        
        if current_config is None:
            current_config = self.get_optimal_config()
        
        modules_setting = current_config.get("modules", "all")
        
        if modules_setting == "all":
            return False
        elif modules_setting == "essential":
            # Skip non-essential modules
            non_essential = ["reflection", "detailed_planning"]
            return module_name in non_essential
        elif modules_setting == "minimal":
            # Only keep core modules
            essential = ["perception", "conversation", "basic_action"]
            return module_name not in essential
        elif modules_setting == "essential_only":
            # Only keep absolutely critical modules
            critical = ["conversation"]
            return module_name not in critical
        
        return False
    
    def adjust_prompt_for_load(self, original_prompt: str, task_type: str) -> str:
        """Adjust prompt complexity based on current load"""
        
        config = self.get_optimal_config()
        max_tokens = config["max_tokens"]
        
        # Estimate prompt token usage (rough approximation)
        estimated_tokens = len(original_prompt.split()) * 1.3
        
        if estimated_tokens > max_tokens * 0.7:  # If prompt uses >70% of token budget
            # Simplify prompt
            if task_type == "conversation":
                return self._simplify_conversation_prompt(original_prompt)
            elif task_type == "perception":
                return self._simplify_perception_prompt(original_prompt)
            elif task_type == "action":
                return self._simplify_action_prompt(original_prompt)
            else:
                # Generic simplification
                return self._generic_simplify(original_prompt, max_tokens)
        
        return original_prompt
    
    def record_response_time(self, response_time: float):
        """Record response time for load assessment"""
        
        self.response_times.append({
            "time": response_time,
            "timestamp": datetime.now()
        })
        
        # Keep only recent times (last hour)
        cutoff = datetime.now() - timedelta(hours=1)
        self.response_times = [
            rt for rt in self.response_times 
            if rt["timestamp"] > cutoff
        ]
    
    def get_degradation_status(self) -> Dict[str, Any]:
        """Get current degradation status and recommendations"""
        
        current_metrics = self._get_current_metrics()
        config = self.get_optimal_config()
        
        return {
            "current_load_level": self.current_load.value,
            "degradation_active": config["degradation_active"],
            "system_metrics": current_metrics,
            "configuration": config,
            "recommendations": self._get_optimization_recommendations(),
            "load_trend": self._analyze_load_trend()
        }
    
    def _get_avg_response_time(self) -> float:
        """Calculate average response time from recent history"""
        
        if not self.response_times:
            return 0.0
        
        # Get times from last 10 minutes
        cutoff = datetime.now() - timedelta(minutes=10)
        recent_times = [
            rt["time"] for rt in self.response_times
            if rt["timestamp"] > cutoff
        ]
        
        return sum(recent_times) / len(recent_times) if recent_times else 0.0
    
    def _update_load_tracking(self, load_level: LoadLevel, cpu: float, memory: float, response_time: float):
        """Update load tracking history"""
        
        entry = {
            "timestamp": datetime.now(),
            "load_level": load_level,
            "cpu_percent": cpu,
            "memory_percent": memory,
            "avg_response_time": response_time
        }
        
        self.load_history.append(entry)
        
        # Keep only last 100 entries
        if len(self.load_history) > 100:
            self.load_history = self.load_history[-100:]
    
    def _get_current_metrics(self) -> Dict[str, Any]:
        """Get current system metrics"""
        
        return {
            "cpu_percent": psutil.cpu_percent(),
            "memory_percent": psutil.virtual_memory().percent,
            "available_memory_gb": psutil.virtual_memory().available / (1024**3),
            "avg_response_time": self._get_avg_response_time(),
            "active_processes": len(psutil.pids())
        }
    
    def _simplify_conversation_prompt(self, prompt: str) -> str:
        """Simplify conversation prompt for high load"""
        
        # Extract key elements
        if ":" in prompt and "said" in prompt.lower():
            # Find the actual message
            lines = prompt.split('\n')
            speaker_line = None
            message_line = None
            
            for line in lines:
                if "said:" in line.lower() or "says:" in line.lower():
                    message_line = line
                elif "Character:" in line:
                    speaker_line = line
            
            if speaker_line and message_line:
                char_name = speaker_line.split(":")[1].strip()
                return f"{char_name}, respond to: {message_line}"
        
        # Generic simplification
        return prompt[:200] + "..."
    
    def _simplify_perception_prompt(self, prompt: str) -> str:
        """Simplify perception prompt for high load"""
        
        # Extract location and events
        simplified = "Observe and list 3 key things: "
        
        if "Location:" in prompt:
            loc_start = prompt.find("Location:") + 9
            loc_end = prompt.find("\n", loc_start)
            if loc_end > loc_start:
                location = prompt[loc_start:loc_end].strip()
                simplified += f"at {location}"
        
        return simplified
    
    def _simplify_action_prompt(self, prompt: str) -> str:
        """Simplify action prompt for high load"""
        
        return f"Next action: {prompt[:100]}..."
    
    def _generic_simplify(self, prompt: str, max_tokens: int) -> str:
        """Generic prompt simplification"""
        
        # Rough token estimation
        max_chars = max_tokens * 3  # Approximate chars per token
        
        if len(prompt) <= max_chars:
            return prompt
        
        # Truncate and add ellipsis
        return prompt[:max_chars] + "..."
    
    def _get_optimization_recommendations(self) -> List[str]:
        """Generate optimization recommendations"""
        
        recommendations = []
        metrics = self._get_current_metrics()
        
        if metrics["cpu_percent"] > 70:
            recommendations.append("Consider reducing concurrent AI operations")
        
        if metrics["memory_percent"] > 80:
            recommendations.append("Clear agent memory caches")
            recommendations.append("Reduce batch processing size")
        
        if metrics["avg_response_time"] > 20:
            recommendations.append("Enable fast mode for non-critical agents")
            recommendations.append("Use cached responses where possible")
        
        if len(recommendations) == 0:
            recommendations.append("System performing optimally")
        
        return recommendations
    
    def _analyze_load_trend(self) -> str:
        """Analyze recent load trend"""
        
        if len(self.load_history) < 3:
            return "insufficient_data"
        
        recent_levels = [entry["load_level"] for entry in self.load_history[-5:]]
        
        # Convert to numeric for trend analysis
        level_values = {
            LoadLevel.LOW: 1,
            LoadLevel.MEDIUM: 2,
            LoadLevel.HIGH: 3,
            LoadLevel.CRITICAL: 4
        }
        
        numeric_levels = [level_values[level] for level in recent_levels]
        
        # Simple trend analysis
        if len(set(numeric_levels)) == 1:
            return "stable"
        elif numeric_levels[-1] > numeric_levels[0]:
            return "increasing"
        elif numeric_levels[-1] < numeric_levels[0]:
            return "decreasing"
        else:
            return "fluctuating"

# Global degradation manager
smart_degradation = SmartDegradation()