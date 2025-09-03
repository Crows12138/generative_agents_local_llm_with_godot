"""
Monitor raw LLM outputs to diagnose parsing issues
Monitor raw LLM outputs to diagnose parsing issues
"""

import json
import time
from datetime import datetime
from pathlib import Path
import logging
from typing import Dict, List, Any
import re

class RawOutputMonitor:
    """Monitor and record raw LLM outputs for debugging"""
    
    def __init__(self, log_dir="debug_logs/raw_outputs"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup logging
        self.logger = logging.getLogger("raw_output_monitor")
        handler = logging.FileHandler(self.log_dir / "raw_outputs.log", encoding='utf-8')
        handler.setFormatter(logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s'
        ))
        if not self.logger.handlers:  # Avoid duplicate handlers
            self.logger.addHandler(handler)
        self.logger.setLevel(logging.DEBUG)
        
        # Real-time monitoring data
        self.recent_outputs = []  # Last 10 outputs
        self.max_recent = 10
        
        # Statistics
        self.stats = {
            "total_requests": 0,
            "format_types": {},  # Track different output formats
            "common_patterns": {},
            "avg_length": 0,
            "parse_success_by_pattern": {},
            "error_patterns": {}  # Track common error patterns
        }
        
    def log_raw_output(self, prompt: str, raw_output: str, 
                       expected_format: str = None,
                       parse_success: bool = False,
                       parse_error: str = None,
                       context_type: str = None):
        """Record raw output and context information"""
        
        self.stats["total_requests"] += 1
        
        # Create detailed log entry
        entry = {
            "id": self.stats["total_requests"],
            "timestamp": datetime.now().isoformat(),
            "prompt_preview": prompt[:200] + "..." if len(prompt) > 200 else prompt,
            "raw_output": raw_output,
            "output_length": len(raw_output),
            "expected_format": expected_format,
            "context_type": context_type,  # perceive, plan, reflect, etc.
            "parse_success": parse_success,
            "parse_error": parse_error,
            "detected_format": self._detect_format(raw_output),
            "language": self._detect_language(raw_output),
            "contains_json": self._contains_json(raw_output),
            "structure_analysis": self._analyze_structure(raw_output)
        }
        
        # Update recent outputs (for dashboard)
        self.recent_outputs.append(entry)
        if len(self.recent_outputs) > self.max_recent:
            self.recent_outputs.pop(0)
        
        # Log to file
        self.logger.info(f"RAW OUTPUT: {json.dumps(entry, indent=2, ensure_ascii=False)}")
        
        # Save individual output for analysis
        output_file = self.log_dir / f"output_{self.stats['total_requests']:05d}.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(entry, f, indent=2, ensure_ascii=False)
        
        # Update statistics
        self._update_stats(entry)
        
        # Print to console for immediate viewing
        self._print_analysis(entry)
        
        return entry
    
    def _detect_format(self, output: str) -> str:
        """Detect output format type"""
        output_lower = output.lower().strip()
        
        # Check JSON format
        if output_lower.startswith('{') or output_lower.startswith('['):
            try:
                json.loads(output)
                return "valid_json"
            except:
                return "invalid_json"
        
        # Check markdown JSON
        if '```json' in output_lower or '```' in output:
            return "json_in_markdown"
        
        # Check list format
        if re.match(r'^\d+[\.\)]', output_lower) or output_lower.startswith('-') or output_lower.startswith('•'):
            return "numbered_list"
        
        # Check Chinese natural language
        chinese_patterns = [
            r'^(I|he|she|it|this|that|based on|based|considering|due to)',
            r'(should|can|need|will|will|is|has|been|do|do|say|look|listen|walk|stop|wait|talk|work|rest)',
            r'(,|.|!|?|;|:)'
        ]
        if any(re.search(pattern, output) for pattern in chinese_patterns):
            return "chinese_natural_language"
        
        # Check English natural language
        if '.' in output and len(output.split()) > 5:
            return "english_natural_language"
        
        # Check short response
        if len(output.split()) <= 3:
            return "short_phrase"
        
        # Check action words
        action_patterns = [
            r'(analyze|think|decide|act|go|do|say|look|listen|walk|stop|wait|talk|work|rest)',
            r'(analyze|think|decide|act|go|do|say|look|listen|walk|stop|wait|talk|work|rest)'
        ]
        if any(re.search(pattern, output.lower()) for pattern in action_patterns):
            return "action_statement"
        
        return "unknown"
    
    def _detect_language(self, output: str) -> str:
        """Detect the main language of the output"""
        # Simple language detection
        chinese_chars = len(re.findall(r'[\u4e00-\u9fff]', output))
        english_words = len(re.findall(r'[a-zA-Z]+', output))
        
        if chinese_chars > english_words:
            return "chinese"
        elif english_words > chinese_chars:
            return "english"
        else:
            return "mixed"
    
    def _contains_json(self, output: str) -> bool:
        """Check if JSON structure is contained"""
        return '{' in output and '}' in output
    
    def _analyze_structure(self, output: str) -> Dict[str, Any]:
        """Analyze the structural features of the output"""
        analysis = {
            "line_count": len(output.split('\n')),
            "word_count": len(output.split()),
            "sentence_count": len(re.split(r'[.!?。！？]', output)),
            "has_quotes": '"' in output or "'" in output,
            "has_brackets": '{' in output or '[' in output,
            "starts_with_prefix": self._check_common_prefixes(output),
            "ends_with_punctuation": output.strip()[-1:] in '.!?。！？',
            "contains_code_block": '```' in output,
            "contains_numbering": bool(re.search(r'^\d+[\.\)]', output, re.MULTILINE))
        }
        return analysis
    
    def _check_common_prefixes(self, output: str) -> str:
        """Check common prefix patterns"""
        prefixes = [
            "Based on", "According to", "As an AI", "I think", "I believe",
            "Based on", "According to", "As an AI", "I think", "I believe",
            "Considering", "Due to"
        ]
        
        for prefix in prefixes:
            if output.strip().startswith(prefix):
                return prefix
        return None
    
    def _update_stats(self, entry: Dict):
        """Update run statistics"""
        format_type = entry["detected_format"]
        self.stats["format_types"][format_type] = \
            self.stats["format_types"].get(format_type, 0) + 1
        
        # Track parsing success rate by format
        if format_type not in self.stats["parse_success_by_pattern"]:
            self.stats["parse_success_by_pattern"][format_type] = {
                "success": 0, "fail": 0, "total": 0
            }
        
        if entry["parse_success"]:
            self.stats["parse_success_by_pattern"][format_type]["success"] += 1
        else:
            self.stats["parse_success_by_pattern"][format_type]["fail"] += 1
            
            # Record error patterns
            if entry["parse_error"]:
                error_key = f"{format_type}:{entry['parse_error'][:50]}"
                self.stats["error_patterns"][error_key] = \
                    self.stats["error_patterns"].get(error_key, 0) + 1
        
        self.stats["parse_success_by_pattern"][format_type]["total"] += 1
        
        # Update average length
        total_length = self.stats["avg_length"] * (self.stats["total_requests"] - 1)
        self.stats["avg_length"] = (total_length + entry["output_length"]) / \
                                   self.stats["total_requests"]
    
    def _print_analysis(self, entry: Dict):
        """Print real-time analysis to the console"""
        print("\n" + "="*60)
        print(f"🔍 Raw Output Analysis #{entry['id']}")
        print("="*60)
        print(f"📝 Prompt Preview: {entry['prompt_preview'][:100]}...")
        print(f"�� Expected Format: {entry['expected_format']} | Context: {entry.get('context_type', 'N/A')}")
        print(f"🔍 Detected Format: {entry['detected_format']} | Language: {entry['language']}")
        print(f"📤 Raw Output ({entry['output_length']} characters):")
        print("-"*40)
        print(entry['raw_output'][:300])  # Display first 300 characters
        if len(entry['raw_output']) > 300:
            print("... [Truncated]")
        print("-"*40)
        
        # Display structure analysis
        structure = entry["structure_analysis"]
        print(f"📊 Structure: {structure['line_count']} lines, {structure['word_count']} words, {structure['sentence_count']} sentences")
        print(f"🏷️ Features: JSON Block={structure['contains_code_block']}, Numbering={structure['contains_numbering']}")
        
        # Display parsing results
        success_icon = "✅" if entry["parse_success"] else "❌"
        print(f"{success_icon} Parsing Status: {'Success' if entry['parse_success'] else 'Failed'}")
        if entry["parse_error"]:
            print(f"❌ Error Message: {entry['parse_error'][:100]}")
        
        print("="*60 + "\n")
    
    def get_dashboard_data(self) -> Dict:
        """Get data for the dashboard display"""
        return {
            "recent_outputs": self.recent_outputs,
            "stats": self.stats,
            "recommendations": self._generate_recommendations(),
            "success_rates": self._calculate_success_rates()
        }
    
    def _calculate_success_rates(self) -> Dict[str, float]:
        """Calculate success rates for different formats"""
        rates = {}
        for format_type, data in self.stats["parse_success_by_pattern"].items():
            if data["total"] > 0:
                rates[format_type] = (data["success"] / data["total"]) * 100
        return rates
    
    def _generate_recommendations(self) -> List[str]:
        """Generate recommendations based on observed patterns"""
        recommendations = []
        
        # Check most common format
        if self.stats["format_types"]:
            most_common = max(self.stats["format_types"].items(), 
                            key=lambda x: x[1])
            recommendations.append(
                f"Most Common Output Format: {most_common[0]} (Appeared {most_common[1]} times)"
            )
            
            # Provide parsing suggestions for the format
            if most_common[0] == "chinese_natural_language":
                recommendations.append(
                    "Suggestion: Add keyword extraction for Chinese natural language outputs"
                )
            elif most_common[0] == "json_in_markdown":
                recommendations.append(
                    "Suggestion: Remove markdown markers before parsing JSON"
                )
            elif most_common[0] == "action_statement":
                recommendations.append(
                    "Suggestion: Improve parsing using action word recognition"
                )
        
        # Check parsing success rates
        success_rates = self._calculate_success_rates()
        for format_type, rate in success_rates.items():
            if rate < 50:
                recommendations.append(
                    f"Warning: Low parsing success rate for {format_type} format ({rate:.1f}%)"
                )
        
        # Check common errors
        if self.stats["error_patterns"]:
            top_error = max(self.stats["error_patterns"].items(), key=lambda x: x[1])
            recommendations.append(
                f"Most Common Error: {top_error[0]} (Appeared {top_error[1]} times)"
            )
        
        return recommendations
    
    def export_analysis_report(self, filename: str = None) -> str:
        """Export detailed analysis report"""
        if filename is None:
            filename = f"analysis_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        report = {
            "summary": {
                "total_requests": self.stats["total_requests"],
                "avg_output_length": self.stats["avg_length"],
                "monitoring_period": {
                    "start": self.recent_outputs[0]["timestamp"] if self.recent_outputs else None,
                    "end": self.recent_outputs[-1]["timestamp"] if self.recent_outputs else None
                }
            },
            "format_distribution": self.stats["format_types"],
            "success_rates": self._calculate_success_rates(),
            "error_patterns": self.stats["error_patterns"],
            "recommendations": self._generate_recommendations(),
            "detailed_outputs": self.recent_outputs
        }
        
        report_path = self.log_dir / filename
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        print(f"📄 Analysis report exported to: {report_path}")
        return str(report_path)

# Global monitor instance
raw_monitor = RawOutputMonitor()