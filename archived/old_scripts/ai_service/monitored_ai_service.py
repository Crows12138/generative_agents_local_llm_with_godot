"""
AI Service with Monitoring
AI Service with raw output monitoring
"""

from ai_service.ai_service import local_llm_generate
from monitoring.raw_output_monitor import raw_monitor
from ai_service.unified_parser import get_unified_parser
from agents.llm_output_adapter import get_llm_adapter
import traceback
import json
import re
from typing import Dict, Any, Tuple, Optional, List

class MonitoredAIService:
    """AI Service wrapper with monitoring"""
    
    def __init__(self, model_key="qwen3"):
        self.model_key = model_key
        self.parser = get_unified_parser()
        self.adapter = get_llm_adapter()
        
        # Monitoring statistics
        self.total_requests = 0
        self.successful_parses = 0
        
    def generate_with_monitoring(self, prompt: str, 
                                expected_format: str = None,
                                context_type: str = None,
                                max_tokens: int = 200) -> Tuple[str, Any, bool]:
        """Generate response with complete monitoring"""
        
        self.total_requests += 1
        
        try:
            # Generate response
            raw_output = local_llm_generate(
                prompt, 
                model_key=self.model_key
            )
            
            # Try to parse
            parse_success = False
            parse_error = None
            parsed_result = None
            
            try:
                if expected_format == "json":
                    parsed_result = self._parse_json_output(raw_output)
                    parse_success = True
                elif expected_format == "list":
                    parsed_result = self._parse_list_output(raw_output)
                    parse_success = True
                elif expected_format == "action":
                    parsed_result = self.parser.parse_action(raw_output)
                    parse_success = parsed_result is not None
                elif context_type:
                    # Use LLM adapter
                    parsed_result = self.adapter.adapt_output(
                        raw_output, context_type, "MonitoredAgent"
                    )
                    parse_success = parsed_result is not None
                else:
                    # Default parsing
                    parsed_result = raw_output.strip()
                    parse_success = len(parsed_result) > 0
                    
                if parse_success:
                    self.successful_parses += 1
                    
            except Exception as e:
                parse_error = str(e)
                parsed_result = None
                parse_success = False
                print(f"[MonitoredAIService] Parsing error: {e}")
            
            # Log to monitor
            raw_monitor.log_raw_output(
                prompt=prompt,
                raw_output=raw_output,
                expected_format=expected_format,
                context_type=context_type,
                parse_success=parse_success,
                parse_error=parse_error
            )
            
            return raw_output, parsed_result, parse_success
            
        except Exception as e:
            # LLM generation failed
            error_msg = f"LLM generation failed: {str(e)}"
            print(f"[MonitoredAIService] {error_msg}")
            
            raw_monitor.log_raw_output(
                prompt=prompt,
                raw_output="",
                expected_format=expected_format,
                context_type=context_type,
                parse_success=False,
                parse_error=error_msg
            )
            
            return "", None, False
    
    def _parse_json_output(self, output: str) -> Dict[str, Any]:
        """Enhanced JSON parsing, supports multiple strategies"""
        
        # Strategy 1: Direct JSON parsing
        try:
            return json.loads(output)
        except:
            pass
        
        # Strategy 2: Extract JSON from markdown
        json_match = re.search(r'```(?:json)?\s*(\{.*?\}|\[.*?\])\s*```', 
                              output, re.DOTALL | re.IGNORECASE)
        if json_match:
            try:
                return json.loads(json_match.group(1))
            except:
                pass
        
        # Strategy 3: Find JSON structure
        json_match = re.search(r'(\{[^{}]*\}|\[[^\[\]]*\])', output)
        if json_match:
            try:
                return json.loads(json_match.group(1))
            except:
                pass
        
        # Strategy 4: Fix common issues
        fixed = output.strip()
        
        # Remove trailing comma
        fixed = re.sub(r',\s*}', '}', fixed)
        fixed = re.sub(r',\s*]', ']', fixed)
        
        # Add quotes around unquoted keys
        fixed = re.sub(r'(\w+):', r'"\1":', fixed)
        
        try:
            return json.loads(fixed)
        except:
            pass
        
        # Strategy 5: Try to create simple key-value pairs
        if ':' in output:
            try:
                result = {}
                lines = output.split('\n')
                for line in lines:
                    if ':' in line:
                        key, value = line.split(':', 1)
                        result[key.strip()] = value.strip()
                if result:
                    return result
            except:
                pass
        
        raise ValueError(f"Could not parse as JSON: {output[:100]}...")
    
    def _parse_list_output(self, output: str) -> List[str]:
        """Parse list format output"""
        lines = output.strip().split('\n')
        result = []
        
        for line in lines:
            # Remove numbering
            line = re.sub(r'^\d+[\.\)]\s*', '', line)
            # Remove bullet points
            line = re.sub(r'^[-*•]\s*', '', line)
            # Remove Chinese numbering (一二三四五六七八九十)
            line = re.sub(r'^[一二三四五六七八九十]\s*[\.\)]\s*', '', line)
            
            if line.strip():
                result.append(line.strip())
        
        # If no list items found, try splitting by comma
        if not result and ',' in output:
            result = [item.strip() for item in output.split(',') if item.strip()]
        
        # If still no result, return the entire output as a single item
        if not result:
            result = [output.strip()]
        
        return result
    
    def get_monitoring_stats(self) -> Dict[str, Any]:
        """Get monitoring statistics"""
        success_rate = (self.successful_parses / self.total_requests * 100) if self.total_requests > 0 else 0
        
        return {
            "total_requests": self.total_requests,
            "successful_parses": self.successful_parses,
            "success_rate": success_rate,
            "raw_monitor_data": raw_monitor.get_dashboard_data()
        }
    
    def test_parsing_strategies(self, test_outputs: List[str]) -> Dict[str, Any]:
        """Test different parsing strategies"""
        results = {
            "json_parsing": [],
            "list_parsing": [],
            "adapter_parsing": []
        }
        
        for i, output in enumerate(test_outputs):
            print(f"\nTest output {i+1}: {output[:100]}...")
            
            # Test JSON parsing
            try:
                json_result = self._parse_json_output(output)
                results["json_parsing"].append({
                    "input": output[:50] + "...",
                    "result": json_result,
                    "success": True
                })
                print(f"✅ JSON parsing successful: {type(json_result)}")
            except Exception as e:
                results["json_parsing"].append({
                    "input": output[:50] + "...",
                    "error": str(e),
                    "success": False
                })
                print(f"❌ JSON parsing failed: {e}")
            
            # Test list parsing
            try:
                list_result = self._parse_list_output(output)
                results["list_parsing"].append({
                    "input": output[:50] + "...",
                    "result": list_result,
                    "success": True,
                    "item_count": len(list_result)
                })
                print(f"✅ List parsing successful: {len(list_result)} items")
            except Exception as e:
                results["list_parsing"].append({
                    "input": output[:50] + "...",
                    "error": str(e),
                    "success": False
                })
                print(f"❌ List parsing failed: {e}")
            
            # Test adapter parsing
            for context_type in ["perceive", "plan", "reflect", "converse"]:
                try:
                    adapter_result = self.adapter.adapt_output(output, context_type, "TestAgent")
                    results["adapter_parsing"].append({
                        "input": output[:50] + "...",
                        "context": context_type,
                        "result": adapter_result,
                        "result_type": type(adapter_result).__name__,
                        "success": True
                    })
                    print(f"✅ Adapter parsing ({context_type}): {type(adapter_result).__name__}")
                except Exception as e:
                    results["adapter_parsing"].append({
                        "input": output[:50] + "...",
                        "context": context_type,
                        "error": str(e),
                        "success": False
                    })
                    print(f"❌ Adapter parsing ({context_type}) failed: {e}")
        
        return results

# Global monitoring service instance
monitored_service = MonitoredAIService()