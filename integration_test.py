#!/usr/bin/env python
"""
Integration Test - Verify all components work together
Tests AI service, Godot bridge, and simple agents
"""

import sys
import time
import json
import asyncio
import requests
from pathlib import Path
from typing import Dict, Any, List, Optional

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

# Import components
from ai_service.ai_service import unified_ai_service
from ai_service.config_enhanced import get_config
from simple_agents import SimpleAgent, create_demo_characters, Location

# Test configuration
TEST_CONFIG = {
    "ai_bridge_url": "http://127.0.0.1:8080",
    "test_timeout": 30,
    "verbose": True
}

class IntegrationTest:
    """Integration test suite"""
    
    def __init__(self, verbose: bool = True):
        self.verbose = verbose
        self.results = []
        self.config = get_config()
        
    def log(self, message: str, level: str = "INFO"):
        """Log message"""
        if self.verbose:
            print(f"[{level}] {message}")
    
    def test_ai_service(self) -> bool:
        """Test AI service directly"""
        self.log("Testing AI Service...")
        
        try:
            # Test simple generation
            response = unified_ai_service(
                "Hello, this is a test. Please respond briefly.",
                max_new_tokens=50,
                temperature=0.5
            )
            
            if response and len(response) > 0:
                self.log(f"✓ AI Service responded: {response[:100]}...")
                return True
            else:
                self.log("✗ AI Service returned empty response", "ERROR")
                return False
                
        except Exception as e:
            self.log(f"✗ AI Service error: {e}", "ERROR")
            return False
    
    def test_config_system(self) -> bool:
        """Test configuration system"""
        self.log("Testing Configuration System...")
        
        try:
            # Test model detection
            detected = self.config.config.model.detect_models()
            self.log(f"  Detected {len(detected)} models")
            
            # Test config validation
            is_valid = self.config.validate_config()
            if is_valid:
                self.log("✓ Configuration validated successfully")
            else:
                self.log("✗ Configuration validation failed", "WARNING")
            
            # Test config save/load
            test_config_path = PROJECT_ROOT / "test_config.yaml"
            if self.config.save_config(str(test_config_path)):
                self.log("✓ Configuration saved successfully")
                test_config_path.unlink()  # Clean up
            else:
                self.log("✗ Failed to save configuration", "WARNING")
            
            return True
            
        except Exception as e:
            self.log(f"✗ Config system error: {e}", "ERROR")
            return False
    
    def test_simple_agents(self) -> bool:
        """Test simple agent system"""
        self.log("Testing Simple Agents...")
        
        try:
            # Create demo characters
            characters = create_demo_characters()
            self.log(f"  Created {len(characters)} demo characters")
            
            # Test character interactions
            alice = characters[0]
            bob = characters[1]
            
            # Test perception
            alice.perceive("A customer enters the store")
            memories = alice.memory.get_recent_memories(1)
            if memories:
                self.log(f"✓ Perception working: {memories[0].content}")
            
            # Test thinking
            thought = alice.think("the weather")
            if thought:
                self.log(f"✓ Thinking working: {thought[:100]}...")
            
            # Test decision making
            actions = ["greet customer", "continue working", "take break"]
            action, reason = alice.decide_action(actions)
            self.log(f"✓ Decision made: {action} - {reason[:50]}...")
            
            # Test conversation
            response = alice.respond_to("Bob", "Hello Alice!")
            if response:
                self.log(f"✓ Conversation working: {response[:100]}...")
            
            # Test needs update
            initial_energy = alice.energy
            alice.update_needs(1.0)
            if alice.energy != initial_energy:
                self.log("✓ Needs system working")
            
            return True
            
        except Exception as e:
            self.log(f"✗ Simple agents error: {e}", "ERROR")
            return False
    
    def test_ai_bridge_server(self) -> bool:
        """Test AI Bridge server endpoints"""
        self.log("Testing AI Bridge Server...")
        
        base_url = TEST_CONFIG["ai_bridge_url"]
        
        # Check if server is running
        try:
            response = requests.get(f"{base_url}/health", timeout=5)
            if response.status_code != 200:
                self.log("✗ AI Bridge server not running", "WARNING")
                self.log("  Please run: python godot_ai_bridge.py")
                return False
        except:
            self.log("✗ Cannot connect to AI Bridge server", "WARNING")
            self.log("  Please run: python godot_ai_bridge.py")
            return False
        
        try:
            # Test status endpoint
            response = requests.get(f"{base_url}/ai/status")
            if response.status_code == 200:
                data = response.json()
                self.log(f"✓ Status endpoint working: {data['status']}")
                self.log(f"  Active model: {data['active_model']}")
            
            # Test chat endpoint
            chat_data = {
                "character_name": "TestBot",
                "message": "Hello, how are you?",
                "context": {"location": "test"},
                "max_length": 50
            }
            response = requests.post(f"{base_url}/ai/chat", json=chat_data)
            if response.status_code == 200:
                data = response.json()
                self.log(f"✓ Chat endpoint working: {data['response'][:50]}...")
            
            # Test decide endpoint
            decide_data = {
                "character_name": "TestBot",
                "situation": "You are hungry",
                "options": ["eat", "wait", "sleep"],
                "context": {}
            }
            response = requests.post(f"{base_url}/ai/decide", json=decide_data)
            if response.status_code == 200:
                data = response.json()
                self.log(f"✓ Decide endpoint working: chose '{data['chosen_option']}'")
            
            # Test think endpoint
            think_data = {
                "character_name": "TestBot",
                "topic": "the meaning of life",
                "context": {},
                "depth": "quick"
            }
            response = requests.post(f"{base_url}/ai/think", json=think_data)
            if response.status_code == 200:
                data = response.json()
                self.log(f"✓ Think endpoint working: {data['thought'][:50]}...")
            
            return True
            
        except Exception as e:
            self.log(f"✗ AI Bridge test error: {e}", "ERROR")
            return False
    
    def test_performance(self) -> bool:
        """Test performance metrics"""
        self.log("Testing Performance...")
        
        try:
            import time
            
            # Test AI response time
            start = time.time()
            response = unified_ai_service("Quick test", max_new_tokens=20)
            elapsed = time.time() - start
            
            if elapsed < 5.0:  # Should respond within 5 seconds
                self.log(f"✓ AI response time: {elapsed:.2f}s")
            else:
                self.log(f"⚠ AI response slow: {elapsed:.2f}s", "WARNING")
            
            # Test memory usage
            import psutil
            process = psutil.Process()
            memory_mb = process.memory_info().rss / 1024 / 1024
            
            if memory_mb < 4000:  # Less than 4GB
                self.log(f"✓ Memory usage: {memory_mb:.0f}MB")
            else:
                self.log(f"⚠ High memory usage: {memory_mb:.0f}MB", "WARNING")
            
            return True
            
        except ImportError:
            self.log("⚠ psutil not installed, skipping memory test", "WARNING")
            return True
        except Exception as e:
            self.log(f"✗ Performance test error: {e}", "ERROR")
            return False
    
    def run_all_tests(self) -> Dict[str, bool]:
        """Run all integration tests"""
        self.log("\n" + "="*60)
        self.log("INTEGRATION TEST SUITE")
        self.log("="*60 + "\n")
        
        tests = [
            ("Configuration System", self.test_config_system),
            ("AI Service", self.test_ai_service),
            ("Simple Agents", self.test_simple_agents),
            ("AI Bridge Server", self.test_ai_bridge_server),
            ("Performance", self.test_performance)
        ]
        
        results = {}
        
        for test_name, test_func in tests:
            self.log(f"\n--- {test_name} ---")
            try:
                result = test_func()
                results[test_name] = result
            except Exception as e:
                self.log(f"✗ Test crashed: {e}", "ERROR")
                results[test_name] = False
            time.sleep(0.5)  # Brief pause between tests
        
        # Summary
        self.log("\n" + "="*60)
        self.log("TEST SUMMARY")
        self.log("="*60)
        
        passed = sum(1 for r in results.values() if r)
        total = len(results)
        
        for test_name, result in results.items():
            status = "✓ PASS" if result else "✗ FAIL"
            self.log(f"{test_name}: {status}")
        
        self.log(f"\nTotal: {passed}/{total} tests passed")
        
        if passed == total:
            self.log("\n🎉 All tests passed! System is ready.", "SUCCESS")
        elif passed > total / 2:
            self.log("\n⚠ Some tests failed. System may work with limitations.", "WARNING")
        else:
            self.log("\n❌ Many tests failed. Please check your setup.", "ERROR")
        
        return results

def main():
    """Main test runner"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Integration Test Suite")
    parser.add_argument("--verbose", action="store_true", help="Verbose output")
    parser.add_argument("--test", help="Run specific test")
    
    args = parser.parse_args()
    
    # Create test suite
    tester = IntegrationTest(verbose=args.verbose or True)
    
    if args.test:
        # Run specific test
        test_map = {
            "config": tester.test_config_system,
            "ai": tester.test_ai_service,
            "agents": tester.test_simple_agents,
            "bridge": tester.test_ai_bridge_server,
            "performance": tester.test_performance
        }
        
        if args.test in test_map:
            print(f"Running {args.test} test...")
            result = test_map[args.test]()
            sys.exit(0 if result else 1)
        else:
            print(f"Unknown test: {args.test}")
            print(f"Available tests: {', '.join(test_map.keys())}")
            sys.exit(1)
    else:
        # Run all tests
        results = tester.run_all_tests()
        
        # Exit with appropriate code
        all_passed = all(results.values())
        sys.exit(0 if all_passed else 1)

if __name__ == "__main__":
    main()
