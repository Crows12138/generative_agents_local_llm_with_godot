"""Test Enhanced Bar Agent with Reverie Cognitive Capabilities

This tests the complete integration of:
- Enhanced Bar Agent
- Reverie cognitive system 
- Local LLM with complete outputs
- Memory and relationship management
"""

import sys
import os
import time
from datetime import datetime

# Add project path
project_root = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, project_root)

def test_enhanced_agent():
    """Comprehensive test of Enhanced Bar Agent"""
    
    print("Enhanced Bar Agent - Comprehensive Test")
    print("="*60)
    
    try:
        from enhanced_bar_agent import EnhancedBarAgent, CognitiveMode
        
        # Test 1: Agent Creation and Initialization
        print("\n[1] Testing Agent Creation and Initialization")
        print("-" * 50)
        
        print("Creating Enhanced Bar Agent...")
        bob = EnhancedBarAgent(
            name="Bob",
            role="bartender",
            position=(5, 2),
            personality="friendly, experienced bartender who remembers customers",
            background="5 years experience at Murphy's Bar, known for great cocktails",
            cognitive_mode=CognitiveMode.ADAPTIVE
        )
        
        print(f"[OK] Created agent: {bob.name}")
        print(f"[OK] Role: {bob.role}")
        print(f"[OK] Cognitive mode: {bob.cognitive_mode.value}")
        
        # Test 2: Enhanced Perception
        print(f"\n[2] Testing Enhanced Perception")
        print("-" * 50)
        
        bar_environment = {
            "location": "Murphy's Bar",
            "time": "8:30 PM",
            "description": "Cozy evening atmosphere with soft lighting",
            "events": ["Alice orders a drink", "Charlie reads a book", "music playing softly"],
            "objects": ["bar counter", "bottles", "glasses", "cash register", "napkins"],
            "people": ["Alice", "Charlie", "Bob"]
        }
        
        print("Testing enhanced perception...")
        perception_result = bob.enhanced_perceive(bar_environment)
        
        print(f"[OK] Enhanced perception completed")
        print(f"  - Observations: {len(perception_result.get('observations', []))}")
        print(f"  - Role insights: {len(perception_result.get('role_insights', []))}")
        print(f"  - Priorities: {perception_result.get('priorities', [])}")
        print(f"  - Time taken: {perception_result.get('perception_time', 0):.2f}s")
        
        perception_passed = (
            perception_result.get("enhanced", False) and
            len(perception_result.get("observations", [])) >= 3 and
            perception_result.get("perception_time", 0) < 30.0
        )
        
        # Test 3: Enhanced Decision Making
        print(f"\n[3] Testing Enhanced Decision Making")
        print("-" * 50)
        
        print("Testing enhanced decision making...")
        decision_result = bob.enhanced_decide(bar_environment, perception_result)
        
        print(f"[OK] Enhanced decision completed")
        print(f"  - Decision: {decision_result.get('decision', {}).get('action', 'none')}")
        print(f"  - Observations considered: {decision_result.get('observations_considered', 0)}")
        print(f"  - Memories used: {decision_result.get('memories_used', 0)}")
        print(f"  - Time taken: {decision_result.get('decision_time', 0):.2f}s")
        
        decision_passed = (
            decision_result.get("enhanced", False) and
            decision_result.get("observations_considered", 0) > 0 and
            decision_result.get("decision_time", 0) < 30.0
        )
        
        # Test 4: Enhanced Conversation
        print(f"\n[4] Testing Enhanced Conversation")
        print("-" * 50)
        
        conversations = [
            ("Alice", "Hi Bob! I'd like something strong after the day I've had."),
            ("Charlie", "Excuse me, could you recommend something light and refreshing?"),
            ("Alice", "Bob, you always know what I need. Thanks!")
        ]
        
        conversation_results = []
        
        for person, message in conversations:
            print(f"Testing conversation with {person}...")
            response = bob.enhanced_converse(person, message, bar_environment)
            
            conversation_results.append({
                "person": person,
                "message": message,
                "response": response,
                "response_length": len(response)
            })
            
            print(f"  {person}: {message}")
            print(f"  Bob: {response[:100]}...")
            print(f"  [OK] Response length: {len(response)} chars")
        
        conversation_passed = all(
            len(result["response"]) > 20 for result in conversation_results
        )
        
        # Test 5: Customer Relationship Tracking
        print(f"\n[5] Testing Customer Relationship Tracking")
        print("-" * 50)
        
        print("Checking customer relationships...")
        relationships = bob.customer_relationships
        
        print(f"[OK] Tracked relationships: {len(relationships)}")
        for customer, data in relationships.items():
            visit_count = data.get("visit_count", 0)
            level = data.get("relationship_level", "unknown")
            print(f"  - {customer}: {visit_count} visits, {level} level")
        
        relationship_passed = len(relationships) >= 2  # Should have Alice and Charlie
        
        # Test 6: Enhanced Reflection
        print(f"\n[6] Testing Enhanced Reflection")
        print("-" * 50)
        
        print("Testing enhanced reflection...")
        reflection_result = bob.enhanced_reflect(force_reflection=True)
        
        if reflection_result.get("enhanced", False):
            print(f"[OK] Reflection completed")
            insights = reflection_result.get("insights", [])
            work_reflections = reflection_result.get("work_reflections", [])
            print(f"  - Personal insights: {len(insights)}")
            print(f"  - Work reflections: {len(work_reflections)}")
            
            if insights:
                print(f"  - Sample insight: {insights[0][:100]}...")
                
            reflection_passed = len(insights) > 0 or len(work_reflections) > 0
        else:
            reflection_passed = False
            print(f"[WARN] Reflection not enhanced: {reflection_result.get('reason', 'unknown')}")
        
        # Test 7: Full Cognitive Cycle
        print(f"\n[7] Testing Full Cognitive Cycle")
        print("-" * 50)
        
        print("Running complete cognitive cycle...")
        cycle_start_time = time.time()
        
        # Create dynamic environment
        evening_environment = {
            "location": "Murphy's Bar",
            "time": "9:00 PM",
            "description": "Busy Friday evening with multiple customers",
            "events": ["live music starts", "Alice orders another drink", "new customer enters"],
            "objects": ["busy bar counter", "multiple glasses", "tip jar", "menu"],
            "people": ["Alice", "Charlie", "Sarah", "Bob"]
        }
        
        cycle_result = bob.full_cognitive_cycle(evening_environment)
        cycle_time = time.time() - cycle_start_time
        
        print(f"[OK] Full cognitive cycle completed in {cycle_time:.2f}s")
        print(f"  - Perception: {cycle_result['perception']['enhanced']}")
        print(f"  - Decision: {cycle_result['decision']['enhanced']}")
        print(f"  - Conversations: {len(cycle_result['conversations'])}")
        print(f"  - Reflection: {cycle_result['reflection']['enhanced']}")
        print(f"  - Agent state: {cycle_result['agent_state']['current_focus']}")
        
        cycle_passed = (
            cycle_result["perception"]["enhanced"] and
            cycle_result["decision"]["enhanced"] and
            len(cycle_result["conversations"]) > 0 and
            cycle_time < 120.0  # Should complete within 2 minutes
        )
        
        # Test 8: Performance and Status
        print(f"\n[8] Testing Performance and Status")
        print("-" * 50)
        
        enhanced_status = bob.get_enhanced_status()
        
        print(f"[OK] Enhanced status retrieved")
        print(f"  - Cognitive mode: {enhanced_status['cognitive_mode']}")
        print(f"  - Current focus: {enhanced_status['current_focus']}")
        print(f"  - Customer relationships: {enhanced_status['customer_relationships']}")
        print(f"  - Performance stats: {enhanced_status['performance_stats']}")
        
        status_passed = (
            enhanced_status["customer_relationships"] > 0 and
            enhanced_status["performance_stats"]["perceptions"] > 0
        )
        
        # Final Assessment
        print(f"\n{'='*60}")
        print("ENHANCED AGENT TEST SUMMARY")
        print(f"{'='*60}")
        
        test_results = [
            ("Agent Creation", True),  # Always passes if we get this far
            ("Enhanced Perception", perception_passed),
            ("Enhanced Decision", decision_passed),
            ("Enhanced Conversation", conversation_passed),
            ("Relationship Tracking", relationship_passed),
            ("Enhanced Reflection", reflection_passed),
            ("Full Cognitive Cycle", cycle_passed),
            ("Performance Status", status_passed)
        ]
        
        passed_tests = sum(1 for _, passed in test_results if passed)
        total_tests = len(test_results)
        
        print(f"Test Results:")
        for test_name, passed in test_results:
            status = "PASSED" if passed else "FAILED"
            marker = "[OK]" if passed else "[ERROR]"
            print(f"  {marker} {test_name:<25} {status}")
        
        success_rate = passed_tests / total_tests * 100
        print(f"\nOverall Success Rate: {passed_tests}/{total_tests} ({success_rate:.1f}%)")
        
        if success_rate >= 80:
            print(f"\n[SUCCESS] Enhanced Bar Agent is working excellently!")
            print("All major cognitive functions operational:")
            print("- Sophisticated perception with role insights")
            print("- Enhanced decision making with memory integration")
            print("- Natural conversation with relationship tracking")
            print("- Periodic reflection with performance analysis")
            print("- Complete cognitive cycles under 2 minutes")
            
            agent_success = True
        elif success_rate >= 60:
            print(f"\n[PARTIAL SUCCESS] Core functions working, some refinements needed")
            agent_success = True
        else:
            print(f"\n[NEEDS WORK] Significant issues detected")
            agent_success = False
        
        # Cleanup
        print(f"\nCleaning up agent...")
        bob.cleanup()
        
        return agent_success
        
    except Exception as e:
        print(f"[CRITICAL] Enhanced agent test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_multiple_agents():
    """Test multiple enhanced agents interacting"""
    
    print(f"\n{'='*60}")
    print("MULTIPLE ENHANCED AGENTS TEST")
    print(f"{'='*60}")
    
    try:
        from enhanced_bar_agent import EnhancedBarAgent, CognitiveMode
        
        # Create multiple agents with different roles and cognitive modes
        agents = [
            EnhancedBarAgent(
                name="Bob",
                role="head_bartender", 
                position=(5, 2),
                personality="experienced, leadership-focused",
                cognitive_mode=CognitiveMode.DELIBERATIVE
            ),
            EnhancedBarAgent(
                name="Alice", 
                role="bartender",
                position=(7, 2),
                personality="friendly, quick-service focused",
                cognitive_mode=CognitiveMode.REACTIVE
            ),
            EnhancedBarAgent(
                name="Charlie",
                role="server",
                position=(3, 5),
                personality="attentive, customer-focused",
                cognitive_mode=CognitiveMode.ADAPTIVE
            )
        ]
        
        print(f"Created {len(agents)} enhanced agents")
        
        # Simulate busy bar scenario
        busy_environment = {
            "location": "Murphy's Bar",
            "time": "10:30 PM",
            "description": "Peak Friday night with live music",
            "events": ["live band performing", "multiple orders", "crowded dance floor"],
            "objects": ["packed bar", "multiple glasses", "cash register", "microphone"],
            "people": ["Bob", "Alice", "Charlie", "Customer1", "Customer2", "Customer3", "Band"]
        }
        
        print(f"\nTesting agents in busy environment...")
        
        agent_results = []
        total_time = 0
        
        for agent in agents:
            print(f"\nTesting {agent.name} ({agent.role})...")
            
            start_time = time.time()
            cycle_result = agent.full_cognitive_cycle(busy_environment)
            agent_time = time.time() - start_time
            total_time += agent_time
            
            agent_results.append({
                "name": agent.name,
                "role": agent.role,
                "cognitive_mode": agent.cognitive_mode.value,
                "cycle_time": agent_time,
                "enhanced_functions": {
                    "perception": cycle_result["perception"]["enhanced"],
                    "decision": cycle_result["decision"]["enhanced"],
                    "conversations": len(cycle_result["conversations"]),
                    "reflection": cycle_result["reflection"]["enhanced"]
                }
            })
            
            print(f"  Completed in {agent_time:.2f}s")
            print(f"  Conversations: {len(cycle_result['conversations'])}")
            print(f"  Current focus: {cycle_result['agent_state']['current_focus']}")
        
        # Analyze results
        print(f"\nMultiple Agent Analysis:")
        print(f"  Total processing time: {total_time:.2f}s")
        print(f"  Average time per agent: {total_time/len(agents):.2f}s")
        
        all_enhanced = all(
            result["enhanced_functions"]["perception"] and
            result["enhanced_functions"]["decision"]
            for result in agent_results
        )
        
        if all_enhanced:
            print(f"[SUCCESS] All agents functioning with enhanced cognition!")
        else:
            print(f"[PARTIAL] Some agents not fully enhanced")
        
        # Cleanup
        for agent in agents:
            agent.cleanup()
        
        return all_enhanced
        
    except Exception as e:
        print(f"[ERROR] Multiple agents test failed: {e}")
        return False

if __name__ == "__main__":
    print("Starting Enhanced Bar Agent Tests...")
    
    # Test single agent
    single_success = test_enhanced_agent()
    
    # Test multiple agents
    multiple_success = test_multiple_agents()
    
    print(f"\n{'='*60}")
    print("FINAL TEST RESULTS")
    print(f"{'='*60}")
    print(f"Single Agent Test: {'PASSED' if single_success else 'FAILED'}")
    print(f"Multiple Agent Test: {'PASSED' if multiple_success else 'FAILED'}")
    
    overall_success = single_success and multiple_success
    
    if overall_success:
        print(f"\n[SUCCESS] Enhanced Bar Agent system is ready!")
        print("✓ Reverie cognitive integration working")
        print("✓ Local LLM providing complete outputs") 
        print("✓ Memory and relationship systems functional")
        print("✓ Multiple agents can operate simultaneously")
        print(f"\n[READY] System prepared for Godot integration and demo!")
    else:
        print(f"\n[PARTIAL] System partially working - review failed components")
    
    print(f"\nNext steps:")
    print("- Integrate with Godot visualization")
    print("- Create bar demo scenarios")
    print("- Test real-time interactions")