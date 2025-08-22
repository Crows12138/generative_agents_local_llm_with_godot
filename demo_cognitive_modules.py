"""
Demo script for cognitive module integration
Demonstrates the enhanced capabilities of agents with reverie cognitive modules
"""

import sys
sys.path.append('.')

from agents.simple_agents import SimpleAgent, Location, EmotionalState

def demo_cognitive_capabilities():
    """Demonstrate cognitive module capabilities"""
    print("🧠 === Cognitive Module Integration Demo ===\\n")
    
    # Create an enhanced agent
    print("1. Creating Enhanced Agent...")
    agent = SimpleAgent(
        name="CognitiveBot",
        personality="Thoughtful and analytical AI assistant with advanced reasoning",
        background="An AI agent equipped with sophisticated cognitive modules for perception, memory, planning, and reflection",
        location=Location("Cozy Cafe", 15, 10, "social_zone"),
        emotional_state=EmotionalState.HAPPY
    )
    
    print(f"   ✅ Agent: {agent.name}")
    print(f"   🧠 Cognitive Modules: {'✅ Enabled' if agent.use_cognitive_modules else '❌ Disabled'}")
    print(f"   💾 Enhanced Memory: {'✅ Enabled' if agent.use_reverie_memory else '❌ Disabled'}")
    
    # Show capabilities summary
    print("\\n2. Cognitive Capabilities Summary:")
    summary = agent.get_cognitive_summary()
    for key, value in summary.items():
        icon = "✅" if value in [True, "Active"] else "📊" if isinstance(value, (int, str)) else "ℹ️"
        print(f"   {icon} {key.replace('_', ' ').title()}: {value}")
    
    # Demonstrate enhanced perception
    print("\\n3. Enhanced Perception Demonstration:")
    environment = {
        "description": "A bustling cafe with customers chatting and soft music playing",
        "events": [
            "A customer is ordering a latte at the counter",
            "Someone is reading a newspaper by the window", 
            "The barista is grinding coffee beans",
            "A couple is having a quiet conversation",
            "Jazz music is playing softly in the background"
        ],
        "objects": ["coffee machine", "tables", "chairs", "menu board", "cash register"],
        "other_agents": ["Barista", "Customer1", "Customer2", "Couple"],
        "available_actions": ["observe", "approach_counter", "find_seat", "listen_music", "order_coffee"]
    }
    
    perception_result = agent.perceive_with_cognition(environment)
    print(f"   🔍 Enhanced Perception: {'✅' if perception_result.get('enhanced') else '❌'}")
    
    if perception_result.get('cognitive_events'):
        print(f"   📊 Events Perceived: {len(perception_result['cognitive_events'])}")
        for i, event in enumerate(perception_result['cognitive_events'][:3], 1):
            desc = event.get('description', str(event))
            print(f"      {i}. {desc}")
    
    # Demonstrate enhanced thinking
    print("\\n4. Enhanced Cognitive Processing:")
    try:
        thinking_result = agent.think_with_cognition(environment)
        print(f"   🤔 Enhanced Thinking: {'✅' if thinking_result.get('enhanced') else '❌'}")
        
        if thinking_result.get('reflection'):
            reflections = thinking_result['reflection'].get('reflections', [])
            print(f"   💭 Reflections Generated: {len(reflections)}")
            for i, reflection in enumerate(reflections[:2], 1):
                print(f"      {i}. {reflection}")
        
        if thinking_result.get('memories'):
            print(f"   🧠 Memory Retrieval: {len(thinking_result['memories'])} focal points processed")
        
        if thinking_result.get('plan'):
            plan = thinking_result['plan']
            print(f"   📋 Planning: {plan.get('type', 'N/A')}")
    except Exception as e:
        print(f"   ⚠️ Cognitive processing: Using fallback mode ({str(e)[:30]}...)")
    
    # Demonstrate enhanced decision making
    print("\\n5. Enhanced Decision Making:")
    try:
        decision_result = agent.decide_with_cognition(environment)
        print(f"   🎯 Enhanced Decision: {'✅' if decision_result.get('enhanced') else '❌'}")
        
        if decision_result.get('cognitive_action'):
            cog_action = decision_result['cognitive_action']
            print(f"   🧠 Cognitive Choice: {cog_action.get('action', 'N/A')}")
        
        if decision_result.get('normal_action'):
            norm_action = decision_result['normal_action']
            print(f"   🔄 Baseline Choice: {norm_action.get('action', 'N/A')}")
            print(f"      Reasoning: {norm_action.get('reason', 'N/A')[:60]}...")
    except Exception as e:
        print(f"   ⚠️ Decision making: Using fallback mode ({str(e)[:30]}...)")
    
    # Demonstrate enhanced conversation
    print("\\n6. Enhanced Conversation:")
    try:
        response = agent.converse_with_cognition("Barista", "Hello! What would you recommend?", environment)
        print(f"   💬 Conversation Response: {response[:80]}...")
    except Exception as e:
        print(f"   ⚠️ Conversation: Using fallback mode ({str(e)[:30]}...)")
    
    # Show memory state
    print("\\n7. Memory System Status:")
    memory_summary = agent.get_memory_summary()
    simple_mem = memory_summary.get('simple_memory', {})
    print(f"   📝 Simple Memory: {simple_mem.get('short_term_count', 0)} short-term, {simple_mem.get('long_term_count', 0)} long-term")
    
    reverie_mem = memory_summary.get('reverie_memory', {})
    if isinstance(reverie_mem, dict):
        events = reverie_mem.get('associative_events', 'N/A')
        thoughts = reverie_mem.get('associative_thoughts', 'N/A')
        print(f"   🧠 Reverie Memory: {events} events, {thoughts} thoughts")
    
    # Architecture summary
    print("\\n8. System Architecture:")
    print("   📊 Integration Stack:")
    print("      ├── SimpleAgent (Base behavioral system)")
    print("      ├── ReverieMemoryAdapter (Advanced memory)")
    print("      ├── CognitiveModuleWrapper (Reverie cognitive functions)")
    print("      ├── Enhanced perception, planning, reflection")
    print("      └── Graceful fallback mechanisms")
    
    print("\\n✨ === Demo Complete ===")
    print("\\n💡 Key Benefits:")
    print("   • Enhanced environmental perception")
    print("   • Sophisticated memory retrieval")
    print("   • Advanced planning and reflection")
    print("   • Robust fallback mechanisms")
    print("   • Full reverie cognitive module integration")
    
    return True

if __name__ == "__main__":
    demo_cognitive_capabilities()