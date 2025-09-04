#!/usr/bin/env python3
"""
Optimized WebSocket Decision Server using bit flags for state compression
Handles NPC decision making with minimal network overhead
"""

import asyncio
import json
import logging
import time
import websockets
from typing import Dict, Optional, Any
from pathlib import Path
from gpt4all import GPT4All

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)


class BitStateDecoder:
    """Decode compressed bit flags to state dictionary"""
    
    # Bit position to state name mapping
    STATE_FLAGS = {
        0: "counter_dirty",
        1: "counter_has_customers", 
        2: "table_dirty",
        3: "table_has_customers",
        4: "shelf_low",
        5: "shelf_empty",
        6: "pool_waiting",
        7: "music_playing",
        # Can extend up to 31 flags (32-bit integer)
    }
    
    def decode_state(self, state_int: int) -> Dict[str, bool]:
        """Convert bit flags to boolean dictionary"""
        result = {}
        for bit, name in self.STATE_FLAGS.items():
            result[name] = bool(state_int & (1 << bit))
        return result
    
    def encode_state(self, state_dict: Dict[str, bool]) -> int:
        """Convert boolean dictionary to bit flags"""
        result = 0
        for bit, name in self.STATE_FLAGS.items():
            if state_dict.get(name, False):
                result |= (1 << bit)
        return result


class NPCDecisionEngine:
    """Fast decision engine for NPCs using local LLM"""
    
    def __init__(self, model_name: str = "Llama-3.2-3B-Instruct-Q4_0.gguf"):
        # Initialize model
        model_path = Path(__file__).parent.parent.parent / "models" / "llms"
        self.model = GPT4All(
            model_name=model_name,
            model_path=str(model_path),
            allow_download=False,
            verbose=False
        )
        
        self.decoder = BitStateDecoder()
        
        # Action priority mapping
        self.action_priority = {
            "serve_customer": 10,
            "table_has_customers": 8,
            "restock_urgent": 7,
            "clean_counter": 5,
            "clear_table": 4,
            "restock": 3,
            "observe": 1,
            "take_break": 0
        }
        
        # Cache for recent decisions (avoid redundant LLM calls)
        self.decision_cache = {}
        self.cache_ttl = 2.0  # Cache for 2 seconds
    
    def state_to_perception(self, state_dict: Dict[str, bool], npc_name: str) -> str:
        """Convert state to natural language perception"""
        perceptions = []
        
        # Priority order for perception
        if state_dict.get("counter_has_customers"):
            perceptions.append("customers are waiting at the bar")
        if state_dict.get("table_has_customers"):
            perceptions.append("customers are at a table")
        if state_dict.get("shelf_empty"):
            perceptions.append("the liquor shelf is empty")
        if state_dict.get("counter_dirty"):
            perceptions.append("the bar counter is dirty")
        if state_dict.get("table_dirty"):
            perceptions.append("a table needs cleaning")
        if state_dict.get("shelf_low"):
            perceptions.append("supplies are running low")
        if state_dict.get("pool_waiting"):
            perceptions.append("someone is waiting for a pool partner")
        if state_dict.get("music_playing"):
            perceptions.append("music is playing")
        
        if not perceptions:
            return "Everything looks normal"
        
        return "You notice that " + " and ".join(perceptions)
    
    def get_npc_role(self, npc_name: str) -> str:
        """Get NPC role description"""
        roles = {
            "bob": "an experienced bartender",
            "alice": "a helpful bar regular", 
            "sam": "a cool musician"
        }
        return roles.get(npc_name.lower(), "a bar worker")
    
    def decide_action(self, npc_name: str, state_int: int) -> Dict[str, Any]:
        """Make decision based on compressed state"""
        
        # Check cache first
        cache_key = f"{npc_name}_{state_int}"
        if cache_key in self.decision_cache:
            cached_time, cached_decision = self.decision_cache[cache_key]
            if time.time() - cached_time < self.cache_ttl:
                logger.info(f"[{npc_name}] Using cached decision: {cached_decision['action']}")
                return cached_decision
        
        # Decode state
        state_dict = self.decoder.decode_state(state_int)
        
        # Use LLM for all decisions
        perception = self.state_to_perception(state_dict, npc_name)
        role = self.get_npc_role(npc_name)
        
        # Detailed prompt for LLM
        prompt = f"""You are {npc_name}, {role}.
{perception}

Choose ONE action: serve_customer, clean_counter, clear_table, restock, observe, take_break

Reply with just the action word."""
        
        # Fast generation with low temperature
        response = self.model.generate(
            prompt,
            max_tokens=20,  # Much smaller for speed
            temp=0.3,
            top_k=40,
            top_p=0.95
        )
        
        # Extract action from response
        action = self.extract_action(response)
        priority = self.action_priority.get(action, 1)
        decision = {"action": action, "priority": priority}
        
        # Cache decision
        self.decision_cache[cache_key] = (time.time(), decision)
        
        # Clean old cache entries
        if len(self.decision_cache) > 100:
            current_time = time.time()
            self.decision_cache = {
                k: v for k, v in self.decision_cache.items()
                if current_time - v[0] < self.cache_ttl
            }
        
        return decision
    
    def extract_action(self, response: str) -> str:
        """Extract action from LLM response"""
        response_lower = response.lower().strip()
        
        actions = ["serve_customer", "clean_counter", "clear_table", 
                  "restock", "observe", "take_break"]
        
        for action in actions:
            if action in response_lower:
                return action
        
        return "observe"  # Default


class DecisionServer:
    """WebSocket server for NPC decisions"""
    
    def __init__(self, port: int = 9998):
        self.port = port
        self.engine = NPCDecisionEngine()
        self.clients = set()
        self.stats = {
            "requests": 0,
            "avg_response_time": 0,
            "cache_hits": 0
        }
    
    async def handle_client(self, websocket):
        """Handle WebSocket client connection"""
        self.clients.add(websocket)
        logger.info(f"Client connected. Total clients: {len(self.clients)}")
        
        try:
            async for message in websocket:
                start_time = time.time()
                
                try:
                    data = json.loads(message)
                    
                    # Handle batch requests
                    if "batch" in data:
                        responses = {}
                        for npc_name, state_int in data["batch"].items():
                            decision = self.engine.decide_action(npc_name, state_int)
                            responses[npc_name] = decision
                        
                        await websocket.send(json.dumps({
                            "type": "batch_decisions",
                            "decisions": responses,
                            "timestamp": time.time()
                        }))
                    
                    # Handle single request
                    else:
                        npc_name = data.get("npc", "unknown")
                        state_int = data.get("state", 0)
                        
                        decision = self.engine.decide_action(npc_name, state_int)
                        
                        await websocket.send(json.dumps({
                            "type": "decision",
                            "npc": npc_name,
                            "action": decision["action"],
                            "priority": decision["priority"],
                            "timestamp": time.time()
                        }))
                    
                    # Update stats
                    response_time = time.time() - start_time
                    self.stats["requests"] += 1
                    self.stats["avg_response_time"] = (
                        (self.stats["avg_response_time"] * (self.stats["requests"] - 1) + response_time) 
                        / self.stats["requests"]
                    )
                    
                    logger.info(f"Decision made in {response_time:.3f}s (avg: {self.stats['avg_response_time']:.3f}s)")
                    
                except json.JSONDecodeError:
                    await websocket.send(json.dumps({
                        "type": "error",
                        "message": "Invalid JSON"
                    }))
                except Exception as e:
                    logger.error(f"Error processing request: {e}")
                    await websocket.send(json.dumps({
                        "type": "error",
                        "message": str(e)
                    }))
        
        except websockets.exceptions.ConnectionClosed:
            pass
        finally:
            self.clients.remove(websocket)
            logger.info(f"Client disconnected. Total clients: {len(self.clients)}")
    
    async def start(self):
        """Start the WebSocket server"""
        logger.info(f"Starting Decision Server on port {self.port}")
        logger.info("Optimizations enabled: bit flags, caching, batch processing")
        
        async with websockets.serve(self.handle_client, "localhost", self.port):
            logger.info(f"Decision Server ready at ws://localhost:{self.port}")
            
            # Print stats periodically
            while True:
                await asyncio.sleep(30)
                if self.stats["requests"] > 0:
                    logger.info(f"Stats: {self.stats['requests']} requests, "
                              f"avg response: {self.stats['avg_response_time']:.3f}s, "
                              f"clients: {len(self.clients)}")


def main():
    """Main entry point"""
    server = DecisionServer(port=9998)
    
    print("="*60)
    print("NPC DECISION SERVER (Optimized)")
    print("="*60)
    print("Features:")
    print("- Bit flag state compression")
    print("- Decision caching (2s TTL)")
    print("- Batch request support")
    print("- Rule-based fast paths")
    print("="*60)
    
    try:
        asyncio.run(server.start())
    except KeyboardInterrupt:
        logger.info("Server stopped by user")


if __name__ == "__main__":
    main()