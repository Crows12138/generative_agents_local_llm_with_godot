import time
import random
from typing import Any, List, Tuple, TYPE_CHECKING
from datetime import datetime

if TYPE_CHECKING:
    from basic_functions.maze import Object
from basic_functions.memory.memory import Memory, MemoryType
from basic_functions.plan.plan import generate_daily_schedule
from basic_functions.decider.decider import BottomDecider, set_enhanced_reflection_for_decider
from basic_functions.reflection.reflection import reflect, get_enhanced_reflection
from basic_functions.decider.decider import ActionIntent
from ai_service.ai_service import local_llm_generate
from basic_functions.perception.embedding import get_single_embedding


class Persona:
    def __init__(
        self, name: str, initial_location: Tuple[float, ...] = (0.0, 0.0, 0.0)
    ):
        self.name = name
        if len(initial_location) == 2:
            loc = (float(initial_location[0]), float(initial_location[1]), 0.0)
        else:
            loc = (
                float(initial_location[0]),
                float(initial_location[1]),
                float(initial_location[2]),
            )
        self.location: Tuple[float, float, float] = loc
        self.x, self.y, self.z = self.location
        
        # Short-term memory (original Memory system)
        from basic_functions.memory.memory import Memory
        self.memory = Memory()
        
        # Enhanced memory retriever
        from basic_functions.memory.enhanced_memory_retriever import EnhancedMemoryRetriever
        self.memory_retriever = EnhancedMemoryRetriever(self.memory)
        
        # Enhanced reflection system (includes medium-term and long-term memory)
        from basic_functions.reflection.reflection import get_enhanced_reflection
        self.enhanced_reflection = get_enhanced_reflection()
        
        # Set decider to use enhanced reflection system
        from basic_functions.decider.decider import set_enhanced_reflection_for_decider
        set_enhanced_reflection_for_decider(self.enhanced_reflection)
        
        # Personality description for self-identification
        self.personality_description = f"I am {name}."
        
        self.inventory = []
        self.previous_action = None
        self.long_term_goals = []
        self.today_schedule = []
        self.step_counter = 0
        self.game_hours_passed = 0
        self.summarize_every = 10
        self.short_term_events = []
        self.last_daily_reflection_date = None
        self.last_weekly_reflection_date = None
        
        # Failure tracking to avoid repeated failed actions
        self.failed_actions = {}  # Track failed actions and their counts
        self.last_failed_action = None
        self.consecutive_failures = 0
        
        # instantiate the low-level planner
        from basic_functions.decider.decider import BottomDecider
        self.planner = BottomDecider()

    def _set_location(self, loc: Tuple[float, float, float]) -> None:
        """Update location tuple and individual coordinates."""
        self.location = (float(loc[0]), float(loc[1]), float(loc[2]))
        self.x, self.y, self.z = self.location

    def describe(self) -> str:
        """Return a short description used by the maze."""
        return f"Persona {self.name}"

    def _get_current_high_level_task(self) -> str:
        """Get current high-level task from long-term goals and refined goals."""
        # Start with long-term goals from settings
        high_level_goals = self.long_term_goals.copy() if self.long_term_goals else []
        
        # Add refined goals from long-term memory if available
        if self.enhanced_reflection and hasattr(self.enhanced_reflection, 'long_term_memory'):
            refined_goals = self.enhanced_reflection.long_term_memory.get_latest_goals()
            if refined_goals:
                high_level_goals.extend([goal["content"] for goal in refined_goals])
        
        return "; ".join(high_level_goals) if high_level_goals else "Survive and thrive in this environment"

    def _get_current_task_from_schedule(self, current_time: str) -> str:
        """Get the current task based on time and schedule."""
        if not self.today_schedule or not isinstance(self.today_schedule, list):
            return "No current task"
        
        try:
            # Parse current time (format: "HH:MM")
            current_hour, current_minute = map(int, current_time.split(":"))
            current_minutes = current_hour * 60 + current_minute
            
            for task in self.today_schedule:
                if isinstance(task, dict):
                    start_time = task.get("start", "")
                    end_time = task.get("end", "")
                    task_desc = task.get("task", "")
                    
                    if start_time and end_time:
                        # Parse start and end times
                        start_hour, start_minute = map(int, start_time.split(":"))
                        end_hour, end_minute = map(int, end_time.split(":"))
                        
                        start_minutes = start_hour * 60 + start_minute
                        end_minutes = end_hour * 60 + end_minute
                        
                        # Handle overnight tasks (e.g., 22:00 to 06:00)
                        if end_minutes < start_minutes:
                            if current_minutes >= start_minutes or current_minutes < end_minutes:
                                return task_desc
                        else:
                            if start_minutes <= current_minutes < end_minutes:
                                return task_desc
            
            return "Free time"
        except:
            return "No current task"

    def _format_daily_schedule(self) -> str:
        """Format the daily schedule for the decider."""
        if not self.today_schedule:
            return "No specific schedule for today."
        
        # Format the schedule as a readable string
        if isinstance(self.today_schedule, list):
            tasks = []
            for task in self.today_schedule:
                if isinstance(task, dict):
                    # Handle plan.py format: start, end, task
                    start_time = task.get("start", task.get("time", ""))
                    task_desc = task.get("task", task.get("activity", ""))
                    end_time = task.get("end", "")
                    
                    if start_time and task_desc:
                        if end_time:
                            tasks.append(f"{start_time}-{end_time}: {task_desc}")
                        else:
                            tasks.append(f"{start_time}: {task_desc}")
                    elif task_desc:
                        tasks.append(task_desc)
                else:
                    tasks.append(str(task))
            return "; ".join(tasks) if tasks else "No specific schedule for today."
        else:
            return str(self.today_schedule) if self.today_schedule else "No specific schedule for today."

    def start_new_day(self, current_date: str):
        """
        Generate today's schedule for the agent and store it in memory.
        """
        # Load historical memories
        self.enhanced_reflection.load_memories_from_disk(self.name)
        
        # Get current high-level goals
        current_goals = self._get_current_high_level_task()
        high_level_goals_list = [goal.strip() for goal in current_goals.split(";") if goal.strip()]
        
        # Get medium-term memories for context
        medium_term_context = ""
        if self.enhanced_reflection and hasattr(self.enhanced_reflection, 'get_behavior_context_for_decider'):
            medium_term_context = self.enhanced_reflection.get_behavior_context_for_decider()
        
        self.today_schedule = generate_daily_schedule(
            agent_name=self.name,
            personality_description=self.personality_description,
            high_level_goals=high_level_goals_list,
            medium_term_memories=medium_term_context,
            current_date=current_date,
        )

        schedule_txt = f"{current_date} schedule: {self.today_schedule}"
        schedule_emb = get_single_embedding(schedule_txt)
        self.memory.add(
            text=schedule_txt,
            embedding=schedule_emb,
            event_type="daily_schedule",
            metadata={"date": current_date},
            ttl=24 * 3600,
            memory_type=MemoryType.REFLECTION,
        )

    def _summarize_events(self) -> None:
        if not self.short_term_events:
            return
        lines = "\n".join(
            f"{i+1}. {txt}" for i, txt in enumerate(self.short_term_events)
        )
        prompt = (
            "Summarize the following events in one or two sentences, focusing only on the main actions and location. "
            "Do not add any explanations or commentary. Output only the summary:\n"
            f"{lines}"
        )
        summary = local_llm_generate(prompt).strip()
        if not summary:
            summary = "; ".join(self.short_term_events)
        
        # Clean up any explanatory text that might have been generated
        if "Assistant:" in summary:
            summary = summary.split("Assistant:")[-1].strip()
        if "To summarize" in summary:
            summary = summary.split("To summarize")[-1].strip()
        
        emb = get_single_embedding(summary)
        
        # Store using new memory system
        self.memory.add(
            text=summary, 
            embedding=emb, 
            event_type="summary",
            memory_type=MemoryType.REFLECTION,
            ttl=24 * 3600,  # 1 day TTL
        )
        self.short_term_events.clear()

    def step(self, maze: Any, current_time: str):
        """
        One tick:
          1) perceive & remember sight
          2) retrieve memories
          3) low-level decide
          4) perform action
          5) remember action (with dedup)
          6) check for daily/weekly reflection

        Returns the list of memories retrieved in step 2 so that callers can
        reuse them for debugging without performing another retrieval.
        """
        from basic_functions.perception.perceive import perceive
        from basic_functions.perception.describe import describe_cell, get_enhanced_perception_summary

        self.step_counter += 1
        self.game_hours_passed = self.step_counter // 60  # Assume 60 steps per game hour

        # 1) perceive and store in short-term buffer with proper tagging
        percept = perceive(self, maze, radius=3)  # Increased radius for better perception
        descs = percept["descriptions"]
        emb = percept["merged_embedding"]
        
        # Use enhanced perception summary for better context
        enhanced_perception = get_enhanced_perception_summary(self, maze, radius=3)
        
        if descs:
            sight_text = f"{self.name} perceived {enhanced_perception} at {self.location}"
            
            # Store directly to short-term memory, tagged as sight
            self.memory.add_sight(
                text=sight_text,
                embedding=emb,
                location=self.location,
                ttl=24 * 3600,  # 1 day TTL
                persona_name=self.name,
            )
            
            self.short_term_events.append(sight_text)

        # 2) retrieve summaries relevant to the current perception
        # Use enhanced memory retriever for better contextual retrieval
        if self.enhanced_reflection:
            # Use enhanced memory retriever for contextual memories
            relevant = self.memory_retriever.retrieve_contextual_memories(
                query=enhanced_perception,
                persona_name=self.name,
                current_location=self.location,
                current_time=current_time,
                top_k=3,
                time_window_hours=24
            )
        else:
            # Fallback to basic retrieval
            relevant = self.memory.retrieve_similar(
                query_emb=emb,
                top_k=3,
                filter_event="summary",
            )

        # 3) low-level decide; use enhanced perception summary
        intent = self.planner.decide(
            persona_name=self.name,
            location=self.location,
            surroundings_desc=enhanced_perception,  # Use enhanced perception
            similar_memories=relevant,
            high_level_task=self._get_current_high_level_task(),
            self_identification=self.personality_description,
            previous_action=self.previous_action,
            use_enhanced_memory=True,  # Use enhanced memory system
            daily_schedule=self._get_current_task_from_schedule(current_time),
            failed_actions=self.failed_actions,
        )

        print(f"[LLM RAW RESPONSE] {intent.description!r}")

        # 4) perform action
        self.perform_action(intent, maze)

        # 5) remember action in short-term buffer, but skip if duplicated
        action_text = f"{self.name} did {intent.action_type}" + (
            f" -> {intent.target}" if getattr(intent, "target", None) else ""
        )
        now_ts = time.time()
        window_sec = 60.0
        dup_found = False
        for e in reversed(self.memory.entries[-10:]):
            if (
                e.event_type == "action"
                and e.text == action_text
                and (now_ts - e.timestamp) < window_sec
            ):
                dup_found = True
                break

        if not dup_found:
            # Store to short-term memory, tagged as action
            action_emb = get_single_embedding(action_text)
            self.memory.add_action(
                text=action_text,
                embedding=action_emb,
                location=self.location,
                ttl=24 * 3600,  # 1 day TTL
                persona_name=self.name,
            )
            
            self.short_term_events.append(action_text)

        if (
            len(self.short_term_events) >= 5
            or self.step_counter % self.summarize_every == 0
        ):
            self._summarize_events()

        # 6) Check for daily/weekly reflection
        current_date = datetime.now().strftime("%Y-%m-%d")
        self._check_and_perform_reflections(current_date)

        # Return retrieved memories so external callers can inspect them
        return relevant

    def _check_and_perform_reflections(self, current_date: str):
        """Check and perform daily and weekly reflections"""
        # Daily reflection - every 24 game hours (24 real minutes)
        if (self.game_hours_passed % 24 == 0 and 
            self.game_hours_passed > 0 and 
            self.last_daily_reflection_date != current_date):
            
            self.perform_daily_reflection(current_date)
            self.last_daily_reflection_date = current_date

        # Weekly reflection - every 7 game days (7*24 real minutes)
        if (self.game_hours_passed % (24 * 7) == 0 and 
            self.game_hours_passed > 0 and 
            self.last_weekly_reflection_date != current_date):
            
            self.perform_weekly_reflection(current_date)
            self.last_weekly_reflection_date = current_date

    def perform_daily_reflection(self, current_date: str):
        """Perform daily reflection and generate medium-term memory."""
        print(f"[Daily Reflection] {self.name} performing daily reflection for {current_date}")
        
        # Get today's short-term memories
        recent_memories = self.memory.get_recent_memories(hours=24)
        
        if not recent_memories:
            print(f"[Daily Reflection] No recent memories found for {self.name}")
            return
        
        # Perform daily reflection
        reflection_result = self.enhanced_reflection.daily_reflection(
            agent_name=self.name,
            current_date=current_date,
            short_term_memories=recent_memories,
            save_to_medium_term=True,
        )
        
        # Save reflection results to short-term memory
        reflection_text = f"Daily reflection: {reflection_result['behavior_summary']}"
        reflection_emb = get_single_embedding(reflection_text)
        self.memory.add(
            text=reflection_text,
            embedding=reflection_emb,
            event_type="daily_reflection",
            metadata={"date": current_date},
            ttl=24 * 3600,
            memory_type=MemoryType.REFLECTION,
        )
        
        # Save to disk
        self.enhanced_reflection.save_memories_to_disk(self.name)
        
        print(f"[Daily Reflection] Completed for {self.name}")

    def perform_weekly_reflection(self, current_date: str):
        """Perform weekly reflection and generate long-term memory"""
        print(f"[Weekly Reflection] {self.name} performing weekly reflection for {current_date}")
        
        # Perform weekly reflection
        reflection_result = self.enhanced_reflection.weekly_reflection(
            agent_name=self.name,
            current_date=current_date,
            days_back=7,
            save_to_long_term=True,
        )
        
        # Save reflection results to short-term memory
        reflection_text = f"Weekly reflection insights: {len(reflection_result['personality_insights'])} personality insights, {len(reflection_result['values_discovered'])} values discovered"
        reflection_emb = get_single_embedding(reflection_text)
        self.memory.add(
            text=reflection_text,
            embedding=reflection_emb,
            event_type="weekly_reflection",
            metadata={"date": current_date},
            ttl=24 * 3600,
            memory_type=MemoryType.REFLECTION,
        )
        
        # Save to disk
        self.enhanced_reflection.save_memories_to_disk(self.name)
        
        print(f"[Weekly Reflection] Completed for {self.name}")

    def perform_action(self, intent, maze: Any):
        """
        Execute the low-level intent, using coordinates not Tile objects.
        """
        action = getattr(intent, "action_type", None)
        target = getattr(intent, "target", None)

        if action == "move" and target:
            # move toward named target
            path = maze.find_path(self.location, target)
            if path:
                # path contains list of (x,y,z) tuples
                self._set_location(path[0])

        elif action == "move_towards" and target:
            # move toward named target
            path = maze.find_path(self.location, target)
            if path:
                # path contains list of (x,y,z) tuples
                self._set_location(path[0])
                self.short_term_events.append(f"Moved towards {target}")
            else:
                self.short_term_events.append(f"Failed to move towards {target} - no path found")
                self._record_failed_action(action, target, "no path found")

        elif action == "pickup" and target:
            # Try to pickup object from current location
            from basic_functions.maze import Object
            
            # Check if object exists at current location and is portable
            target_found = False
            action_success = False
            
            # This would require checking maze objects at current location
            # For now, we'll simulate based on object properties
            if "tree" in str(target).lower() or "sink" in str(target).lower() or "faucet" in str(target).lower() or "stove" in str(target).lower():
                self.short_term_events.append(f"Failed to pickup {target} - not portable")
                self._record_failed_action(action, target, "not portable")
            else:
                # Assume other objects can be picked up
                self.short_term_events.append(f"Successfully picked up {target}")
                action_success = True
            
            if action_success:
                self.consecutive_failures = 0

        elif action == "drop" and target:
            # Drop object from inventory
            target_found = False
            for obj in list(self.inventory):
                if isinstance(obj, Object) and obj.name.lower() == str(target).lower():
                    self.inventory.remove(obj)
                    self.short_term_events.append(f"Dropped {obj.name}")
                    target_found = True
                    break
            
            if not target_found:
                self.short_term_events.append(f"Failed to drop {target} - not in inventory")
                self._record_failed_action(action, target, "not in inventory")

        elif action == "talk_to" and target:
            # Talk to target (could be another agent or object)
            # For now, assume this always succeeds
            self.short_term_events.append(f"Talked to {target}")

        elif action == "create" and target:
            from basic_functions.maze import Object

            new_object = Object(name=str(target), x=self.x, y=self.y, z=self.z)
            maze.place_object(new_object, int(self.x), int(self.y), int(self.z))
            self.inventory.append(new_object)
            self.short_term_events.append(f"Created {target}")

        elif action == "wander":
            # random neighbor move via get_walkable_neighbors
            x, y, z = self.location
            neighbors = maze.get_walkable_neighbors(x, y, z)
            if neighbors:
                self._set_location(random.choice(neighbors))
                self.short_term_events.append("Wandered to a new location")
            else:
                self.short_term_events.append("Failed to wander - no available neighbors")
                self._record_failed_action(action, "wander", "no available neighbors")

        elif action == "wait":
            # do nothing this tick
            self.short_term_events.append("Waited")
            pass

        elif action == "eat" and target:
            from basic_functions.maze import Object

            # Check if target is in inventory and is edible
            target_found = False
            action_success = False
            for obj in list(self.inventory):
                if isinstance(obj, Object) and obj.name.lower() == str(target).lower():
                    if obj.edible:
                        self.inventory.remove(obj)
                        self.short_term_events.append(f"Successfully ate {obj.name}")
                        action_success = True
                    else:
                        self.short_term_events.append(f"Failed to eat {obj.name} - not edible")
                        self._record_failed_action(action, target, "not edible")
                    target_found = True
                    break
            
            if not target_found:
                # Target not in inventory, check if it's an environment object
                self.short_term_events.append(f"Failed to eat {target} - not in inventory")
                self._record_failed_action(action, target, "not in inventory")
            
            # If action was successful, reset failure counter
            if action_success:
                self.consecutive_failures = 0

        # store the executed intent for the next planning step
        self.previous_action = intent

    def end_of_day_reflection(self, current_date: str):
        """
        Backward-compatible daily reflection function
        """
        # Execute original reflection logic
        result = reflect(
            agent_name=self.name,
            current_date=current_date,
            today_schedule=self.today_schedule,
            memory=self.memory.entries,
            top_k=5,
        )
        
        # Also execute new daily reflection
        self.perform_daily_reflection(current_date)
        
        # Save original reflection results
        reflection_emb = get_single_embedding(f"Reflection insights: {result['insights']}; adjustments: {result['adjustments']}")
        self.memory.add(
            text=f"Reflection insights: {result['insights']}; adjustments: {result['adjustments']}",
            embedding=reflection_emb,
            event_type="reflection",
            metadata={"date": current_date},
            ttl=24 * 3600,
            memory_type=MemoryType.REFLECTION,
        )

    def display_all_memories(self) -> str:
        """Display all three layers of memory"""
        result = f"=== {self.name} Memory System ===\n\n"
        
        # Short-term memory
        result += "SHORT-TERM MEMORIES:\n"
        result += self.memory.display_memories(limit=20)
        result += "\n\n"
        
        # Enhanced memory summary
        result += "ENHANCED MEMORY SUMMARY:\n"
        result += self.memory_retriever.get_memory_summary(self.name, time_window_hours=24)
        result += "\n\n"
        
        # Medium-term and long-term memory
        result += self.enhanced_reflection.display_all_memories()
        
        return result
    
    def get_emotional_memories(self, emotion_keywords: List[str], top_k: int = 5) -> List[Any]:
        """Retrieve memories based on emotional context."""
        return self.memory_retriever.retrieve_by_emotional_context(
            emotion_keywords, self.name, top_k=top_k
        )
    
    def get_relationship_memories(self, target_name: str, top_k: int = 5) -> List[Any]:
        """Retrieve memories related to a specific person or object."""
        return self.memory_retriever.retrieve_by_relationship(
            target_name, self.name, top_k=top_k
        )
    
    def get_location_memories(self, location: Any, top_k: int = 5) -> List[Any]:
        """Retrieve memories related to a specific location."""
        return self.memory_retriever.retrieve_by_location(
            location, self.name, top_k=top_k
        )
    
    def get_recent_important_memories(self, top_k: int = 5, hours: int = 24) -> List[Any]:
        """Retrieve recent memories with high importance scores."""
        return self.memory_retriever.retrieve_recent_important_memories(
            self.name, top_k=top_k, hours=hours
        )
    
    def get_pattern_memories(self, pattern_description: str, top_k: int = 5) -> List[Any]:
        """Retrieve memories that match a behavioral pattern."""
        return self.memory_retriever.retrieve_pattern_memories(
            pattern_description, self.name, top_k=top_k
        )
    
    def get_relationship_with(self, target_name: str) -> str:
        """Get relationship summary with another entity."""
        from basic_functions.conversation.relationship_tracker import relationship_tracker
        return relationship_tracker.get_relationship_summary(self.name, target_name)
    
    def get_all_relationships(self) -> List[Any]:
        """Get all relationships for this persona."""
        from basic_functions.conversation.relationship_tracker import relationship_tracker
        return relationship_tracker.get_entity_relationships(self.name)
    
    def get_recent_interactions_with(self, target_name: str, hours: int = 24) -> List[Any]:
        """Get recent interactions with a specific entity."""
        from basic_functions.conversation.relationship_tracker import relationship_tracker
        return relationship_tracker.get_recent_interactions(self.name, target_name, hours)

    def get_memory_stats(self) -> dict:
        """Get memory statistics"""
        stats = {
            "short_term_count": len(self.memory.entries),
            "medium_term_count": len(self.enhanced_reflection.medium_term_memory.entries),
            "long_term_count": len(self.enhanced_reflection.long_term_memory.entries),
            "game_hours_passed": self.game_hours_passed,
            "step_counter": self.step_counter,
        }
        
        # Add long-term memory statistics
        lt_stats = self.enhanced_reflection.long_term_memory.get_stats()
        stats.update({"long_term_" + k: v for k, v in lt_stats.items()})
        
        return stats

    def cleanup_expired_memories(self) -> dict:
        """Clean up expired memories"""
        short_cleaned = self.memory.cleanup_expired()
        enhanced_cleaned = self.enhanced_reflection.cleanup_memories()
        
        return {
            "short_term_cleaned": short_cleaned,
            **enhanced_cleaned
        }

    def save_memories_to_disk(self):
        """Save memories to disk"""
        self.enhanced_reflection.save_memories_to_disk(self.name)

    def load_memories_from_disk(self):
        """Load memories from disk"""
        self.enhanced_reflection.load_memories_from_disk(self.name)

    def _record_failed_action(self, action: str, target: str, reason: str):
        """Record a failed action to avoid repeating it"""
        action_key = f"{action}_{target}"
        if action_key not in self.failed_actions:
            self.failed_actions[action_key] = {"count": 0, "reasons": []}
        
        self.failed_actions[action_key]["count"] += 1
        self.failed_actions[action_key]["reasons"].append(reason)
        self.last_failed_action = action_key
        self.consecutive_failures += 1
        
        # Clean up old failed actions (older than 20 steps or after many failures)
        if self.step_counter > 20:
            old_actions = [k for k in self.failed_actions.keys() 
                          if self.failed_actions[k]["count"] > 5]  # Only clean up after 5+ failures
            for old_action in old_actions:
                del self.failed_actions[old_action]

    def _should_avoid_action(self, action: str, target: str) -> bool:
        """Check if we should avoid a specific action based on recent failures"""
        action_key = f"{action}_{target}"
        if action_key in self.failed_actions:
            return self.failed_actions[action_key]["count"] >= 3  # Only avoid after 3+ failures
        return False
