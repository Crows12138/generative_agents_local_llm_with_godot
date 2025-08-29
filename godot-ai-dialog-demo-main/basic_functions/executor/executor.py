import random
from typing import Callable, Dict, List, Optional

from basic_functions.maze import Maze, Object
from basic_functions.persona import Persona
from basic_functions.decider.decider import ActionIntent
try:
    # Try to use cognitive dual model for dialogue (1.7B for fast responses)
    from basic_functions.cognitive.cognitive_llm_service import get_cognitive_llm_service
    def get_dialogue_service():
        return get_cognitive_llm_service()
except ImportError:
    # Fallback to original dialogue service
    try:
        from ai_service.dialogue_service import get_dialogue_service
    except ImportError:
        # Final fallback
        def get_dialogue_service():
            return None
from basic_functions.logger.action_logger import log_action
from basic_functions.conversation.relationship_tracker import relationship_tracker, InteractionType
from basic_functions.conversation.enhanced_dialogue_generator import dialogue_generator, DialogueContext

# Map each built-in action name to a handler method signature:
#   handler(intent: ActionIntent, persona: Persona, maze: Maze) -> None
ACTION_HANDLERS: Dict[str, Callable[[ActionIntent, Persona, Maze], None]] = {}


class SimpleExecutor:
    """
    Execute a low-level ActionIntent against the maze and persona.
    """

    def __init__(self):
        # register all handlers
        handlers = {
            "move_towards": self._handle_move_towards,
            "move_away": self._handle_move_away,
            "throw_towards": self._handle_throw_towards,
            "pickup": self._handle_pickup,
            "drop": self._handle_drop,
            "talk_to": self._handle_talk_to,
            "create": self._handle_create,
            "eat": self._handle_eat,
            "sleep": self._handle_sleep,
            "read": self._handle_read,
            "write": self._handle_write,
            "search": self._handle_search,
            "observe": self._handle_observe,
            "think": self._handle_think,
            "rest": self._handle_rest,
            "exercise": self._handle_exercise,
            "work": self._handle_work,
            "play": self._handle_play,
            "wander": self._handle_wander,
            "wait": self._handle_wait,
        }
        ACTION_HANDLERS.update(handlers)

    def execute(self, intent: ActionIntent, persona: Persona, maze: Maze) -> None:
        # 1) Try the exact action
        handler = ACTION_HANDLERS.get(intent.action_type)

        # 2) If not found, try borrowed_action
        if (
            handler is None
            and hasattr(intent, "borrowed_action")
            and getattr(intent, "borrowed_action")
        ):
            handler = ACTION_HANDLERS.get(getattr(intent, "borrowed_action"))

        # 3) If still none, record as custom
        if handler is None:
            persona.memory.add(
                text=f"Performed custom action: {intent.action_type} -> {intent.target}",
                embedding=[],
                location=persona.location,
                event_type="action",
                importance=0.1,
                metadata={
                    "action": intent.action_type,
                    "target": intent.target,
                    "borrowed": getattr(intent, "borrowed_action", None),
                },
                ttl=None,
            )
            return

        # 4) Call the resolved handler
        handler(intent, persona, maze)
        persona.previous_action = intent
    
    def execute_batch_dialogue(self, persona: Persona, maze: Maze, targets: List[str]) -> List[str]:
        """
        Execute dialogue with multiple targets in parallel.
        
        Args:
            persona: The persona initiating the dialogue
            maze: The maze containing all entities
            targets: List of target names to talk to
            
        Returns:
            List of responses from each target
        """
        if not targets:
            return []
        
        x, y, z = persona.location
        nearby_entities = maze.spatial.nearby(x, y, 3.0, 0.0)  # Check within 3 units
        
        # Prepare dialogues for parallel processing
        dialogues = []
        valid_targets = []
        
        for target_name in targets:
            speaker = None
            speaker_description = ""
            
            # Find the target
            for entity in nearby_entities:
                if isinstance(entity, Persona) and entity.name.lower() == target_name.lower():
                    speaker = entity
                    speaker_description = f"You are {entity.name}, a persona in this world. You have your own personality, goals, and memories. Respond naturally to conversations."
                    break
                elif isinstance(entity, Object) and entity.name.lower() == target_name.lower():
                    speaker = entity
                    speaker_description = f"You are {entity.name}, {entity.description}. You can speak and interact with others. Respond naturally and in character."
                    break
            
            # Check for self-talk
            if speaker is None and persona.name.lower() == target_name.lower():
                speaker = persona
                speaker_description = f"You are {persona.name}, reflecting on yourself. You have your own personality, goals, and memories. Respond naturally to self-reflection."
            
            if speaker is not None:
                conversation_history = self._get_conversation_history(persona, speaker)
                message = self._generate_message_context(persona, speaker, maze)
                
                dialogues.append((speaker.name, speaker_description, message, conversation_history))
                valid_targets.append(target_name)
        
        if not dialogues:
            return []
        
        # Use parallel dialogue processing
        dialogue_service = get_dialogue_service()
        responses = dialogue_service.batch_dialogue(dialogues)
        
        # Store conversations in memory
        for i, (target_name, response) in enumerate(zip(valid_targets, responses)):
            conversation_text = f"{persona.name}: [message to {target_name}] | {target_name}: {response}"
            persona.memory.add(
                text=conversation_text,
                embedding=[],
                location=persona.location,
                event_type="talk",
                importance=0.8,
                metadata={
                    "target": target_name,
                    "speaker": target_name,
                    "response": response,
                    "batch_dialogue": True
                },
                ttl=None,
            )
        
        return responses

    # --- Handlers for each built-in action ---

    def _handle_move_towards(self, intent, persona, maze):
        """Handle move_towards action - persona moves toward a target location or object."""
        target = intent.target
        
        # Handle case where target is None
        if target is None:
            persona.memory.add(
                text=f"Tried to move towards something but no target was specified",
                embedding=[],
                location=persona.location,
                event_type="move",
                importance=0.3,
                metadata={
                    "target": None,
                    "action": "move_towards",
                    "success": False,
                    "reason": "no_target"
                },
                ttl=None,
            )
            return
            
        target_location = None
        
        # If target is a tuple (coordinates), use it directly
        if isinstance(target, (tuple, list)) and len(target) >= 2:
            target_location = (int(target[0]), int(target[1]))
        else:
            # Search for the target object or persona
            x, y, z = persona.location
            nearby_entities = maze.spatial.nearby(x, y, z, 10.0)  # Search in larger radius
            
            for entity in nearby_entities:
                if isinstance(entity, (Object, Persona)) and entity.name.lower() == str(target).lower():
                    if hasattr(entity, 'location'):
                        target_location = entity.location
                    else:
                        target_location = (entity.x, entity.y)
                    break
        
        if target_location:
            # Simple direct movement towards target
            current_x, current_y, current_z = persona.location
            target_x, target_y, target_z = target_location
            
            # Calculate direction vector
            dx = target_x - current_x
            dy = target_y - current_y
            dz = target_z - current_z
            
            # Normalize and move one step towards target
            distance = (dx**2 + dy**2 + dz**2)**0.5
            if distance > 0:
                step_size = min(1.0, distance)  # Move at most 1 unit
                new_x = current_x + (dx / distance) * step_size
                new_y = current_y + (dy / distance) * step_size
                new_z = current_z + (dz / distance) * step_size
                
                # Update position
                maze.remove_agent(persona)
                persona.location = (new_x, new_y, new_z)
                maze.place_agent(persona, int(new_x), int(new_y), int(new_z))
                
                # Record successful movement
                persona.memory.add(
                    text=f"Moved towards {target}. Now at {persona.location}.",
                    embedding=[],
                    location=persona.location,
                    event_type="move",
                    importance=0.4,
                    metadata={
                        "target": str(target),
                        "destination": persona.location,
                        "action": "move_towards",
                        "success": True
                    },
                    ttl=None,
                )
            else:
                # Path not found
                persona.memory.add(
                    text=f"Tried to move towards {target} but couldn't find a path",
                    embedding=[],
                    location=persona.location,
                    event_type="move",
                    importance=0.3,
                    metadata={
                        "target": str(target),
                        "action": "move_towards",
                        "success": False,
                        "reason": "no_path"
                    },
                    ttl=None,
                )
        else:
            # Target not found
            persona.memory.add(
                text=f"Tried to move towards {target} but couldn't find it",
                embedding=[],
                location=persona.location,
                event_type="move",
                importance=0.3,
                metadata={
                    "target": str(target),
                    "action": "move_towards",
                    "success": False,
                    "reason": "target_not_found"
                },
                ttl=None,
            )

    def _handle_move_away(self, intent, persona, maze):
        """Handle move_away action - persona moves away from a target."""
        target = intent.target
        
        # Handle case where target is None
        if target is None:
            persona.memory.add(
                text=f"Tried to move away from something but no target was specified",
                embedding=[],
                location=persona.location,
                event_type="move",
                importance=0.3,
                metadata={
                    "target": None,
                    "action": "move_away",
                    "success": False,
                    "reason": "no_target"
                },
                ttl=None,
            )
            return
            
        x, y, z = persona.location
        
        # Find target location
        target_location = None
        if isinstance(target, (tuple, list)) and len(target) >= 2:
            target_location = (int(target[0]), int(target[1]))
        else:
            # Search for the target object or persona
            nearby_entities = maze.spatial.nearby(x, y, 10.0, 0.0)
            for entity in nearby_entities:
                if isinstance(entity, (Object, Persona)) and entity.name.lower() == str(target).lower():
                    if hasattr(entity, 'location'):
                        target_location = entity.location
                    else:
                        target_location = (entity.x, entity.y)
                    break
        
        # Get walkable neighbors
        nbrs = maze.get_walkable_neighbors(x, y)
        if not nbrs:
            persona.memory.add(
                text=f"Tried to move away from {target} but no walkable paths available",
                embedding=[],
                location=persona.location,
                event_type="move",
                importance=0.3,
                metadata={
                    "target": str(target),
                    "action": "move_away",
                    "success": False,
                    "reason": "no_walkable_neighbors"
                },
                ttl=None,
            )
            return
        
        # Find the neighbor that maximizes distance from target
        best = None
        best_dist2 = -1.0
        
        if target_location:
            tx, ty = target_location
            for nx, ny in nbrs:
                # Calculate distance from neighbor to target
                dx = nx - tx
                dy = ny - ty
                d2 = dx * dx + dy * dy
                if d2 > best_dist2:
                    best_dist2 = d2
                    best = (nx, ny)
        else:
            # If target not found, just move to a random neighbor
            best = random.choice(nbrs)
        
        if best:
            maze.remove_agent(persona)
            persona.location = best
            maze.place_agent(persona, *best)
            
            persona.memory.add(
                text=f"Moved away from {target}. Now at {persona.location}.",
                embedding=[],
                location=persona.location,
                event_type="move",
                importance=0.4,
                metadata={
                    "target": str(target),
                    "destination": persona.location,
                    "action": "move_away",
                    "success": True
                },
                ttl=None,
            )

    def _handle_throw_towards(self, intent, persona, maze):
        """Handle throw_towards action - persona throws an object toward a target."""
        # intent.target might be tuple (source, dest) or string "object target"
        if isinstance(intent.target, (tuple, list)) and len(intent.target) >= 2:
            src, dst = intent.target[0], intent.target[1]
        elif isinstance(intent.target, str) and " " in intent.target:
            parts = intent.target.split(" ", 1)
            src, dst = parts[0], parts[1]
        else:
            src, dst = intent.target, "somewhere"
        
        # Check if persona has the object to throw
        thrown_object = None
        for item in list(persona.inventory):
            if isinstance(item, Object) and item.name.lower() == str(src).lower():
                thrown_object = item
                persona.inventory.remove(item)
                break
        
        if thrown_object:
            # Find destination location
            x, y, z = persona.location
            dest_location = None
            
            # Search for destination target
            nearby_entities = maze.spatial.nearby(x, y, 10.0, 0.0)
            for entity in nearby_entities:
                if isinstance(entity, (Object, Persona)) and entity.name.lower() == str(dst).lower():
                    if hasattr(entity, 'location'):
                        dest_location = entity.location
                    else:
                        dest_location = (entity.x, entity.y)
                    break
            
            # Place the thrown object at destination or nearby
            if dest_location:
                maze.place_object(thrown_object, dest_location[0], dest_location[1])
                result_text = f"Threw {src} towards {dst}. The object landed near the target."
                importance = 0.6
            else:
                # Throw in a random nearby location
                neighbors = maze.get_walkable_neighbors(x, y)
                if neighbors:
                    throw_location = random.choice(neighbors)
                    maze.place_object(thrown_object, throw_location[0], throw_location[1])
                    result_text = f"Threw {src} towards {dst}. The object landed somewhere nearby."
                else:
                    # Can't throw, put back in inventory
                    persona.inventory.append(thrown_object)
                    result_text = f"Tried to throw {src} towards {dst} but couldn't find a place to throw it."
                importance = 0.4
        else:
            result_text = f"Tried to throw {src} towards {dst} but don't have that object."
            importance = 0.3
        
        persona.memory.add(
            text=result_text,
            embedding=[],
            location=persona.location,
            event_type="throw",
            importance=importance,
            metadata={
                "source": str(src),
                "destination": str(dst),
                "action": "throw_towards",
                "success": thrown_object is not None
            },
            ttl=None,
        )

    def _handle_pickup(self, intent, persona, maze):
        # identical to your old pickup logic
        x, y, z = persona.location
        ents = maze.spatial.nearby(x, y, 0.0, 0.0)
        for ent in ents:
            if isinstance(ent, Object) and ent.name.lower() == intent.target.lower():
                maze.spatial.remove(ent)
                persona.inventory.append(ent)
                persona.memory.add(
                    text=f"Picked up {ent.name}",
                    embedding=[],
                    location=persona.location,
                    event_type="pickup",
                    importance=1.0,
                    metadata={"object": ent.name},
                    ttl=None,
                )
                break

    def _handle_drop(self, intent, persona, maze):
        x, y, z = persona.location
        for ent in list(persona.inventory):
            if isinstance(ent, Object) and ent.name.lower() == intent.target.lower():
                persona.inventory.remove(ent)
                maze.place_object(ent, x, y)
                persona.memory.add(
                    text=f"Dropped {ent.name}",
                    embedding=[],
                    location=persona.location,
                    event_type="drop",
                    importance=0.5,
                    metadata={"object": ent.name},
                    ttl=None,
                )
                break

    def _handle_talk_to(self, intent, persona, maze):
        """Handle talk_to action with enhanced dialogue generation."""
        target_name = intent.target
        
        # Handle case where target is None
        if target_name is None:
            response = f"[{persona.name}] I need to specify who to talk to."
            persona.memory.add(
                text=f"Tried to talk but no target was specified",
                embedding=[],
                location=persona.location,
                event_type="talk",
                importance=0.3,
                metadata={"target": None, "status": "no_target"},
                ttl=None,
                persona_name=persona.name,
            )
            return
            
        speaker = None
        speaker_description = ""
        
        # Find the target (could be a persona or object)
        x, y, z = persona.location
        
        # First, check if target is a nearby persona
        nearby_entities = maze.spatial.nearby(x, y, z, 10.0)  # Check within 10 units
        print(f"🔍 {persona.name} at {persona.location} looking for {target_name}")
        print(f"🔍 Found {len(nearby_entities)} nearby entities")
        for entity in nearby_entities:
            print(f"🔍 Entity: {entity.name} at {getattr(entity, 'location', 'unknown')}")
            if isinstance(entity, Persona) and entity.name.lower() == target_name.lower():
                speaker = entity
                speaker_description = f"You are {entity.name}, a persona in this world. You have your own personality, goals, and memories. Respond naturally to conversations."
                print(f"✅ Found speaker: {speaker.name}")
                break
            elif isinstance(entity, Object) and entity.name.lower() == target_name.lower():
                speaker = entity
                speaker_description = f"You are {entity.name}, {entity.description}. You can speak and interact with others. Respond naturally and in character."
                print(f"✅ Found object: {speaker.name}")
                break
        
        # If target not found nearby, check if it's the persona itself (self-talk)
        if speaker is None and persona.name.lower() == target_name.lower():
            speaker = persona
            speaker_description = f"You are {persona.name}, reflecting on yourself. You have your own personality, goals, and memories. Respond naturally to self-reflection."
        
        # If still not found, create a generic response
        if speaker is None:
            response = f"[{persona.name}] I don't see {target_name} nearby to talk to."
            persona.memory.add(
                text=f"Tried to talk to {target_name} but they weren't found",
                embedding=[],
                location=persona.location,
                event_type="talk",
                importance=0.3,
                metadata={"target": target_name, "status": "not_found"},
                ttl=None,
                persona_name=persona.name,
            )
            return
        
        # Set up relationship tracking
        relationship_tracker.set_entity_personality(persona.name, persona.personality_description)
        if isinstance(speaker, Persona):
            relationship_tracker.set_entity_personality(speaker.name, speaker.personality_description)
        else:
            relationship_tracker.set_entity_personality(speaker.name, speaker_description)
        
        # Get conversation history for this speaker
        conversation_history = self._get_conversation_history(persona, speaker)
        
        # Get relationship context
        relationship_context = relationship_tracker.get_conversation_context(persona.name, speaker.name)
        
        # Get recent memories for context
        recent_memories = persona.get_recent_important_memories(top_k=3, hours=24)
        
        # Create dialogue context
        dialogue_context = DialogueContext(
            speaker=persona.name,
            listener=speaker.name,
            speaker_personality=persona.personality_description,
            listener_personality=speaker_description,
            relationship_context=relationship_context,
            recent_memories=recent_memories,
            current_location=persona.location,
            current_time="current_time",  # TODO: Get actual time
            conversation_history=conversation_history,
            emotional_state="neutral"  # TODO: Determine emotional state
        )
        
        # Generate message using enhanced dialogue generator
        message = dialogue_generator.generate_dialogue(
            context=dialogue_context,
            message_type="contextual"
        )
        
        # Generate response using enhanced dialogue generator
        response_context = DialogueContext(
            speaker=speaker.name,
            listener=persona.name,
            speaker_personality=speaker_description,
            listener_personality=persona.personality_description,
            relationship_context=relationship_context,
            recent_memories=speaker.get_recent_important_memories(top_k=3, hours=24) if isinstance(speaker, Persona) else [],
            current_location=speaker.location,
            current_time="current_time",
            conversation_history=conversation_history,
            emotional_state="neutral"
        )
        
        response = dialogue_generator.generate_dialogue(
            context=response_context,
            message_type="contextual"
        )
        
        # Analyze conversation sentiment
        sentiment = dialogue_generator.analyze_conversation_sentiment(message + " " + response)
        
        # Store the conversation in memory
        conversation_text = f"{persona.name}: {message} | {speaker.name}: {response}"
        persona.memory.add(
            text=conversation_text,
            embedding=[],
            location=persona.location,
            event_type="talk",
            importance=0.8,
            metadata={
                "target": target_name,
                "speaker": speaker.name,
                "message": message,
                "response": response,
                "speaker_type": "persona" if isinstance(speaker, Persona) else "object",
                "sentiment": sentiment
            },
            ttl=None,
            persona_name=persona.name,
        )
        
        # Record interaction in relationship tracker
        relationship_tracker.add_interaction(
            initiator=persona.name,
            target=speaker.name,
            interaction_type=InteractionType.CONVERSATION,
            description=conversation_text,
            emotional_tone=sentiment,
            location=persona.location,
            metadata={
                "message": message,
                "response": response,
                "speaker_type": "persona" if isinstance(speaker, Persona) else "object"
            }
        )
        
        # Log conversation to file
        log_action(
            persona_name=persona.name,
            action_type="talk",
                            content=f"Conversation with {speaker.name}\n{persona.name}: {message}\n{speaker.name}: {response}",
            metadata={
                "target": target_name,
                "speaker": speaker.name,
                "message": message,
                "response": response,
                "speaker_type": "persona" if isinstance(speaker, Persona) else "object",
                "sentiment": sentiment
            },
            location=persona.location
        )
        
        # If speaker is a persona, also add to their memory
        if isinstance(speaker, Persona):
            speaker.memory.add(
                text=conversation_text,
                embedding=[],
                location=speaker.location,
                event_type="talk",
                importance=0.8,
                metadata={
                    "target": persona.name,
                    "speaker": persona.name,
                    "message": message,
                    "response": response,
                    "speaker_type": "persona",
                    "sentiment": sentiment
                },
                ttl=None,
                persona_name=speaker.name,
            )
    
    def _get_conversation_history(self, persona: Persona, speaker) -> List[Dict]:
        """Get conversation history between persona and speaker."""
        history = []
        
        # Get recent talk events from persona's memory
        recent_talks = []
        for entry in persona.memory.entries[-10:]:  # Last 10 entries
            if entry.event_type == "talk" and entry.metadata:
                target = entry.metadata.get("target", "")
                speaker_name = entry.metadata.get("speaker", "")
                message = entry.metadata.get("message", "")
                response = entry.metadata.get("response", "")
                
                if (target == speaker.name or speaker_name == speaker.name) and message and response:
                    recent_talks.append({
                        "user": message,
                        "response": response
                    })
        
        # If speaker is a persona, also check their memory
        if isinstance(speaker, Persona):
            for entry in speaker.memory.entries[-10:]:
                if entry.event_type == "talk" and entry.metadata:
                    target = entry.metadata.get("target", "")
                    speaker_name = entry.metadata.get("speaker", "")
                    message = entry.metadata.get("message", "")
                    response = entry.metadata.get("response", "")
                    
                    if (target == persona.name or speaker_name == persona.name) and message and response:
                        recent_talks.append({
                            "user": message,
                            "response": response
                        })
        
        # Return the most recent conversations
        return recent_talks[-5:]  # Last 5 conversations
    
    def _generate_message_context(self, persona: Persona, speaker, maze) -> str:
        """Generate a contextual message based on the situation."""
        # Get current location and nearby objects
        x, y, z = persona.location
        nearby_objects = [obj for obj in maze.spatial.nearby(x, y, 2.0, 0.0) if isinstance(obj, Object)]
        
        # Get persona's recent memories and goals
        recent_memories = [entry.text for entry in persona.memory.entries[-3:] if entry.text]
        current_goals = persona.long_term_goals[:2] if persona.long_term_goals else []
        
        # Create context-aware message
        context_parts = []
        
        if nearby_objects:
            object_names = [obj.name for obj in nearby_objects[:3]]
            context_parts.append(f"I see {', '.join(object_names)} nearby")
        
        if recent_memories:
            context_parts.append(f"I've been thinking about: {recent_memories[-1]}")
        
        if current_goals:
            context_parts.append(f"My current goal is: {current_goals[0]}")
        
        # Generate different types of messages based on context
        if isinstance(speaker, Persona):
            if context_parts:
                return f"Hi {speaker.name}! {'. '.join(context_parts)}. How are you doing?"
            else:
                return f"Hello {speaker.name}! How are you today?"
        else:  # Object
            if context_parts:
                return f"Hello {speaker.name}! {'. '.join(context_parts)}. What can you tell me about yourself?"
            else:
                return f"Hello {speaker.name}! What can you tell me about yourself?"

    def _handle_create(self, intent, persona, maze):
        x, y, z = persona.location
        new_obj = Object(intent.target, x, y, 0.0, description="(created)")
        maze.place_object(new_obj, x, y)
        persona.inventory.append(new_obj)
        persona.memory.add(
            text=f"Created {new_obj.name}",
            embedding=[],
            location=persona.location,
            event_type="create",
            importance=1.0,
            metadata={"object": new_obj.name},
            ttl=None,
        )

    def _handle_eat(self, intent, persona, maze):
        for ent in list(persona.inventory):
            if (
                isinstance(ent, Object)
                and ent.name.lower() == str(intent.target).lower()
            ):
                persona.inventory.remove(ent)
                persona.memory.add(
                    text=f"Ate {ent.name}",
                    embedding=[],
                    location=persona.location,
                    event_type="eat",
                    importance=1.0,
                    metadata={"object": ent.name},
                    ttl=None,
                )
                break

    def _handle_wander(self, intent, persona, maze):
        """Handle wander action - persona moves randomly to explore."""
        x, y, z = persona.location
        nbrs = maze.get_walkable_neighbors(x, y)
        
        if nbrs:
            # Choose a random neighbor, but prefer unexplored areas
            old_location = persona.location
            chosen_location = random.choice(nbrs)
            
            maze.remove_agent(persona)
            persona.location = chosen_location
            maze.place_agent(persona, *persona.location)
            
            # Check what's around the new location
            nearby_entities = maze.spatial.nearby(chosen_location[0], chosen_location[1], 1.0, 0.0)
            nearby_objects = [e for e in nearby_entities if isinstance(e, Object)]
            nearby_personas = [e for e in nearby_entities if isinstance(e, Persona) and e != persona]
            
            # Create description of what was discovered
            discovery_parts = [f"Wandered from {old_location} to {chosen_location}."]
            if nearby_objects:
                discovery_parts.append(f"Discovered: {', '.join([obj.name for obj in nearby_objects[:3]])}")
            if nearby_personas:
                discovery_parts.append(f"Encountered: {', '.join([p.name for p in nearby_personas[:2]])}")
            
            discovery_text = " ".join(discovery_parts)
            
            persona.memory.add(
                text=discovery_text,
                embedding=[],
                location=persona.location,
                event_type="wander",
                importance=0.3,
                metadata={
                    "old_location": old_location,
                    "new_location": chosen_location,
                    "objects_found": len(nearby_objects),
                    "personas_found": len(nearby_personas),
                    "action": "wander"
                },
                ttl=None,
            )
        else:
            persona.memory.add(
                text="Tried to wander but no walkable paths available. Staying in place.",
                embedding=[],
                location=persona.location,
                event_type="wander",
                importance=0.2,
                metadata={
                    "action": "wander",
                    "success": False,
                    "reason": "no_walkable_neighbors"
                },
                ttl=None,
            )

    def _handle_wait(self, intent, persona, maze):
        """Handle wait action - persona waits and observes surroundings."""
        x, y, z = persona.location
        
        # Observe what's happening around while waiting
        nearby_entities = maze.spatial.nearby(x, y, 3.0, 0.0)
        nearby_objects = [e for e in nearby_entities if isinstance(e, Object)]
        nearby_personas = [e for e in nearby_entities if isinstance(e, Persona) and e != persona]
        
        # Create waiting description with observations
        wait_parts = ["Waited and observed the surroundings."]
        
        if nearby_objects:
            wait_parts.append(f"Noticed nearby objects: {', '.join([obj.name for obj in nearby_objects[:3]])}")
        if nearby_personas:
            wait_parts.append(f"Observed people: {', '.join([p.name for p in nearby_personas[:2]])}")
        
        if not nearby_objects and not nearby_personas:
            wait_parts.append("The area is quiet and peaceful.")
        
        wait_text = " ".join(wait_parts)
        
        persona.memory.add(
            text=wait_text,
            embedding=[],
            location=persona.location,
            event_type="wait",
            importance=0.2,
            metadata={
                "action": "wait",
                "objects_observed": len(nearby_objects),
                "personas_observed": len(nearby_personas),
                "observation": True
            },
            ttl=None,
        )

    def _handle_sleep(self, intent, persona, maze):
        """Handle sleep action - persona rests and recovers energy."""
        duration = intent.target if intent.target else "short time"
        
        # Record sleep in memory with importance based on duration
        importance = 0.3
        if isinstance(duration, str):
            if "long" in duration.lower() or "night" in duration.lower():
                importance = 0.8
            elif "short" in duration.lower() or "nap" in duration.lower():
                importance = 0.4
        
        persona.memory.add(
            text=f"Slept for {duration}. Feeling rested and refreshed.",
            embedding=[],
            location=persona.location,
            event_type="sleep",
            importance=importance,
            metadata={
                "duration": str(duration),
                "action": "sleep",
                "energy_restored": True
            },
            ttl=None,
        )
        
        # Log sleep to file
        log_action(
            persona_name=persona.name,
            action_type="sleep",
                            content=f"Sleep duration: {duration}\nStatus: Feeling well-rested and refreshed",
            metadata={
                "duration": str(duration),
                "energy_restored": True,
                "importance_level": importance
            },
            location=persona.location
        )

    def _handle_read(self, intent, persona, maze):
        """Handle read action - persona reads an object or text."""
        target = intent.target
        x, y, z = persona.location
        
        # Check if target is in inventory
        found_item = None
        for item in persona.inventory:
            if isinstance(item, Object) and item.name.lower() == str(target).lower():
                found_item = item
                break
        
        # Check if target is nearby
        if not found_item:
            nearby_entities = maze.spatial.nearby(x, y, 2.0, 0.0)
            for entity in nearby_entities:
                if isinstance(entity, Object) and entity.name.lower() == str(target).lower():
                    found_item = entity
                    break
        
        if found_item:
            # Generate reading content based on object description
            content = f"Reading {found_item.name}: {found_item.description}"
            if "book" in found_item.name.lower():
                content += " The pages contain interesting stories and knowledge."
            elif "note" in found_item.name.lower():
                content += " The note contains important information."
            elif "sign" in found_item.name.lower():
                content += " The sign provides directions or warnings."
            
            persona.memory.add(
                text=f"Read {found_item.name}. {content}",
                embedding=[],
                location=persona.location,
                event_type="read",
                importance=0.7,
                metadata={
                    "target": found_item.name,
                    "content": content,
                    "action": "read"
                },
                ttl=None,
            )
            
            # Log reading content to file
            log_action(
                persona_name=persona.name,
                action_type="read",
                content=f"Reading item: {found_item.name}\nReading content: {content}",
                metadata={
                    "target": found_item.name,
                    "item_description": found_item.description,
                    "item_type": "book" if "book" in found_item.name.lower() else "other"
                },
                location=persona.location
            )
        else:
            persona.memory.add(
                text=f"Tried to read {target} but couldn't find it",
                embedding=[],
                location=persona.location,
                event_type="read",
                importance=0.3,
                metadata={
                    "target": str(target),
                    "status": "not_found",
                    "action": "read"
                },
                ttl=None,
            )

    def _handle_write(self, intent, persona, maze):
        """Handle write action - persona writes or creates written content."""
        target = intent.target
        x, y, z = persona.location
        
        # Create a written object (note, letter, etc.)
        written_object = Object(
            name=f"{target}",
            x=x, y=y, z=0.0,
            description=f"A {target} written by {persona.name}"
        )
        
        # Add to inventory first
        persona.inventory.append(written_object)
        
        persona.memory.add(
            text=f"Wrote {target}. Put thoughts and ideas into written form.",
            embedding=[],
            location=persona.location,
            event_type="write",
            importance=0.6,
            metadata={
                "target": str(target),
                "created_object": written_object.name,
                "action": "write"
            },
            ttl=None,
        )
        
        # Log writing content to file
        log_action(
            persona_name=persona.name,
            action_type="write",
                            content=f"Writing content: {target}\nDescription: Converting thoughts and ideas into written form\nCreated item: {written_object.name}",
            metadata={
                "target": str(target),
                "created_object": written_object.name,
                "object_description": written_object.description
            },
            location=persona.location
        )

    def _handle_search(self, intent, persona, maze):
        """Handle search action - persona searches for something in the area."""
        target = intent.target
        x, y, z = persona.location
        
        # Search nearby area
        nearby_entities = maze.spatial.nearby(x, y, 3.0, 0.0)
        found_objects = []
        found_personas = []
        
        for entity in nearby_entities:
            if isinstance(entity, Object):
                if str(target).lower() in entity.name.lower() or str(target).lower() in entity.description.lower():
                    found_objects.append(entity)
            elif isinstance(entity, Persona):
                if str(target).lower() in entity.name.lower():
                    found_personas.append(entity)
        
        search_results = []
        if found_objects:
            search_results.extend([f"Found object: {obj.name}" for obj in found_objects[:3]])
        if found_personas:
            search_results.extend([f"Found person: {p.name}" for p in found_personas[:3]])
        
        if search_results:
            result_text = f"Searched for {target}. " + "; ".join(search_results)
            importance = 0.6
        else:
            result_text = f"Searched for {target} but found nothing matching"
            importance = 0.3
        
        persona.memory.add(
            text=result_text,
            embedding=[],
            location=persona.location,
            event_type="search",
            importance=importance,
            metadata={
                "target": str(target),
                "found_objects": len(found_objects),
                "found_personas": len(found_personas),
                "action": "search"
            },
            ttl=None,
        )
        
        # Log search results to file
        search_details = f"Search target: {target}\nSearch results: {result_text}"
        if found_objects:
            search_details += f"\nFound items: {', '.join([obj.name for obj in found_objects[:5]])}"
        if found_personas:
            search_details += f"\nFound characters: {', '.join([p.name for p in found_personas[:3]])}"
        
        log_action(
            persona_name=persona.name,
            action_type="search",
            content=search_details,
            metadata={
                "target": str(target),
                "found_objects": len(found_objects),
                "found_personas": len(found_personas),
                "success": len(found_objects) > 0 or len(found_personas) > 0
            },
            location=persona.location
        )

    def _handle_observe(self, intent, persona, maze):
        """Handle observe action - persona carefully observes something."""
        target = intent.target
        x, y, z = persona.location
        
        # Get detailed information about surroundings or specific target
        nearby_entities = maze.spatial.nearby(x, y, 2.0, 0.0)
        
        if target and str(target).lower() != "surroundings":
            # Observe specific target
            observed_entity = None
            for entity in nearby_entities:
                if isinstance(entity, (Object, Persona)) and entity.name.lower() == str(target).lower():
                    observed_entity = entity
                    break
            
            if observed_entity:
                if isinstance(observed_entity, Persona):
                    observation = f"Observed {observed_entity.name}. They appear to be {observed_entity.description if hasattr(observed_entity, 'description') else 'a person in this world'}."
                else:
                    observation = f"Observed {observed_entity.name}. {observed_entity.description}"
                importance = 0.5
            else:
                observation = f"Tried to observe {target} but couldn't see it clearly"
                importance = 0.2
        else:
            # Observe general surroundings
            objects_nearby = [e for e in nearby_entities if isinstance(e, Object)]
            personas_nearby = [e for e in nearby_entities if isinstance(e, Persona) and e != persona]
            
            observation_parts = ["Observed the surroundings carefully."]
            if objects_nearby:
                observation_parts.append(f"Objects visible: {', '.join([obj.name for obj in objects_nearby[:5]])}")
            if personas_nearby:
                observation_parts.append(f"People nearby: {', '.join([p.name for p in personas_nearby[:3]])}")
            
            observation = " ".join(observation_parts)
            importance = 0.4
        
        persona.memory.add(
            text=observation,
            embedding=[],
            location=persona.location,
            event_type="observe",
            importance=importance,
            metadata={
                "target": str(target) if target else "surroundings",
                "action": "observe"
            },
            ttl=None,
        )
        
        # Log observation content to file
        log_action(
            persona_name=persona.name,
            action_type="observe",
                            content=f"Observation target: {target if target else 'surrounding environment'}\nObservation result: {observation}",
            metadata={
                "target": str(target) if target else "surroundings",
                "objects_nearby": len([e for e in nearby_entities if isinstance(e, Object)]),
                "personas_nearby": len([e for e in nearby_entities if isinstance(e, Persona) and e != persona])
            },
            location=persona.location
        )

    def _handle_think(self, intent, persona, maze):
        """Handle think action - persona reflects on a topic."""
        topic = intent.target if intent.target else "life"
        
        # Get recent memories to inform thinking
        recent_memories = persona.memory.entries[-5:] if len(persona.memory.entries) >= 5 else persona.memory.entries
        memory_contexts = [entry.text for entry in recent_memories if entry.text]
        
        # Generate thoughtful reflection
        if memory_contexts:
            thought_context = f"Reflecting on recent experiences: {'; '.join(memory_contexts[-3:])}"
        else:
            thought_context = f"Deep in thought about {topic}"
        
        thinking_content = f"Spent time thinking about {topic}. {thought_context}. Gained new insights and perspective."
        
        persona.memory.add(
            text=thinking_content,
            embedding=[],
            location=persona.location,
            event_type="think",
            importance=0.7,
            metadata={
                "topic": str(topic),
                "reflection": True,
                "action": "think"
            },
            ttl=None,
        )
        
        # Log thinking content to file
        log_action(
            persona_name=persona.name,
            action_type="think",
                            content=f"Thinking topic: {topic}\nThinking content: {thinking_content}",
            metadata={
                "topic": str(topic),
                "reflection": True,
                "recent_memories": len(memory_contexts)
            },
            location=persona.location
        )

    def _handle_rest(self, intent, persona, maze):
        """Handle rest action - persona takes a break to recover."""
        persona.memory.add(
            text="Took a rest to recover energy. Feeling more relaxed and ready for activities.",
            embedding=[],
            location=persona.location,
            event_type="rest",
            importance=0.4,
            metadata={
                "action": "rest",
                "energy_recovered": True
            },
            ttl=None,
        )

    def _handle_exercise(self, intent, persona, maze):
        """Handle exercise action - persona does physical activity."""
        exercise_types = ["stretching", "walking", "basic exercises", "physical training"]
        chosen_exercise = random.choice(exercise_types)
        
        persona.memory.add(
            text=f"Did some {chosen_exercise}. Feeling more energetic and healthy after the physical activity.",
            embedding=[],
            location=persona.location,
            event_type="exercise",
            importance=0.5,
            metadata={
                "exercise_type": chosen_exercise,
                "action": "exercise",
                "health_benefit": True
            },
            ttl=None,
        )

    def _handle_work(self, intent, persona, maze):
        """Handle work action - persona focuses on a specific task."""
        task = intent.target if intent.target else "general work"
        
        # Check if there are relevant objects nearby that could be work-related
        x, y, z = persona.location
        nearby_objects = [obj for obj in maze.spatial.nearby(x, y, 2.0, 0.0) if isinstance(obj, Object)]
        work_objects = [obj for obj in nearby_objects if any(keyword in obj.name.lower() for keyword in ["tool", "desk", "computer", "book", "paper"])]
        
        work_description = f"Focused on {task}."
        if work_objects:
            work_description += f" Used {work_objects[0].name} to help with the work."
        work_description += " Made progress on the task."
        
        persona.memory.add(
            text=work_description,
            embedding=[],
            location=persona.location,
            event_type="work",
            importance=0.6,
            metadata={
                "task": str(task),
                "tools_used": [obj.name for obj in work_objects[:2]],
                "action": "work"
            },
            ttl=None,
        )
        
        # Log work content to file
        log_action(
            persona_name=persona.name,
            action_type="work",
                            content=f"Work task: {task}\nWork description: {work_description}\nTools used: {', '.join([obj.name for obj in work_objects[:2]]) if work_objects else 'none'}",
            metadata={
                "task": str(task),
                "tools_used": [obj.name for obj in work_objects[:2]],
                "tools_count": len(work_objects)
            },
            location=persona.location
        )

    def _handle_play(self, intent, persona, maze):
        """Handle play action - persona engages in recreational activity."""
        activity = intent.target if intent.target else "games"
        
        # Check for nearby objects that could be used for play
        x, y, z = persona.location
        nearby_objects = [obj for obj in maze.spatial.nearby(x, y, 2.0, 0.0) if isinstance(obj, Object)]
        play_objects = [obj for obj in nearby_objects if any(keyword in obj.name.lower() for keyword in ["ball", "toy", "game", "card", "book"])]
        
        play_description = f"Played {activity}."
        if play_objects:
            play_description += f" Used {play_objects[0].name} for entertainment."
        play_description += " Had a fun and enjoyable time."
        
        persona.memory.add(
            text=play_description,
            embedding=[],
            location=persona.location,
            event_type="play",
            importance=0.5,
            metadata={
                "activity": str(activity),
                "objects_used": [obj.name for obj in play_objects[:2]],
                "action": "play",
                "mood_boost": True
            },
            ttl=None,
        )
        
        # Log entertainment activities to file
        log_action(
            persona_name=persona.name,
            action_type="play",
                            content=f"Entertainment activity: {activity}\nActivity description: {play_description}\nItems used: {', '.join([obj.name for obj in play_objects[:2]]) if play_objects else 'none'}",
            metadata={
                "activity": str(activity),
                "objects_used": [obj.name for obj in play_objects[:2]],
                "mood_boost": True,
                "objects_count": len(play_objects)
            },
            location=persona.location
        )
