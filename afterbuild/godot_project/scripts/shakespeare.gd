extends CharacterBody2D

@export var speed = 100
@export var target: Node2D

@onready var NA := $NavigationAgent2D as NavigationAgent2D
@onready var AP := $AnimatedSprite2D
@onready var server = get_node("/root/Server")
@onready var decision_timer := $DicisionTimer  # Reference to the decision timer

# Preload BubbleManager for static access
const BubbleManager = preload("res://scripts/bubble_manager.gd")

var is_observing := false  # Flag to pause movement during observation
var observation_duration := 3.0  # How long to observe
var observation_timer := 0.0
var is_acting := false  # Flag to indicate if Shakespeare is performing an action
var is_walking := false  # Flag to indicate if Shakespeare is walking to a target
var current_observation_bubble = null  # Reference to current observation bubble
var current_action_bubble = null  # Reference to current action bubble
var decision_enabled := false          # Flag to enable/disable decision system (default: OFF)


func set_decision_enabled(enabled: bool):
	"""Enable or disable the decision system for this NPC"""
	decision_enabled = enabled
	if decision_timer:
		if enabled:
			# Resume decision timer if it was stopped
			if decision_timer.is_stopped():
				decision_timer.start()
			# print("Shakespeare: Decision system enabled")
		else:
			# Stop decision timer
			decision_timer.stop()
			# print("Shakespeare: Decision system disabled")


func _ready():
	
	if target:
		makepath()
	
	# Stagger initial decision timing
	if decision_timer:
		decision_timer.wait_time = 20.0  # Set to 20 seconds
		# Shakespeare starts after 12 seconds
		await get_tree().create_timer(12.0).timeout
		decision_timer.start()

func _physics_process(_delta: float) -> void:
	# Handle observation timer
	if is_observing:
		observation_timer -= _delta
		if observation_timer <= 0:
			is_observing = false
			# print("Shakespeare finished observing")
			# Clear bubble reference when observation ends
			current_observation_bubble = null
			# Resume decision timer after observation
			finish_action()
	
	# Check if walk action is completed
	if is_walking and NA.is_navigation_finished():
		# print("Shakespeare reached destination")
		is_walking = false
		finish_action()
	
	# Only move if not observing
	if not is_observing:
		if not NA.is_navigation_finished():
			var next_pos = NA.get_next_path_position()
			var dir = global_position.direction_to(next_pos)
			velocity = dir.normalized() * speed
		else:
			velocity = Vector2.ZERO
		move_and_slide()
	else:
		# Stop movement during observation
		velocity = Vector2.ZERO
		
	update_animation()
	
	# Update observation bubble position to follow Shakespeare
	update_observation_bubble_position()
	# Update action bubble position to follow Shakespeare
	update_action_bubble_position()

func finish_action():
	# Resume decision timer after action completes
	if is_acting and decision_timer:
		is_acting = false
		decision_timer.wait_time = 20.0  # Set to standard 20 second delay
		decision_timer.start()
		# print("Decision timer resumed with 20 seconds")

func update_animation() -> void:
	if velocity.length() < 0.1:
		AP.play("idle")
	elif abs(velocity.x) > abs(velocity.y):
		if velocity.x < 0:
			AP.play("walk left")
		else:
			AP.play("walk right")
	else:
		if velocity.y > 0:
			AP.play("walk down")
		else:
			AP.play("walk up")

func makepath() -> void:
	if target:
		NA.target_position = target.global_position


func _on_dicision_timer_timeout() -> void:
	# Check if decision system is enabled
	if not decision_enabled:
		return
		
	if is_acting:
		return
		
	if not server:
		server = get_node_or_null("/root/Server")
		if not server:
			print("Server node not found")
			return
	
	if not server.connected:
		print("Server not connected yet, waiting...")
		return
		
	var context = generate_context()
	# print("Shakespeare sending decision request: " + context)  # Debug output disabled
	server.request_decision_with_callback("Shakespeare", context, on_decision_received)
	

func generate_context() -> String:
	var context_parts = []
	var npcs = get_tree().get_nodes_in_group("npcs")
	for npc in npcs:
		if npc == self:
			continue
		var distance = global_position.distance_to(npc.global_position)
		if distance < 150:
			context_parts.append(npc.name + " is nearby")
		elif distance < 300:
			context_parts.append(npc.name + " is in sight")
	
	# Shakespeare-specific context: check if near guitar
	var guitar = find_target_by_name("guitar")
	if guitar:
		var distance = global_position.distance_to(guitar.global_position)
		if distance < 50:
			context_parts.append("Guitar is within reach")
		elif distance < 150:
			context_parts.append("Guitar is nearby")
	
	if context_parts.is_empty():
		return "Quiet evening at the bar"
	else:
		return ". ".join(context_parts)
	

func _on_path_finding_timer_timeout() -> void:
	makepath()

# Shakespeare's observations - musician perspective
var observations = {
	"einstein": "Einstein seems lost in thought, pondering the universe.",
	"leonardo": "Leonardo's keeping busy behind the bar, always creating.",
	"dog": "That little dog looks like it wants to dance!",
	"customer": "A new face in the crowd, wonder what songs they like.",
	"bar": "The bar, where I've played so many nights.",
	"guitar": "My trusty guitar, ready for another performance.",
	# Shakespeare-specific observations
	"stage": "The stage, my second home.",
	"piano": "That old piano, maybe I should try playing it sometime."
}

func on_decision_received(action: String, target_name: String):
	# Double-check we're not already acting before executing
	if is_acting:
		print("Shakespeare received decision but is already acting, ignoring: " + action + " " + target_name)
		return
		
	if target_name != "" and target_name != "self":
		print("Shakespeare decided to: " + action + " " + target_name)
	else:
		print("Shakespeare decided to: " + action)
	
	# Stop decision timer during action execution
	if decision_timer:
		decision_timer.stop()
		is_acting = true
	
	if action == "walk":
		execute_walk(target_name)
	elif action == "chat":
		execute_chat(target_name)
	elif action == "observe":
		execute_observe(target_name)
	elif action == "play":
		execute_play(target_name)
		
func execute_walk(target_name: String):
	var target_node = find_target_by_name(target_name)
	if target_node:
		# Show action bubble for walking
		show_action_bubble("*walking to " + target_name + "*")
		target = target_node
		makepath()
		is_walking = true  # Set walking flag
	else:
		print("Target not found: " + target_name)
		finish_action()  # Finish if target not found

func execute_chat(target_name: String):
	# print("Shakespeare initiating chat with " + target_name)
	
	# Check if target is player or NPC
	if target_name == "customer":
		# Chat with player - show a greeting
		show_chat_bubble_to_player("Hey there! Any song requests?")
		await get_tree().create_timer(3.0).timeout
		finish_action()
	else:
		# Chat with another NPC - first walk to them
		var target_node = find_target_by_name(target_name)
		if target_node:
			# Walk to the NPC first
			# print("Shakespeare walking to " + target_name + " for chat")
			target = target_node
			makepath()
			is_walking = true
			
			# Wait until Shakespeare reaches the NPC
			while not NA.is_navigation_finished():
				await get_tree().process_frame
			
			is_walking = false
			# print("Shakespeare reached " + target_name)
			
			# Now start dialogue
			start_npc_dialogue(target_name)
		else:
			print("Target not found for chat: " + target_name)
			finish_action()

func execute_play(target_name: String):
	print("Shakespeare playing " + target_name)
	
	# Walk to the guitar/instrument first
	var instrument = find_target_by_name(target_name)
	if instrument:
		target = instrument
		makepath()
		is_walking = true
		
		# Wait until Shakespeare reaches the instrument
		while not NA.is_navigation_finished():
			await get_tree().process_frame
		
		is_walking = false
		
		# Show playing animation/bubble
		show_action_bubble("*strums a melody on the " + target_name + "*")
		
		# Longer duration for playing music
		await get_tree().create_timer(5.0).timeout
	else:
		print("Instrument not found: " + target_name)
	
	finish_action()

func show_action_bubble(text: String):
	"""Show an action bubble using BubbleManager"""
	var bubble = BubbleManager.show_action_bubble("Shakespeare", text, global_position)
	
	if bubble:
		current_action_bubble = bubble
		
		# Auto-remove after 3 seconds
		await get_tree().create_timer(3.0).timeout
		if bubble and is_instance_valid(bubble):
			bubble.queue_free()
			current_action_bubble = null

func start_npc_dialogue(target_npc: String):
	"""Start a dialogue session between Shakespeare and another NPC"""
	# Ensure is_acting is true to prevent new decisions during dialogue
	is_acting = true
	
	var server = get_node_or_null("/root/Server")
	if not server or not server.connected:
		print("Server not available for NPC dialogue")
		finish_action()
		return
	
	# Generate Shakespeare's greeting dynamically based on context
	var context_prompts = {
		"Einstein": "You are William Shakespeare, the playwright. Start a conversation with Einstein, the physicist.\nBe dramatic and poetic.\nReply with ONE short sentence only.",
		"Leonardo": "You are William Shakespeare. Start a conversation with Leonardo da Vinci.\nBe theatrical about art.\nReply with ONE short sentence only.",
		"Socrates": "You are William Shakespeare. Start a conversation with Socrates, the philosopher dog.\nBe philosophical and dramatic.\nReply with ONE short sentence only.", 
		"dog": "You are Shakespeare, a cool musician. Talk to the dog in a playful way.\nBe cool.\nReply with ONE short sentence only."
	}
	
	var prompt = context_prompts.get(target_npc, "You are Shakespeare the musician. Start a conversation with " + target_npc + ". Reply with ONE short sentence only.")
	
	# print("Shakespeare is thinking of what to say to " + target_npc + "...")
	
	# Request generation for Shakespeare's initial greeting
	# Store the target so we know who to send it to after generation
	set_meta("pending_chat_target", target_npc)
	
	# Send to server to generate Shakespeare's greeting
	server.send_npc_dialogue("system", "Shakespeare", prompt)
	
	# The chat action will be completed in show_dialogue_bubble() after the conversation
	# Don't call finish_action() here as it causes the decision timer to restart too early

func show_chat_bubble_to_player(text: String):
	"""Show Shakespeare's chat bubble when talking to player"""
	show_shakespeare_speech_bubble(text)

func show_shakespeare_speech_bubble(text: String):
	"""Display a speech bubble for Shakespeare"""
	var ui_layer = get_parent().get_node_or_null("UILayer")
	if not ui_layer:
		print("UILayer not found for speech bubble")
		return
	
	# Remove any existing speech bubble
	var bubble_name = "SpeechBubble_Shakespeare"
	var existing = ui_layer.get_node_or_null(bubble_name)
	if existing:
		existing.queue_free()
	
	# Use SpeechBubbleExample as template (shakespearee as dialogue_client.gd)
	var example = ui_layer.get_node_or_null("SpeechBubbleExample")
	var bubble
	
	if example:
		# Duplicate the template
		bubble = example.duplicate()
		bubble.name = bubble_name
		bubble.visible = true
		
		# Attach the streaming_bubble script for arrow functionality
		var script_path = "res://scripts/streaming_bubble.gd"
		var script = load(script_path)
		if script:
			bubble.set_script(script)
		
		# Update the text
		var text_label = find_rich_text_label(bubble)
		if text_label:
			text_label.text = text
	else:
		# Fallback: create manually with consistent style (matching dialogue_client.gd)
		bubble = Panel.new()
		bubble.name = bubble_name
		bubble.size = Vector2(200, 80)
		bubble.modulate = Color(1, 1, 1, 0.9)
		
		# Attach the streaming_bubble script for arrow functionality
		var script_path = "res://scripts/streaming_bubble.gd"
		var script = load(script_path)
		if script:
			bubble.set_script(script)
		
		# Create ScrollContainer
		var scroll_container = ScrollContainer.new()
		scroll_container.position = Vector2(5, 5)
		scroll_container.size = Vector2(190, 70)
		
		# Create text label
		var label = RichTextLabel.new()
		label.name = "Label"
		label.custom_minimum_size = Vector2(180, 0)
		label.bbcode_enabled = false
		label.fit_content = true
		label.scroll_active = false
		label.autowrap_mode = TextServer.AUTOWRAP_WORD_SMART
		label.add_theme_font_size_override("normal_font_size", 14)
		label.add_theme_color_override("default_color", Color(0.1, 0.1, 0.1, 1.0))
		label.text = text
		
		scroll_container.add_child(label)
		bubble.add_child(scroll_container)
		
		# Use consistent style matching ui_theme.tres
		var style_box = StyleBoxFlat.new()
		style_box.bg_color = Color(0.05, 0.05, 0.05, 0.85)  # Match theme
		style_box.border_color = Color(0.2, 0.2, 0.2, 0.5)
		style_box.border_width_left = 1
		style_box.border_width_right = 1
		style_box.border_width_top = 1
		style_box.border_width_bottom = 1
		style_box.corner_radius_top_left = 10  # Match theme
		style_box.corner_radius_top_right = 10
		style_box.corner_radius_bottom_left = 10
		style_box.corner_radius_bottom_right = 10
		bubble.add_theme_stylebox_override("panel", style_box)
	
	# Add to UI layer
	ui_layer.add_child(bubble)
	
	# Position above Shakespeare
	bubble.position = Vector2(global_position.x - 100, global_position.y - 100)
	
	# Auto-remove after 8 seconds for better readability
	var tween = get_tree().create_tween()
	tween.tween_interval(8.0)
	tween.tween_property(bubble, "modulate:a", 0.0, 0.5)
	tween.tween_callback(func(): 
		if is_instance_valid(bubble):
			bubble.queue_free()
	)

func find_rich_text_label(node: Node) -> RichTextLabel:
	"""Recursively find RichTextLabel in node tree"""
	if node is RichTextLabel:
		return node
	
	for child in node.get_children():
		var result = find_rich_text_label(child)
		if result:
			return result
	
	return null
	
func execute_observe(target_name: String):
	var target_node = find_target_by_name(target_name)
	
	# Start observation state
	is_observing = true
	observation_timer = observation_duration
	
	if target_node:
		# Face the target without rotating the sprite
		var direction = target_node.global_position - global_position
		# Update animation to face the target
		if abs(direction.x) > abs(direction.y):
			if direction.x < 0:
				AP.play("idle")  # or "idle_left" if you have directional idle animations
			else:
				AP.play("idle")  # or "idle_right"
		else:
			if direction.y > 0:
				AP.play("idle")  # or "idle_down"
			else:
				AP.play("idle")  # or "idle_up"
		
	# Get observation description
	var description = observations.get(target_name.to_lower(), "Shakespeare observes " + target_name + " carefully.")
	# print("Observation: " + description)
	
	# Show observation in thought bubble with "Observation:" prefix and target
	show_observation_bubble("Observing " + target_name + ": " + description)

func find_target_by_name(target_name: String) -> Node2D:
	# Check NPCs
	var npcs = get_tree().get_nodes_in_group("npcs")
	for npc in npcs:
		if npc.name.to_lower() == target_name.to_lower():
			return npc
	
	# Check for objects (bar, guitar, etc.)
	var all_nodes = get_tree().get_nodes_in_group("objects")
	for node in all_nodes:
		if node.name.to_lower() == target_name.to_lower():
			return node
	
	return null

func show_observation_bubble(text: String):
	"""Show an observation bubble using BubbleManager"""
	var bubble = BubbleManager.show_observation_bubble("Shakespeare", text, global_position)
	
	if bubble:
		current_observation_bubble = bubble
		
		# Auto-fade after observation duration (3 seconds)
		await get_tree().create_timer(observation_duration - 0.5).timeout
		
		# Create tween after waiting
		if is_instance_valid(bubble):
			var tween = get_tree().create_tween()
			tween.tween_property(bubble, "modulate:a", 0.0, 0.5)
			tween.tween_callback(func():
				if is_instance_valid(current_observation_bubble):
					current_observation_bubble.queue_free()
					current_observation_bubble = null
			)

func update_observation_bubble_position():
	# Update bubble position if it exists
	if current_observation_bubble and is_instance_valid(current_observation_bubble):
		var shakespeare_pos = global_position
		
		# Position above Shakespeare (offset for visibility)
		current_observation_bubble.position = Vector2(shakespeare_pos.x - 100, shakespeare_pos.y - 100)

func update_action_bubble_position():
	# Update action bubble position if it exists
	if current_action_bubble and is_instance_valid(current_action_bubble):
		var shakespeare_pos = global_position
		# Position above character
		current_action_bubble.position = Vector2(shakespeare_pos.x - 70, shakespeare_pos.y - 80)

# Variables for streaming dialogue
var current_streaming_response: String = ""
var is_streaming: bool = false
var conversation_turn_count: int = 0  # Track conversation turns
const MAX_CONVERSATION_TURNS: int = 5  # Limit to 5 exchanges per conversation

func handle_dialogue_token(token: String, from_speaker: String):
	"""Handle streaming tokens for dialogue"""
	# Only handle if we're expecting a response
	if not is_acting:
		return
		
	# Accumulate tokens
	current_streaming_response += token
	is_streaming = true
	
	# Update or create speech bubble with streaming text
	update_streaming_bubble(current_streaming_response)

func update_streaming_bubble(text: String):
	"""Update speech bubble with streaming text"""
	var ui_layer = get_parent().get_node_or_null("UILayer")
	if not ui_layer:
		return
	
	var bubble_name = "SpeechBubble_Shakespeare"
	var bubble = ui_layer.get_node_or_null(bubble_name)
	
	if not bubble:
		# Create new bubble
		show_shakespeare_speech_bubble(text)
	else:
		# Update existing bubble text
		var text_label = find_rich_text_label(bubble)
		if text_label:
			text_label.text = text

func show_dialogue_bubble(response: String, from_speaker: String = ""):
	"""Called by server when Shakespeare receives a dialogue response"""
	# Reset streaming state  
	current_streaming_response = ""
	is_streaming = false
	
	# Check if this is Shakespeare generating his own greeting for an NPC
	if has_meta("pending_chat_target"):
		var target_npc = get_meta("pending_chat_target")
		remove_meta("pending_chat_target")
		
		# This is Shakespeare's generated greeting - send it to the target NPC
		# print("Shakespeare says to " + target_npc + ": " + response)
		show_shakespeare_speech_bubble(response)
		
		# Now send this to the target NPC to get their response
		var server = get_node_or_null("/root/Server")
		if server and server.connected:
			server.send_npc_dialogue("Shakespeare", target_npc, response)
		
		# Don't call finish_action here - let the conversation complete naturally
		# The action will finish when conversation ends after MAX_CONVERSATION_TURNS
		return
	
	# Normal case: Shakespeare responding to someone else
	# If Shakespeare is doing something, interrupt it to respond
	if is_acting:
		print("Shakespeare interrupted to respond to dialogue")
		is_observing = false
		is_walking = false
		velocity = Vector2.ZERO
	# Store remaining time and add 20 seconds
	var remaining_time = 0.0
	if decision_timer and decision_timer.time_left > 0:
		remaining_time = decision_timer.time_left + 20.0
		decision_timer.stop()
		is_acting = true
	else:
		remaining_time = 20.0
		is_acting = true
	show_shakespeare_speech_bubble(response)
	
	# If this is from another NPC (not user or system), send Shakespeare's response back to continue the conversation
	if from_speaker != "" and from_speaker != "user" and from_speaker != "system":
		conversation_turn_count += 1
		if conversation_turn_count < MAX_CONVERSATION_TURNS:
			await get_tree().create_timer(2.0).timeout  # Short pause before responding
			var server = get_node_or_null("/root/Server")
			if server and server.connected:
				# Send Shakespeare's response back to the NPC who spoke to him
				server.send_npc_dialogue("Shakespeare", from_speaker, response)
				print("Shakespeare continuing conversation with " + from_speaker + " (turn " + str(conversation_turn_count) + "/" + str(MAX_CONVERSATION_TURNS) + ")")
		else:
			print("Shakespeare ending conversation with " + from_speaker + " after " + str(MAX_CONVERSATION_TURNS) + " turns")
			conversation_turn_count = 0  # Reset for next conversation
			# End the chat action after max turns reached
			await get_tree().create_timer(2.0).timeout
			# Always ensure Shakespeare can resume decisions after conversation ends
			if decision_timer:
				is_acting = false
				decision_timer.wait_time = 20.0
				decision_timer.start()
			return
	else:
		conversation_turn_count = 0  # Reset when talking to user or starting new conversation
	
	# Resume after dialogue with extended time (only for user dialogue)
	if from_speaker == "user":
		await get_tree().create_timer(4.0).timeout
		if is_acting and decision_timer:
			is_acting = false
			decision_timer.wait_time = remaining_time
			decision_timer.start()
			print("Decision timer resumed with " + str(remaining_time) + " seconds")

# Handle user clicking on Shakespeare for dialogue
func handle_user_dialogue():
	# Interrupt current action
	if is_acting:
		print("Shakespeare interrupted by user dialogue")
		is_observing = false
		is_walking = false
		velocity = Vector2.ZERO
	# Pause decision timer
	if decision_timer and decision_timer.time_left > 0:
		decision_timer.stop()
		is_acting = true
	# print("Shakespeare is listening to user...")
