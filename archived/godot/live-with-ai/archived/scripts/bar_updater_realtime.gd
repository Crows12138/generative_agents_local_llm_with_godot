# bar_updater_realtime.gd - Real-time LLM calls, no caching
extends Node2D

var npcs = {}
var python_path = "python"
var is_processing = false
var llm_loaded = false

# Track conversations for context (not cache)
var conversation_history = {}

func _ready():
	print("Bar scene - Real-time LLM mode (no cache)!")
	
	# Get NPC nodes
	npcs["Bob"] = $Bob if has_node("Bob") else null
	npcs["Alice"] = $Alice if has_node("Alice") else null
	npcs["Sam"] = $Sam if has_node("Sam") else null
	
	# Initialize conversation history
	for npc in npcs:
		conversation_history[npc] = []
	
	# Preload model in background
	_preload_model()

func _preload_model():
	"""Preload the model once at startup"""
	print("Preloading LLM model...")
	var thread = Thread.new()
	thread.start(_preload_thread)

func _preload_thread():
	"""Load model in background thread"""
	var output = []
	# Make one call to load the model into memory
	OS.execute(python_path, ["quick_llm_nocache.py", "init"], output, true)
	llm_loaded = true
	print("LLM model loaded and ready!")

func _input(event):
	if event is InputEventMouseButton and event.pressed:
		if event.button_index == MOUSE_BUTTON_LEFT:
			handle_npc_click(event)
		elif event.button_index == MOUSE_BUTTON_RIGHT:
			show_menu()

func handle_npc_click(event):
	"""Handle clicking on NPCs"""
	if is_processing:
		print("Still processing...")
		return
	
	var mouse_pos = get_global_mouse_position()
	
	for npc_name in npcs:
		var npc = npcs[npc_name]
		if npc:
			var distance = npc.position.distance_to(mouse_pos)
			
			if distance < 50:
				print("\nInteracting with ", npc_name)
				
				# Generate unique message based on context
				var message = generate_contextual_message(npc_name)
				interact_realtime(npc_name, message)
				break

func generate_contextual_message(npc_name: String) -> String:
	"""Generate context-aware messages"""
	var messages = [
		"Hello, how are you today?",
		"What's new?",
		"Tell me something interesting.",
		"How's your day going?",
		"What do you think about this place?",
		"Any recommendations?",
		"What's on your mind?",
		"How can I help you?"
	]
	
	# Use conversation history to vary messages
	var history_count = conversation_history[npc_name].size()
	
	if history_count == 0:
		return "Hello!"
	elif history_count == 1:
		return messages[randi() % 3]
	else:
		return messages[3 + (randi() % (messages.size() - 3))]

func interact_realtime(npc_name: String, message: String):
	"""Real-time LLM interaction - no cache"""
	
	if is_processing:
		return
	
	is_processing = true
	
	# Show thinking immediately
	show_thinking(npc_name)
	
	# Record start time
	var start_time = Time.get_ticks_msec()
	
	# Create thread for LLM call
	var thread = Thread.new()
	thread.start(_llm_realtime_thread.bind(npc_name, message, start_time))

func _llm_realtime_thread(npc_name: String, message: String, start_time: int):
	"""Call LLM in thread - always fresh responses"""
	
	# Build contextual prompt
	var context = ""
	if conversation_history[npc_name].size() > 0:
		# Include last exchange for context
		var last = conversation_history[npc_name][-1]
		context = "Previous: " + last.substr(0, 50) + "\n"
	
	# Create role-specific prompt
	var role_prompts = {
		"Bob": "You are Bob, a friendly bartender. ",
		"Alice": "You are Alice, a regular customer who loves this place. ",
		"Sam": "You are Sam, a musician who performs here. "
	}
	
	var full_prompt = role_prompts.get(npc_name, "") + context + "Customer: " + message
	
	# Call Python script - no cache version
	var output = []
	var args = ["quick_llm_nocache.py", full_prompt]
	var exit_code = OS.execute(python_path, args, output, true, false)
	
	var elapsed = (Time.get_ticks_msec() - start_time) / 1000.0
	
	if output.size() > 0 and exit_code == 0:
		var response = output[0].strip_edges()
		
		# Store in history (not cache)
		conversation_history[npc_name].append(message + " -> " + response)
		
		# Keep history limited
		if conversation_history[npc_name].size() > 5:
			conversation_history[npc_name].pop_front()
		
		call_deferred("_on_realtime_response", npc_name, response, elapsed)
	else:
		call_deferred("_on_realtime_response", npc_name, "...", 0.0)

func _on_realtime_response(npc_name: String, response: String, time_taken: float):
	"""Handle response in main thread"""
	is_processing = false
	
	# Hide thinking
	hide_thinking(npc_name)
	
	# Show response
	show_speech_bubble(npc_name, response)
	
	# Log performance
	print("[%s] Real-time response (%.2fs): %s" % [
		npc_name, 
		time_taken, 
		response.substr(0, 50)
	])

func show_thinking(npc_name: String):
	"""Show thinking animation"""
	var npc = npcs.get(npc_name)
	if npc and npc.get_child_count() > 0:
		var label = npc.get_child(0)
		if label is Label:
			label.text = npc_name + "\n💭..."
			label.modulate = Color(0.8, 0.8, 0.8, 0.7)
			
			# Animate dots
			var tween = create_tween()
			tween.set_loops()
			tween.tween_callback(func():
				var dots = label.text.count(".")
				if dots >= 3:
					label.text = npc_name + "\n💭."
				else:
					label.text = label.text + "."
			).set_delay(0.3)

func hide_thinking(npc_name: String):
	"""Stop thinking animation"""
	var npc = npcs.get(npc_name)
	if npc and npc.get_child_count() > 0:
		var label = npc.get_child(0)
		if label is Label:
			label.modulate = Color.WHITE
			# Stop any tweens
			var tween = create_tween()
			tween.kill()

func show_speech_bubble(npc_name: String, text: String):
	"""Show speech with animation"""
	var npc = npcs.get(npc_name)
	if npc and npc.get_child_count() > 0:
		var label = npc.get_child(0)
		if label is Label:
			# Clean and limit text
			var clean_text = text.strip_edges()
			if clean_text.length() > 60:
				clean_text = clean_text.substr(0, 57) + "..."
			
			label.text = npc_name + "\n💬 " + clean_text
			
			# Fade in animation
			label.modulate.a = 0
			var tween = create_tween()
			tween.tween_property(label, "modulate:a", 1.0, 0.3)
			
			# Auto-hide after 5 seconds
			tween.tween_interval(5.0)
			tween.tween_property(label, "modulate:a", 0.5, 0.5)

func show_menu():
	"""Show interaction menu"""
	print("\n=== Real-time LLM Info ===")
	print("Model loaded: ", llm_loaded)
	print("No caching - all responses are fresh!")
	print("Conversation histories:")
	for npc in conversation_history:
		print("  %s: %d exchanges" % [npc, conversation_history[npc].size()])
	print("========================\n")