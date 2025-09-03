# bar_server_client.gd - Bar scene using server-client architecture
extends Node2D

var npcs = {}
var llm_client: LLMClientIntegration
var is_processing = false

# Track conversation for context
var conversation_history = {}

func _ready():
	print("Bar scene - Server-Client mode!")
	
	# Initialize LLM client
	llm_client = LLMClientIntegration.new()
	add_child(llm_client)
	
	# Connect signals
	llm_client.response_received.connect(_on_response_received)
	llm_client.server_status_changed.connect(_on_server_status_changed)
	
	# Get NPC nodes - check both direct children and NPCs group
	if has_node("NPCs"):
		npcs["Bob"] = $NPCs/Bob if has_node("NPCs/Bob") else null
		npcs["Alice"] = $NPCs/Alice if has_node("NPCs/Alice") else null
		npcs["Sam"] = $NPCs/Sam if has_node("NPCs/Sam") else null
	else:
		npcs["Bob"] = $Bob if has_node("Bob") else null
		npcs["Alice"] = $Alice if has_node("Alice") else null
		npcs["Sam"] = $Sam if has_node("Sam") else null
	
	# Initialize conversation history
	for npc in npcs:
		conversation_history[npc] = []
	
	# Check/start server
	llm_client.check_server_status()
	
	# Auto-start server if not running
	if not llm_client.is_server_running:
		print("Auto-starting LLM server...")
		llm_client.start_server()

func _on_server_status_changed(running: bool):
	"""Handle server status changes"""
	if running:
		print("✅ LLM Server is ready!")
		show_server_status("Server Ready", Color.GREEN)
	else:
		print("❌ LLM Server is not running")
		show_server_status("Server Offline", Color.RED)

func show_server_status(text: String, color: Color):
	"""Show server status on screen"""
	# You can add a Label node to show this visually
	print("[STATUS] " + text)

func _input(event):
	if event is InputEventMouseButton and event.pressed:
		if event.button_index == MOUSE_BUTTON_LEFT:
			handle_npc_click(event)
		elif event.button_index == MOUSE_BUTTON_RIGHT:
			show_menu()

func handle_npc_click(event):
	"""Handle clicking on NPCs"""
	if is_processing:
		print("Still processing previous request...")
		return
	
	if not llm_client.is_server_running:
		print("Server not ready yet...")
		return
	
	var mouse_pos = get_global_mouse_position()
	
	for npc_name in npcs:
		var npc = npcs[npc_name]
		if npc:
			var distance = npc.position.distance_to(mouse_pos)
			
			if distance < 50:
				print("\nInteracting with ", npc_name)
				
				# Generate contextual message
				var message = generate_contextual_message(npc_name)
				interact_with_npc(npc_name, message)
				break

func generate_contextual_message(npc_name: String) -> String:
	"""Generate context-aware messages"""
	var messages = [
		"Hello!",
		"How are you today?",
		"What's new?",
		"Tell me something interesting.",
		"How's business?",
		"Any recommendations?",
		"What's on your mind?",
		"Nice weather today, isn't it?"
	]
	
	var history_count = conversation_history[npc_name].size()
	
	if history_count == 0:
		return messages[0]  # First greeting
	elif history_count < 3:
		return messages[1 + (randi() % 3)]  # Early conversation
	else:
		return messages[4 + (randi() % 4)]  # Later conversation

func interact_with_npc(npc_name: String, message: String):
	"""Interact with NPC using server-client"""
	if is_processing:
		return
	
	is_processing = true
	
	# Show thinking animation
	show_thinking(npc_name)
	
	# Record start time
	var start_time = Time.get_ticks_msec()
	
	# Get response from server (async)
	var response = await llm_client.get_npc_response(npc_name, message)
	
	# Calculate time taken
	var elapsed = (Time.get_ticks_msec() - start_time) / 1000.0
	
	# Process response
	_on_response_received(npc_name, response)
	
	# Log performance
	print("[%s] Response time: %.2fs" % [npc_name, elapsed])
	
	is_processing = false

func _on_response_received(npc_name: String, response: String):
	"""Handle LLM response"""
	# Hide thinking
	hide_thinking(npc_name)
	
	# Clean response
	response = response.strip_edges()
	if response.length() > 60:
		response = response.substr(0, 57) + "..."
	
	# Store in history
	conversation_history[npc_name].append(response)
	if conversation_history[npc_name].size() > 5:
		conversation_history[npc_name].pop_front()
	
	# Show speech bubble
	show_speech_bubble(npc_name, response)

func show_thinking(npc_name: String):
	"""Show thinking animation"""
	var npc = npcs.get(npc_name)
	if npc and npc.get_child_count() > 0:
		var label = npc.get_child(0)
		if label is Label:
			label.text = npc_name + "\n💭..."
			label.modulate = Color(0.8, 0.8, 0.8, 0.7)

func hide_thinking(npc_name: String):
	"""Hide thinking animation"""
	var npc = npcs.get(npc_name)
	if npc and npc.get_child_count() > 0:
		var label = npc.get_child(0)
		if label is Label:
			label.modulate = Color.WHITE

func show_speech_bubble(npc_name: String, text: String):
	"""Show speech with animation"""
	var npc = npcs.get(npc_name)
	if npc and npc.get_child_count() > 0:
		var label = npc.get_child(0)
		if label is Label:
			label.text = npc_name + "\n💬 " + text
			
			# Fade in animation
			label.modulate.a = 0
			var tween = create_tween()
			tween.tween_property(label, "modulate:a", 1.0, 0.3)
			
			# Auto-hide after 5 seconds
			tween.tween_interval(5.0)
			tween.tween_property(label, "modulate:a", 0.5, 0.5)

func show_menu():
	"""Show debug menu"""
	print("\n=== Server-Client Info ===")
	print("Server running: ", llm_client.is_server_running)
	print("Processing: ", is_processing)
	print("Conversation histories:")
	for npc in conversation_history:
		print("  %s: %d messages" % [npc, conversation_history[npc].size()])
	print("=======================\n")
