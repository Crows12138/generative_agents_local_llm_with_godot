extends Node2D

@onready var http = HTTPRequest.new()
@onready var interact_http = HTTPRequest.new()
var update_timer = Timer.new()
var npcs = {}  # Store NPC references
var selected_npc = null
var is_requesting = false
var request_start_time = 0

# CRITICAL FIX: Use 127.0.0.1 instead of localhost (200x faster!)
const SERVER_URL = "http://127.0.0.1:8000"

func _ready():
	print("=== Bar Scene with REAL LLM (Fixed Networking) ===")
	print("Using 127.0.0.1 for 200x faster connection")
	
	# Get existing NPC nodes
	npcs["Bob"] = $Bob
	npcs["Alice"] = $Alice  
	npcs["Sam"] = $Sam
	
	# Create speech bubbles for each NPC
	for npc_name in npcs:
		create_speech_bubble(npcs[npc_name])
	
	# Setup HTTP requests with proper configuration
	setup_http_clients()
	
	# Setup update timer
	add_child(update_timer)
	update_timer.wait_time = 3.0  # Update every 3 seconds
	update_timer.one_shot = false
	update_timer.timeout.connect(_fetch_npc_states)
	update_timer.start()
	
	# Initial fetch
	_fetch_npc_states()

func setup_http_clients():
	"""Configure HTTP clients for optimal performance"""
	# Status fetcher
	add_child(http)
	http.timeout = 5.0  # 5 second timeout
	http.use_threads = true  # Enable threading
	http.request_completed.connect(_on_data_received)
	
	# Interaction client
	add_child(interact_http)
	interact_http.timeout = 5.0  # 5 second timeout  
	interact_http.use_threads = true  # Enable threading
	interact_http.request_completed.connect(_on_interaction_response)
	
	print("HTTP clients configured: timeout=5s, threading=enabled")

func create_speech_bubble(npc_node):
	"""Create persistent speech bubble for NPC"""
	var bubble = Label.new()
	bubble.name = "SpeechBubble"
	bubble.text = ""
	bubble.visible = false
	bubble.add_theme_color_override("font_color", Color.WHITE)
	bubble.add_theme_color_override("font_shadow_color", Color.BLACK)
	bubble.add_theme_constant_override("shadow_offset_x", 1)
	bubble.add_theme_constant_override("shadow_offset_y", 1)
	bubble.position = Vector2(-50, -40)
	bubble.z_index = 10
	npc_node.add_child(bubble)

func _fetch_npc_states():
	"""Request NPC states from LLM server"""
	if http.get_http_client_status() == HTTPClient.STATUS_DISCONNECTED:
		var url = SERVER_URL + "/npcs"
		http.request(url)

func _on_data_received(result, response_code, headers, body):
	"""Handle NPC state updates"""
	if response_code != 200:
		print("[WARNING] Failed to get NPC states: ", response_code)
		return
	
	var json = JSON.new()
	var parse_result = json.parse(body.get_string_from_utf8())
	
	if parse_result != OK:
		print("[ERROR] Failed to parse NPC data")
		return
	
	var data = json.data
	_update_npcs(data)

func _update_npcs(data):
	"""Update NPC positions and LLM-generated thoughts"""
	for npc_name in data:
		if npc_name in npcs and npcs[npc_name]:
			var npc_data = data[npc_name]
			var npc_node = npcs[npc_name]
			
			# Smoothly move to new position
			var target_pos = Vector2(npc_data.x, npc_data.y)
			npc_node.position = npc_node.position.lerp(target_pos, 0.15)
			
			# Update thought bubble with LLM-generated content
			var label = npc_node.get_child(0)
			if label and label.name != "SpeechBubble":
				var thought = str(npc_data.get("thought", "..."))
				label.text = npc_name + "\n[" + thought + "]"

func _input(event):
	"""Handle mouse clicks for NPC interaction"""
	if event is InputEventMouseButton and event.pressed:
		var mouse_pos = get_global_mouse_position()
		
		if event.button_index == MOUSE_BUTTON_LEFT:
			# Check which NPC was clicked
			for npc_name in npcs:
				var npc = npcs[npc_name]
				if npc:
					var npc_rect = Rect2(npc.position, Vector2(32, 32))
					if npc_rect.has_point(mouse_pos):
						print("\n[INTERACTION] Clicked on ", npc_name)
						selected_npc = npc_name
						interact_with_llm_npc(npc_name, "Hello!")
						break
		
		elif event.button_index == MOUSE_BUTTON_RIGHT and selected_npc:
			# Right click for custom interaction
			interact_with_llm_npc(selected_npc, "How are you today?")

func interact_with_llm_npc(npc_name: String, message: String):
	"""Send interaction to LLM-powered NPC"""
	
	if is_requesting:
		print("[BUSY] Previous request still processing...")
		return
	
	is_requesting = true
	request_start_time = Time.get_ticks_msec()
	
	# Show immediate feedback
	show_thinking_bubble(npc_name)
	print("[SENDING] '%s' to %s..." % [message, npc_name])
	
	# Send to LLM server using fast 127.0.0.1
	var url = "http://127.0.0.1:8000/interact/" + npc_name
	var headers = ["Content-Type: application/json"]
	var body = JSON.stringify({"message": message})
	
	var error = interact_http.request(url, headers, HTTPClient.METHOD_POST, body)
	if error != OK:
		print("[ERROR] Failed to send request: ", error)
		is_requesting = false
		hide_thinking_bubble(npc_name)
	else:
		print("[REQUEST] Sent to LLM server, waiting for AI response...")

func _on_interaction_response(result, response_code, headers, body):
	"""Handle LLM-generated response"""
	is_requesting = false
	var elapsed_ms = Time.get_ticks_msec() - request_start_time
	
	print("\n[RESPONSE] Received in %.3f seconds" % [elapsed_ms / 1000.0])
	
	if response_code != 200:
		print("[ERROR] Server returned code: ", response_code)
		# Hide thinking bubbles on error
		for npc_name in npcs:
			hide_thinking_bubble(npc_name)
		return
	
	# Parse LLM response
	var json = JSON.new()
	var parse_result = json.parse(body.get_string_from_utf8())
	
	if parse_result != OK:
		print("[ERROR] Failed to parse LLM response")
		return
	
	var response_data = json.data
	var npc_name = response_data.get("npc", "")
	var llm_response = response_data.get("response", "...")
	var server_time = response_data.get("time", 0)
	var is_cached = response_data.get("cached", false)
	
	print("[%s] LLM says: '%s'" % [npc_name, llm_response])
	print("  Server processing: %.3fs %s" % [server_time, "(cached)" if is_cached else "(generated)"])
	print("  Total round-trip: %.3fs" % [elapsed_ms / 1000.0])
	
	# Display LLM response
	if npc_name != "":
		show_speech_response(npc_name, llm_response)

func show_thinking_bubble(npc_name: String):
	"""Show thinking indicator while LLM processes"""
	if npc_name in npcs:
		var bubble = npcs[npc_name].get_node_or_null("SpeechBubble")
		if bubble:
			bubble.text = "..."
			bubble.visible = true
			bubble.modulate = Color(0.7, 0.7, 1.0)  # Blue tint for thinking

func hide_thinking_bubble(npc_name: String):
	"""Hide thinking indicator"""
	if npc_name in npcs:
		var bubble = npcs[npc_name].get_node_or_null("SpeechBubble")
		if bubble and bubble.text == "...":
			bubble.visible = false

func show_speech_response(npc_name: String, text: String):
	"""Display LLM-generated speech"""
	if npc_name in npcs:
		var bubble = npcs[npc_name].get_node_or_null("SpeechBubble")
		if bubble:
			bubble.text = text
			bubble.visible = true
			bubble.modulate = Color.WHITE  # White for actual speech
			
			# Keep visible for duration based on text length
			var display_time = max(3.0, len(text) * 0.05)  # At least 3 seconds
			await get_tree().create_timer(display_time).timeout
			
			# Fade out
			var tween = create_tween()
			tween.tween_property(bubble, "modulate:a", 0.0, 0.5)
			await tween.finished
			
			bubble.visible = false
			bubble.modulate.a = 1.0  # Reset alpha

func _exit_tree():
	"""Cleanup on exit"""
	print("Bar scene closing, cleaning up HTTP connections...")
	if http:
		http.cancel_request()
	if interact_http:
		interact_http.cancel_request()
