extends Node2D

@onready var http = HTTPRequest.new()
@onready var interact_http = HTTPRequest.new()
var update_timer = Timer.new()
var npcs = {}  # Store NPC references
var selected_npc = null
var is_requesting = false
var request_start_time = 0

func _ready():
	print("Bar scene started with FIXED networking!")
	
	# Get existing NPC nodes
	npcs["Bob"] = $Bob
	npcs["Alice"] = $Alice  
	npcs["Sam"] = $Sam
	
	# Create speech bubbles for each NPC
	for npc_name in npcs:
		create_speech_bubble(npcs[npc_name])
	
	# Setup HTTP requests with timeout fix
	add_child(http)
	http.timeout = 2.0  # 2 second timeout
	http.request_completed.connect(_on_data_received)
	
	add_child(interact_http)
	interact_http.timeout = 2.0  # 2 second timeout
	interact_http.request_completed.connect(_on_interaction_response)
	
	# Setup timer
	add_child(update_timer)
	update_timer.wait_time = 5.0  # Update every 5 seconds
	update_timer.one_shot = false
	update_timer.timeout.connect(_fetch_npc_states)
	update_timer.start()
	
	# Request immediately
	_fetch_npc_states()
	
	print("HTTP timeout set to 2 seconds")
	print("Using 127.0.0.1 instead of localhost")

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
	"""Request NPC states"""
	if http.get_http_client_status() == HTTPClient.STATUS_DISCONNECTED:
		# Use 127.0.0.1 instead of localhost
		http.request("http://127.0.0.1:8000/npcs")

func _on_data_received(result, response_code, headers, body):
	"""Handle received data"""
	if response_code != 200:
		print("Failed to get data, code: ", response_code)
		return
	
	var json = JSON.new()
	var parse_result = json.parse(body.get_string_from_utf8())
	
	if parse_result != OK:
		print("Failed to parse JSON")
		return
	
	var data = json.data
	_update_npcs(data)

func _update_npcs(data):
	"""Update NPC positions and thoughts"""
	for npc_name in data:
		if npc_name in npcs and npcs[npc_name]:
			var npc_data = data[npc_name]
			var npc_node = npcs[npc_name]
			
			# Smoothly move to new position
			var target_pos = Vector2(npc_data.x, npc_data.y)
			npc_node.position = npc_node.position.lerp(target_pos, 0.1)
			
			# Update name label with thought
			var label = npc_node.get_child(0)
			if label and label.name != "SpeechBubble":
				label.text = npc_name + "\n" + str(npc_data.thought)

func _input(event):
	"""Handle mouse clicks for NPC interaction"""
	if event is InputEventMouseButton and event.pressed and event.button_index == MOUSE_BUTTON_LEFT:
		var mouse_pos = get_global_mouse_position()
		
		# Check which NPC was clicked
		for npc_name in npcs:
			var npc = npcs[npc_name]
			if npc:
				var npc_rect = Rect2(npc.position, Vector2(32, 32))
				if npc_rect.has_point(mouse_pos):
					print("Clicked on ", npc_name)
					selected_npc = npc_name
					# Use local response for instant feedback
					interact_with_npc_hybrid(npc_name, "Hello")
					break
	
	# Right click to send custom message
	if event is InputEventMouseButton and event.pressed and event.button_index == MOUSE_BUTTON_RIGHT:
		if selected_npc:
			interact_with_npc_hybrid(selected_npc, "How are you today?")

func interact_with_npc_hybrid(npc_name: String, message: String):
	"""Hybrid approach: instant local response + background server update"""
	
	# Instant local response
	var quick_response = get_local_response(npc_name, message)
	show_speech_response(npc_name, quick_response)
	print("[LOCAL] ", npc_name, " says: ", quick_response, " (instant)")
	
	# Also send to server in background (optional)
	if not is_requesting:
		send_to_server_async(npc_name, message)

func get_local_response(npc_name: String, message: String) -> String:
	"""Generate instant local response"""
	var responses = {
		"Bob": {
			"hello": ["Welcome!", "What'll it be?", "Good to see you!"],
			"how are": ["Great, busy night!", "Can't complain!", "Better now!"],
			"default": ["Coming right up!", "Sure thing!", "You got it!"]
		},
		"Alice": {
			"hello": ["Oh, hello!", "Hi there!", "Nice to see you!"],
			"how are": ["Feeling relaxed!", "This drink is perfect!", "Wonderful!"],
			"default": ["That's interesting!", "I agree!", "Tell me more!"]
		},
		"Sam": {
			"hello": ["Hey there!", "Welcome!", "Hi friend!"],
			"how are": ["Ready to rock!", "Music is life!", "Feeling great!"],
			"default": ["Any requests?", "Let's play!", "Rock on!"]
		}
	}
	
	var msg_lower = message.to_lower()
	var npc_responses = responses.get(npc_name, {})
	
	# Find matching response category
	for key in npc_responses:
		if key in msg_lower:
			var options = npc_responses[key]
			return options[randi() % options.size()]
	
	# Default response
	var defaults = npc_responses.get("default", ["Hello!"])
	return defaults[randi() % defaults.size()]

func send_to_server_async(npc_name: String, message: String):
	"""Send to server in background (non-blocking)"""
	is_requesting = true
	request_start_time = Time.get_ticks_msec()
	
	# Use 127.0.0.1 instead of localhost
	var url = "http://127.0.0.1:8000/interact/" + npc_name
	var headers = ["Content-Type: application/json"]
	var body = JSON.stringify({"message": message})
	
	var error = interact_http.request(url, headers, HTTPClient.METHOD_POST, body)
	if error != OK:
		print("Failed to send request: ", error)
		is_requesting = false

func _on_interaction_response(result, response_code, headers, body):
	"""Handle server response (background update)"""
	is_requesting = false
	var actual_time = Time.get_ticks_msec() - request_start_time
	
	if response_code == 200:
		var json = JSON.new()
		var parse_result = json.parse(body.get_string_from_utf8())
		
		if parse_result == OK:
			var response_data = json.data
			print("[SERVER] Updated from server in %.3fs" % [actual_time / 1000.0])
			# Optionally update with server response if it's different/better
			# But user already saw instant local response

func show_thinking_bubble(npc_name: String):
	"""Show thinking indicator"""
	if npc_name in npcs:
		var bubble = npcs[npc_name].get_node_or_null("SpeechBubble")
		if bubble:
			bubble.text = "..."
			bubble.visible = true
			bubble.modulate = Color(0.8, 0.8, 0.8)

func show_speech_response(npc_name: String, text: String):
	"""Display speech response"""
	if npc_name in npcs:
		var bubble = npcs[npc_name].get_node_or_null("SpeechBubble")
		if bubble:
			bubble.text = text
			bubble.visible = true
			bubble.modulate = Color.WHITE
			
			# Auto-hide after 3 seconds
			await get_tree().create_timer(3.0).timeout
			bubble.visible = false