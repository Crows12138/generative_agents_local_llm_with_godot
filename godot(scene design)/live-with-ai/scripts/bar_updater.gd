extends Node2D

@onready var http = HTTPRequest.new()
@onready var interact_http = HTTPRequest.new()
var update_timer = Timer.new()
var npcs = {}  # Store NPC references
var selected_npc = null
var is_requesting = false  # Prevent duplicate requests
var thinking_bubbles = {}  # Store thinking bubble references

func _ready():
	print("Bar scene started!")
	
	# Get existing NPC nodes
	npcs["Bob"] = $Bob
	npcs["Alice"] = $Alice  
	npcs["Sam"] = $Sam
	
	# Create speech bubbles for each NPC
	for npc_name in npcs:
		create_speech_bubble(npcs[npc_name])
	
	# Setup HTTP requests
	add_child(http)
	http.request_completed.connect(_on_data_received)
	
	add_child(interact_http)
	interact_http.request_completed.connect(_on_interaction_response)
	
	# Setup timer (update every 2 seconds to reduce load)
	add_child(update_timer)
	update_timer.wait_time = 2.0
	update_timer.one_shot = false
	update_timer.timeout.connect(_fetch_npc_states)
	update_timer.start()
	
	# Request immediately
	_fetch_npc_states()

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
		http.request("http://localhost:8000/npcs")

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
					interact_with_npc(npc_name, "Hello")
					break
	
	# Right click to send custom message
	if event is InputEventMouseButton and event.pressed and event.button_index == MOUSE_BUTTON_RIGHT:
		if selected_npc:
			interact_with_npc(selected_npc, "How are you today?")

func interact_with_npc(npc_name: String, message: String):
	"""Send interaction to NPC with visual feedback"""
	
	# Prevent duplicate requests
	if is_requesting:
		print("Request in progress, skipping...")
		return
	
	is_requesting = true
	
	# Show thinking bubble immediately
	show_thinking_bubble(npc_name)
	
	# Send request
	var url = "http://localhost:8000/interact/" + npc_name
	var headers = ["Content-Type: application/json"]
	var body = JSON.stringify({"message": message})
	
	print("Sending to ", npc_name, ": ", message)
	interact_http.request(url, headers, HTTPClient.METHOD_POST, body)

func show_thinking_bubble(npc_name: String):
	"""Show thinking indicator"""
	if npc_name in npcs:
		var bubble = npcs[npc_name].get_node_or_null("SpeechBubble")
		if bubble:
			bubble.text = "..."
			bubble.visible = true
			bubble.modulate = Color(0.8, 0.8, 0.8)  # Gray for thinking

func hide_thinking_bubble(npc_name: String):
	"""Hide thinking indicator"""
	if npc_name in npcs:
		var bubble = npcs[npc_name].get_node_or_null("SpeechBubble")
		if bubble and bubble.text == "...":
			bubble.visible = false

func _on_interaction_response(result, response_code, headers, body):
	"""Handle NPC response"""
	is_requesting = false
	
	if response_code != 200:
		print("Interaction failed, code: ", response_code)
		# Hide all thinking bubbles on failure
		for npc_name in npcs:
			hide_thinking_bubble(npc_name)
		return
	
	var json = JSON.new()
	var parse_result = json.parse(body.get_string_from_utf8())
	
	if parse_result != OK:
		print("Failed to parse response")
		return
	
	var response_data = json.data
	var response_time = response_data.get("time", 0)
	var is_cached = response_data.get("cached", false)
	
	print("[", response_data.npc, "] says: ", response_data.response)
	print("  Response time: ", "%.3f" % response_time, "s", " (cached)" if is_cached else "")
	
	# Show response
	show_speech_response(response_data.npc, response_data.response)

func show_speech_response(npc_name: String, text: String):
	"""Display speech response"""
	if npc_name in npcs:
		var bubble = npcs[npc_name].get_node_or_null("SpeechBubble")
		if bubble:
			bubble.text = text
			bubble.visible = true
			bubble.modulate = Color.WHITE  # White for actual speech
			
			# Auto-hide after 3 seconds
			await get_tree().create_timer(3.0).timeout
			bubble.visible = false