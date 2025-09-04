extends Node2D

var npcs = {}
var is_requesting = false

func _ready():
	print("Bar scene started!")
	
	# Fix 1: Adjust scene position
	setup_camera()
	setup_scene_position()
	
	# Get or create NPCs
	setup_npcs()
	
	# Initialize interaction (no status updates needed)
	setup_interaction()

func setup_camera():
	"""Setup camera to ensure scene is visible"""
	var camera = Camera2D.new()
	camera.name = "MainCamera"
	camera.position = Vector2(400, 300)  # Scene center
	camera.zoom = Vector2(1, 1)
	camera.enabled = true
	add_child(camera)
	print("Camera setup at center")

func setup_scene_position():
	"""Ensure scene is at correct position"""
	position = Vector2(0, 0)
	
	# If there's a background, adjust it
	if has_node("Background"):
		$Background.position = Vector2(0, 0)

func setup_npcs():
	"""Setup NPC positions"""
	# Get existing nodes or create new ones
	var npc_data = {
		"Bob": Vector2(400, 80),
		"Alice": Vector2(200, 300),
		"Sam": Vector2(600, 400)
	}
	
	for npc_name in npc_data:
		var npc = get_node_or_null(npc_name)
		
		if not npc:
			# Create simple ColorRect as NPC
			npc = ColorRect.new()
			npc.name = npc_name
			npc.size = Vector2(32, 32)
			npc.color = Color(randf(), randf(), randf())
			add_child(npc)
			
			# Add label
			var label = Label.new()
			label.text = npc_name
			label.position = Vector2(-10, -20)
			npc.add_child(label)
		
		# Set position
		npc.position = npc_data[npc_name]
		npcs[npc_name] = npc
		
		print("Setup NPC: ", npc_name, " at ", npc.position)

func setup_interaction():
	"""Setup interaction system"""
	# Only for click interaction, no periodic updates needed
	print("Interaction system ready")

func _input(event):
	if event is InputEventMouseButton and event.pressed:
		if event.button_index == MOUSE_BUTTON_LEFT:
			var mouse_pos = get_global_mouse_position()
			
			for npc_name in npcs:
				var npc = npcs[npc_name]
				var distance = npc.position.distance_to(mouse_pos)
				
				if distance < 50:
					print("Clicked on ", npc_name)
					interact_with_npc(npc_name)
					break

func interact_with_npc(npc_name: String):
	"""Interact with NPC"""
	if is_requesting:
		print("Request in progress...")
		return
	
	is_requesting = true
	
	# Create HTTP request
	var http = HTTPRequest.new()
	add_child(http)
	
	# Connect signal
	http.request_completed.connect(_on_interaction_response.bind(npc_name, http))
	
	# Send request - use 127.0.0.1
	var url = "http://127.0.0.1:8000/interact/" + npc_name
	var headers = ["Content-Type: application/json"]
	var body = JSON.stringify({"message": "Hello, how are you?"})
	
	print("Sending request to: ", url)
	var error = http.request(url, headers, HTTPClient.METHOD_POST, body)
	
	if error != OK:
		print("Request failed: ", error)
		is_requesting = false
		http.queue_free()

func _on_interaction_response(result, response_code, headers, body, npc_name, http):
	"""Handle response"""
	is_requesting = false
	
	print("Response code: ", response_code)
	
	if response_code == 200:
		var json = JSON.new()
		var parse_result = json.parse(body.get_string_from_utf8())
		
		if parse_result == OK:
			var data = json.data
			print("[", npc_name, "] says: ", data.response)
			show_speech_bubble(npc_name, data.response)
	else:
		print("Request failed with code: ", response_code)
	
	# Clean up HTTP node
	http.queue_free()

func show_speech_bubble(npc_name: String, text: String):
	"""Show speech bubble"""
	if npc_name in npcs:
		var npc = npcs[npc_name]
		
		# Create or get bubble
		var bubble = npc.get_node_or_null("Bubble")
		if not bubble:
			bubble = Label.new()
			bubble.name = "Bubble"
			npc.add_child(bubble)
		
		bubble.text = text.substr(0, 50) + "..."
		bubble.position = Vector2(-50, -40)
		bubble.visible = true
		
		# Hide after 5 seconds
		await get_tree().create_timer(5.0).timeout
		bubble.visible = false