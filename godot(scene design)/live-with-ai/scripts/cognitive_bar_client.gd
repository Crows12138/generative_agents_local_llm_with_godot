# cognitive_bar_client.gd - Enhanced client for cognitive AI server
extends Node2D

signal npc_state_updated(npc_name: String, state: Dictionary)
signal dialogue_received(npc_name: String, response: String)
signal perception_received(npc_name: String, perceptions: Array)

var socket: StreamPeerTCP
var is_connected: bool = false
var npcs = {}
var reconnect_timer: Timer

# Server configuration
const SERVER_HOST = "127.0.0.1"
const SERVER_PORT = 9999
const RECONNECT_DELAY = 5.0

func _ready():
	print("[COGNITIVE] Initializing Cognitive AI Client...")
	
	# Create socket
	socket = StreamPeerTCP.new()
	
	# Setup reconnect timer
	reconnect_timer = Timer.new()
	reconnect_timer.wait_time = RECONNECT_DELAY
	reconnect_timer.timeout.connect(_attempt_reconnect)
	add_child(reconnect_timer)
	
	# Get NPC nodes
	_find_npcs()
	
	# Connect to server
	connect_to_server()

func _find_npcs():
	"""Find all NPC nodes in the scene"""
	# Check for NPCs group
	if has_node("NPCs"):
		var npcs_group = get_node("NPCs")
		for child in npcs_group.get_children():
			npcs[child.name] = child
			print("[COGNITIVE] Found NPC: " + child.name)
	
	# Also check direct children
	for child in get_children():
		if child.name in ["Bob", "Alice", "Sam", "Charlie", "Eve"]:
			npcs[child.name] = child
			print("[COGNITIVE] Found NPC: " + child.name)

func connect_to_server():
	"""Connect to the cognitive AI server"""
	print("[COGNITIVE] Connecting to server at %s:%d..." % [SERVER_HOST, SERVER_PORT])
	
	var error = socket.connect_to_host(SERVER_HOST, SERVER_PORT)
	if error == OK:
		is_connected = true
		print("[COGNITIVE] Connected to AI server!")
		
		# Request initial states
		_send_command("GET_ALL_STATES")
		
		# Stop reconnect timer
		reconnect_timer.stop()
	else:
		print("[COGNITIVE] Failed to connect: " + str(error))
		is_connected = false
		
		# Start reconnect timer
		reconnect_timer.start()

func _attempt_reconnect():
	"""Try to reconnect to server"""
	if not is_connected:
		print("[COGNITIVE] Attempting to reconnect...")
		connect_to_server()

func _process(_delta):
	"""Process incoming messages from server"""
	if not is_connected:
		return
	
	# Check connection status
	if socket.get_status() != StreamPeerTCP.STATUS_CONNECTED:
		if socket.get_status() == StreamPeerTCP.STATUS_ERROR:
			print("[COGNITIVE] Connection error, disconnecting...")
			_on_disconnected()
		return
	
	# Read available data
	var available = socket.get_available_bytes()
	if available > 0:
		var data = socket.get_string(available)
		_process_server_message(data)

func _process_server_message(data: String):
	"""Process message from server"""
	# Handle multiple messages in one packet
	var messages = data.split("\n")
	
	for message in messages:
		if message.is_empty():
			continue
		
		var parts = message.split("|")
		if parts.size() < 1:
			continue
		
		var msg_type = parts[0]
		
		match msg_type:
			"ALL_STATES":
				if parts.size() > 1:
					_handle_all_states(parts[1])
			
			"STATE_UPDATE":
				if parts.size() > 1:
					_handle_state_update(parts[1])
			
			"DIALOGUE_RESPONSE":
				if parts.size() > 2:
					_handle_dialogue_response(parts[1], parts[2])
			
			"PERCEPTIONS":
				if parts.size() > 2:
					_handle_perceptions(parts[1], parts[2])
			
			"REFLECTION":
				if parts.size() > 2:
					_handle_reflection(parts[1], parts[2])
			
			"STATE":
				if parts.size() > 1:
					_handle_single_state(parts[1])
			
			"ERROR":
				if parts.size() > 1:
					print("[COGNITIVE] Server error: " + parts[1])
			
			_:
				print("[COGNITIVE] Unknown message type: " + msg_type)

func _handle_all_states(json_data: String):
	"""Handle all NPC states"""
	var json = JSON.new()
	var result = json.parse(json_data)
	
	if result == OK:
		var states = json.data
		for npc_name in states:
			var state = states[npc_name]
			_update_npc_state(npc_name, state)
			npc_state_updated.emit(npc_name, state)

func _handle_state_update(json_data: String):
	"""Handle state update broadcast"""
	_handle_all_states(json_data)

func _handle_single_state(json_data: String):
	"""Handle single NPC state"""
	var json = JSON.new()
	var result = json.parse(json_data)
	
	if result == OK:
		var state = json.data
		if state.has("name"):
			_update_npc_state(state["name"], state)
			npc_state_updated.emit(state["name"], state)

func _update_npc_state(npc_name: String, state: Dictionary):
	"""Update NPC visual state based on AI state"""
	if not npc_name in npcs:
		return
	
	var npc = npcs[npc_name]
	
	# Update position
	if state.has("position"):
		var target_pos = Vector2(
			state["position"]["x"] * 64,  # Assuming 64px tile size
			state["position"]["y"] * 64
		)
		# Smooth movement
		npc.position = npc.position.lerp(target_pos, 0.1)
	
	# Update activity label if exists
	if npc.has_node("ActivityLabel"):
		var label = npc.get_node("ActivityLabel")
		label.text = state.get("activity", "idle")
	
	# Update emotion sprite if exists
	if npc.has_node("EmotionSprite"):
		var sprite = npc.get_node("EmotionSprite")
		# Set emotion sprite based on state
		_set_emotion_sprite(sprite, state.get("emotion", "neutral"))
	
	# Store state in NPC
	npc.set_meta("ai_state", state)

func _set_emotion_sprite(sprite: Sprite2D, emotion: String):
	"""Set emotion sprite based on emotion state"""
	# This would map emotions to sprite frames
	# You'd need emotion sprites in your project
	pass

func _handle_dialogue_response(npc_name: String, response: String):
	"""Handle dialogue response from NPC"""
	print("[COGNITIVE] %s says: %s" % [npc_name, response])
	dialogue_received.emit(npc_name, response)
	
	# Show dialogue bubble if exists
	if npc_name in npcs:
		var npc = npcs[npc_name]
		if npc.has_method("show_dialogue"):
			npc.show_dialogue(response)

func _handle_perceptions(npc_name: String, json_data: String):
	"""Handle NPC perceptions"""
	var json = JSON.new()
	var result = json.parse(json_data)
	
	if result == OK:
		var perceptions = json.data
		print("[COGNITIVE] %s perceives: %s" % [npc_name, str(perceptions)])
		perception_received.emit(npc_name, perceptions)

func _handle_reflection(npc_name: String, reflection: String):
	"""Handle NPC reflection"""
	print("[COGNITIVE] %s reflects: %s" % [npc_name, reflection.substr(0, 100) + "..."])

func _on_disconnected():
	"""Handle disconnection from server"""
	is_connected = false
	print("[COGNITIVE] Disconnected from server")
	
	# Start reconnect timer
	reconnect_timer.start()

# Public API functions

func send_dialogue(npc_name: String, user_input: String):
	"""Send dialogue to an NPC"""
	_send_command("DIALOGUE|%s|%s" % [npc_name, user_input])

func request_npc_state(npc_name: String):
	"""Request state of specific NPC"""
	_send_command("GET_STATE|%s" % npc_name)

func request_all_states():
	"""Request all NPC states"""
	_send_command("GET_ALL_STATES")

func trigger_perception(npc_name: String):
	"""Trigger perception for an NPC"""
	_send_command("PERCEIVE|%s" % npc_name)

func trigger_reflection(npc_name: String):
	"""Trigger reflection for an NPC"""
	_send_command("REFLECT|%s" % npc_name)

func execute_action(npc_name: String, action: String, target: String = ""):
	"""Make NPC execute specific action"""
	if target.is_empty():
		_send_command("ACTION|%s|%s" % [npc_name, action])
	else:
		_send_command("ACTION|%s|%s|%s" % [npc_name, action, target])

func _send_command(command: String):
	"""Send command to server"""
	if not is_connected:
		print("[COGNITIVE] Not connected to server")
		return
	
	if socket.get_status() != StreamPeerTCP.STATUS_CONNECTED:
		print("[COGNITIVE] Socket not ready")
		return
	
	socket.put_data(command.to_utf8_buffer())

# Input handling for testing

func _input(event):
	if event is InputEventMouseButton and event.pressed:
		if event.button_index == MOUSE_BUTTON_LEFT:
			_handle_click(event.position)

func _handle_click(position: Vector2):
	"""Handle clicking on NPCs"""
	var mouse_pos = get_global_mouse_position()
	
	for npc_name in npcs:
		var npc = npcs[npc_name]
		var distance = npc.position.distance_to(mouse_pos)
		
		if distance < 50:  # Click radius
			print("[COGNITIVE] Clicked on " + npc_name)
			
			# Send a test dialogue
			var greetings = [
				"Hello! How are you?",
				"What are you doing?",
				"Nice to see you!",
				"How's it going?"
			]
			var greeting = greetings[randi() % greetings.size()]
			send_dialogue(npc_name, greeting)
			break
