# Godot JSON Protocol Client for Cozy Bar Server
# Handles bidirectional communication with Python simulation server
extends Node

signal npc_position_updated(npc_name, position)
signal npc_action_completed(npc_name, action, success)
signal dialogue_received(speaker, text)
signal state_updated(state_data)
signal connection_status_changed(connected)

var socket := StreamPeerTCP.new()
var is_connected := false
var server_host := "127.0.0.1"
var server_port := 9999
var receive_buffer := ""
var message_queue := []
var reconnect_timer := 0.0
var reconnect_delay := 5.0

# Animation interpolation
var npc_positions := {}  # Current positions
var npc_targets := {}    # Target positions
var interpolation_speed := 2.0  # Units per second

# Priority queue for player actions
var player_action_queue := []
var npc_action_queue := []
var max_queue_size := 10

func _ready():
	set_process(true)
	connect_to_server()

func _process(delta):
	# Handle reconnection
	if not is_connected:
		reconnect_timer += delta
		if reconnect_timer >= reconnect_delay:
			reconnect_timer = 0.0
			connect_to_server()
		return
	
	# Process incoming messages
	_process_incoming_messages()
	
	# Process message queue
	_process_message_queue()
	
	# Update NPC positions with interpolation
	_update_npc_positions(delta)

func connect_to_server():
	print("[CLIENT] Connecting to server at %s:%d" % [server_host, server_port])
	
	var result = socket.connect_to_host(server_host, server_port)
	if result == OK:
		is_connected = true
		emit_signal("connection_status_changed", true)
		print("[CLIENT] Connected to server")
		
		# Send initialization message
		send_message({"type": "INIT"})
	else:
		is_connected = false
		emit_signal("connection_status_changed", false)
		print("[CLIENT] Failed to connect: %d" % result)

func disconnect_from_server():
	if is_connected:
		socket.disconnect_from_host()
		is_connected = false
		emit_signal("connection_status_changed", false)
		print("[CLIENT] Disconnected from server")

func _process_incoming_messages():
	if not is_connected:
		return
	
	# Check connection status
	var status = socket.get_status()
	if status != StreamPeerTCP.STATUS_CONNECTED:
		if status == StreamPeerTCP.STATUS_ERROR or status == StreamPeerTCP.STATUS_NONE:
			is_connected = false
			emit_signal("connection_status_changed", false)
			print("[CLIENT] Connection lost")
		return
	
	# Read available data
	var available = socket.get_available_bytes()
	if available > 0:
		var data = socket.get_string(available)
		receive_buffer += data
		
		# Process complete messages (separated by newlines)
		while "\n" in receive_buffer:
			var parts = receive_buffer.split("\n", true, 1)
			var message = parts[0]
			receive_buffer = parts[1] if parts.size() > 1 else ""
			
			if message.length() > 0:
				_handle_server_message(message)

func _handle_server_message(message: String):
	# Try to parse as JSON
	var json = JSON.new()
	var parse_result = json.parse(message)
	
	if parse_result != OK:
		# Try legacy format for backward compatibility
		_handle_legacy_message(message)
		return
	
	var data = json.data
	var msg_type = data.get("type", "")
	
	match msg_type:
		"STATE_UPDATE":
			_handle_state_update(data)
		"ACTION_RESULT":
			_handle_action_result(data)
		"DIALOGUE":
			_handle_dialogue(data)
		_:
			# Generic response handling
			if data.has("status"):
				_handle_generic_response(data)

func _handle_legacy_message(message: String):
	# Handle old pipe-separated format
	var parts = message.split("|")
	if parts.size() >= 2:
		var status = parts[0]
		var content = parts[1] if parts.size() > 1 else ""
		
		print("[CLIENT] Legacy response: %s - %s" % [status, content])

func _handle_state_update(data: Dictionary):
	# Update NPC positions
	if data.has("positions"):
		for npc_name in data.positions:
			var pos = data.positions[npc_name]
			_set_npc_target_position(npc_name, Vector2(pos[0], pos[1]))
	
	# Emit general state update
	emit_signal("state_updated", data)

func _handle_action_result(data: Dictionary):
	var npc_name = data.get("npc", "")
	var action = data.get("action", "")
	var success = data.get("status", "") == "OK"
	
	if data.has("position"):
		var pos = data.position
		_set_npc_target_position(npc_name, Vector2(pos[0], pos[1]))
	
	emit_signal("npc_action_completed", npc_name, action, success)

func _handle_dialogue(data: Dictionary):
	var speaker = data.get("speaker", "Unknown")
	var text = data.get("dialogue", "...")
	
	emit_signal("dialogue_received", speaker, text)

func _handle_generic_response(data: Dictionary):
	var status = data.get("status", "")
	
	if status == "OK":
		print("[CLIENT] Request successful")
	else:
		var error = data.get("error", "Unknown error")
		print("[CLIENT] Request failed: %s" % error)

# Public API

func request_npc_action(npc_name: String, action: String, target = null, is_player_action: bool = false):
	var message = {
		"type": "ACTION",
		"npc": npc_name,
		"action": action
	}
	
	if target != null:
		message["target"] = str(target)
	
	# Add to appropriate queue based on priority
	if is_player_action:
		player_action_queue.append(message)
	else:
		npc_action_queue.append(message)

func request_dialogue(speaker: String, listener: String, context: String = ""):
	var message = {
		"type": "TALK",
		"speaker": speaker,
		"listener": listener,
		"message": context
	}
	
	# Player dialogues have priority
	player_action_queue.append(message)

func update_npc_position(npc_name: String, position: Vector2):
	var message = {
		"type": "MOVE",
		"npc": npc_name,
		"position": [position.x, position.y]
	}
	
	send_message(message)

func query_npcs():
	send_message({"type": "QUERY", "query": "npcs"})

func query_memories(npc_name: String):
	send_message({"type": "QUERY", "query": "memories", "npc": npc_name})

func save_game_state():
	send_message({"type": "SAVE"})

func load_game_state():
	send_message({"type": "LOAD"})

func request_simulation_tick():
	send_message({"type": "TICK"})

# Internal helpers

func send_message(data):
	if not is_connected:
		print("[CLIENT] Cannot send - not connected")
		return false
	
	var message: String
	if data is Dictionary:
		message = JSON.stringify(data) + "\n"
	else:
		message = str(data) + "\n"
	
	socket.put_data(message.to_utf8_buffer())
	return true

func _process_message_queue():
	# Process player actions first (priority)
	if player_action_queue.size() > 0:
		var message = player_action_queue.pop_front()
		send_message(message)
		return
	
	# Then process NPC actions
	if npc_action_queue.size() > 0:
		var message = npc_action_queue.pop_front()
		send_message(message)
		return

func _set_npc_target_position(npc_name: String, target_pos: Vector2):
	# Initialize if needed
	if not npc_positions.has(npc_name):
		npc_positions[npc_name] = target_pos
	
	# Set target for interpolation
	npc_targets[npc_name] = target_pos

func _update_npc_positions(delta: float):
	# Interpolate all NPC positions
	for npc_name in npc_positions:
		if not npc_targets.has(npc_name):
			continue
		
		var current = npc_positions[npc_name]
		var target = npc_targets[npc_name]
		
		if current.distance_to(target) > 0.1:
			# Interpolate position
			var direction = (target - current).normalized()
			var distance = interpolation_speed * delta
			
			if current.distance_to(target) <= distance:
				# Reached target
				npc_positions[npc_name] = target
			else:
				# Move towards target
				npc_positions[npc_name] = current + direction * distance
			
			# Emit position update
			emit_signal("npc_position_updated", npc_name, npc_positions[npc_name])

func get_npc_position(npc_name: String) -> Vector2:
	if npc_positions.has(npc_name):
		return npc_positions[npc_name]
	return Vector2.ZERO

# Utility functions for common actions

func make_npc_greet(npc_name: String, target: String):
	request_npc_action(npc_name, "greet", target)

func make_npc_move_to_bar(npc_name: String):
	request_npc_action(npc_name, "move_to_counter")

func make_npc_order_drink(npc_name: String):
	request_npc_action(npc_name, "order_drink")

func make_npc_perform(npc_name: String):
	request_npc_action(npc_name, "start_performance")

func make_npc_sit(npc_name: String):
	request_npc_action(npc_name, "sit_down")

func make_npc_chat(speaker: String, listener: String, topic: String = ""):
	request_dialogue(speaker, listener, topic)