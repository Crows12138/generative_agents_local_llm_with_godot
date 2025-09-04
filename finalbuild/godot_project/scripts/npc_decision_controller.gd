extends Node
# NPC Decision Controller - Main integration point
# Connects state detector, WebSocket client, and action executor

@export var npc_name: String = "bob"  # NPC identifier
@export var decision_server_url: String = "ws://localhost:9998"
@export var enable_batch_mode: bool = false  # Send with other NPCs
@export var debug_mode: bool = true

# Component references
@onready var state_detector = $StateDetector
@onready var action_executor = $ActionExecutor

# WebSocket client
var socket = WebSocketPeer.new()
var connected_to_server: bool = false
var reconnect_timer: Timer
var last_state_sent: int = -1

# Statistics
var stats = {
	"decisions_requested": 0,
	"decisions_received": 0,
	"actions_executed": 0,
	"connection_failures": 0,
	"avg_response_time": 0.0
}

var request_timestamp: float = 0.0

func _ready():
	print("[DecisionController] Initializing for NPC: ", npc_name)
	
	# Setup reconnect timer
	reconnect_timer = Timer.new()
	reconnect_timer.wait_time = 5.0
	reconnect_timer.timeout.connect(_attempt_reconnect)
	add_child(reconnect_timer)
	
	# Connect component signals
	if state_detector:
		state_detector.state_changed.connect(_on_state_changed)
	else:
		push_error("[DecisionController] StateDetector not found!")
	
	if action_executor:
		action_executor.action_completed.connect(_on_action_completed)
	else:
		push_error("[DecisionController] ActionExecutor not found!")
	
	# Connect to decision server
	_connect_to_server()

func _process(_delta):
	if socket.get_ready_state() != WebSocketPeer.STATE_CLOSED:
		socket.poll()
	
	var state = socket.get_ready_state()
	
	match state:
		WebSocketPeer.STATE_OPEN:
			if not connected_to_server:
				connected_to_server = true
				reconnect_timer.stop()
				print("[DecisionController] Connected to decision server")
			
			# Check for incoming messages
			while socket.get_available_packet_count():
				_handle_server_message()
		
		WebSocketPeer.STATE_CLOSED:
			if connected_to_server:
				connected_to_server = false
				stats.connection_failures += 1
				print("[DecisionController] Disconnected from server")
				reconnect_timer.start()

func _connect_to_server():
	"""Connect to the decision server"""
	print("[DecisionController] Connecting to: ", decision_server_url)
	
	var error = socket.connect_to_url(decision_server_url)
	if error != OK:
		push_error("[DecisionController] Failed to connect: " + str(error))
		reconnect_timer.start()
	else:
		print("[DecisionController] Connection initiated...")

func _attempt_reconnect():
	"""Try to reconnect to server"""
	if not connected_to_server:
		print("[DecisionController] Attempting reconnection...")
		_connect_to_server()

func _on_state_changed(state_flags: int):
	"""Handle state changes from detector"""
	if not connected_to_server:
		if debug_mode:
			print("[DecisionController] State changed but not connected: ", state_flags)
		return
	
	# Don't send duplicate states
	if state_flags == last_state_sent:
		return
	
	last_state_sent = state_flags
	
	# Create request
	var request = {
		"npc": npc_name,
		"state": state_flags,
		"timestamp": Time.get_ticks_msec() / 1000.0
	}
	
	# Send to server
	_send_to_server(request)
	request_timestamp = Time.get_ticks_msec() / 1000.0
	stats.decisions_requested += 1
	
	if debug_mode:
		var decoded = state_detector.decode_state(state_flags)
		print("[DecisionController] Sent state: ", decoded)

func _send_to_server(data: Dictionary):
	"""Send data to decision server"""
	if socket.get_ready_state() != WebSocketPeer.STATE_OPEN:
		push_warning("[DecisionController] Cannot send - not connected")
		return
	
	var json_string = JSON.stringify(data)
	socket.send_text(json_string)

func _handle_server_message():
	"""Process message from decision server"""
	var packet = socket.get_packet()
	var json_string = packet.get_string_from_utf8()
	
	var json = JSON.new()
	var parse_result = json.parse(json_string)
	
	if parse_result != OK:
		push_error("[DecisionController] Failed to parse server message")
		return
	
	var data = json.data
	
	# Update response time stat
	if request_timestamp > 0:
		var response_time = Time.get_ticks_msec() / 1000.0 - request_timestamp
		stats.avg_response_time = (stats.avg_response_time * stats.decisions_received + response_time) / (stats.decisions_received + 1)
		request_timestamp = 0
	
	stats.decisions_received += 1
	
	# Handle different message types
	match data.get("type", ""):
		"decision":
			_handle_decision(data)
		"batch_decisions":
			_handle_batch_decisions(data)
		"error":
			push_error("[DecisionController] Server error: " + data.get("message", "Unknown"))
		_:
			if debug_mode:
				print("[DecisionController] Unknown message type: ", data)

func _handle_decision(data: Dictionary):
	"""Handle single decision from server"""
	var action = data.get("action", "observe")
	var priority = data.get("priority", 0)
	
	if debug_mode:
		print("[DecisionController] Received decision: ", action, " (priority: ", priority, ")")
	
	# Queue action for execution
	if action_executor:
		action_executor.queue_action({
			"action": action,
			"priority": priority,
			"params": data.get("params", {})
		})
		stats.actions_executed += 1
	else:
		push_warning("[DecisionController] No action executor to handle decision")

func _handle_batch_decisions(data: Dictionary):
	"""Handle batch decisions (if this NPC is included)"""
	var decisions = data.get("decisions", {})
	
	if npc_name in decisions:
		var decision = decisions[npc_name]
		_handle_decision({
			"action": decision.get("action", "observe"),
			"priority": decision.get("priority", 0)
		})

func _on_action_completed(action_name: String, result: String):
	"""Handle action completion feedback"""
	if debug_mode:
		print("[DecisionController] Action completed: ", action_name, " - ", result)
	
	# Send completion feedback to server (optional)
	if connected_to_server:
		_send_to_server({
			"type": "action_complete",
			"npc": npc_name,
			"action": action_name,
			"result": result,
			"timestamp": Time.get_ticks_msec() / 1000.0
		})
	
	# Request new decision after action completes
	if state_detector:
		var current_state = state_detector.get_compressed_state()
		if current_state != last_state_sent:
			_on_state_changed(current_state)

# Public API
func force_decision_request():
	"""Manually request a decision based on current state"""
	if state_detector:
		var current_state = state_detector.get_compressed_state()
		last_state_sent = -1  # Force send even if same
		_on_state_changed(current_state)

func get_stats() -> Dictionary:
	"""Get controller statistics"""
	return stats.duplicate()

func reset_stats():
	"""Reset statistics"""
	stats.decisions_requested = 0
	stats.decisions_received = 0
	stats.actions_executed = 0
	stats.connection_failures = 0
	stats.avg_response_time = 0.0

func is_server_connected() -> bool:
	"""Check if connected to decision server"""
	return connected_to_server

# Debug UI helpers
func _input(event):
	if not debug_mode:
		return
	
	# Debug hotkeys
	if event.is_action_pressed("ui_page_up"):  # PageUp - Print stats
		print("[DecisionController] Stats: ", stats)
	
	if event.is_action_pressed("ui_page_down"):  # PageDown - Force decision
		print("[DecisionController] Forcing decision request...")
		force_decision_request()
