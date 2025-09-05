extends Node
# NPC Decision Controller - Main integration point
# Connects state detector, WebSocket client, and action executor

@export var npc_name: String = "bob"  # NPC identifier
@export var decision_server_url: String = "ws://localhost:9998"
@export var enable_batch_mode: bool = false  # Send with other NPCs
@export var debug_mode: bool = true

# Component references
var state_detector: Node
var action_executor: Node

# WebSocket client
var socket = WebSocketPeer.new()
var connected_to_server: bool = false
var reconnect_timer: Timer
var last_state_sent: int = -1

# Request deduplication and rate limiting
var last_request_time: float = 0.0
var min_request_interval: float = 1.0  # Minimum 1 second between requests
var request_queue: Array = []  # Queue for delayed requests
var queue_timer: Timer
var last_state_hash: int = 0  # Track state changes
var pending_request: bool = false  # Track if request is in progress

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
	
	# Setup queue timer for request rate limiting
	queue_timer = Timer.new()
	queue_timer.wait_time = 0.1  # Check queue every 100ms
	queue_timer.timeout.connect(_process_request_queue)
	queue_timer.autostart = true
	add_child(queue_timer)
	
	# Setup request timeout timer
	var timeout_timer = Timer.new()
	timeout_timer.name = "RequestTimeoutTimer"
	timeout_timer.wait_time = 3.0  # 3 second timeout
	timeout_timer.one_shot = true
	timeout_timer.timeout.connect(_on_request_timeout)
	add_child(timeout_timer)
	
	
	# Connect to decision server
	_connect_to_server()
	
	# Delay component connection until child nodes are ready
	call_deferred("_connect_components")

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

func _connect_components():
	"""Connect to child components after they're ready"""
	print("[DecisionController-", npc_name, "] Connecting to components...")
	
	# Find component references after scene is ready
	state_detector = get_node_or_null("StateDetector")
	action_executor = get_node_or_null("ActionExecutor")
	
	# If ActionExecutor not found as child, try finding it as sibling
	if not action_executor:
		action_executor = get_parent().get_node_or_null("ActionExecutor")
	
	# If still not found, try finding it in the parent's parent (NPC node)
	if not action_executor:
		var npc_parent = get_parent()
		if npc_parent:
			action_executor = npc_parent.get_node_or_null("ActionExecutor")
	
	# Connect component signals
	if state_detector:
		if state_detector.has_signal("state_changed"):
			state_detector.state_changed.connect(_on_state_changed)
			print("[DecisionController-", npc_name, "] Connected to StateDetector signal")
		else:
			push_error("[DecisionController-", npc_name, "] StateDetector has no state_changed signal!")
	else:
		push_error("[DecisionController-", npc_name, "] StateDetector not found!")
	
	if action_executor:
		if action_executor.has_signal("action_completed"):
			action_executor.action_completed.connect(_on_action_completed)
		print("[DecisionController-", npc_name, "] Connected to ActionExecutor")
	else:
		push_error("[DecisionController-", npc_name, "] ActionExecutor not found!")

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
	"""Handle state changes from detector with improved deduplication"""
	print("[DecisionController-", npc_name, "] State changed: ", state_flags, " Connected: ", connected_to_server)
	
	if not connected_to_server:
		print("[DecisionController-", npc_name, "] State changed but not connected: ", state_flags)
		return
	
	# Don't send if we're already waiting for a response
	if pending_request:
		print("[DecisionController-", npc_name, "] Already waiting for response, skipping: ", state_flags)
		return
	
	# Improved duplicate detection using hash
	var current_time = Time.get_ticks_msec() / 1000.0
	var state_hash = hash(str(state_flags) + str(int(current_time * 10)))  # Include time component
	
	# Don't send duplicate states or very recent similar states
	if (state_flags == last_state_sent and 
		current_time - last_request_time < min_request_interval):
		print("[DecisionController-", npc_name, "] Duplicate/too frequent, skipping: ", state_flags)
		return
	
	# Check rate limiting
	if current_time - last_request_time < min_request_interval:
		print("[DecisionController-", npc_name, "] Rate limited, queuing request: ", state_flags)
		# Clear old queue items for same state
		request_queue = request_queue.filter(func(req): return req.state != state_flags)
		# Add to queue
		request_queue.append({
			"state": state_flags,
			"timestamp": current_time
		})
		return
	
	# Send immediately
	_send_state_request(state_flags)

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
	
	# Clear pending request flag
	pending_request = false
	
	# Reset any timeout timer if exists
	if has_node("RequestTimeoutTimer"):
		$RequestTimeoutTimer.stop()
	
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
	
	print("[DecisionController-", npc_name, "] Received decision: ", action, " (priority: ", priority, ")")
	
	# Queue action for execution
	if action_executor:
		print("[DecisionController-", npc_name, "] Queuing action to executor")
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

func test_manual_decision():
	"""Test sending a manual decision request"""
	print("[DecisionController-", npc_name, "] Testing manual decision...")
	print("  Connected: ", connected_to_server)
	print("  Socket state: ", socket.get_ready_state())
	
	if connected_to_server:
		# Send test state: counter dirty
		print("[DecisionController-", npc_name, "] Sending test state (counter_dirty)...")
		_on_state_changed(1)  # Bit flag 1 = counter_dirty
	else:
		print("[DecisionController-", npc_name, "] Not connected, cannot test")

func _send_state_request(state_flags: int):
	"""Send state request with proper tracking"""
	last_state_sent = state_flags
	last_request_time = Time.get_ticks_msec() / 1000.0
	pending_request = true  # Mark as pending
	
	# Start timeout timer
	if has_node("RequestTimeoutTimer"):
		$RequestTimeoutTimer.start()
	
	# Create request
	var request = {
		"npc": npc_name,
		"state": state_flags,
		"timestamp": last_request_time
	}
	
	print("[DecisionController-", npc_name, "] Sending request: ", request)
	
	# Send to server
	_send_to_server(request)
	request_timestamp = last_request_time
	stats.decisions_requested += 1
	
	if state_detector and state_detector.has_method("decode_state"):
		var decoded = state_detector.decode_state(state_flags)
		print("[DecisionController-", npc_name, "] Sent state: ", decoded)

func _process_request_queue():
	"""Process queued requests with rate limiting"""
	if request_queue.is_empty() or pending_request:
		return
	
	var current_time = Time.get_ticks_msec() / 1000.0
	if current_time - last_request_time >= min_request_interval:
		# Process next request in queue
		var queued_request = request_queue.pop_front()
		var state_flags = queued_request.state
		
		# Remove any older duplicate states from queue
		request_queue = request_queue.filter(func(req): return req.state != state_flags)
		
		print("[DecisionController-", npc_name, "] Processing queued request: ", state_flags)
		_send_state_request(state_flags)

func _on_request_timeout():
	"""Handle request timeout"""
	print("[DecisionController-", npc_name, "] Request timeout! Clearing pending flag")
	pending_request = false
	stats.connection_failures += 1
	
	# Try to process next queued request if any
	if not request_queue.is_empty():
		_process_request_queue()
