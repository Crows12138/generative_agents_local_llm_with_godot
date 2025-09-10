extends Node

var socket = WebSocketPeer.new()
var url = "ws://127.0.0.1:9999"
var connected = false
var decision_callbacks = {}
var dialogue_callbacks = {}  # Callbacks for NPC dialogue responses

var pending_decisions = {}

var action_duration = 5.0
var wait_after_action = 10.0
var timer = 0.0
var current_state = "waiting"

func request_decision(npc_name: String, context: String):
	var request = {
		"type": "decision",
		"npc": npc_name,
		"context": context
	}
	send_message(request)

func _ready():
	print("initial decision system")
	connect_to_websocket_server()

func connect_to_websocket_server():
	print("[Server] Attempting to connect to: " + url)
	var err = socket.connect_to_url(url)
	if err !=OK:
		set_process(false)
		print("[Server] Unable to connect, error: " + str(err))
	else:
		print("[Server] Connection initiated...")
	
func _process(delta):
	socket.poll()
	var state = socket.get_ready_state()
	
	match state:
		WebSocketPeer.STATE_OPEN:
			if not connected:
				connected = true
				print("[Server] WebSocket connected successfully!")
			while socket.get_available_packet_count() > 0:
				receive_message()
				
			update_decision_timer(delta)
				
		WebSocketPeer.STATE_CLOSED:
			if connected:
				connected = false
				print("Disconnected")
				
				await get_tree().create_timer(3.0).timeout
				connect_to_websocket_server()

func update_decision_timer(delta):
	timer += delta
	
	match current_state:
		"waiting":
			if timer >= 1.0:
				current_state = "acting"
				timer = 0.0
				
		"acting":
			if timer >= action_duration:
				current_state = "cooldown"
				timer = 0.0
				
		"cooldown":
			if timer >= wait_after_action:
				current_state = "waiting"
				timer = 0.0

func receive_message():
	var packet = socket.get_packet()
	var message = packet.get_string_from_utf8()
	
	var json = JSON.new()
	var result = json.parse(message)
	if result == OK:
		if json.data.type == "decision_result":
			handle_decision_response(json.data)
		elif json.data.type == "complete":
			# Handle dialogue completion from NPC
			handle_dialogue_response(json.data)
		elif json.data.type == "token":
			# Handle streaming tokens for NPC dialogue
			var npc_name = json.data.get("npc", "")
			var token = json.data.get("content", "")
			var from_speaker = json.data.get("from", "user")
			
			# Find the NPC node to handle streaming
			var npcs = get_tree().get_nodes_in_group("npcs")
			for npc in npcs:
				var node_name_lower = npc.name.to_lower()
				var npc_name_lower = npc_name.to_lower()
				if node_name_lower == npc_name_lower:
					if npc.has_method("handle_dialogue_token"):
						npc.handle_dialogue_token(token, from_speaker)
					break
			
func request_decision_with_callback(npc_name: String, context: String, callback: Callable):
	decision_callbacks[npc_name] = callback
	request_decision(npc_name, context)

func handle_decision_response(data):
	var npc_name = data.npc
	var action = data.action
	var target = data.target
	if npc_name in decision_callbacks:
		decision_callbacks[npc_name].call(action, target)
		decision_callbacks.erase(npc_name)

func handle_dialogue_response(data):
	"""Handle dialogue completion response from server"""
	var npc_name = data.get("npc", "")
	var response = data.get("content", "")  # Changed from "response" to "content"
	var from_speaker = data.get("from", "user")  # Who initiated the dialogue
	
	# Only print user dialogues
	if from_speaker == "user":
		print(npc_name + " responding to user: " + response)
	# Don't print system-generated responses or NPC-to-NPC responses
	
	# Check if there's a callback for this NPC's dialogue
	if npc_name in dialogue_callbacks:
		dialogue_callbacks[npc_name].call(response)
		dialogue_callbacks.erase(npc_name)
	
	# Find the NPC node and show their response bubble
	var npcs = get_tree().get_nodes_in_group("npcs")
	for npc in npcs:
		# Check both exact name match and case-insensitive match
		var node_name_lower = npc.name.to_lower()
		var npc_name_lower = npc_name.to_lower()
		if node_name_lower == npc_name_lower:
			if npc.has_method("show_dialogue_bubble"):
				npc.show_dialogue_bubble(response, from_speaker)
				if from_speaker == "user":
					print(npc_name + " showing dialogue bubble with response to user: " + response)
			break


func send_message(data):
	if not connected:
		print("not connected")
		return
	var json_str = JSON.stringify(data)
	socket.send_text(json_str)

func send_npc_dialogue(from_npc: String, to_npc: String, message: String):
	"""Send dialogue request from one NPC to another"""
	if not connected:
		print("Server not connected for NPC dialogue")
		return
	
	# New format: include 'from' field to identify speaker
	var data = {
		"type": "dialogue",
		"npc": to_npc,
		"message": message,
		"from": from_npc  # Important: identifies who is speaking
	}
	
	if from_npc != "system":
		print("NPC Dialogue: " + from_npc + " -> " + to_npc + ": " + message)
	send_message(data)

func _exit_tree():
	if socket:
		socket.close()

	
	
