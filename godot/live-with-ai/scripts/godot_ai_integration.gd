# Main AI Integration Controller for Godot
# Manages NPCs, dialogue, and synchronization with Python AI server
extends Node

# Dependencies
var server_client = preload("res://scripts/bar_server_client_json.gd").new()

# Signals
signal npc_spawned(npc_data)
signal dialogue_started(speaker, text)
signal action_executed(agent, action, result)

# NPC Management
var npcs = {}  # name -> NPC node
var npc_scene = preload("res://scenes/NPC.tscn")  # You'll need to create this scene

# Dialogue System
var dialogue_ui = null
var current_speaker = ""
var dialogue_queue = []

# State
var connected = false
var simulation_state = {}

func _ready():
	# Add server client to scene
	add_child(server_client)
	
	# Connect signals
	server_client.connect("connection_status_changed", self, "_on_connection_changed")
	server_client.connect("state_updated", self, "_on_state_updated")
	server_client.connect("dialogue_received", self, "_on_dialogue_received")
	server_client.connect("npc_action_completed", self, "_on_action_completed")
	
	# Initialize connection
	_initialize_connection()

func _initialize_connection():
	"""Initialize connection to Python AI server"""
	print("[AI] Connecting to Python server...")
	server_client.connect_to_server()
	
	# Wait for connection
	yield(get_tree().create_timer(1.0), "timeout")
	
	if server_client.is_connected:
		print("[AI] Connected! Sending INIT...")
		server_client.send_message({
			"type": "INIT",
			"data": {"client": "godot", "version": "1.0"}
		})
		
		# Request initial state
		yield(get_tree().create_timer(0.5), "timeout")
		server_client.send_message({"type": "GET_STATE"})
	else:
		print("[AI] Failed to connect. Retrying...")
		yield(get_tree().create_timer(5.0), "timeout")
		_initialize_connection()

func _on_connection_changed(is_connected):
	"""Handle connection status changes"""
	connected = is_connected
	if is_connected:
		print("[AI] Connection established")
	else:
		print("[AI] Connection lost")

func _on_state_updated(state_data):
	"""Handle state updates from server"""
	simulation_state = state_data
	
	# Update or spawn NPCs
	if "agents" in state_data:
		for agent_data in state_data["agents"]:
			_update_or_spawn_npc(agent_data)

func _update_or_spawn_npc(agent_data):
	"""Update existing NPC or spawn new one"""
	var npc_name = agent_data["name"]
	
	if npc_name in npcs:
		# Update existing NPC
		var npc = npcs[npc_name]
		npc.update_from_data(agent_data)
	else:
		# Spawn new NPC
		var npc = npc_scene.instance()
		npc.initialize(agent_data)
		get_parent().add_child(npc)
		npcs[npc_name] = npc
		emit_signal("npc_spawned", agent_data)

func _on_dialogue_received(speaker, text):
	"""Handle dialogue from AI"""
	print("[DIALOGUE] %s: %s" % [speaker, text])
	
	# Add to queue
	dialogue_queue.append({"speaker": speaker, "text": text})
	
	# Display if UI available
	if dialogue_ui:
		dialogue_ui.show_dialogue(speaker, text)
	
	emit_signal("dialogue_started", speaker, text)

func _on_action_completed(agent_name, action, success):
	"""Handle action completion"""
	print("[ACTION] %s: %s (%s)" % [agent_name, action, "success" if success else "failed"])
	emit_signal("action_executed", agent_name, action, success)

# ========== PUBLIC API ==========

func request_dialogue(speaker_name: String, target_name: String, message: String):
	"""Request AI-generated dialogue"""
	if not connected:
		print("[AI] Not connected to server")
		return
	
	server_client.send_message({
		"type": "TALK",
		"agent": speaker_name,
		"target": target_name,
		"message": message
	})

func request_action(agent_name: String, action: String, params = {}):
	"""Request NPC action"""
	if not connected:
		print("[AI] Not connected to server")
		return
	
	server_client.send_message({
		"type": "ACTION",
		"agent": agent_name,
		"action": action,
		"params": params
	})

func request_move(agent_name: String, destination):
	"""Request NPC movement"""
	if not connected:
		print("[AI] Not connected to server")
		return
	
	server_client.send_message({
		"type": "MOVE",
		"agent": agent_name,
		"destination": destination
	})

func query_agent(agent_name: String):
	"""Query agent state"""
	if not connected:
		print("[AI] Not connected to server")
		return
	
	server_client.send_message({
		"type": "QUERY",
		"agent": agent_name
	})

func get_full_state():
	"""Get full simulation state"""
	if not connected:
		print("[AI] Not connected to server")
		return
	
	server_client.send_message({"type": "GET_STATE"})

func save_simulation():
	"""Save simulation state"""
	if not connected:
		print("[AI] Not connected to server")
		return
	
	server_client.send_message({"type": "SAVE"})

func load_simulation():
	"""Load simulation state"""
	if not connected:
		print("[AI] Not connected to server")
		return
	
	server_client.send_message({"type": "LOAD"})
