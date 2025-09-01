# bar_server_client_simple.gd - Simplified version designed for existing scenes
extends Node2D

var npcs = {}
var python_path = ".venv/Scripts/python.exe"
var is_processing = false

func _ready():
	print("Bar scene - Server-Client mode!")
	
	# Get existing NPC nodes (ColorRect type)
	npcs["Bob"] = $Bob if has_node("Bob") else null
	npcs["Alice"] = $Alice if has_node("Alice") else null
	npcs["Sam"] = $Sam if has_node("Sam") else null
	
	print("Found NPCs: ", npcs.keys())
	
	# Check server status
	check_server()

func check_server():
	"""Check if server is running"""
	var output = []
	var args = ["../../llm_client.py", "ping"]
	OS.execute(python_path, args, output, true, false)
	
	if output.size() > 0 and not output[0].contains("Server not running"):
		print("✅ LLM Server is ready!")
	else:
		print("❌ Server not running. Please run: start_llm_server.bat")

func _input(event):
	if event is InputEventMouseButton and event.pressed:
		if event.button_index == MOUSE_BUTTON_LEFT:
			handle_npc_click(event.position)

func handle_npc_click(click_pos):
	"""Handle NPC click"""
	if is_processing:
		return
	
	# Debug output
	print("Click position: ", click_pos)
	
	for npc_name in npcs:
		var npc = npcs[npc_name]
		if npc and npc is ColorRect:
			# ColorRect uses global coordinates
			var rect = Rect2(npc.global_position, npc.size)
			print(npc_name, " rect: ", rect)
			
			if rect.has_point(click_pos):
				print("\nClicked on ", npc_name, "!")
				interact_with_npc(npc_name)
				break

func interact_with_npc(npc_name: String):
	"""Interact with NPC"""
	if is_processing:
		return
	
	is_processing = true
	
	# Show thinking state
	show_thinking(npc_name)
	
	# Prepare message
	var messages = {
		"Bob": "Hello Bob the bartender!",
		"Alice": "Hi Alice, how are you?",
		"Sam": "Hey Sam, play us a song!"
	}
	
	var message = messages.get(npc_name, "Hello!")
	
	# Call LLM
	var thread = Thread.new()
	thread.start(_llm_thread.bind(npc_name, message))

func _llm_thread(npc_name: String, message: String):
	"""Call LLM in thread"""
	var output = []
	var args = ["../../llm_client.py", message]
	
	var start_time = Time.get_ticks_msec()
	OS.execute(python_path, args, output, true, false)
	var elapsed = (Time.get_ticks_msec() - start_time) / 1000.0
	
	var response = "..."
	if output.size() > 0:
		response = output[0].strip_edges()
		if response.length() > 50:
			response = response.substr(0, 47) + "..."
	
	call_deferred("_on_response", npc_name, response, elapsed)

func _on_response(npc_name: String, response: String, time_taken: float):
	"""Handle response"""
	is_processing = false
	show_response(npc_name, response)
	print("[%s] Response (%.2fs): %s" % [npc_name, time_taken, response])

func show_thinking(npc_name: String):
	"""Show thinking state"""
	var npc = npcs.get(npc_name)
	if npc and npc.has_node(npc_name):  # Label child node
		var label = npc.get_node(npc_name)
		if label is Label:
			label.text = npc_name + "\n💭..."

func show_response(npc_name: String, text: String):
	"""Show response"""
	var npc = npcs.get(npc_name)
	if npc and npc.has_node(npc_name):  # Label child node
		var label = npc.get_node(npc_name)
		if label is Label:
			label.text = npc_name + "\n💬 " + text
			
			# Restore after 5 seconds
			await get_tree().create_timer(5.0).timeout
			label.text = npc_name
