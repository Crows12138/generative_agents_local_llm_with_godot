# bar_server_client_fixed.gd - Fixed version, properly handles clicks
extends Node2D

# Speech bubble settings (for programmatic creation fallback)
var bubble_font_size: int = 8
var bubble_container_width: int = 160
var bubble_container_height: int = 60

# Signals for streaming
signal token_received(npc_name: String, token: String)
signal response_completed(npc_name: String, full_response: String)

var npcs = {}
var is_dialogue_processing = false
var input_dialog = null
var current_npc_for_input = ""

# Memory viewer UI elements (from scene)
var memory_panel = null
var memory_text = null
var speech_bubbles = {}  # Store speech bubbles for each NPC

# WebSocket streaming support
var websocket = WebSocketPeer.new()
var is_websocket_connected = false
var current_streaming_npc = ""
var current_streaming_response = ""
var use_websocket = true  # Enable WebSocket streaming by default
var last_update_time = 0.0
var update_interval = 0.05  # Update every 50ms for smooth streaming

func _ready():
	print("Bar scene - Dialogue System (WebSocket mode)!")
	
	# Get NPC nodes
	setup_npcs()
	
	# Check server
	check_server()
	
	# Connect to WebSocket for streaming
	if use_websocket:
		connect_websocket()
	
	# Setup UI from scene nodes
	setup_ui_from_scene()

func setup_ui_from_scene():
	"""Connect UI elements from the scene"""
	var ui_layer = get_node_or_null("UILayer")
	if not ui_layer:
		print("Error: UILayer not found in scene!")
		return
	
	# Connect memory panel
	memory_panel = ui_layer.get_node_or_null("MemoryPanel")
	if memory_panel:
		var close_btn = memory_panel.get_node_or_null("CloseButton")
		if close_btn:
			close_btn.pressed.connect(func(): memory_panel.visible = false)
		
		var scroll = memory_panel.get_node_or_null("ScrollContainer")
		if scroll:
			memory_text = scroll.get_node_or_null("MemoryText")
	
	# Connect Bob's buttons
	var bob_view = ui_layer.get_node_or_null("ViewBobMemoryButton")
	if bob_view:
		bob_view.pressed.connect(_on_memory_button_pressed.bind("Bob"))
	
	var bob_clear = ui_layer.get_node_or_null("ClearBobMemoryButton")
	if bob_clear:
		bob_clear.pressed.connect(_on_clear_memory_button_pressed.bind("Bob"))
	
	# Connect Alice's buttons
	var alice_view = ui_layer.get_node_or_null("ViewAliceMemoryButton")
	if alice_view:
		alice_view.pressed.connect(_on_memory_button_pressed.bind("Alice"))
	
	var alice_clear = ui_layer.get_node_or_null("ClearAliceMemoryButton")
	if alice_clear:
		alice_clear.pressed.connect(_on_clear_memory_button_pressed.bind("Alice"))
	
	# Connect Sam's buttons
	var sam_view = ui_layer.get_node_or_null("ViewSamMemoryButton")
	if sam_view:
		sam_view.pressed.connect(_on_memory_button_pressed.bind("Sam"))
	
	var sam_clear = ui_layer.get_node_or_null("ClearSamMemoryButton")
	if sam_clear:
		sam_clear.pressed.connect(_on_clear_memory_button_pressed.bind("Sam"))
	
	print("UI setup from scene complete")

func setup_npcs():
	"""Setup NPC nodes and click detection"""
	# Find CharacterBody2D NPCs (with case correction)
	var bob = $bob if has_node("bob") else null
	var alice = $Alice if has_node("Alice") else null
	var sam = $sam if has_node("sam") else null
	
	print("Found nodes - Bob: ", bob != null, ", Alice: ", alice != null, ", Sam: ", sam != null)
	
	# Setup click detection for CharacterBody2D
	if bob and bob is CharacterBody2D:
		npcs["Bob"] = bob
		setup_character_click(bob, "Bob")
		# Also check for Area2D child for expanded click area
		for child in bob.get_children():
			if child is Area2D:
				setup_area_click(child, "Bob")
				break
		
	if alice and alice is CharacterBody2D:
		npcs["Alice"] = alice
		setup_character_click(alice, "Alice")
		# Also check for Area2D child for expanded click area
		for child in alice.get_children():
			if child is Area2D:
				setup_area_click(child, "Alice")
				break
		
	if sam and sam is CharacterBody2D:
		npcs["Sam"] = sam
		setup_character_click(sam, "Sam")
		# Also check for Area2D child for expanded click area
		for child in sam.get_children():
			if child is Area2D:
				setup_area_click(child, "Sam")
				break
	
	print("NPCs setup complete: ", npcs.keys())

func setup_character_click(character: CharacterBody2D, npc_name: String):
	"""Setup CharacterBody2D for click detection"""
	# Connect input_event signal for CharacterBody2D
	if not character.input_event.is_connected(_on_character_input_event):
		character.input_event.connect(_on_character_input_event.bind(npc_name))
	
	# Make sure it can receive input
	character.input_pickable = true
	
	print(npc_name, " CharacterBody2D is now clickable")

func setup_area_click(area: Area2D, npc_name: String):
	"""Setup Area2D for expanded click detection"""
	# Connect input_event signal for Area2D
	if not area.input_event.is_connected(_on_area_input_event):
		area.input_event.connect(_on_area_input_event.bind(npc_name))
	
	# Make sure it can receive input
	area.input_pickable = true
	
	print(npc_name, " Area2D expanded click area is now active")

func _on_character_input_event(_viewport: Node, event: InputEvent, _shape_idx: int, npc_name: String):
	"""Handle CharacterBody2D click events"""
	if event is InputEventMouseButton:
		if event.button_index == MOUSE_BUTTON_LEFT and event.pressed:
			print("\nLeft-clicked on ", npc_name, "!")
			interact_with_npc(npc_name)
		elif event.button_index == MOUSE_BUTTON_RIGHT and event.pressed:
			print("\nRight-clicked on ", npc_name, "! Opening input dialog...")
			show_custom_input_dialog(npc_name)

func _on_area_input_event(_viewport: Node, event: InputEvent, _shape_idx: int, npc_name: String):
	"""Handle Area2D click events (expanded click area)"""
	if event is InputEventMouseButton:
		if event.button_index == MOUSE_BUTTON_LEFT and event.pressed:
			print("\nLeft-clicked on ", npc_name, " (Area2D)!")
			interact_with_npc(npc_name)
		elif event.button_index == MOUSE_BUTTON_RIGHT and event.pressed:
			print("\nRight-clicked on ", npc_name, " (Area2D)! Opening input dialog...")
			show_custom_input_dialog(npc_name)

func check_server():
	"""Check server status"""
	# For WebSocket, we'll just try to connect
	print("WebSocket server should be running on port 9999")
	print("Start with: python afterbuild/server/dialogue_server.py")

func connect_websocket():
	"""Connect to WebSocket server for streaming"""
	var url = "ws://127.0.0.1:9999"
	var err = websocket.connect_to_url(url)
	if err == OK:
		set_process(true)  # Enable _process for WebSocket updates
		print("✅ Connecting to WebSocket server...")
	else:
		print("❌ Failed to initiate WebSocket connection")
		use_websocket = false

func _process(_delta):
	"""Process WebSocket connection and data"""
	if not use_websocket:
		return
	
	# Poll WebSocket
	websocket.poll()
	
	# Check connection state
	var state = websocket.get_ready_state()
	
	if state == WebSocketPeer.STATE_OPEN:
		if not is_websocket_connected:
			is_websocket_connected = true
			print("✅ WebSocket connected!")
		
		# Process incoming packets
		while websocket.get_available_packet_count() > 0:
			var packet = websocket.get_packet()
			var text = packet.get_string_from_utf8()
			_process_websocket_message(text)
			
	elif state == WebSocketPeer.STATE_CLOSING:
		print("WebSocket closing...")
	elif state == WebSocketPeer.STATE_CLOSED:
		if is_websocket_connected:
			is_websocket_connected = false
			print("❌ WebSocket disconnected")
			var code = websocket.get_close_code()
			var reason = websocket.get_close_reason()
			print("Close code: ", code, " Reason: ", reason)

func _process_websocket_message(data: String):
	"""Process WebSocket message"""
	var json = JSON.new()
	var parse_result = json.parse(data)
	
	if parse_result != OK:
		print("Failed to parse WebSocket message: ", data)
		return
	
	var message = json.data
	var msg_type = message.get("type", "")
	var content = message.get("content", "")
	var npc = message.get("npc", current_streaming_npc)
	
	if msg_type == "token":
		# Append token to response
		current_streaming_response += content
		emit_signal("token_received", npc, content)
		# Direct update without deferral for better streaming
		update_streaming_bubble(npc, current_streaming_response)
		
	elif msg_type == "complete":
		# Response complete
		emit_signal("response_completed", npc, content)
		
		# Just update the existing streaming bubble with final text, don't create new one
		update_streaming_bubble(npc, content)
		
		# Add auto-fade timer for the streaming bubble
		if npc in speech_bubbles and is_instance_valid(speech_bubbles[npc]):
			var bubble = speech_bubbles[npc]
			var tween = create_tween()
			tween.tween_interval(8.0)  # Display for 8 seconds
			tween.tween_property(bubble, "modulate:a", 0.0, 0.5).set_ease(Tween.EASE_IN_OUT)
			tween.tween_callback(func():
				if is_instance_valid(bubble):
					bubble.queue_free()
				if npc in speech_bubbles:
					speech_bubbles.erase(npc)
			)
		
		current_streaming_response = ""
		current_streaming_npc = ""
		is_dialogue_processing = false
		
	elif msg_type == "error":
		print("WebSocket error: ", content)
		is_dialogue_processing = false

func update_streaming_bubble(npc_name: String, partial_text: String):
	"""Update speech bubble with streaming text"""
	
	# Keep NPC label  
	var npc_node = npcs.get(npc_name)
	if npc_node and npc_node.has_node(npc_name):
		var label = npc_node.get_node(npc_name)
		if label is Label:
			label.text = npc_name
	
	# Get UI layer from scene
	var ui_layer = get_node_or_null("UILayer")
	if not ui_layer:
		print("Error: UILayer not found!")
		return
	
	# Look for existing bubble or create new one
	var bubble_name = "SpeechBubble_" + npc_name
	var bubble = ui_layer.get_node_or_null(bubble_name)
	
	if not bubble:
		# Create bubble using the SpeechBubbleExample as template
		var example = ui_layer.get_node_or_null("SpeechBubbleExample")
		if example:
			bubble = example.duplicate()
			bubble.name = bubble_name
			ui_layer.add_child(bubble)
		else:
			# Fallback: create bubble programmatically
			var npc = npcs.get(npc_name)
			if npc:
				create_streaming_bubble_in_layer(ui_layer, npc_name, partial_text, npc.global_position)
			return
	
	# Update bubble text and position
	if bubble and is_instance_valid(bubble):
		var text_label = find_rich_text_label(bubble)
		if text_label:
			# Only update if text is different to avoid flicker
			if text_label.text != partial_text:
				text_label.text = partial_text
		
		# Update position relative to NPC
		var npc = npcs.get(npc_name)
		if npc:
			var npc_pos = npc.global_position
			bubble.position = Vector2(npc_pos.x - 80, npc_pos.y - 100)
		
		bubble.visible = true

func _update_label_text(label: RichTextLabel, text: String):
	"""Deferred update of label text to ensure UI refresh"""
	if label and is_instance_valid(label):
		label.text = text
		label.queue_redraw()

func find_rich_text_label(node: Node) -> RichTextLabel:
	"""Recursively find RichTextLabel in node tree"""
	if node is RichTextLabel:
		return node
	
	for child in node.get_children():
		var result = find_rich_text_label(child)
		if result:
			return result
	
	return null

func create_streaming_bubble_in_layer(ui_layer: CanvasLayer, npc_name: String, text: String, position: Vector2):
	"""Create a speech bubble in the given UI layer"""
	if not ui_layer:
		print("Error: No UI layer provided")
		return
	
	# Remove existing bubble if any
	var bubble_name = "SpeechBubble_" + npc_name
	var existing = ui_layer.get_node_or_null(bubble_name)
	if existing:
		existing.queue_free()
	
	# Create bubble container
	var bubble = Panel.new()
	bubble.name = bubble_name
	bubble.size = Vector2(bubble_container_width, bubble_container_height)
	bubble.modulate = Color(1, 1, 1, 0.9)
	
	# Position bubble above NPC
	var npc = npcs.get(npc_name)
	if npc:
		var npc_pos = npc.global_position
		bubble.position = Vector2(npc_pos.x - 80, npc_pos.y - 100)
	else:
		bubble.position = Vector2(200, 100)
	
	
	# Create ScrollContainer
	var scroll_container = ScrollContainer.new()
	scroll_container.position = Vector2(5, 5)
	scroll_container.size = Vector2(150, 50)
	
	# Create text label
	var bubble_text = RichTextLabel.new()
	bubble_text.custom_minimum_size = Vector2(140, 0)
	bubble_text.bbcode_enabled = false
	bubble_text.fit_content = true
	bubble_text.text = text
	bubble_text.add_theme_font_size_override("normal_font_size", bubble_font_size)
	bubble_text.add_theme_color_override("default_color", Color(0, 0, 0, 1))
	
	scroll_container.add_child(bubble_text)
	bubble.add_child(scroll_container)
	
	ui_layer.add_child(bubble)
	
	# Store reference
	speech_bubbles[npc_name] = bubble
	
	# Simple fade in
	bubble.modulate.a = 0
	var tween = create_tween()
	tween.tween_property(bubble, "modulate:a", 1.0, 0.3)

func show_custom_input_dialog(npc_name: String):
	"""Show custom message input dialog"""
	current_npc_for_input = npc_name
	
	# Create input dialog
	if input_dialog:
		input_dialog.queue_free()
	
	input_dialog = AcceptDialog.new()
	input_dialog.title = "Send Custom Message to " + npc_name
	input_dialog.dialog_text = ""  # Remove duplicate text
	input_dialog.size = Vector2(400, 150)
	
	# Create input field
	var vbox = VBoxContainer.new()
	var line_edit = LineEdit.new()
	line_edit.placeholder_text = "Type your message here..."
	line_edit.name = "CustomInput"
	# Connect Enter key to confirm
	line_edit.text_submitted.connect(_on_line_edit_text_submitted)
	vbox.add_child(line_edit)
	
	input_dialog.add_child(vbox)
	
	# Connect confirmation signals
	input_dialog.confirmed.connect(_on_custom_input_confirmed)
	input_dialog.canceled.connect(_on_custom_input_canceled)
	
	# Add to scene and show
	get_tree().root.add_child(input_dialog)
	input_dialog.popup_centered()
	
	# Focus on input field
	line_edit.grab_focus()

func _on_line_edit_text_submitted(text: String):
	"""Handle Enter key press in input field"""
	if text.strip_edges() != "":
		_on_custom_input_confirmed()

func _on_custom_input_confirmed():
	"""Handle custom input confirmation"""
	if input_dialog and current_npc_for_input:
		var line_edit = input_dialog.find_child("CustomInput", true, false)
		if line_edit and line_edit is LineEdit:
			var custom_message = line_edit.text.strip_edges()
			if custom_message != "":
				print("Custom message for ", current_npc_for_input, ": ", custom_message)
				interact_with_npc(current_npc_for_input, custom_message)
			else:
				print("Empty message, ignoring...")
		
		input_dialog.queue_free()
		input_dialog = null
		current_npc_for_input = ""

func _on_custom_input_canceled():
	"""Handle custom input cancellation"""
	if input_dialog:
		input_dialog.queue_free()
		input_dialog = null
	current_npc_for_input = ""
	print("Custom input canceled")

func interact_with_npc(npc_name: String, custom_message: String = ""):
	"""Interact with NPC"""
	if is_dialogue_processing:
		print("Still processing previous request...")
		return
	
	is_dialogue_processing = true
	
	# Clear existing bubble for this NPC when starting new conversation
	if npc_name in speech_bubbles and is_instance_valid(speech_bubbles[npc_name]):
		var old_bubble = speech_bubbles[npc_name]
		speech_bubbles.erase(npc_name)
		
		# Fade out animation before removing
		var tween = create_tween()
		tween.tween_property(old_bubble, "modulate:a", 0.0, 0.3).set_ease(Tween.EASE_IN_OUT)
		tween.tween_callback(old_bubble.queue_free)
	
	# Add click feedback - flash the NPC
	var npc = npcs.get(npc_name)
	if npc:
		var original_modulate = npc.modulate
		var tween = create_tween()
		tween.tween_property(npc, "modulate", Color(1.2, 1.2, 1.2), 0.1)
		tween.tween_property(npc, "modulate", original_modulate, 0.1)
	
	# No thinking bubble needed for WebSocket (it's fast)
	
	# CLEAN PROTOCOL: Use simple format NPC_NAME|MESSAGE
	var message = ""
	if custom_message != "":
		# Custom message with clean protocol
		message = npc_name + "|" + custom_message
		print("Using clean protocol: ", message)
	else:
		# Preset message with clean protocol
		message = npc_name + "|Hello!"
		print("Using clean protocol (preset): ", message)
	
	# Choose WebSocket or traditional mode
	if use_websocket and is_websocket_connected:
		# Send via WebSocket for streaming
		current_streaming_npc = npc_name
		current_streaming_response = ""
		
		var data = {
			"npc": npc_name,
			"message": message
		}
		
		var json_str = JSON.stringify(data)
		websocket.send_text(json_str)
		print("Sent WebSocket request for ", npc_name)
	else:
		# WebSocket not connected
		print("WebSocket not connected. Please start afterbuild/server/dialogue_server.py")
		is_dialogue_processing = false
		# Don't show any bubble when not connected






func _on_memory_button_pressed(npc_name: String = "Bob"):
	"""Handle memory button press for specific NPC"""
	print("Memory button pressed for ", npc_name)
	if memory_panel:
		# Toggle visibility
		memory_panel.visible = !memory_panel.visible
		if memory_panel.visible:
			load_npc_memories(npc_name)
	else:
		print("Error: Memory panel not found in scene!")

func _on_clear_memory_button_pressed(npc_name: String = "Bob"):
	"""Handle clear memory button press for specific NPC"""
	print("Clear memory button pressed for ", npc_name)
	
	# Show confirmation dialog
	var confirm_dialog = ConfirmationDialog.new()
	confirm_dialog.dialog_text = "Are you sure you want to clear " + npc_name + "'s memories?\nThis cannot be undone!"
	confirm_dialog.title = "Confirm Clear Memories"
	confirm_dialog.size = Vector2(400, 150)
	
	# Connect confirmation signal with NPC name
	confirm_dialog.confirmed.connect(_on_clear_memory_confirmed.bind(npc_name))
	
	# Add to scene and show
	add_child(confirm_dialog)
	confirm_dialog.popup_centered()

func _on_clear_memory_confirmed(npc_name: String = "Bob"):
	"""Actually clear the memories after confirmation"""
	print("Clearing ", npc_name, "'s memories...")
	
	# Clear memory file for the specific NPC
	var project_root = ProjectSettings.globalize_path("res://").replace("afterbuild/godot_project/", "")
	var mem_file1 = project_root + "afterbuild/npc_memories/" + npc_name + ".json"
	
	# Clear memory file
	var file1 = FileAccess.open(mem_file1, FileAccess.WRITE)
	if file1:
		file1.store_string("[]")  # Empty array
		file1.close()
		print("Cleared memory file for ", npc_name)
	
	# Silent success - no dialog needed
	print(npc_name + "'s memories have been cleared successfully")
	
	# Update memory viewer if open
	if memory_panel and memory_panel.visible:
		load_npc_memories(npc_name)

func create_memory_panel():
	"""This function is deprecated - memory panel should be in scene"""
	print("Warning: create_memory_panel() called but UI should be in scene")
	# Memory panel is now in the scene file, no need to create dynamically

func load_npc_memories(npc_name: String = "Bob"):
	"""Load NPC's memories using Python tool"""
	memory_text.clear()
	memory_text.append_text("=== " + npc_name + "'s Conversation History ===\n\n")
	
	# Read memory file directly
	var output = []
	var project_root = ProjectSettings.globalize_path("res://").replace("afterbuild/godot_project/", "")
	var memory_file = project_root + "afterbuild/npc_memories/" + npc_name + ".json"
	print("Loading memories for ", npc_name, " from: ", memory_file)
	
	# Read memory file directly
	var file = FileAccess.open(memory_file, FileAccess.READ)
	if file:
		var content = file.get_as_text()
		file.close()
		output.append(content)
	var exit_code = 0 if output.size() > 0 else 1
	
	print("Memory load exit code: ", exit_code)
	print("Output size: ", output.size())
	if output.size() > 0:
		print("First 200 chars of output: ", output[0].substr(0, 200))
	
	if exit_code == 0 and output.size() > 0:
		# Parse and format memories
		var memories_text = output[0]
		var lines = memories_text.split("\n")
		
		print("Processing ", lines.size(), " lines of memory data")
		
		# Simple plain text display
		for line in lines:
			memory_text.append_text(line + "\n")
		
		print("Memory text content length: ", memory_text.get_parsed_text().length())
	else:
		memory_text.append_text("Failed to load memories\n")
		if output.size() > 0:
			memory_text.append_text(output[0])
