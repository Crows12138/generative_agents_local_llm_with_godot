# bar_server_client_fixed.gd - Fixed version, properly handles clicks
extends Node2D

# Signals for streaming
signal token_received(npc_name: String, token: String)
signal response_completed(npc_name: String, full_response: String)

var npcs = {}
var python_path = ""
var project_root = ""
var is_processing = false
var input_dialog = null
var current_npc_for_input = ""

# Memory viewer UI elements
var memory_button = null
var clear_memory_button = null
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
	print("Bar scene - Server-Client mode (Fixed)!")
	
	# Setup paths
	var exe_path = OS.get_executable_path()
	var exe_dir = exe_path.get_base_dir()
	
	if OS.get_name() == "Windows":
		# Running in Godot editor
		if exe_dir.contains("Godot"):
			project_root = ProjectSettings.globalize_path("res://").replace("finalbuild/godot_project/", "")
			python_path = project_root + ".venv/Scripts/python.exe"
		else:
			# Running after export
			project_root = exe_dir + "/../.."
			python_path = project_root + "/.venv/Scripts/python.exe"
	
	print("Python path: ", python_path)
	print("Project root: ", project_root)
	
	# INTEGRATE AI DECISION SYSTEM
	print("\n[AI SYSTEM] Integrating AI decision system...")
	var IntegrateAI = load("res://scripts/integrate_ai_system.gd")
	IntegrateAI.integrate_ai_to_scene(self)
	print("[AI SYSTEM] Integration complete! NPCs now have intelligent decision making.")
	print("[AI SYSTEM] Make sure decision_server.py is running on port 9998!")
	
	# Get NPC nodes
	setup_npcs()
	
	# Check server
	check_server()
	
	# Connect to WebSocket for streaming
	if use_websocket:
		connect_websocket()
	
	# Create memory viewer button
	create_memory_button()

func setup_npcs():
	"""Setup NPC nodes and click detection"""
	print("=== Setting up NPCs ===")
	
	# Find CharacterBody2D NPCs by their actual names
	var bob = $bob if has_node("bob") else null
	var alice = $Alice if has_node("Alice") else null  
	var sam = $sam if has_node("sam") else null
	
	print("Found nodes - bob: ", bob, " alice: ", alice, " sam: ", sam)
	
	# Setup click detection for CharacterBody2D nodes
	if bob and bob is CharacterBody2D:
		npcs["Bob"] = bob
		make_npc_clickable(bob, "Bob")
		print("Bob setup as clickable at position: ", bob.global_position)
		
	if alice and alice is CharacterBody2D:
		npcs["Alice"] = alice
		make_npc_clickable(alice, "Alice")
		print("Alice setup as clickable at position: ", alice.global_position)
		
	if sam and sam is CharacterBody2D:
		npcs["Sam"] = sam
		make_npc_clickable(sam, "Sam")
		print("Sam setup as clickable at position: ", sam.global_position)
	
	print("NPCs setup complete: ", npcs.keys())
	
	# Debug: List all clickable NPCs
	var clickable_npcs = get_tree().get_nodes_in_group("clickable_npcs")
	print("Clickable NPCs in group: ", clickable_npcs.size())
	for npc in clickable_npcs:
		print("  - ", npc.name, " at ", npc.global_position)

func make_npc_clickable(npc: CharacterBody2D, npc_name: String):
	"""Make CharacterBody2D clickable"""
	# Add to clickable NPCs list
	npc.add_to_group("clickable_npcs")
	npc.set_meta("npc_name", npc_name)
	print(npc_name, " CharacterBody2D is now clickable")

func _input(event: InputEvent):
	"""Handle global input for NPC clicking"""
	if event is InputEventMouseButton:
		if event.button_index == MOUSE_BUTTON_LEFT and event.pressed:
			# Convert screen coordinates to world coordinates
			var world_pos = get_global_mouse_position()
			_check_npc_click(world_pos, false)
		elif event.button_index == MOUSE_BUTTON_RIGHT and event.pressed:
			# Convert screen coordinates to world coordinates
			var world_pos = get_global_mouse_position()
			_check_npc_click(world_pos, true)

func _check_npc_click(click_pos: Vector2, is_right_click: bool):
	"""Check if click position hits any NPC"""
	# Configurable click threshold - adjust this value to change click sensitivity
	# Smaller value = more precise clicking required
	# Larger value = easier to click but may activate wrong NPC
	var click_threshold = 20.0  # Small value for precise clicking on character body only
	var found_npc = false
	var closest_npc = ""
	var closest_distance = click_threshold + 1.0
	
	for npc_name in npcs.keys():
		var npc_node = npcs[npc_name]
		var distance = npc_node.global_position.distance_to(click_pos)
		
		# Only activate if within reasonable distance
		if distance <= click_threshold:
			found_npc = true
			if is_right_click:
				_on_npc_right_click(npc_name)
			else:
				_on_npc_left_click(npc_name)
			break
	
	# If no NPC within threshold, do nothing (don't activate random NPCs)

func _on_npc_left_click(npc_name: String):
	"""Handle left click on NPC"""
	interact_with_npc(npc_name)

func _on_npc_right_click(npc_name: String):
	"""Handle right click on NPC"""
	show_custom_input_dialog(npc_name)

func _calculate_bubble_position(npc_node: Node) -> Vector2:
	"""Calculate bubble position above NPC"""
	if not npc_node:
		# Fallback to center of screen
		var viewport_size = get_viewport().size
		return Vector2(viewport_size.x * 0.5 - 150, viewport_size.y * 0.3)
	
	# Use Godot's built-in coordinate conversion for CanvasLayer
	var npc_screen_pos: Vector2
	
	# Try to get the main camera
	var camera = get_viewport().get_camera_2d()
	
	if camera:
		# Use Godot's built-in method to convert world to screen coordinates
		# This handles camera position, zoom, and viewport transforms automatically
		npc_screen_pos = camera.get_screen_center_position() + (npc_node.global_position - camera.global_position) * camera.zoom
	else:
		# No camera - use world coordinates directly
		npc_screen_pos = npc_node.global_position
	
	print("NPC ", npc_node.name, " world pos: ", npc_node.global_position, " screen pos: ", npc_screen_pos)
	print("Camera pos: ", camera.global_position if camera else "No camera", " zoom: ", camera.zoom if camera else "No camera")
	
	# Position bubble above and centered on NPC with per-character adjustments
	var bubble_x = npc_screen_pos.x - 150  # Center bubble (300px width / 2)
	var bubble_y = npc_screen_pos.y - 150  # Above NPC (120px height + some margin)
	
	# Apply per-character position corrections if needed
	match npc_node.name.to_lower():
		"alice":
			bubble_x -= 50  # Move Alice's bubble left
			bubble_y -= 20  # Move Alice's bubble up more
		"sam":
			bubble_x -= 30  # Move Sam's bubble left slightly
			bubble_y -= 10  # Move Sam's bubble up slightly
		"bob":
			pass  # Bob's position is already correct
	
	# Keep bubble within screen bounds
	var viewport_size = get_viewport().size
	bubble_x = clamp(bubble_x, 10, viewport_size.x - 310)  # 10px margin, 300px bubble width
	bubble_y = clamp(bubble_y, 10, viewport_size.y - 130)  # 10px margin, 120px bubble height
	
	return Vector2(bubble_x, bubble_y)

func _calculate_thinking_bubble_position(npc_node: Node) -> Vector2:
	"""Calculate thinking bubble position above NPC (smaller bubble)"""
	if not npc_node:
		# Fallback to center of screen
		var viewport_size = get_viewport().size
		return Vector2(viewport_size.x * 0.5 - 50, viewport_size.y * 0.35)
	
	# Get NPC position in screen coordinates
	var npc_screen_pos = npc_node.global_position
	
	# Position smaller bubble above and centered on NPC
	var bubble_x = npc_screen_pos.x - 50   # Center small bubble (100px width / 2)
	var bubble_y = npc_screen_pos.y - 80   # Above NPC (50px height + margin)
	
	# Keep bubble within screen bounds
	var viewport_size = get_viewport().size
	bubble_x = clamp(bubble_x, 10, viewport_size.x - 110)  # 10px margin, 100px bubble width
	bubble_y = clamp(bubble_y, 10, viewport_size.y - 60)   # 10px margin, 50px bubble height
	
	return Vector2(bubble_x, bubble_y)

func make_clickable(color_rect: ColorRect, npc_name: String):
	"""Make ColorRect clickable"""
	# Connect gui_input signal
	if not color_rect.gui_input.is_connected(_on_npc_gui_input):
		color_rect.gui_input.connect(_on_npc_gui_input.bind(npc_name))
	
	# Set mouse filter
	color_rect.mouse_filter = Control.MOUSE_FILTER_PASS
	
	print(npc_name, " is now clickable")

func _on_npc_gui_input(event: InputEvent, npc_name: String):
	"""Handle NPC click events"""
	if event is InputEventMouseButton:
		if event.button_index == MOUSE_BUTTON_LEFT and event.pressed:
			print("\nLeft-clicked on ", npc_name, "!")
			interact_with_npc(npc_name)
		elif event.button_index == MOUSE_BUTTON_RIGHT and event.pressed:
			print("\nRight-clicked on ", npc_name, "! Opening input dialog...")
			show_custom_input_dialog(npc_name)

func check_server():
	"""Check server status"""
	# For WebSocket, we'll just try to connect
	print("WebSocket server should be running on port 9999")
	print("Start with: python finalbuild/server/gpt4all_server.py")

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
		is_processing = false
		
	elif msg_type == "error":
		print("WebSocket error: ", content)
		is_processing = false

func update_streaming_bubble(npc_name: String, partial_text: String):
	"""Update speech bubble with streaming text"""
	
	# Keep NPC label  
	var npc_node = npcs.get(npc_name)
	if npc_node and npc_node.has_node(npc_name):
		var label = npc_node.get_node(npc_name)
		if label is Label:
			label.text = npc_name
	
	# Always look up bubble by name directly from UILayer
	var bubble_name = "SpeechBubble_" + npc_name
	var ui_layer = get_node_or_null("UILayer")
	
	if not ui_layer:
		if npc_node:
			create_streaming_bubble(npc_node, partial_text, npc_name)
		return
	
	# Find bubble directly by name
	var bubble = ui_layer.get_node_or_null(bubble_name)
	if bubble and is_instance_valid(bubble):
		var text_label = find_rich_text_label(bubble)
		if text_label:
			# Direct update without any deferral
			# Only update if text is different to avoid flicker
			if text_label.text != partial_text:
				text_label.text = partial_text
			return
	else:
		if npc_node:
			create_streaming_bubble(npc_node, partial_text, npc_name)

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

func create_streaming_bubble(npc_node: Node, text: String, npc_name: String):
	"""Create a streaming bubble for ongoing conversation"""
	# Remove existing bubble if any
	if npc_name in speech_bubbles:
		speech_bubbles[npc_name].queue_free()
		speech_bubbles.erase(npc_name)
	
	# Get or create UI layer for top-level display
	var ui_layer = get_node_or_null("UILayer")
	if not ui_layer:
		ui_layer = CanvasLayer.new()
		ui_layer.name = "UILayer"
		ui_layer.layer = 100  # High layer to ensure it's on top
		add_child(ui_layer)
	
	# Proper coordinate conversion: world to screen for CanvasLayer
	var camera = get_viewport().get_camera_2d()
	var viewport_rect = get_viewport().get_visible_rect()
	var screen_center = viewport_rect.size / 2
	
	var world_pos = npc_node.global_position
	var screen_pos: Vector2
	
	if camera:
		# Convert world coordinates to screen coordinates accounting for camera
		var offset_from_camera = world_pos - camera.global_position
		screen_pos = screen_center + offset_from_camera * camera.zoom
	else:
		# No camera, use world coordinates directly
		screen_pos = world_pos
	
	# Position bubble - center it above character
	screen_pos.x -= 250  # Shift left MORE to compensate for rightward offset  
	screen_pos.y -= 20  # Position above character with space for arrow
	
	# Create simple bubble
	var bubble_panel = Panel.new()
	bubble_panel.name = "StreamingBubble_" + npc_name
	bubble_panel.size = Vector2(300, 120)  # Much larger bubble
	bubble_panel.position = screen_pos
	
	# Semi-transparent white bubble style
	var style_box = StyleBoxFlat.new()
	style_box.bg_color = Color(1.0, 1.0, 1.0, 0.6)  # White with 60% opacity (more transparent)
	style_box.corner_radius_top_left = 8
	style_box.corner_radius_top_right = 8
	style_box.corner_radius_bottom_left = 8
	style_box.corner_radius_bottom_right = 8
	style_box.border_width_left = 1
	style_box.border_width_right = 1
	style_box.border_width_top = 1
	style_box.border_width_bottom = 1
	style_box.border_color = Color.BLACK
	bubble_panel.add_theme_stylebox_override("panel", style_box)
	
	# Create a smaller, more elegant arrow pointing down
	var arrow = Polygon2D.new()
	arrow.name = "BubbleArrow"
	# Small triangle tail pointing to character
	var arrow_points = PackedVector2Array([
		Vector2(145, 118),  # Left point, slightly inside bubble
		Vector2(155, 118),  # Right point, slightly inside bubble  
		Vector2(150, 128)   # Bottom tip pointing down (smaller arrow)
	])
	arrow.polygon = arrow_points
	arrow.color = Color(1.0, 1.0, 1.0, 0.6)  # Same semi-transparent white as bubble
	bubble_panel.add_child(arrow)
	
	# Create text with RichTextLabel for better text handling
	var bubble_text = RichTextLabel.new()
	bubble_text.text = text
	bubble_text.size = Vector2(290, 110)  # Much larger text area to match bubble
	bubble_text.position = Vector2(5, 5)
	bubble_text.add_theme_font_size_override("normal_font_size", 13)  # Larger font as requested
	bubble_text.add_theme_color_override("default_color", Color.BLACK)
	bubble_text.autowrap_mode = TextServer.AUTOWRAP_WORD_SMART
	bubble_text.fit_content = false  # Fixed height for scrolling
	bubble_text.scroll_active = true  # Enable scrolling
	bubble_text.scroll_following = true  # Auto scroll to bottom
	bubble_text.bbcode_enabled = false  # Plain text only
	bubble_text.clip_contents = true  # Prevent text overflow
	
	bubble_panel.add_child(bubble_text)
	ui_layer.add_child(bubble_panel)  # Add to UI layer instead of scene
	speech_bubbles[npc_name] = bubble_panel
	
	
	return  # Skip old code

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
	if is_processing:
		print("Still processing previous request...")
		return
	
	is_processing = true
	
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
	
	# Skip thinking state for WebSocket streaming (it's fast enough)
	# show_thinking(npc_name)
	
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
		
		var json = JSON.new()
		var json_str = JSON.stringify(data)
		websocket.send_text(json_str)
		print("Sent WebSocket request for ", npc_name)
	else:
		# Traditional mode - Call LLM in thread
		var thread = Thread.new()
		thread.start(_llm_thread.bind(npc_name, message))

func _llm_thread(npc_name: String, message: String):
	"""Call LLM in thread"""
	var output = []
	# Call Python client with the message
	var args = [project_root + "llm_client_cognitive.py", "dialogue", npc_name, message]
	
	print("Calling LLM with: ", python_path, " ", args)
	
	var start_time = Time.get_ticks_msec()
	var exit_code = OS.execute(python_path, args, output, true, false)
	var elapsed = (Time.get_ticks_msec() - start_time) / 1000.0
	
	var response = "..."
	if exit_code == 0 and output.size() > 0:
		response = output[0].strip_edges()
		
		# Clean response
		if response.contains("Server not running"):
			response = "Server offline"
		elif response.length() > 1000:
			response = response.substr(0, 997) + "..."
	else:
		response = "No response"
		print("LLM call failed. Exit code: ", exit_code)
	
	call_deferred("_on_response", npc_name, response, elapsed)

func _on_response(npc_name: String, response: String, time_taken: float):
	"""Handle response"""
	is_processing = false
	show_response(npc_name, response)
	print("[%s] Response (%.2fs): %s" % [npc_name, time_taken, response])
	
	# Auto-refresh memory display if panel is open for this NPC
	if memory_panel and memory_panel.visible and memory_text:
		# Check if we're viewing this NPC's memories
		var current_title = memory_text.get_parsed_text()
		if current_title.contains(npc_name + "'s Conversation History"):
			# Wait a bit for file to be written, then reload
			await get_tree().create_timer(0.5).timeout
			load_npc_memories(npc_name)

func show_thinking(npc_name: String):
	"""Show thinking indicator as a bubble"""
	# Keep NPC label
	var npc = npcs.get(npc_name)
	if npc and npc.has_node(npc_name):  # Label child node
		var label = npc.get_node(npc_name)
		if label is Label:
			label.text = npc_name
			label.modulate = Color(0.9, 0.9, 1.0)  # Slightly blue tint
	
	# Show thinking bubble
	if npc:
		show_thinking_bubble(npc, npc_name)

func show_response(npc_name: String, text: String):
	"""Show response as speech bubble above NPC"""
	# Keep NPC label showing only name
	var npc = npcs.get(npc_name)
	if npc and npc.has_node(npc_name):  # Label child node
		var label = npc.get_node(npc_name)
		if label is Label:
			# Keep showing only the NPC name
			label.text = npc_name
			label.modulate = Color.WHITE
	
	# Show speech bubble above NPC
	if npc:
		show_speech_bubble(npc, text, npc_name)

func create_memory_button():
	"""Create buttons for memory management"""
	# Create a CanvasLayer for UI elements to ensure they're on top
	var ui_layer = CanvasLayer.new()
	ui_layer.name = "UILayer"
	add_child(ui_layer)
	
	# Create memory buttons for each NPC
	var npc_list = ["Bob", "Alice", "Sam"]
	var button_y = 10
	
	for npc_name in npc_list:
		# View memories button
		var view_button = Button.new()
		view_button.text = "View " + npc_name + "'s Memories"
		view_button.position = Vector2(10, button_y)
		view_button.size = Vector2(180, 30)  # Slightly wider for better text fit
		view_button.add_theme_font_size_override("font_size", 12)
		view_button.pressed.connect(_on_memory_button_pressed.bind(npc_name))
		ui_layer.add_child(view_button)
		
		# Clear memories button
		var clear_button = Button.new()
		clear_button.text = "Clear " + npc_name
		clear_button.position = Vector2(200, button_y)  # More space from view button
		clear_button.size = Vector2(120, 30)  # Slightly wider
		clear_button.add_theme_font_size_override("font_size", 12)
		clear_button.modulate = Color(1, 0.8, 0.8)  # Slightly red to indicate danger
		clear_button.pressed.connect(_on_clear_memory_button_pressed.bind(npc_name))
		ui_layer.add_child(clear_button)
		
		button_y += 35  # Move down for next NPC's buttons
	
	print("Memory management buttons created on UI layer")

func _on_memory_button_pressed(npc_name: String = "Bob"):
	"""Handle memory button press for specific NPC"""
	print("Memory button pressed for ", npc_name)
	if memory_panel:
		# Toggle visibility
		memory_panel.visible = !memory_panel.visible
		if memory_panel.visible:
			load_npc_memories(npc_name)
	else:
		# Create panel first time
		create_memory_panel()
		load_npc_memories(npc_name)

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
	
	# Clear unified memory file (single source of truth)
	var mem_file = project_root + "finalbuild/npc_memories/" + npc_name + ".json"
	var file = FileAccess.open(mem_file, FileAccess.WRITE)
	if file:
		file.store_string("[]")  # Empty array
		file.close()
		print("Cleared memory file for ", npc_name)
	
	# Silent success - no dialog needed
	print(npc_name + "'s memories have been cleared successfully")
	
	# Update memory viewer if open
	if memory_panel and memory_panel.visible:
		load_npc_memories(npc_name)

func create_memory_panel():
	"""Create panel to display memories"""
	# Get or reuse the UI layer
	var ui_layer = get_node_or_null("UILayer")
	if not ui_layer:
		ui_layer = CanvasLayer.new()
		ui_layer.name = "UILayer"
		add_child(ui_layer)
	
	# Create background panel
	memory_panel = Panel.new()
	memory_panel.position = Vector2(10, 50)
	memory_panel.size = Vector2(400, 300)
	
	# Create scroll container
	var scroll = ScrollContainer.new()
	scroll.position = Vector2(5, 5)
	scroll.size = Vector2(390, 290)
	memory_panel.add_child(scroll)
	
	# Create rich text label for formatted text
	memory_text = RichTextLabel.new()
	memory_text.custom_minimum_size = Vector2(380, 1000)  # Tall for scrolling
	memory_text.bbcode_enabled = false  # Disable BBCode, use plain text
	memory_text.fit_content = true
	memory_text.scroll_active = true
	scroll.add_child(memory_text)
	
	# Add close button
	var close_btn = Button.new()
	close_btn.text = "X"
	close_btn.position = Vector2(370, 5)
	close_btn.size = Vector2(25, 25)
	close_btn.pressed.connect(func(): memory_panel.visible = false)
	memory_panel.add_child(close_btn)
	
	ui_layer.add_child(memory_panel)
	print("Memory panel created on UI layer")

func load_npc_memories(npc_name: String = "Bob"):
	"""Load NPC's memories using Python tool"""
	memory_text.clear()
	memory_text.append_text("=== " + npc_name + "'s Conversation History ===\n\n")
	
	# Call Python memory viewer - use absolute path
	var output = []
	var memory_script = project_root + "finalbuild/tools/view_memories.py"
	print("Loading memories for ", npc_name, " from: ", memory_script)
	print("Using Python: ", python_path)
	
	var args = [memory_script, npc_name]
	var exit_code = OS.execute(python_path, args, output, true, false)
	
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

func show_thinking_bubble(npc_node: Node, npc_name: String):
	"""Thinking bubble disabled - no longer shows anything"""
	pass

func show_speech_bubble(npc_node: Node, text: String, npc_name: String):
	"""Speech bubble disabled - streaming bubble handles all text display"""
	pass
