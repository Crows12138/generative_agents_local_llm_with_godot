# bar_server_client_fixed.gd - Fixed version, properly handles clicks
extends Node2D

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

func _ready():
	print("Bar scene - Server-Client mode (Fixed)!")
	
	# Setup paths
	var exe_path = OS.get_executable_path()
	var exe_dir = exe_path.get_base_dir()
	
	if OS.get_name() == "Windows":
		# Running in Godot editor
		if exe_dir.contains("Godot"):
			project_root = ProjectSettings.globalize_path("res://").replace("godot/live-with-ai/", "")
			python_path = project_root + ".venv/Scripts/python.exe"
		else:
			# Running after export
			project_root = exe_dir + "/../.."
			python_path = project_root + "/.venv/Scripts/python.exe"
	
	print("Python path: ", python_path)
	print("Project root: ", project_root)
	
	# Get NPC nodes
	setup_npcs()
	
	# Check server
	check_server()
	
	# Create memory viewer button
	create_memory_button()

func setup_npcs():
	"""Setup NPC nodes and click detection"""
	# Find ColorRect type NPCs
	var bob = $Bob if has_node("Bob") else null
	var alice = $Alice if has_node("Alice") else null
	var sam = $Sam if has_node("Sam") else null
	
	# Setup click detection
	if bob and bob is ColorRect:
		npcs["Bob"] = bob
		make_clickable(bob, "Bob")
		
	if alice and alice is ColorRect:
		npcs["Alice"] = alice
		make_clickable(alice, "Alice")
		
	if sam and sam is ColorRect:
		npcs["Sam"] = sam
		make_clickable(sam, "Sam")
	
	print("NPCs setup complete: ", npcs.keys())

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
	var output = []
	var args = [project_root + "llm_client_cognitive.py", "ping"]
	var exit_code = OS.execute(python_path, args, output, true, false)
	
	if exit_code == 0 and output.size() > 0:
		var response = output[0].strip_edges()
		if response.contains("Server is running"):
			print("✅ Optimized Cozy Bar Server is ready on port 9999!")
			return
	
	print("❌ Server not running. Please run: START_OPTIMIZED_COZY_BAR.bat")
	print("   Or start: python server_client/optimized_cozy_bar_server.py 9999")

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
	
	# Add click feedback - flash the NPC
	var npc = npcs.get(npc_name)
	if npc:
		var original_modulate = npc.modulate
		var tween = create_tween()
		tween.tween_property(npc, "modulate", Color(1.2, 1.2, 1.2), 0.1)
		tween.tween_property(npc, "modulate", original_modulate, 0.1)
	
	# Show thinking state
	show_thinking(npc_name)
	
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
	
	# Call LLM in thread
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
		view_button.size = Vector2(150, 30)
		view_button.add_theme_font_size_override("font_size", 12)
		view_button.pressed.connect(_on_memory_button_pressed.bind(npc_name))
		ui_layer.add_child(view_button)
		
		# Clear memories button
		var clear_button = Button.new()
		clear_button.text = "Clear " + npc_name
		clear_button.position = Vector2(170, button_y)  # Right of view button
		clear_button.size = Vector2(100, 30)
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
	
	# Clear both memory files directly for the specific NPC
	var mem_file1 = project_root + "finalbuild/npc_memories/" + npc_name + ".json"
	var mem_file2 = project_root + "finalbuild/server/npc_gpt4all_conversations/" + npc_name + ".json"
	
	# Clear standard memory file
	var file1 = FileAccess.open(mem_file1, FileAccess.WRITE)
	if file1:
		file1.store_string("[]")  # Empty array
		file1.close()
		print("Cleared standard memory file")
	
	# Clear GPT4All conversation file
	var file2 = FileAccess.open(mem_file2, FileAccess.WRITE)
	if file2:
		file2.store_string('{"npc": "' + npc_name + '", "conversation": []}')  # Empty conversation
		file2.close()
		print("Cleared GPT4All conversation file for ", npc_name)
	
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
	"""Create and show a thinking bubble (simpler, smaller)"""
	# Remove existing bubble if any
	if npc_name in speech_bubbles:
		speech_bubbles[npc_name].queue_free()
		speech_bubbles.erase(npc_name)
	
	# Get or create UI layer
	var ui_layer = get_node_or_null("UILayer")
	if not ui_layer:
		ui_layer = CanvasLayer.new()
		ui_layer.name = "UILayer"
		add_child(ui_layer)
	
	# Create bubble container
	var bubble_container = Control.new()
	bubble_container.name = "ThinkingBubble_" + npc_name
	
	# Position bubble in center-ish area
	var viewport_size = get_viewport().size
	var bubble_x = viewport_size.x * 0.5 - 50  # Smaller bubble, so less offset
	var bubble_y = viewport_size.y * 0.35
	
	# Adjust position based on which NPC
	if npc_name == "Bob":
		bubble_x = viewport_size.x * 0.4 - 50
	elif npc_name == "Alice":
		bubble_x = viewport_size.x * 0.6 - 50
	
	bubble_container.position = Vector2(bubble_x, bubble_y)
	
	# Create smaller bubble panel
	var bubble_panel = Panel.new()
	bubble_panel.size = Vector2(100, 50)
	bubble_panel.position = Vector2(0, 0)
	
	# Create bubble style
	var style_box = StyleBoxFlat.new()
	style_box.bg_color = Color(0.9, 0.9, 1.0, 0.9)  # Light blue background
	style_box.corner_radius_top_left = 10
	style_box.corner_radius_top_right = 10
	style_box.corner_radius_bottom_left = 10
	style_box.corner_radius_bottom_right = 10
	style_box.border_width_left = 2
	style_box.border_width_right = 2
	style_box.border_width_top = 2
	style_box.border_width_bottom = 2
	style_box.border_color = Color(0.7, 0.7, 0.9, 0.8)
	bubble_panel.add_theme_stylebox_override("panel", style_box)
	
	# Create animated dots label
	var thinking_text = Label.new()
	thinking_text.position = Vector2(25, 10)
	thinking_text.size = Vector2(50, 30)
	thinking_text.add_theme_font_size_override("font_size", 20)
	thinking_text.add_theme_color_override("font_color", Color(0.3, 0.3, 0.5))
	thinking_text.text = "..."
	
	# Assemble the bubble
	bubble_panel.add_child(thinking_text)
	bubble_container.add_child(bubble_panel)
	
	# Add to UI layer
	ui_layer.add_child(bubble_container)
	
	# Store reference
	speech_bubbles[npc_name] = bubble_container
	
	# Animate the dots - simple animation without complex callbacks
	var tween = create_tween()
	tween.set_loops()  # Loop animation
	tween.tween_property(thinking_text, "modulate:a", 0.5, 0.3)
	tween.tween_property(thinking_text, "modulate:a", 1.0, 0.3)
	tween.tween_property(thinking_text, "modulate:a", 0.5, 0.3)
	tween.tween_property(thinking_text, "modulate:a", 1.0, 0.3)

func show_speech_bubble(npc_node: Node, text: String, npc_name: String):
	"""Create and show a speech bubble above the NPC"""
	# Remove existing bubble if any
	if npc_name in speech_bubbles:
		speech_bubbles[npc_name].queue_free()
		speech_bubbles.erase(npc_name)
	
	# Get or create UI layer
	var ui_layer = get_node_or_null("UILayer")
	if not ui_layer:
		ui_layer = CanvasLayer.new()
		ui_layer.name = "UILayer"
		add_child(ui_layer)
	
	# Create bubble container
	var bubble_container = Control.new()
	bubble_container.name = "SpeechBubble_" + npc_name
	
	# Position bubble in center-ish area but calculate tail direction
	var viewport_size = get_viewport().size
	var bubble_x = viewport_size.x * 0.5 - 200  # Center horizontally (minus half of 400px width)
	var bubble_y = viewport_size.y * 0.3  # Upper-middle of screen
	
	# Adjust position based on which NPC (left/right bias)
	if npc_name == "Bob":
		bubble_x = viewport_size.x * 0.4 - 200  # Slightly left
	elif npc_name == "Alice":
		bubble_x = viewport_size.x * 0.6 - 200  # Slightly right
	
	bubble_container.position = Vector2(bubble_x, bubble_y)
	
	# Create the bubble panel (larger size)
	var bubble_panel = Panel.new()
	bubble_panel.size = Vector2(400, 200)  # Increased width and height
	bubble_panel.position = Vector2(0, 0)
	
	# Create bubble style with rounded corners
	var style_box = StyleBoxFlat.new()
	style_box.bg_color = Color(1, 1, 1, 0.95)  # White background, slightly transparent
	style_box.corner_radius_top_left = 15
	style_box.corner_radius_top_right = 15
	style_box.corner_radius_bottom_left = 15
	style_box.corner_radius_bottom_right = 15
	style_box.border_width_left = 2
	style_box.border_width_right = 2
	style_box.border_width_top = 2
	style_box.border_width_bottom = 2
	style_box.border_color = Color(0.2, 0.2, 0.2, 0.8)
	bubble_panel.add_theme_stylebox_override("panel", style_box)
	
	# Create ScrollContainer for scrollable text
	var scroll_container = ScrollContainer.new()
	scroll_container.position = Vector2(10, 10)
	scroll_container.size = Vector2(380, 180)  # Match panel interior size
	scroll_container.vertical_scroll_mode = ScrollContainer.SCROLL_MODE_AUTO
	scroll_container.horizontal_scroll_mode = ScrollContainer.SCROLL_MODE_DISABLED
	
	# Create text label inside scroll container
	var bubble_text = RichTextLabel.new()
	if bubble_text:
		bubble_text.bbcode_enabled = true
		bubble_text.custom_minimum_size = Vector2(370, 0)  # Width slightly less than scroll container
		bubble_text.fit_content = true  # Auto-adjust height based on content
		bubble_text.add_theme_font_size_override("normal_font_size", 13)  # Good size for readability
		bubble_text.add_theme_color_override("default_color", Color(0, 0, 0, 1))  # Pure black for better contrast
		bubble_text.autowrap_mode = TextServer.AUTOWRAP_WORD_SMART
		bubble_text.append_text(text)  # Use append_text instead of text property
		
	# Add text to scroll container
	scroll_container.add_child(bubble_text)
	
	# Add tail/pointer to bubble pointing toward NPC
	var tail = Polygon2D.new()
	var tail_offset_x = 0
	
	# Adjust tail position based on NPC
	if npc_name == "Bob":
		tail_offset_x = -50  # Point left toward Bob
	elif npc_name == "Alice":
		tail_offset_x = 50  # Point right toward Alice
	elif npc_name == "Sam":
		tail_offset_x = 0  # Center for Sam
	
	# Create tail shape pointing down and slightly to the side
	var points = PackedVector2Array([
		Vector2(-10, 200),  # Left corner at bottom of bubble (200 = panel height)
		Vector2(10, 200),   # Right corner at bottom of bubble
		Vector2(tail_offset_x, 230)  # Point toward NPC
	])
	tail.polygon = points
	tail.color = Color(1, 1, 1, 0.95)
	tail.position = Vector2(200, 0)  # Center horizontally on bubble (200 = half of 400px width)
	
	# Assemble the bubble
	bubble_panel.add_child(scroll_container)
	bubble_container.add_child(bubble_panel)
	bubble_container.add_child(tail)
	
	# Add to UI layer for proper display
	ui_layer.add_child(bubble_container)
	
	# Store reference
	speech_bubbles[npc_name] = bubble_container
	
	# Animate appearance
	bubble_container.modulate.a = 0
	var tween = create_tween()
	tween.tween_property(bubble_container, "modulate:a", 1.0, 0.3)
	tween.tween_interval(8.0)  # Display for 8 seconds
	tween.tween_property(bubble_container, "modulate:a", 0.0, 0.5)
	tween.tween_callback(func(): 
		if is_instance_valid(bubble_container):
			bubble_container.queue_free()
		if npc_name in speech_bubbles:
			speech_bubbles.erase(npc_name)
	)
