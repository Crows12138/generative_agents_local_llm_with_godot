# bar_server_client_fixed.gd
extends Node2D

var npcs = {}
var python_path = ""
var project_root = ""
var is_processing = false
var input_dialog = null
var current_npc_for_input = ""

# Memory viewer UI elements
var memory_button = null
var memory_panel = null
var memory_text = null

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
		elif response.length() > 60:
			response = response.substr(0, 57) + "..."
	else:
		response = "No response"
		print("LLM call failed. Exit code: ", exit_code)
	
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
			label.modulate = Color(0.8, 0.8, 1.0)

func show_response(npc_name: String, text: String):
	"""Show response"""
	var npc = npcs.get(npc_name)
	if npc and npc.has_node(npc_name):  # Label child node
		var label = npc.get_node(npc_name)
		if label is Label:
			label.text = npc_name + "\n💬 " + text
			label.modulate = Color.WHITE
			
			# Create fade out animation
			var tween = create_tween()
			tween.tween_interval(5.0)
			tween.tween_property(label, "modulate:a", 0.5, 1.0)
			tween.tween_callback(func(): label.text = npc_name)

func create_memory_button():
	"""Create button to view Bob's memories"""
	memory_button = Button.new()
	memory_button.text = "View Bob's Memories"
	memory_button.position = Vector2(10, 10)
	memory_button.size = Vector2(150, 30)
	memory_button.add_theme_font_size_override("font_size", 12)
	memory_button.pressed.connect(_on_memory_button_pressed)
	add_child(memory_button)
	print("Memory viewer button created")

func _on_memory_button_pressed():
	"""Handle memory button press"""
	print("Memory button pressed")
	if memory_panel:
		# Toggle visibility
		memory_panel.visible = !memory_panel.visible
	else:
		# Create panel first time
		create_memory_panel()
		load_bob_memories()

func create_memory_panel():
	"""Create panel to display memories"""
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
	memory_text.size = Vector2(380, 1000)  # Tall for scrolling
	memory_text.bbcode_enabled = true
	memory_text.fit_content = true
	scroll.add_child(memory_text)
	
	# Add close button
	var close_btn = Button.new()
	close_btn.text = "X"
	close_btn.position = Vector2(370, 5)
	close_btn.size = Vector2(25, 25)
	close_btn.pressed.connect(func(): memory_panel.visible = false)
	memory_panel.add_child(close_btn)
	
	add_child(memory_panel)
	print("Memory panel created")

func load_bob_memories():
	"""Load Bob's memories using Python tool"""
	memory_text.clear()
	memory_text.append_text("[b][color=yellow]Bob's Conversation History[/color][/b]\n\n")
	
	# Call Python memory viewer
	var output = []
	var args = [project_root + "finalbuild/tools/view_memories.py", "Bob"]
	var exit_code = OS.execute(python_path, args, output, true, false)
	
	if exit_code == 0 and output.size() > 0:
		# Parse and format memories
		var memories_text = output[0]
		var lines = memories_text.split("\n")
		
		for line in lines:
			if line.begins_with("#") and line.contains(" - "):
				# Memory entry header
				memory_text.append_text("[color=white][b]" + line + "[/b][/color]\n")
			elif line.contains("User:"):
				memory_text.append_text("[color=cyan]" + line + "[/color]\n")
			elif line.contains("NPC:"):
				memory_text.append_text("[color=lime]" + line + "[/color]\n")
			elif line.contains("Importance:"):
				memory_text.append_text("[color=yellow]" + line + "[/color]\n")
			elif line.contains("Response time:"):
				memory_text.append_text("[color=gray]" + line + "[/color]\n")
			elif line.contains("----"):
				memory_text.append_text("[color=gray]" + line + "[/color]\n")
			elif line.contains("Summary:") or line.contains("Deep thinking:") or line.contains("Average importance:"):
				memory_text.append_text("[color=aqua][i]" + line + "[/i][/color]\n")
			else:
				memory_text.append_text(line + "\n")
	else:
		memory_text.append_text("[color=red]Failed to load memories[/color]\n")
		if output.size() > 0:
			memory_text.append_text(output[0])
	
	print("Memories loaded")
