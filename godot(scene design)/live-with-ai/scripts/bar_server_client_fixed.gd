# bar_server_client_fixed.gd - 修复版本，正确处理点击
extends Node2D

var npcs = {}
var python_path = ""
var project_root = ""
var is_processing = false
var input_dialog = null
var current_npc_for_input = ""

func _ready():
	print("Bar scene - Server-Client mode (Fixed)!")
	
	# 设置路径
	var exe_path = OS.get_executable_path()
	var exe_dir = exe_path.get_base_dir()
	
	if OS.get_name() == "Windows":
		# Godot编辑器中运行
		if exe_dir.contains("Godot"):
			project_root = ProjectSettings.globalize_path("res://").replace("godot/live-with-ai/", "")
			python_path = project_root + ".venv/Scripts/python.exe"
		else:
			# 导出后运行
			project_root = exe_dir + "/../.."
			python_path = project_root + "/.venv/Scripts/python.exe"
	
	print("Python path: ", python_path)
	print("Project root: ", project_root)
	
	# 获取 NPC 节点
	setup_npcs()
	
	# 检查服务器
	check_server()

func setup_npcs():
	"""设置 NPC 节点和点击检测"""
	# 查找 ColorRect 类型的 NPC
	var bob = $Bob if has_node("Bob") else null
	var alice = $Alice if has_node("Alice") else null
	var sam = $Sam if has_node("Sam") else null
	
	# 设置点击检测
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
	"""让 ColorRect 可点击"""
	# 连接 gui_input 信号
	if not color_rect.gui_input.is_connected(_on_npc_gui_input):
		color_rect.gui_input.connect(_on_npc_gui_input.bind(npc_name))
	
	# 设置鼠标过滤
	color_rect.mouse_filter = Control.MOUSE_FILTER_PASS
	
	print(npc_name, " is now clickable")

func _on_npc_gui_input(event: InputEvent, npc_name: String):
	"""处理 NPC 点击事件"""
	if event is InputEventMouseButton:
		if event.button_index == MOUSE_BUTTON_LEFT and event.pressed:
			print("\nLeft-clicked on ", npc_name, "!")
			interact_with_npc(npc_name)
		elif event.button_index == MOUSE_BUTTON_RIGHT and event.pressed:
			print("\nRight-clicked on ", npc_name, "! Opening input dialog...")
			show_custom_input_dialog(npc_name)

func check_server():
	"""检查服务器状态"""
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
	"""显示自定义消息输入对话框"""
	current_npc_for_input = npc_name
	
	# 创建输入对话框
	if input_dialog:
		input_dialog.queue_free()
	
	input_dialog = AcceptDialog.new()
	input_dialog.title = "Send Custom Message to " + npc_name
	input_dialog.dialog_text = "Enter your message:"
	input_dialog.size = Vector2(400, 150)
	
	# 创建输入框
	var vbox = VBoxContainer.new()
	var line_edit = LineEdit.new()
	line_edit.placeholder_text = "Type your message here..."
	line_edit.name = "CustomInput"
	vbox.add_child(line_edit)
	
	input_dialog.add_child(vbox)
	
	# 连接确认信号
	input_dialog.confirmed.connect(_on_custom_input_confirmed)
	input_dialog.canceled.connect(_on_custom_input_canceled)
	
	# 添加到场景并显示
	get_tree().root.add_child(input_dialog)
	input_dialog.popup_centered()
	
	# 聚焦到输入框
	line_edit.grab_focus()

func _on_custom_input_confirmed():
	"""处理自定义输入确认"""
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
	"""处理自定义输入取消"""
	if input_dialog:
		input_dialog.queue_free()
		input_dialog = null
	current_npc_for_input = ""
	print("Custom input canceled")

func interact_with_npc(npc_name: String, custom_message: String = ""):
	"""与 NPC 交互"""
	if is_processing:
		print("Still processing previous request...")
		return
	
	is_processing = true
	
	# 显示思考状态
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
	
	# 在线程中调用 LLM
	var thread = Thread.new()
	thread.start(_llm_thread.bind(npc_name, message))

func _llm_thread(npc_name: String, message: String):
	"""在线程中调用 LLM"""
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
		
		# 清理响应
		if response.contains("Server not running"):
			response = "Server offline"
		elif response.length() > 60:
			response = response.substr(0, 57) + "..."
	else:
		response = "No response"
		print("LLM call failed. Exit code: ", exit_code)
	
	call_deferred("_on_response", npc_name, response, elapsed)

func _on_response(npc_name: String, response: String, time_taken: float):
	"""处理响应"""
	is_processing = false
	show_response(npc_name, response)
	print("[%s] Response (%.2fs): %s" % [npc_name, time_taken, response])

func show_thinking(npc_name: String):
	"""显示思考状态"""
	var npc = npcs.get(npc_name)
	if npc and npc.has_node(npc_name):  # Label 子节点
		var label = npc.get_node(npc_name)
		if label is Label:
			label.text = npc_name + "\n💭..."
			label.modulate = Color(0.8, 0.8, 1.0)

func show_response(npc_name: String, text: String):
	"""显示响应"""
	var npc = npcs.get(npc_name)
	if npc and npc.has_node(npc_name):  # Label 子节点
		var label = npc.get_node(npc_name)
		if label is Label:
			label.text = npc_name + "\n💬 " + text
			label.modulate = Color.WHITE
			
			# 创建淡出动画
			var tween = create_tween()
			tween.tween_interval(5.0)
			tween.tween_property(label, "modulate:a", 0.5, 1.0)
			tween.tween_callback(func(): label.text = npc_name)
