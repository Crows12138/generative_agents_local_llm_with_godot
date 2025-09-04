# bar_server_client_simple.gd - 为现有场景设计的简化版本
extends Node2D

var npcs = {}
var python_path = ".venv/Scripts/python.exe"
var is_processing = false

func _ready():
	print("Bar scene - Server-Client mode!")
	
	# 获取现有的 NPC 节点（ColorRect 类型）
	npcs["Bob"] = $Bob if has_node("Bob") else null
	npcs["Alice"] = $Alice if has_node("Alice") else null
	npcs["Sam"] = $Sam if has_node("Sam") else null
	
	print("Found NPCs: ", npcs.keys())
	
	# 检查服务器状态
	check_server()

func check_server():
	"""检查服务器是否运行"""
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
	"""处理点击 NPC"""
	if is_processing:
		return
	
	# 调试输出
	print("Click position: ", click_pos)
	
	for npc_name in npcs:
		var npc = npcs[npc_name]
		if npc and npc is ColorRect:
			# ColorRect 使用全局坐标
			var rect = Rect2(npc.global_position, npc.size)
			print(npc_name, " rect: ", rect)
			
			if rect.has_point(click_pos):
				print("\nClicked on ", npc_name, "!")
				interact_with_npc(npc_name)
				break

func interact_with_npc(npc_name: String):
	"""与 NPC 交互"""
	if is_processing:
		return
	
	is_processing = true
	
	# 显示思考状态
	show_thinking(npc_name)
	
	# 准备消息
	var messages = {
		"Bob": "Hello Bob the bartender!",
		"Alice": "Hi Alice, how are you?",
		"Sam": "Hey Sam, play us a song!"
	}
	
	var message = messages.get(npc_name, "Hello!")
	
	# 调用 LLM
	var thread = Thread.new()
	thread.start(_llm_thread.bind(npc_name, message))

func _llm_thread(npc_name: String, message: String):
	"""在线程中调用 LLM"""
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

func show_response(npc_name: String, text: String):
	"""显示响应"""
	var npc = npcs.get(npc_name)
	if npc and npc.has_node(npc_name):  # Label 子节点
		var label = npc.get_node(npc_name)
		if label is Label:
			label.text = npc_name + "\n💬 " + text
			
			# 5秒后恢复
			await get_tree().create_timer(5.0).timeout
			label.text = npc_name
