# llm_client_integration.gd - Simple client integration using OS.execute
extends Node

class_name LLMClientIntegration

var python_path = "python"
var project_root = ""
var is_server_running = false

signal response_received(npc_name: String, response: String)
signal server_status_changed(running: bool)

func _ready():
	# Get project root path (2 levels up from Godot project)
	var exe_path = OS.get_executable_path()
	var exe_dir = exe_path.get_base_dir()
	
	# Adjust path based on OS
	if OS.get_name() == "Windows":
		python_path = exe_dir + "/../../.venv/Scripts/python.exe"
		project_root = exe_dir + "/../.."
	else:
		python_path = "python"
		project_root = exe_dir + "/../.."
	
	print("LLM Client Integration ready")
	print("Python path: ", python_path)
	print("Project root: ", project_root)
	
	# Check if server is running
	check_server_status()

func check_server_status():
	"""Check if LLM server is running"""
	var output = []
	# Use cognitive client instead
	var args = [project_root + "/llm_client_cognitive.py", "ping"]
	var exit_code = OS.execute(python_path, args, output, true, false)
	
	if exit_code == 0 and output.size() > 0:
		var response = output[0].strip_edges()
		is_server_running = not response.contains("Server not running")
		server_status_changed.emit(is_server_running)
		
		if is_server_running:
			print("Cognitive AI Server is running")
		else:
			print("Cognitive AI Server is not running")
			print("Start it with: python unified_cozy_bar_server.py 9999")

func start_server():
	"""Start the Cognitive AI server in background"""
	if is_server_running:
		print("Server already running")
		return
	
	print("Starting Cognitive AI server...")
	
	# Start cognitive server in background (Windows specific)
	if OS.get_name() == "Windows":
		var args = ["/c", "start", "/min", python_path, project_root + "/unified_cozy_bar_server.py", "9999"]
		OS.execute("cmd", args, [], false)
	else:
		# Linux/Mac
		var args = [project_root + "/unified_cozy_bar_server.py", "9999"]
		OS.create_process(python_path, args)
	
	# Wait for server to start (AI models need more time to load)
	print("Waiting for AI models to load...")
	await get_tree().create_timer(15.0).timeout
	check_server_status()

func get_llm_response(message: String) -> String:
	"""Get response from LLM server via client"""
	if not is_server_running:
		print("Server not running, attempting to start...")
		await start_server()
		
		if not is_server_running:
			return "Failed to start LLM server"
	
	# Call cognitive Python client
	var output = []
	var args = [project_root + "/llm_client_cognitive.py", message]
	var exit_code = OS.execute(python_path, args, output, true, false)
	
	if exit_code == 0 and output.size() > 0:
		var response = output[0].strip_edges()
		return response
	else:
		return "Error getting response"

func get_npc_response(npc_name: String, message: String) -> String:
	"""Get response for specific NPC using cognitive AI"""
	# Use cognitive dialogue system
	var output = []
	var args = [project_root + "/llm_client_cognitive.py", "dialogue", npc_name, message]
	var exit_code = OS.execute(python_path, args, output, true, false)
	
	if exit_code == 0 and output.size() > 0:
		var response = output[0].strip_edges()
		response_received.emit(npc_name, response)
		return response
	else:
		return "Error getting response"

func _exit_tree():
	# Optional: stop server when game closes
	pass
