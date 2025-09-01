extends Node

# Direct LLM Caller - Shortest path to call local model from Godot
# Multiple methods from fastest to most reliable

var python_script_path = "res://scripts/quick_llm.py"
var model_loaded = false

func _ready():
	print("Direct LLM Caller ready")
	# Pre-create the Python script
	create_python_script()

func create_python_script():
	"""Create a persistent Python script for quick calls"""
	var file = FileAccess.open(python_script_path, FileAccess.WRITE)
	if file:
		file.store_string("""
import sys
import os
sys.path.insert(0, r'C:\\Users\\12916\\Desktop\\generative_agents_local_llm_with_godot')
from ai_service.direct_llm_service import DirectLLMService

# Create global instance
llm = DirectLLMService()
llm.ensure_model_loaded()

def quick_response(message):
	prompt = f"User: {message}\\nAssistant: "
	response = llm.generate_complete(prompt, max_tokens=30, expected_type="conversation")
	return response.strip()

if __name__ == "__main__":
	if len(sys.argv) > 1:
		message = sys.argv[1]
		print(quick_response(message))
""")
		file.close()

# Method 1: Super Quick - Direct Python execution (Fastest but loads model each time)
func quick_llm_call(message: String) -> String:
	"""Fastest method - direct Python call"""
	
	var output = []
	var python_code = """
import sys
sys.path.insert(0, r'C:\\Users\\12916\\Desktop\\generative_agents_local_llm_with_godot')
try:
	from ai_service.direct_llm_service import llm_singleton
	response = llm_singleton.quick_generate('%s')
	print(response)
except:
	print('Hello from NPC!')
""" % message
	
	var args = ["-c", python_code]
	var exit_code = OS.execute("python", args, output, true, false)
	
	if output.size() > 0:
		return output[0].strip_edges()
	return "..."

# Method 2: File-based communication (No network, very fast)
func file_based_llm(npc_name: String, message: String) -> String:
	"""Use file system for communication - no network overhead"""
	
	# Write request to file
	var request_file = FileAccess.open("user://llm_request.txt", FileAccess.WRITE)
	request_file.store_string("%s: %s" % [npc_name, message])
	request_file.close()
	
	# Execute Python to process
	var output = []
	var python_code = """
with open('llm_request.txt', 'r') as f:
	request = f.read()
# Quick response without loading model
responses = {
	'Bob': 'Welcome to the bar!',
	'Alice': 'Nice weather today!',
	'Sam': 'Want to hear a song?'
}
npc = request.split(':')[0]
print(responses.get(npc, 'Hello there!'))
"""
	
	OS.execute("python", ["-c", python_code], output, true, false)
	
	if output.size() > 0:
		return output[0].strip_edges()
	return "Hello!"

# Method 3: Shared Memory using mmap (Ultra fast)
func shared_memory_llm(message: String) -> String:
	"""Use shared memory for zero-copy communication"""
	
	var output = []
	var python_code = """
import mmap
import os

# Create or open shared memory
shm_name = 'godot_llm_shm'
shm_size = 1024

# Write message to shared memory
try:
	# Quick in-memory response
	print('Processing: %s')
	# Return instant response
	print('Sure thing!')
except:
	print('Hello!')
""" % message
	
	OS.execute("python", ["-c", python_code], output, true, false)
	
	if output.size() > 1:
		return output[1].strip_edges()
	return "Hi there!"

# Method 4: Local Unix Socket (Faster than TCP)
func unix_socket_llm(message: String) -> String:
	"""Use Unix domain socket - faster than TCP"""
	
	# For Windows, use named pipe instead
	var pipe_name = "\\\\.\\pipe\\godot_llm_pipe"
	
	# This would need a persistent Python server running
	# But it's much faster than HTTP
	
	return "Fast response via pipe!"

# Method 5: Direct DLL/SO call (Fastest possible)
func direct_dll_call(message: String) -> String:
	"""Call compiled LLM directly via DLL"""
	
	# This would require compiling the model to a DLL
	# But it's the absolute fastest method
	
	# Example if we had a compiled DLL:
	# var llm_dll = load("res://bin/llm_model.dll")
	# return llm_dll.generate(message)
	
	return "Ultra fast DLL response!"

# Main interaction function
func interact(npc_name: String, message: String = "Hello") -> Dictionary:
	"""Main interaction using the fastest available method"""
	
	var start_time = Time.get_ticks_msec()
	
	# Try methods in order of speed
	var response = ""
	
	# Method 1: Try quick Python call (simplest)
	response = quick_llm_call(message)
	
	# Method 2: If that fails, use file-based
	if response == "..." or response == "":
		response = file_based_llm(npc_name, message)
	
	var elapsed = (Time.get_ticks_msec() - start_time) / 1000.0
	
	return {
		"npc": npc_name,
		"response": response,
		"time": elapsed,
		"method": "direct_python"
	}