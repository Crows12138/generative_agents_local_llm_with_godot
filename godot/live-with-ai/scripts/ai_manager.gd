extends Node
# AI Manager - Handles communication with Python AI service

signal ai_response_received(response)
signal ai_error(error_message)

const AI_SERVICE_URL = "http://127.0.0.1:8080"
const TIMEOUT = 30.0

var http_request: HTTPRequest
var pending_requests = {}
var request_counter = 0

func _ready():
	print("AI Manager initialized")
	http_request = HTTPRequest.new()
	add_child(http_request)
	http_request.timeout = TIMEOUT
	
	# Check AI service status on startup
	check_service_status()

func check_service_status():
	"""Check if AI service is running"""
	var url = AI_SERVICE_URL + "/ai/status"
	
	var request = HTTPRequest.new()
	add_child(request)
	request.timeout = 5.0
	request.request_completed.connect(_on_status_check_completed.bind(request))
	
	var error = request.request(url)
	if error != OK:
		print("Failed to check AI service status")
		emit_signal("ai_error", "Cannot connect to AI service")

func _on_status_check_completed(result: int, response_code: int, headers: PackedStringArray, body: PackedByteArray, request: HTTPRequest):
	request.queue_free()
	
	if response_code == 200:
		var json = JSON.new()
		var parse_result = json.parse(body.get_string_from_utf8())
		if parse_result == OK:
			var data = json.data
			print("AI Service Status: ", data.status)
			print("Active Model: ", data.active_model)
			print("Available Models: ", data.available_models)
	else:
		print("AI service not responding. Please run: python -m api.godot_bridge")
		emit_signal("ai_error", "AI service not available")

func chat(character_name: String, message: String, context: Dictionary = {}) -> int:
	"""Send chat request to AI service"""
	var url = AI_SERVICE_URL + "/ai/chat"
	
	var body = {
		"character_name": character_name,
		"message": message,
		"context": context,
		"max_length": 150,
		"temperature": 0.7
	}
	
	return _make_request(url, body, "_on_chat_response")

func decide(character_name: String, situation: String, options: Array, context: Dictionary = {}) -> int:
	"""Send decision request to AI service"""
	var url = AI_SERVICE_URL + "/ai/decide"
	
	var body = {
		"character_name": character_name,
		"situation": situation,
		"options": options,
		"context": context
	}
	
	return _make_request(url, body, "_on_decide_response")

func think(character_name: String, topic: String, context: Dictionary = {}, depth: String = "normal") -> int:
	"""Send thinking request to AI service"""
	var url = AI_SERVICE_URL + "/ai/think"
	
	var body = {
		"character_name": character_name,
		"topic": topic,
		"context": context,
		"depth": depth
	}
	
	return _make_request(url, body, "_on_think_response")

func _make_request(url: String, body: Dictionary, callback: String) -> int:
	"""Make HTTP request to AI service"""
	request_counter += 1
	var request_id = request_counter
	
	var request = HTTPRequest.new()
	add_child(request)
	request.timeout = TIMEOUT
	
	# Store request info
	pending_requests[request_id] = {
		"request": request,
		"callback": callback,
		"start_time": Time.get_ticks_msec()
	}
	
	# Connect signal
	request.request_completed.connect(_on_request_completed.bind(request_id))
	
	# Make request
	var headers = ["Content-Type: application/json"]
	var json_body = JSON.stringify(body)
	
	var error = request.request(url, headers, HTTPClient.METHOD_POST, json_body)
	if error != OK:
		print("Failed to make request: ", error)
		emit_signal("ai_error", "Request failed")
		request.queue_free()
		pending_requests.erase(request_id)
		return -1
	
	return request_id

func _on_request_completed(result: int, response_code: int, headers: PackedStringArray, body: PackedByteArray, request_id: int):
	"""Handle completed HTTP request"""
	if not pending_requests.has(request_id):
		return
	
	var request_info = pending_requests[request_id]
	var request = request_info.request
	var callback = request_info.callback
	var elapsed_time = (Time.get_ticks_msec() - request_info.start_time) / 1000.0
	
	# Clean up
	request.queue_free()
	pending_requests.erase(request_id)
	
	# Check for errors
	if result != HTTPRequest.RESULT_SUCCESS:
		print("Request failed with result: ", result)
		emit_signal("ai_error", "Network error")
		return
	
	if response_code != 200:
		print("Request failed with status code: ", response_code)
		emit_signal("ai_error", "Server error: " + str(response_code))
		return
	
	# Parse response
	var json = JSON.new()
	var parse_result = json.parse(body.get_string_from_utf8())
	if parse_result != OK:
		print("Failed to parse JSON response")
		emit_signal("ai_error", "Invalid response format")
		return
	
	var data = json.data
	data["request_id"] = request_id
	data["elapsed_time"] = elapsed_time
	
	# Call specific handler
	if has_method(callback):
		call(callback, data)
	
	# Emit general signal
	emit_signal("ai_response_received", data)

func _on_chat_response(data: Dictionary):
	"""Handle chat response"""
	print("Chat response for ", data.character_name, ": ", data.response)
	if data.has("emotion"):
		print("  Emotion: ", data.emotion)

func _on_decide_response(data: Dictionary):
	"""Handle decision response"""
	print("Decision for ", data.character_name, ": ", data.chosen_option)
	print("  Reasoning: ", data.reasoning)
	print("  Confidence: ", data.confidence)

func _on_think_response(data: Dictionary):
	"""Handle thinking response"""
	print("Thought from ", data.character_name, ": ", data.thought)
	if data.has("mood"):
		print("  Mood: ", data.mood)

func cancel_request(request_id: int):
	"""Cancel a pending request"""
	if pending_requests.has(request_id):
		var request_info = pending_requests[request_id]
		request_info.request.cancel_request()
		request_info.request.queue_free()
		pending_requests.erase(request_id)
		print("Request ", request_id, " cancelled")

func get_pending_requests() -> Array:
	"""Get list of pending request IDs"""
	return pending_requests.keys()

func is_request_pending(request_id: int) -> bool:
	"""Check if a request is still pending"""
	return pending_requests.has(request_id)


