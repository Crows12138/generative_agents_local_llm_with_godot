# llm_socket_client.gd - Direct socket client for Godot
extends Node

class_name LLMSocketClient

var tcp_client: StreamPeerTCP
var is_connected: bool = false
var server_host: String = "127.0.0.1"
var server_port: int = 9999

signal response_received(text: String)
signal connection_status_changed(connected: bool)

func _ready():
	tcp_client = StreamPeerTCP.new()
	print("LLM Socket Client initialized")

func connect_to_server() -> bool:
	"""Connect to LLM server"""
	if is_connected:
		return true
	
	var result = tcp_client.connect_to_host(server_host, server_port)
	if result == OK:
		print("Connecting to LLM server...")
		# Wait for connection
		var timeout = 5.0
		var elapsed = 0.0
		
		while tcp_client.get_status() == StreamPeerTCP.STATUS_CONNECTING and elapsed < timeout:
			await get_tree().create_timer(0.1).timeout
			elapsed += 0.1
		
		if tcp_client.get_status() == StreamPeerTCP.STATUS_CONNECTED:
			is_connected = true
			connection_status_changed.emit(true)
			print("Connected to LLM server!")
			return true
	
	print("Failed to connect to LLM server")
	return false

func send_message(message: String) -> String:
	"""Send message to server and get response"""
	if not is_connected:
		if not await connect_to_server():
			return "Server not available"
	
	# Send message
	var data = message.to_utf8_buffer()
	tcp_client.put_data(data)
	
	# Wait for response
	var response = await _receive_response()
	response_received.emit(response)
	return response

func _receive_response() -> String:
	"""Receive response from server"""
	var timeout = 30.0  # 30 second timeout
	var elapsed = 0.0
	var response_buffer = PackedByteArray()
	
	while elapsed < timeout:
		tcp_client.poll()
		
		if tcp_client.get_available_bytes() > 0:
			var chunk = tcp_client.get_data(tcp_client.get_available_bytes())
			if chunk[0] == OK:
				response_buffer.append_array(chunk[1])
				
				# Check if we have complete response (simple approach)
				var response_str = response_buffer.get_string_from_utf8()
				if response_str.length() > 0:
					return response_str
		
		await get_tree().create_timer(0.1).timeout
		elapsed += 0.1
	
	return "Timeout waiting for response"

func disconnect_from_server():
	"""Disconnect from server"""
	if is_connected:
		tcp_client.disconnect_from_host()
		is_connected = false
		connection_status_changed.emit(false)
		print("Disconnected from LLM server")

func _exit_tree():
	disconnect_from_server()