extends Node

func _ready():
	print("=== Godot Network Diagnostic Test ===")
	print("Testing different connection methods...")
	
	# Test 1: Direct IP
	await test_request("http://127.0.0.1:8000/status", "127.0.0.1")
	await get_tree().create_timer(0.5).timeout
	
	# Test 2: localhost
	await test_request("http://localhost:8000/status", "localhost")
	await get_tree().create_timer(0.5).timeout
	
	# Test 3: Interaction endpoint with 127.0.0.1
	await test_interaction("http://127.0.0.1:8000/interact/Bob", "127.0.0.1 interaction")
	await get_tree().create_timer(0.5).timeout
	
	# Test 4: Interaction endpoint with localhost
	await test_interaction("http://localhost:8000/interact/Bob", "localhost interaction")
	
	print("=== Test Complete ===")

func test_request(url: String, label: String):
	var http = HTTPRequest.new()
	add_child(http)
	
	var start = Time.get_ticks_msec()
	print("Testing %s..." % label)
	
	http.request_completed.connect(func(r,c,h,b):
		var elapsed = Time.get_ticks_msec() - start
		print("  [%s] Response time: %.3f seconds (code: %d)" % [label, elapsed/1000.0, c])
		http.queue_free()
	)
	
	var error = http.request(url)
	if error != OK:
		print("  [%s] Request failed with error: %d" % [label, error])
		http.queue_free()

func test_interaction(url: String, label: String):
	var http = HTTPRequest.new()
	http.timeout = 5.0  # 5 second timeout
	add_child(http)
	
	var start = Time.get_ticks_msec()
	print("Testing %s..." % label)
	
	http.request_completed.connect(func(r,c,h,b):
		var elapsed = Time.get_ticks_msec() - start
		print("  [%s] Response time: %.3f seconds (code: %d)" % [label, elapsed/1000.0, c])
		if c == 200:
			var response = b.get_string_from_utf8()
			print("  Response preview: %s..." % response.substr(0, 50))
		http.queue_free()
	)
	
	var headers = ["Content-Type: application/json"]
	var body = JSON.stringify({"message": "Hello"})
	
	var error = http.request(url, headers, HTTPClient.METHOD_POST, body)
	if error != OK:
		print("  [%s] Request failed with error: %d" % [label, error])
		http.queue_free()