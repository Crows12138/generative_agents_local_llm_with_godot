# test_direct_llm.gd - Test direct LLM integration
extends Node

func _ready():
	print("\n=== Testing Direct LLM Call ===")
	
	# Test 1: Basic Python call
	test_python_call()
	
	# Test 2: Multiple calls (test caching)
	test_caching()
	
	# Test 3: Different messages
	test_different_messages()

func test_python_call():
	"""Test if Python call works"""
	print("\n[Test 1] Basic Python call...")
	
	var output = []
	var result = OS.execute("python", ["quick_llm.py", "Hello test"], output, true)
	
	if result == 0 and output.size() > 0:
		print("✅ Python call works!")
		print("   Response: ", output[0].substr(0, 50))
	else:
		print("❌ Python call failed")
		print("   Exit code: ", result)
		print("   Output size: ", output.size())
		
		# Try with full python path
		print("\n   Trying with python.exe...")
		result = OS.execute("python.exe", ["quick_llm.py", "Hello test"], output, true)
		if result == 0:
			print("   ✅ Works with python.exe")
		else:
			print("   ❌ Still failed. Check:")
			print("   1. Python is installed")
			print("   2. Python is in PATH")
			print("   3. quick_llm.py exists in project root")

func test_caching():
	"""Test if caching works (second call should be instant)"""
	print("\n[Test 2] Testing cache...")
	
	var message = "Cache test message"
	var output = []
	
	# First call (slow)
	var start1 = Time.get_ticks_msec()
	OS.execute("python", ["quick_llm.py", message], output, true)
	var time1 = (Time.get_ticks_msec() - start1) / 1000.0
	
	# Second call (should be cached)
	output.clear()
	var start2 = Time.get_ticks_msec()
	OS.execute("python", ["quick_llm.py", message], output, true)
	var time2 = (Time.get_ticks_msec() - start2) / 1000.0
	
	print("   First call: %.2fs" % time1)
	print("   Second call: %.2fs" % time2)
	
	if time2 < time1 / 2:
		print("✅ Caching works! (%.1fx faster)" % (time1 / max(time2, 0.01)))
	else:
		print("⚠️ Caching might not be working")

func test_different_messages():
	"""Test different types of messages"""
	print("\n[Test 3] Different messages...")
	
	var test_messages = [
		"Hello",
		"How are you?",
		"Tell me a joke",
		"Bob: What's on the menu?"
	]
	
	for msg in test_messages:
		var output = []
		var start = Time.get_ticks_msec()
		var result = OS.execute("python", ["quick_llm.py", msg], output, true)
		var elapsed = (Time.get_ticks_msec() - start) / 1000.0
		
		if result == 0 and output.size() > 0:
			var response = output[0].strip_edges()
			print("   [%.2fs] '%s' -> '%s'" % [elapsed, msg, response.substr(0, 30)])
		else:
			print("   ❌ Failed for: ", msg)
	
	print("\n=== Test Complete ===\n")