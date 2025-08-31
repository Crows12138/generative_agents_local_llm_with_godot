# bar_updater_direct.gd - Direct LLM call version
extends Node2D

@onready var http = HTTPRequest.new()
var npcs = {}
var python_path = "python"
var is_processing = false

# Performance stats
var stats = {
	"total_calls": 0,
	"cached_calls": 0,
	"avg_response_time": 0.0,
	"response_times": []
}

func _ready():
	print("Bar scene started - Direct LLM mode!")
	
	# Get NPC nodes
	npcs["Bob"] = $Bob if has_node("Bob") else null
	npcs["Alice"] = $Alice if has_node("Alice") else null
	npcs["Sam"] = $Sam if has_node("Sam") else null
	
	# Warmup first call (background)
	_warmup_llm()

func _warmup_llm():
	"""Warmup LLM in background to avoid first click loading"""
	print("Warming up LLM in background...")
	var thread = Thread.new()
	thread.start(_warmup_thread)

func _warmup_thread():
	"""Warmup thread"""
	var output = []
	# Call once to load model
	OS.execute(python_path, ["quick_llm.py", "test"], output, true)
	print("LLM warmup complete!")

func _input(event):
	if event is InputEventMouseButton and event.pressed:
		if is_processing:
			print("Still processing previous request...")
			return
			
		var mouse_pos = get_global_mouse_position()
		
		for npc_name in npcs:
			var npc = npcs[npc_name]
			if npc:
				var distance = npc.position.distance_to(mouse_pos)
				
				if distance < 50:
					print("Clicked on ", npc_name)
					interact_with_npc_direct(npc_name)
					break
		
		# Right click for stats
		if event.button_index == MOUSE_BUTTON_RIGHT:
			print_stats()

func interact_with_npc_direct(npc_name: String):
	"""Direct Python LLM call"""
	
	if is_processing:
		return
	
	is_processing = true
	stats.total_calls += 1
	
	# Show thinking state immediately
	show_thinking(npc_name)
	
	# Create thread to avoid blocking
	var thread = Thread.new()
	thread.start(_llm_call_thread.bind(npc_name, "Hello, how are you?"))

func _llm_call_thread(npc_name: String, message: String):
	"""Call LLM in thread"""
	var start_time = Time.get_ticks_msec()
	
	# Build message with NPC role info
	var full_message = npc_name + " (bartender): " + message
	
	# Call Python
	var output = []
	var args = ["quick_llm.py", full_message]
	var exit_code = OS.execute(python_path, args, output, true, false)
	
	var elapsed = (Time.get_ticks_msec() - start_time) / 1000.0
	
	if output.size() > 0 and exit_code == 0:
		var response = output[0].strip_edges()
		call_deferred("_on_llm_response", npc_name, response, elapsed)
	else:
		# Fallback response
		var fallback = get_fallback_response(npc_name)
		call_deferred("_on_llm_response", npc_name, fallback, 0.0)

func _on_llm_response(npc_name: String, response: String, time_taken: float):
	"""Handle response in main thread"""
	is_processing = false
	
	# Update stats
	if time_taken < 0.5:  # Likely cached
		stats.cached_calls += 1
	
	stats.response_times.append(time_taken)
	var total_time = 0.0
	for t in stats.response_times:
		total_time += t
	stats.avg_response_time = total_time / stats.response_times.size()
	
	# Hide thinking state
	hide_thinking(npc_name)
	
	# Show dialogue
	show_speech_bubble(npc_name, response)
	
	# Print performance info
	if time_taken > 0:
		print("[%s] says: %s (%.2fs)" % [npc_name, response.substr(0, 50), time_taken])
		if time_taken < 0.5:
			print("  ^ CACHED RESPONSE!")
	else:
		print("[%s] says: %s (fallback)" % [npc_name, response])

func show_thinking(npc_name: String):
	"""Show thinking state"""
	var npc = npcs.get(npc_name)
	if npc and npc.get_child_count() > 0:
		var label = npc.get_child(0)
		if label is Label:
			label.text = npc_name + "\n[thinking...]"
			label.modulate = Color.GRAY

func hide_thinking(npc_name: String):
	"""Hide thinking state"""
	var npc = npcs.get(npc_name)
	if npc and npc.get_child_count() > 0:
		var label = npc.get_child(0)
		if label is Label:
			label.modulate = Color.WHITE

func show_speech_bubble(npc_name: String, text: String):
	"""Show dialogue bubble"""
	var npc = npcs.get(npc_name)
	if npc and npc.get_child_count() > 0:
		var label = npc.get_child(0)
		if label is Label:
			# Clean text (remove special chars)
			var clean_text = text.replace("\n", " ")
			if clean_text.length() > 50:
				clean_text = clean_text.substr(0, 47) + "..."
			label.text = npc_name + "\n" + clean_text
			
			# Animate text appearance
			label.modulate.a = 0
			var tween = create_tween()
			tween.tween_property(label, "modulate:a", 1.0, 0.3)

func get_fallback_response(npc_name: String) -> String:
	"""Get fallback response"""
	var responses = {
		"Bob": "What can I get you?",
		"Alice": "This place is nice.",
		"Sam": "Music time!"
	}
	return responses.get(npc_name, "Hello!")

func print_stats():
	"""Print performance statistics"""
	print("\n=== Performance Stats ===")
	print("Total calls: ", stats.total_calls)
	print("Cached calls: ", stats.cached_calls, " (%.1f%%)" % (stats.cached_calls * 100.0 / max(stats.total_calls, 1)))
	print("Avg response time: %.2fs" % stats.avg_response_time)
	
	if stats.response_times.size() > 0:
		var min_time = stats.response_times[0]
		var max_time = stats.response_times[0]
		for t in stats.response_times:
			min_time = min(min_time, t)
			max_time = max(max_time, t)
		print("Min time: %.2fs" % min_time)
		print("Max time: %.2fs" % max_time)
	print("========================\n")