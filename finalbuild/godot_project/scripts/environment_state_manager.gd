extends Node
# Environment State Manager - Automatic state evolution system
# Changes environment based on time and interaction events

signal environment_updated(changes: Dictionary)

# Time-based decay rates (per second)
@export var cleanliness_decay_rate: float = 2.0  # Counter gets dirty over time
@export var stock_consumption_rate: float = 1.5  # Stock depletes gradually

# Interaction-based triggers
@export var spill_chance_per_conversation: float = 0.15  # 15% chance per conversation
@export var actions_per_restock_need: int = 10  # Need restock after N serve actions

# Current counters
var conversation_count: int = 0
var serve_action_count: int = 0
var time_elapsed: float = 0.0

# Environment references
var bar_counter: Node2D
var tables: Array[Node2D] = []
var shelf: Node2D
var npcs: Dictionary = {}  # Track NPC activity

# Random event chances
@export var random_event_chance: float = 0.1  # 10% chance per minute
var last_random_event: float = 0.0
var random_event_cooldown: float = 60.0  # 1 minute between random events

func _ready():
	print("[EnvironmentStateManager] Initializing automatic state system")
	
	# Get environment references
	_setup_environment_references()
	
	# Connect to conversation system
	_connect_to_conversation_events()
	
	# Start environment simulation
	set_process(true)

func _setup_environment_references():
	"""Find and store references to environment objects"""
	# Get bar counter
	bar_counter = get_node_or_null("/root/CozyBar/BarCounter")
	if not bar_counter:
		print("[EnvironmentStateManager] Warning: BarCounter not found")
	
	# Get tables
	var tables_node = get_node_or_null("/root/CozyBar/Tables")
	if tables_node:
		for child in tables_node.get_children():
			tables.append(child)
	
	# Get shelf
	shelf = get_node_or_null("/root/CozyBar/LiquorShelf")
	if not shelf:
		print("[EnvironmentStateManager] Warning: LiquorShelf not found")
	
	# Get NPCs
	var bob = get_node_or_null("/root/CozyBar/bob")
	var alice = get_node_or_null("/root/CozyBar/Alice")
	var sam = get_node_or_null("/root/CozyBar/sam")
	
	if bob: npcs["Bob"] = bob
	if alice: npcs["Alice"] = alice
	if sam: npcs["Sam"] = sam

func _connect_to_conversation_events():
	"""Connect to the conversation/dialogue system"""
	# Connect to bar server client for conversation events
	var bar_client = get_node_or_null("/root/CozyBar")
	if bar_client:
		if bar_client.has_signal("response_completed"):
			bar_client.response_completed.connect(_on_conversation_completed)
		
		# Also track NPC action completions
		for npc_name in npcs:
			var npc = npcs[npc_name]
			var executor = npc.get_node_or_null("DecisionController/ActionExecutor")
			if executor:
				executor.action_completed.connect(_on_npc_action_completed.bind(npc_name))

func _process(delta):
	"""Main update loop for time-based changes"""
	time_elapsed += delta
	
	# Apply time-based degradation
	_update_cleanliness(delta)
	_update_stock_levels(delta)
	
	# Random events
	_check_random_events(delta)
	
	# Emit changes every second
	if int(time_elapsed) != int(time_elapsed - delta):
		_emit_environment_status()

func _update_cleanliness(delta):
	"""Gradually make things dirty"""
	if bar_counter and bar_counter.has_method("get_cleanliness"):
		var current = bar_counter.get_cleanliness()
		var new_value = max(0, current - cleanliness_decay_rate * delta)
		bar_counter.set_cleanliness(new_value)
		
		# Mark as dirty if below threshold
		if new_value < 30 and current >= 30:
			print("[EnvironmentStateManager] Bar counter is now dirty!")
			environment_updated.emit({"counter_dirty": true})
	
	# Tables get dirty slower
	for table in tables:
		if table and table.has_method("get_cleanliness"):
			var current = table.get_cleanliness()
			var new_value = max(0, current - cleanliness_decay_rate * delta * 0.5)  # Half rate for tables
			table.call("set_cleanliness", new_value)

func _update_stock_levels(delta):
	"""Gradually consume stock"""
	if shelf and shelf.has_method("get_stock_level"):
		var current = shelf.get_stock_level()
		var new_value = max(0, current - stock_consumption_rate * delta)
		shelf.set_stock_level(new_value)
		
		# Alert when low
		if new_value < 30 and current >= 30:
			print("[EnvironmentStateManager] Stock is running low!")
			environment_updated.emit({"shelf_low": true})
		
		# Alert when empty
		if new_value <= 0 and current > 0:
			print("[EnvironmentStateManager] Stock is empty!")
			environment_updated.emit({"shelf_empty": true})


func _check_random_events(delta):
	"""Trigger random environmental events"""
	last_random_event += delta
	
	if last_random_event < random_event_cooldown:
		return
	
	if randf() < random_event_chance * delta:
		last_random_event = 0.0
		_trigger_random_event()

func _trigger_random_event():
	"""Execute a random event"""
	var events = [
		"spill",
		"dirty_dishes",
		"stock_drain"
	]
	
	var event = events[randi() % events.size()]
	print("[EnvironmentStateManager] Random event: ", event)
	
	match event:
		"spill":
			# Make something extra dirty
			if bar_counter:
				var spill_severity = randi() % 3  # 0=minor, 1=medium, 2=major
				var new_cleanliness = [50, 20, 0][spill_severity]
				bar_counter.set_cleanliness(new_cleanliness)
				var severity_text = ["Minor", "Medium", "Major"][spill_severity]
				print("  -> ", severity_text, " spill at the bar! Cleanliness: ", new_cleanliness)
				environment_updated.emit({"counter_dirty": true})
		
		"dirty_dishes":
			# Tables need clearing
			for table in tables:
				if table and randf() > 0.5:
					table.set("has_dishes", true)
					table.set("cleanliness", 20)
			environment_updated.emit({"table_dirty": true})
			print("  -> Tables have dirty dishes")
		
		"stock_drain":
			# Sudden stock consumption
			if shelf:
				var current = shelf.get_stock_level()
				var drain_amount = 10 + randi() % 20  # Random 10-30 drain
				shelf.set_stock_level(max(0, current - drain_amount))
				print("  -> Stock depleted by ", drain_amount)

func _on_conversation_completed(npc_name: String, response: String):
	"""Handle conversation completion"""
	conversation_count += 1
	print("[EnvironmentStateManager] Conversation #", conversation_count, " with ", npc_name)
	
	# Random chance of spill during conversation
	if randf() < spill_chance_per_conversation:
		if bar_counter:
			var spill_severity = randf()
			var new_cleanliness
			if spill_severity < 0.5:  # 50% minor
				new_cleanliness = bar_counter.get_cleanliness() - 20
			elif spill_severity < 0.8:  # 30% medium  
				new_cleanliness = bar_counter.get_cleanliness() - 40
			else:  # 20% major
				new_cleanliness = 0
			
			bar_counter.set_cleanliness(max(0, new_cleanliness))
			print("  -> Someone spilled during conversation! Cleanliness: ", new_cleanliness)
			environment_updated.emit({"counter_dirty": true})
	
	# Consume some stock when serving mentioned
	if "drink" in response.to_lower() or "serve" in response.to_lower() or "beer" in response.to_lower():
		if shelf:
			var current = shelf.get_stock_level()
			shelf.set_stock_level(max(0, current - 5))
			print("  -> Conversation about drinks, stock reduced")

func _on_npc_action_completed(action_name: String, result: String, npc_name: String):
	"""Track NPC actions and their effects"""
	print("[EnvironmentStateManager] ", npc_name, " completed: ", action_name)
	
	match action_name:
		"serve_customer":
			serve_action_count += 1
			# Just deplete stock when serving
			if shelf:
				var current = shelf.get_stock_level()
				shelf.set_stock_level(max(0, current - 10))
				print("  -> Stock consumed by serving")
		
		"clean_counter":
			# Reset cleanliness
			if bar_counter:
				bar_counter.set_cleanliness(100)
				environment_updated.emit({"counter_dirty": false})
		
		"clear_table":
			# Clean a table
			for table in tables:
				var has_dishes = table.get("has_dishes") if table else null
				if has_dishes:
					table.set("has_dishes", false)
					table.set("cleanliness", 100)
					break
		
		"restock":
			# Refill shelf
			if shelf:
				shelf.set_stock_level(100)
				environment_updated.emit({"shelf_low": false, "shelf_empty": false})


func _emit_environment_status():
	"""Emit current environment status for debugging"""
	var status = {
		"time_elapsed": int(time_elapsed),
		"conversations": conversation_count,
		"serves": serve_action_count,
	}
	
	if bar_counter:
		var cleanliness = bar_counter.get("cleanliness")
		status["counter_cleanliness"] = cleanliness if cleanliness != null else 100
	
	if shelf:
		var stock = shelf.get("stock_level") 
		status["stock_level"] = stock if stock != null else 100
	
	# Only print significant status every 10 seconds
	if int(time_elapsed) % 10 == 0:
		print("[EnvironmentStateManager] Status: ", status)


func reset_environment():
	"""Reset all environment states"""
	conversation_count = 0
	serve_action_count = 0
	time_elapsed = 0.0
	
	if bar_counter:
		bar_counter.set_cleanliness(100)
	
	if shelf:
		shelf.set_stock_level(100)
	
	for table in tables:
		if table:
			table.set("cleanliness", 100)
			table.set("occupied", false)
			table.set("has_dishes", false)
	
	print("[EnvironmentStateManager] Environment reset")
