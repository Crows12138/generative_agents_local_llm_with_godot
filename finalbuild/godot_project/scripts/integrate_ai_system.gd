extends Node
# AI System Integration for Existing Cozy Bar Scene
# Add this to your bar_server_client_fixed.gd or run separately

static func integrate_ai_to_scene(scene_root: Node2D):
	"""Integrate AI decision system to existing NPCs"""
	
	print("[AI Integration] Starting integration...")
	
	# 1. First create missing environment objects if needed
	_create_environment_objects(scene_root)
	
	# 2. Add AI components to each NPC
	_setup_bob_ai(scene_root)
	_setup_alice_ai(scene_root)
	_setup_sam_ai(scene_root)
	
	# 3. Create location markers
	_create_location_markers(scene_root)
	
	# 4. Add Environment State Manager for automatic state changes
	_create_environment_manager(scene_root)
	
	# 5. Customer Manager DISABLED - No automatic customers
	# _create_customer_manager(scene_root)
	
	# 6. Add Invisible Clickable Environment System
	_create_clickable_environment(scene_root)
	
	print("[AI Integration] Complete!")
	print("Click objects with LEFT/RIGHT mouse buttons to change states!")

static func _create_environment_objects(scene_root: Node2D):
	"""Create missing bar objects for state detection"""
	
	# Create BarCounter if missing
	if not scene_root.has_node("BarCounter"):
		var bar_counter = StaticBody2D.new()
		bar_counter.name = "BarCounter"
		bar_counter.position = Vector2(640, 200)  # Center-top of screen (bar usually at back)
		scene_root.add_child(bar_counter)
		
		# Add properties script
		var counter_script = GDScript.new()
		counter_script.source_code = """
extends StaticBody2D

var cleanliness: float = 100.0
var has_customers: bool = false
var customer_count: int = 0

func get_cleanliness() -> float:
	return cleanliness

func set_cleanliness(value: float):
	cleanliness = clamp(value, 0, 100)

func is_dirty() -> bool:
	return cleanliness < 30

func customer_served():
	if customer_count > 0:
		customer_count -= 1
	if customer_count == 0:
		has_customers = false
"""
		bar_counter.set_script(counter_script)
		print("  Created BarCounter")
	
	# Create Tables if missing
	if not scene_root.has_node("Tables"):
		var tables_node = Node2D.new()
		tables_node.name = "Tables"
		scene_root.add_child(tables_node)
		
		# Create Table1
		var table1 = StaticBody2D.new()
		table1.name = "Table1"
		table1.position = Vector2(320, 400)  # Left side of room
		tables_node.add_child(table1)
		
		# Create Table2
		var table2 = StaticBody2D.new()
		table2.name = "Table2"
		table2.position = Vector2(640, 450)  # Center of room
		tables_node.add_child(table2)
		
		# Create Table3
		var table3 = StaticBody2D.new()
		table3.name = "Table3"
		table3.position = Vector2(960, 500)  # Right-bottom area
		tables_node.add_child(table3)
		
		# Add table script
		var table_script = GDScript.new()
		table_script.source_code = """
extends StaticBody2D

var cleanliness: float = 100.0
var occupied: bool = false
var has_dishes: bool = false

func get_cleanliness() -> float:
	return cleanliness

func needs_cleaning() -> bool:
	return cleanliness < 30 or has_dishes

func is_occupied() -> bool:
	return occupied

func has_dirty_dishes() -> bool:
	return has_dishes

func clear_table():
	has_dishes = false
	cleanliness = 100
"""
		table1.set_script(table_script)
		table2.set_script(table_script)
		table3.set_script(table_script)
		print("  Created Tables")
	
	# Create LiquorShelf if missing
	if not scene_root.has_node("LiquorShelf"):
		var shelf = StaticBody2D.new()
		shelf.name = "LiquorShelf"
		shelf.position = Vector2(640, 100)  # Behind bar counter
		scene_root.add_child(shelf)
		
		var shelf_script = GDScript.new()
		shelf_script.source_code = """
extends StaticBody2D

var stock_level: float = 100.0

func get_stock_level() -> float:
	return stock_level

func set_stock_level(value: float):
	stock_level = clamp(value, 0, 100)
"""
		shelf.set_script(shelf_script)
		print("  Created LiquorShelf")

static func _setup_bob_ai(scene_root: Node2D):
	"""Add AI components to Bob"""
	var bob = scene_root.get_node_or_null("bob")
	if not bob:
		push_error("Bob not found!")
		return
	
	# Check if components exist and fix them
	var existing_controller = bob.get_node_or_null("DecisionController")
	if existing_controller:
		print("  Bob has DecisionController, checking components...")
		
		# Ensure correct NPC name
		existing_controller.npc_name = "bob"
		
		# Ensure StateDetector exists
		var detector = existing_controller.get_node_or_null("StateDetector")
		if not detector:
			print("  Creating missing StateDetector for Bob")
			detector = Node.new()
			detector.name = "StateDetector"
			detector.set_script(load("res://scripts/state_detector.gd"))
			existing_controller.add_child(detector)
			
			# Configure detector
			detector.bar_counter = scene_root.get_node_or_null("BarCounter")
			detector.tables.clear()
			var tables_node = scene_root.get_node_or_null("Tables")
			if tables_node:
				for child in tables_node.get_children():
					detector.tables.append(child)
			detector.shelf = scene_root.get_node_or_null("LiquorShelf")
			detector.vision_range = 400.0
			detector.peripheral_vision_range = 600.0
			detector.peripheral_detection_chance = 0.7
		
		# Ensure ActionExecutor has NPC reference
		var executor = existing_controller.get_node_or_null("ActionExecutor")
		if not executor:
			executor = bob.get_node_or_null("ActionExecutor")
		if executor:
			print("  Linking ActionExecutor to Bob")
			executor.npc = bob
			executor.animation_player = bob.get_node_or_null("AnimatedSprite2D")
			
			# Set location markers for movement
			var markers = scene_root.get_node_or_null("LocationMarkers")
			if markers:
				executor.location_markers = {
					"bar_counter": markers.get_node("BarCounterMarker"),
					"shelf": markers.get_node("ShelfMarker"),
					"idle": markers.get_node("IdleMarker")
				}
		
		print("  Bob AI components fixed")
		return
	
	# Create DecisionController
	var controller = Node.new()
	controller.name = "DecisionController"
	controller.set_script(load("res://scripts/npc_decision_controller.gd"))
	bob.add_child(controller)
	
	# Create StateDetector
	var detector = Node.new()
	detector.name = "StateDetector"
	detector.set_script(load("res://scripts/state_detector.gd"))
	controller.add_child(detector)
	
	# Create ActionExecutor
	var executor = Node.new()
	executor.name = "ActionExecutor"
	executor.set_script(load("res://scripts/action_executor.gd"))
	controller.add_child(executor)
	
	# Configure components
	controller.npc_name = "bob"
	controller.decision_server_url = "ws://localhost:9998"
	
	# Link detector to scene objects
	detector.bar_counter = scene_root.get_node_or_null("BarCounter")
	detector.tables.clear()
	var tables_node = scene_root.get_node_or_null("Tables")
	if tables_node:
		for child in tables_node.get_children():
			detector.tables.append(child)
	detector.shelf = scene_root.get_node_or_null("LiquorShelf")
	
	# Bob has excellent vision as bartender (needs to see everything)
	detector.vision_range = 400.0
	detector.peripheral_vision_range = 600.0
	detector.peripheral_detection_chance = 0.7
	
	# Link executor to NPC
	executor.npc = bob
	executor.animation_player = bob.get_node_or_null("AnimatedSprite2D")
	
	print("  Bob AI configured")

static func _setup_alice_ai(scene_root: Node2D):
	"""Add AI components to Alice"""
	var alice = scene_root.get_node_or_null("Alice")
	if not alice:
		push_error("Alice not found!")
		return
	
	# Check if components exist and fix them
	var existing_controller = alice.get_node_or_null("DecisionController")
	if existing_controller:
		print("  Alice has DecisionController, checking components...")
		
		# Ensure correct NPC name
		existing_controller.npc_name = "alice"
		
		# Ensure StateDetector exists
		var detector = existing_controller.get_node_or_null("StateDetector")
		if not detector:
			print("  Creating missing StateDetector for Alice")
			detector = Node.new()
			detector.name = "StateDetector"
			detector.set_script(load("res://scripts/state_detector.gd"))
			existing_controller.add_child(detector)
			
			# Configure detector
			detector.bar_counter = scene_root.get_node_or_null("BarCounter")
			detector.tables.clear()
			var tables_node = scene_root.get_node_or_null("Tables")
			if tables_node:
				for child in tables_node.get_children():
					detector.tables.append(child)
			detector.shelf = scene_root.get_node_or_null("LiquorShelf")
			detector.vision_range = 300.0
			detector.peripheral_vision_range = 450.0
			detector.peripheral_detection_chance = 0.5
		
		# Ensure ActionExecutor has NPC reference
		var executor = existing_controller.get_node_or_null("ActionExecutor")
		if not executor:
			executor = alice.get_node_or_null("ActionExecutor")
		if executor:
			print("  Linking ActionExecutor to Alice")
			executor.npc = alice
			executor.animation_player = alice.get_node_or_null("AnimatedSprite2D")
			
			# Set location markers for movement
			var markers = scene_root.get_node_or_null("LocationMarkers")
			if markers:
				executor.location_markers = {
					"bar_counter": markers.get_node("BarCounterMarker"),
					"shelf": markers.get_node("ShelfMarker"),
					"idle": markers.get_node("IdleMarker")
				}
		
		print("  Alice AI components fixed")
		return
	
	# Create new DecisionController if none exists
	var controller = Node.new()
	controller.name = "DecisionController"
	controller.set_script(load("res://scripts/npc_decision_controller.gd"))
	alice.add_child(controller)
	
	var detector = Node.new()
	detector.name = "StateDetector"
	detector.set_script(load("res://scripts/state_detector.gd"))
	controller.add_child(detector)
	
	var executor = Node.new()
	executor.name = "ActionExecutor"
	executor.set_script(load("res://scripts/action_executor.gd"))
	controller.add_child(executor)
	
	controller.npc_name = "alice"
	controller.decision_server_url = "ws://localhost:9998"
	
	# Same detector links
	detector.bar_counter = scene_root.get_node_or_null("BarCounter")
	detector.tables.clear()
	var tables_node = scene_root.get_node_or_null("Tables")
	if tables_node:
		for child in tables_node.get_children():
			detector.tables.append(child)
	detector.shelf = scene_root.get_node_or_null("LiquorShelf")
	
	# Alice has moderate vision as a regular (familiar with the place)
	detector.vision_range = 300.0
	detector.peripheral_vision_range = 450.0
	detector.peripheral_detection_chance = 0.5
	
	executor.npc = alice
	executor.animation_player = alice.get_node_or_null("AnimatedSprite2D")
	
	print("  Alice AI configured")

static func _setup_sam_ai(scene_root: Node2D):
	"""Add AI components to Sam"""
	var sam = scene_root.get_node_or_null("sam")
	if not sam:
		push_error("Sam not found!")
		return
	
	# Check if components exist and fix them
	var existing_controller = sam.get_node_or_null("DecisionController")
	if existing_controller:
		print("  Sam has DecisionController, checking components...")
		
		# Ensure correct NPC name
		existing_controller.npc_name = "sam"
		
		# Ensure StateDetector exists
		var detector = existing_controller.get_node_or_null("StateDetector")
		if not detector:
			print("  Creating missing StateDetector for Sam")
			detector = Node.new()
			detector.name = "StateDetector"
			detector.set_script(load("res://scripts/state_detector.gd"))
			existing_controller.add_child(detector)
			
			# Configure detector
			detector.bar_counter = scene_root.get_node_or_null("BarCounter")
			detector.tables.clear()
			var tables_node = scene_root.get_node_or_null("Tables")
			if tables_node:
				for child in tables_node.get_children():
					detector.tables.append(child)
			detector.shelf = scene_root.get_node_or_null("LiquorShelf")
			detector.vision_range = 200.0
			detector.peripheral_vision_range = 350.0
			detector.peripheral_detection_chance = 0.3
		
		# Ensure ActionExecutor has NPC reference
		var executor = existing_controller.get_node_or_null("ActionExecutor")
		if not executor:
			executor = sam.get_node_or_null("ActionExecutor")
		if executor:
			print("  Linking ActionExecutor to Sam")
			executor.npc = sam
			executor.animation_player = sam.get_node_or_null("AnimatedSprite2D")
			
			# Set location markers for movement
			var markers = scene_root.get_node_or_null("LocationMarkers")
			if markers:
				executor.location_markers = {
					"bar_counter": markers.get_node("BarCounterMarker"),
					"shelf": markers.get_node("ShelfMarker"),
					"idle": markers.get_node("IdleMarker")
				}
		
		print("  Sam AI components fixed")
		return
		
	# Create new DecisionController if none exists
	var controller = Node.new()
	controller.name = "DecisionController"
	controller.set_script(load("res://scripts/npc_decision_controller.gd"))
	sam.add_child(controller)
	
	var detector = Node.new()
	detector.name = "StateDetector"
	detector.set_script(load("res://scripts/state_detector.gd"))
	controller.add_child(detector)
	
	var executor = Node.new()
	executor.name = "ActionExecutor"
	executor.set_script(load("res://scripts/action_executor.gd"))
	controller.add_child(executor)
	
	controller.npc_name = "sam"
	controller.decision_server_url = "ws://localhost:9998"
	
	# Detector links
	detector.bar_counter = scene_root.get_node_or_null("BarCounter")
	detector.tables.clear()
	var tables_node = scene_root.get_node_or_null("Tables")
	if tables_node:
		for child in tables_node.get_children():
			detector.tables.append(child)
	detector.shelf = scene_root.get_node_or_null("LiquorShelf")
	
	# Sam has limited vision as musician (focused on performance/music)
	detector.vision_range = 200.0
	detector.peripheral_vision_range = 350.0
	detector.peripheral_detection_chance = 0.3
	
	executor.npc = sam
	executor.animation_player = sam.get_node_or_null("AnimatedSprite2D")
	
	print("  Sam AI configured")

static func _create_location_markers(scene_root: Node2D):
	"""Create location markers for NPC movement"""
	
	if not scene_root.has_node("LocationMarkers"):
		var markers = Node2D.new()
		markers.name = "LocationMarkers"
		scene_root.add_child(markers)
		
		# Bar counter position (where NPCs go to serve)
		var bar_marker = Marker2D.new()
		bar_marker.name = "BarCounterMarker"
		bar_marker.position = Vector2(320, 151)  # Correct position from manual testing
		markers.add_child(bar_marker)
		
		# Shelf position (where NPCs go to restock)
		var shelf_marker = Marker2D.new()
		shelf_marker.name = "ShelfMarker"
		shelf_marker.position = Vector2(311, 87)  # Correct shelf position from manual testing
		markers.add_child(shelf_marker)
		
		# Default idle position (center of room)
		var idle_marker = Marker2D.new()
		idle_marker.name = "IdleMarker"
		idle_marker.position = Vector2(640, 360)  # Center
		markers.add_child(idle_marker)
		
		print("  Created LocationMarkers")
		
		# Update all NPCs with marker references
		_update_npc_markers(scene_root, markers)

static func _update_npc_markers(scene_root: Node2D, markers: Node2D):
	"""Update all NPCs with location marker references"""
	
	# Update Bob's executor
	var bob_executor = scene_root.get_node_or_null("bob/DecisionController/ActionExecutor")
	if not bob_executor:
		bob_executor = scene_root.get_node_or_null("bob/ActionExecutor")
	if bob_executor:
		bob_executor.location_markers = {
			"bar_counter": markers.get_node("BarCounterMarker"),
			"shelf": markers.get_node("ShelfMarker"),
			"idle": markers.get_node("IdleMarker")
		}
		print("  Set location markers for Bob's ActionExecutor")
	else:
		print("  WARNING: Bob's ActionExecutor not found!")
	
	# Update Alice's executor
	var alice_executor = scene_root.get_node_or_null("Alice/DecisionController/ActionExecutor")
	if not alice_executor:
		alice_executor = scene_root.get_node_or_null("Alice/ActionExecutor")
	if alice_executor:
		alice_executor.location_markers = {
			"bar_counter": markers.get_node("BarCounterMarker"),
			"shelf": markers.get_node("ShelfMarker"),
			"idle": markers.get_node("IdleMarker")
		}
		print("  Set location markers for Alice's ActionExecutor")
	else:
		print("  WARNING: Alice's ActionExecutor not found!")
	
	# Update Sam's executor
	var sam_executor = scene_root.get_node_or_null("sam/DecisionController/ActionExecutor")
	if not sam_executor:
		sam_executor = scene_root.get_node_or_null("sam/ActionExecutor")
	if sam_executor:
		sam_executor.location_markers = {
			"bar_counter": markers.get_node("BarCounterMarker"),
			"shelf": markers.get_node("ShelfMarker"),
			"idle": markers.get_node("IdleMarker")
		}
		print("  Set location markers for Sam's ActionExecutor")
	else:
		print("  WARNING: Sam's ActionExecutor not found!")

static func _create_environment_manager(scene_root: Node2D):
	"""Create and configure the Environment State Manager"""
	
	# Check if already exists
	if scene_root.has_node("EnvironmentStateManager"):
		print("  Environment State Manager already exists")
		return
	
	# Create the manager
	var manager = Node.new()
	manager.name = "EnvironmentStateManager"
	manager.set_script(load("res://scripts/environment_state_manager.gd"))
	scene_root.add_child(manager)
	
	# Configure settings for balanced gameplay
	manager.cleanliness_decay_rate = 2.0  # Gets dirty gradually
	manager.stock_consumption_rate = 1.5  # Stock depletes slowly
	manager.spill_chance_per_conversation = 0.15  # 15% chance per conversation
	manager.random_event_chance = 0.1  # 10% chance per minute
	
	print("  Created Environment State Manager")
	print("    - Time-based state changes enabled")
	print("    - Conversation-triggered spills (15% chance)")
	print("    - Random events enabled (10% per minute)")

static func _create_customer_manager(scene_root: Node2D):
	"""Create customer manager for service simulation"""
	
	# Check if already exists
	if scene_root.has_node("CustomerManager"):
		print("  Customer Manager already exists")
		return
	
	# Create the manager
	var manager = Node.new()
	manager.name = "CustomerManager"
	manager.set_script(load("res://scripts/customer_manager.gd"))
	scene_root.add_child(manager)
	
	# Configure settings
	manager.spawn_interval_min = 20.0  # Customer every 20-40 seconds
	manager.spawn_interval_max = 40.0
	manager.customer_patience = 60.0  # Wait 60 seconds before leaving
	manager.initial_delay = 5.0  # First customer after 5 seconds
	
	print("  Created Customer Manager")
	print("    - Customers arrive every 20-40 seconds")
	print("    - Customer patience: 60 seconds")
	print("    - Service locations: bar, 3 tables")

static func _create_clickable_environment(scene_root: Node2D):
	"""Create invisible clickable environment system"""
	
	# Check if already exists
	if scene_root.has_node("InvisibleClickHandler"):
		print("  Click Handler already exists")
		return
	
	# Create the invisible click handler
	var click_handler = Node.new()
	click_handler.name = "InvisibleClickHandler"
	click_handler.set_script(load("res://scripts/invisible_click_handler.gd"))
	scene_root.add_child(click_handler)
	
	print("  Created Invisible Click Handler")
	print("    - LEFT CLICK on Bar/Shelf: Make dirty / Deplete stock")
	print("    - RIGHT CLICK on Bar/Shelf: Clean / Restock")
	print("    - No visual indicators")
