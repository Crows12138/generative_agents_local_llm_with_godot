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
	
	print("[AI Integration] Complete!")

static func _create_environment_objects(scene_root: Node2D):
	"""Create missing bar objects for state detection"""
	
	# Create BarCounter if missing
	if not scene_root.has_node("BarCounter"):
		var bar_counter = StaticBody2D.new()
		bar_counter.name = "BarCounter"
		bar_counter.position = Vector2(400, 200)  # Adjust position
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
		table1.position = Vector2(200, 300)  # Adjust position
		tables_node.add_child(table1)
		
		# Create Table2
		var table2 = StaticBody2D.new()
		table2.name = "Table2"
		table2.position = Vector2(600, 300)  # Adjust position
		tables_node.add_child(table2)
		
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
		print("  Created Tables")
	
	# Create LiquorShelf if missing
	if not scene_root.has_node("LiquorShelf"):
		var shelf = StaticBody2D.new()
		shelf.name = "LiquorShelf"
		shelf.position = Vector2(400, 100)  # Adjust position
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
	
	# Skip if already has AI
	if bob.has_node("DecisionController"):
		print("  Bob already has AI")
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
		
	if alice.has_node("DecisionController"):
		print("  Alice already has AI")
		return
	
	# Similar setup as Bob
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
	
	executor.npc = alice
	executor.animation_player = alice.get_node_or_null("AnimatedSprite2D")
	
	print("  Alice AI configured")

static func _setup_sam_ai(scene_root: Node2D):
	"""Add AI components to Sam"""
	var sam = scene_root.get_node_or_null("sam")
	if not sam:
		push_error("Sam not found!")
		return
	
	if sam.has_node("DecisionController"):
		print("  Sam already has AI")
		return
		
	# Similar setup
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
	
	executor.npc = sam
	executor.animation_player = sam.get_node_or_null("AnimatedSprite2D")
	
	print("  Sam AI configured")

static func _create_location_markers(scene_root: Node2D):
	"""Create location markers for NPC movement"""
	
	if not scene_root.has_node("LocationMarkers"):
		var markers = Node2D.new()
		markers.name = "LocationMarkers"
		scene_root.add_child(markers)
		
		# Bar counter position
		var bar_marker = Marker2D.new()
		bar_marker.name = "BarCounterMarker"
		bar_marker.position = Vector2(400, 250)  # Near counter
		markers.add_child(bar_marker)
		
		# Storage area
		var storage_marker = Marker2D.new()
		storage_marker.name = "StorageMarker"
		storage_marker.position = Vector2(100, 100)  # Corner
		markers.add_child(storage_marker)
		
		# Shelf position
		var shelf_marker = Marker2D.new()
		shelf_marker.name = "ShelfMarker"
		shelf_marker.position = Vector2(400, 150)  # Near shelf
		markers.add_child(shelf_marker)
		
		# Break area
		var break_marker = Marker2D.new()
		break_marker.name = "BreakAreaMarker"
		break_marker.position = Vector2(700, 400)  # Quiet corner
		markers.add_child(break_marker)
		
		print("  Created LocationMarkers")
		
		# Update all NPCs with marker references
		_update_npc_markers(scene_root, markers)

static func _update_npc_markers(scene_root: Node2D, markers: Node2D):
	"""Update all NPCs with location marker references"""
	
	# Update Bob's executor
	var bob_executor = scene_root.get_node_or_null("bob/DecisionController/ActionExecutor")
	if bob_executor:
		bob_executor.location_markers = {
			"bar_counter": markers.get_node("BarCounterMarker"),
			"storage": markers.get_node("StorageMarker"),
			"shelf": markers.get_node("ShelfMarker"),
			"break_area": markers.get_node("BreakAreaMarker")
		}
	
	# Update Alice's executor
	var alice_executor = scene_root.get_node_or_null("Alice/DecisionController/ActionExecutor")
	if alice_executor:
		alice_executor.location_markers = {
			"bar_counter": markers.get_node("BarCounterMarker"),
			"storage": markers.get_node("StorageMarker"),
			"shelf": markers.get_node("ShelfMarker"),
			"break_area": markers.get_node("BreakAreaMarker")
		}
	
	# Update Sam's executor
	var sam_executor = scene_root.get_node_or_null("sam/DecisionController/ActionExecutor")
	if sam_executor:
		sam_executor.location_markers = {
			"bar_counter": markers.get_node("BarCounterMarker"),
			"storage": markers.get_node("StorageMarker"),
			"shelf": markers.get_node("ShelfMarker"),
			"break_area": markers.get_node("BreakAreaMarker")
		}
