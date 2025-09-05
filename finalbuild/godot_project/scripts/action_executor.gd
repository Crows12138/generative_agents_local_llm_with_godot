extends Node
# Action Executor - Manages action queue and smooth execution
# Receives decisions from Python and executes them in Godot

signal action_started(action_name: String)
signal action_completed(action_name: String, result: String)
signal queue_updated(queue_size: int)

# Action buffer with priority queue
var action_buffer: Array = []
var is_executing: bool = false
var current_action: Dictionary = {}

# NPC reference
@export var npc: CharacterBody2D
@export var animation_player: AnimatedSprite2D

# Action timing
@export var default_action_duration: float = 2.0
var action_durations = {
	"serve_customer": 3.0,
	"clean_counter": 2.5,
	"clear_table": 2.0,
	"restock": 4.0,
	"observe": 1.0,
	"take_break": 5.0
}

# Movement parameters
@export var movement_speed: float = 50.0
@export var arrival_threshold: float = 10.0

# Target locations for actions
@export var location_markers: Dictionary = {}  # Set in editor

# Thought bubble UI
var thought_bubble: Control
var thought_label: Label

func _ready():
	print("[ActionExecutor] Ready for NPC: ", npc.name if npc else "Unknown")
	
	# Auto-find animation player if not set
	if not animation_player and npc:
		animation_player = npc.get_node("AnimatedSprite2D")
	
	# Create thought bubble UI
	_create_thought_bubble()

func queue_action(action: Dictionary):
	"""Add action to priority queue"""
	var priority = action.get("priority", 0)
	var inserted = false
	
	# Insert by priority (higher priority first)
	for i in range(action_buffer.size()):
		if action_buffer[i].get("priority", 0) < priority:
			action_buffer.insert(i, action)
			inserted = true
			break
	
	if not inserted:
		action_buffer.append(action)
	
	queue_updated.emit(action_buffer.size())
	print("[ActionExecutor] Queued action: ", action.action, " (priority: ", priority, ")")
	
	# Start execution if not busy
	if not is_executing:
		_execute_next_action()

func clear_queue():
	"""Clear all pending actions"""
	action_buffer.clear()
	queue_updated.emit(0)

func interrupt_current_action():
	"""Stop current action and move to next"""
	is_executing = false
	_execute_next_action()

func _execute_next_action():
	"""Execute the next action in queue"""
	if action_buffer.is_empty():
		is_executing = false
		_hide_thought()
		return
	
	is_executing = true
	current_action = action_buffer.pop_front()
	queue_updated.emit(action_buffer.size())
	
	action_started.emit(current_action.action)
	print("[ActionExecutor] Executing: ", current_action.action)
	
	# Show thought bubble with action
	_show_thought(_get_action_display_text(current_action.action))
	
	# Route to appropriate handler
	match current_action.action:
		"serve_customer":
			await _do_serve_customer()
		"clean_counter":
			await _do_clean_counter()
		"clear_table":
			await _do_clear_table()
		"restock":
			await _do_restock()
		"observe":
			await _do_observe()
		"take_break":
			await _do_take_break()
		_:
			print("[ActionExecutor] Unknown action: ", current_action.action)
			await _on_action_complete("unknown")

# Action implementations
func _do_serve_customer():
	"""Serve customers at the bar or tables"""
	# Check customer manager for waiting customers
	var customer_manager = get_node_or_null("/root/CozyBar/CustomerManager")
	if customer_manager:
		var most_urgent = customer_manager.get_most_urgent_customer()
		if most_urgent != "":
			print("[ActionExecutor] Serving customer at ", most_urgent)
			
			# Move to appropriate location
			var target_location = "bar_counter" if most_urgent == "bar" else most_urgent
			
			# For tables, move to the table position
			if most_urgent.begins_with("table"):
				var table_num = most_urgent.trim_prefix("table")
				var table = get_node_or_null("/root/CozyBar/Tables/Table" + table_num)
				if table:
					if await _move_to_position(table.global_position):
						# Play serving animation
						if animation_player:
							animation_player.play("serve")
						
						await get_tree().create_timer(action_durations.get("serve_customer", 3.0)).timeout
						
						# Actually serve the customer
						if customer_manager.serve_customer(most_urgent):
							await _on_action_complete("success")
						else:
							await _on_action_complete("customer_left")
					else:
						await _on_action_complete("failed_movement")
			else:
				# Bar service
				if await _move_to_location("bar_counter"):
					# Play serving animation
					if animation_player:
						animation_player.play("serve")
					
					await get_tree().create_timer(action_durations.get("serve_customer", 3.0)).timeout
					
					# Actually serve the customer
					if customer_manager.serve_customer("bar"):
						await _on_action_complete("success")
					else:
						_on_action_complete("customer_left")
				else:
					await _on_action_complete("failed_movement")
			return
	
	# Fallback: No customer manager or no customers
	print("[ActionExecutor] No customers to serve")
	await _on_action_complete("no_customers")

func _do_clean_counter():
	"""Clean the bar counter"""
	# Move to counter
	if await _move_to_location("bar_counter"):
		# Play cleaning animation
		if animation_player:
			animation_player.play("clean")
		
		await get_tree().create_timer(action_durations.get("clean_counter", 2.5)).timeout
		
		# Update counter cleanliness
		var counter = get_node_or_null("/root/CozyBar/BarCounter")
		if counter and counter.has_method("set_cleanliness"):
			counter.set_cleanliness(100)
		
		await _on_action_complete("success")
	else:
		await _on_action_complete("failed_movement")

func _do_clear_table():
	"""Clear and clean a dirty table"""
	# Find nearest dirty table
	var target_table = _find_nearest_dirty_table()
	
	if target_table:
		# Move to table
		if await _move_to_position(target_table.global_position):
			# Play cleaning animation
			if animation_player:
				animation_player.play("clean")
			
			await get_tree().create_timer(action_durations.get("clear_table", 2.0)).timeout
			
			# Clear table
			if target_table.has_method("clear_table"):
				target_table.clear_table()
			
			await _on_action_complete("success")
		else:
			await _on_action_complete("failed_movement")
	else:
		await _on_action_complete("no_target")

func _do_restock():
	"""Restock the shelf"""
	# Move directly to shelf (no storage room)
	if await _move_to_location("shelf"):
		# Play restock animation
		if animation_player:
			animation_player.play("restock")
		
		await get_tree().create_timer(action_durations.get("restock", 4.0)).timeout
		
		# Update shelf stock
		var shelf = get_node_or_null("/root/CozyBar/LiquorShelf")
		if shelf and shelf.has_method("set_stock_level"):
			shelf.set_stock_level(100)
		
		await _on_action_complete("success")
	else:
		await _on_action_complete("failed_movement")

func _do_observe():
	"""Look around and observe the environment"""
	# Simple observation - just look around
	if animation_player:
		animation_player.play("idle")
	
	# Maybe rotate to look around
	if npc:
		var original_rotation = npc.rotation
		
		# Look left
		var tween = create_tween()
		tween.tween_property(npc, "rotation", original_rotation - PI/4, 0.3)
		await tween.finished
		
		await get_tree().create_timer(0.3).timeout
		
		# Look right
		tween = create_tween()
		tween.tween_property(npc, "rotation", original_rotation + PI/4, 0.3)
		await tween.finished
		
		await get_tree().create_timer(0.3).timeout
		
		# Return to center
		tween = create_tween()
		tween.tween_property(npc, "rotation", original_rotation, 0.3)
		await tween.finished
	
	await _on_action_complete("success")

func _do_take_break():
	"""Take a short break"""
	# Move to idle position
	if location_markers.has("idle"):
		await _move_to_location("idle")
	
	# Play rest animation
	if animation_player:
		animation_player.play("sit")
	
	await get_tree().create_timer(action_durations.get("take_break", 5.0)).timeout
	
	# Stand up
	if animation_player:
		animation_player.play("idle")
	
	await _on_action_complete("success")

# Movement helpers
func _move_to_location(location_name: String) -> bool:
	"""Move NPC to a named location"""
	if not location_markers.has(location_name):
		print("[ActionExecutor] Unknown location: ", location_name)
		print("[ActionExecutor] Available locations: ", location_markers.keys())
		print("[ActionExecutor] location_markers is empty? ", location_markers.is_empty())
		return false
	
	var target_pos = location_markers[location_name].global_position
	return await _move_to_position(target_pos)

func _move_to_position(target_pos: Vector2) -> bool:
	"""Move NPC to specific position"""
	if not npc:
		return false
	
	# Play walk animation
	if animation_player:
		animation_player.play("walk")
	
	# Smooth movement with acceleration
	var acceleration = 0.1
	var current_velocity = Vector2.ZERO
	
	# Simple movement - could be replaced with pathfinding
	while npc.global_position.distance_to(target_pos) > arrival_threshold:
		var direction = (target_pos - npc.global_position).normalized()
		var desired_velocity = direction * movement_speed
		
		# Smooth velocity transition
		current_velocity = current_velocity.lerp(desired_velocity, acceleration)
		npc.velocity = current_velocity
		
		# Update animation direction
		_update_movement_animation(direction)
		
		npc.move_and_slide()
		await get_tree().process_frame
	
	# Stop and play idle
	npc.velocity = Vector2.ZERO
	if animation_player:
		animation_player.play("idle")
	
	return true

func _update_movement_animation(direction: Vector2):
	"""Update animation based on movement direction"""
	if not animation_player:
		return
	
	# Debug output disabled
	# if npc and npc.name.to_lower() == "alice":
	#	print("[", npc.name, "] Direction: (", direction.x, ", ", direction.y, ")")
	
	# Determine primary direction - same for all NPCs
	if abs(direction.x) > abs(direction.y):
		# Horizontal movement
		if direction.x > 0:
			animation_player.play("walk_right")
		else:
			animation_player.play("walk_left")
	else:
		# Vertical movement
		if direction.y > 0:
			animation_player.play("walk_down")
		else:
			animation_player.play("walk_up")

func _find_nearest_dirty_table() -> Node2D:
	"""Find the nearest table that needs cleaning"""
	var tables = get_tree().get_nodes_in_group("tables")
	var nearest_table = null
	var nearest_distance = INF
	
	for table in tables:
		if table.has_method("needs_cleaning") and table.needs_cleaning():
			var distance = npc.global_position.distance_to(table.global_position)
			if distance < nearest_distance:
				nearest_distance = distance
				nearest_table = table
	
	return nearest_table

func _on_action_complete(result: String):
	"""Called when action finishes"""
	action_completed.emit(current_action.action, result)
	print("[ActionExecutor] Completed: ", current_action.action, " - ", result)
	
	# Hide thought bubble
	_hide_thought()
	
	# Clear current action
	current_action = {}
	is_executing = false
	
	# Execute next action if available
	if not action_buffer.is_empty():
		# Small delay between actions
		await get_tree().create_timer(0.5).timeout
		call_deferred("_execute_next_action")

# Debug functions
func get_queue_info() -> Array:
	"""Get information about queued actions"""
	var info = []
	for action in action_buffer:
		info.append({
			"action": action.action,
			"priority": action.get("priority", 0)
		})
	return info

func is_busy() -> bool:
	"""Check if currently executing an action"""
	return is_executing

# Thought Bubble UI System
func _create_thought_bubble():
	"""Create thought bubble UI above NPC"""
	if not npc:
		return
	
	# Create main container
	thought_bubble = Control.new()
	thought_bubble.name = "ThoughtBubble"
	thought_bubble.set_anchors_and_offsets_preset(Control.PRESET_CENTER)
	thought_bubble.visible = false
	
	# Create background panel
	var panel = Panel.new()
	panel.size = Vector2(100, 40)
	panel.position = Vector2(-50, -80)
	
	# Create rounded style
	var style = StyleBoxFlat.new()
	style.bg_color = Color(1, 1, 1, 0.95)
	style.border_color = Color(0.3, 0.3, 0.3, 1)
	style.border_width_bottom = 1
	style.border_width_left = 1
	style.border_width_right = 1
	style.border_width_top = 1
	style.corner_radius_bottom_left = 20
	style.corner_radius_bottom_right = 20
	style.corner_radius_top_left = 20
	style.corner_radius_top_right = 20
	panel.add_theme_stylebox_override("panel", style)
	
	# Create text label
	thought_label = Label.new()
	thought_label.text = "Thinking..."
	thought_label.horizontal_alignment = HORIZONTAL_ALIGNMENT_CENTER
	thought_label.vertical_alignment = VERTICAL_ALIGNMENT_CENTER
	thought_label.size = Vector2(100, 40)
	thought_label.position = Vector2(-50, -80)
	thought_label.add_theme_color_override("font_color", Color.BLACK)
	thought_label.add_theme_font_size_override("font_size", 12)
	
	# Add bubble tail (thought bubble dots)
	for i in range(3):
		var dot = Panel.new()
		dot.size = Vector2(8 - i * 2, 8 - i * 2)
		dot.position = Vector2(-4 + i * 8, -50 + i * 8)
		var dot_style = StyleBoxFlat.new()
		dot_style.bg_color = Color(1, 1, 1, 0.95 - i * 0.2)
		dot_style.border_color = Color(0.3, 0.3, 0.3, 1 - i * 0.3)
		dot_style.border_width_bottom = 1
		dot_style.border_width_left = 1
		dot_style.border_width_right = 1
		dot_style.border_width_top = 1
		dot_style.corner_radius_bottom_left = 4 - i
		dot_style.corner_radius_bottom_right = 4 - i
		dot_style.corner_radius_top_left = 4 - i
		dot_style.corner_radius_top_right = 4 - i
		dot.add_theme_stylebox_override("panel", dot_style)
		thought_bubble.add_child(dot)
	
	# Add to NPC
	thought_bubble.add_child(panel)
	thought_bubble.add_child(thought_label)
	npc.add_child(thought_bubble)
	
	print("[ActionExecutor] Created thought bubble for ", npc.name)

func _show_thought(text: String):
	"""Show thought bubble with text"""
	# Create thought bubble if not exists (lazy creation)
	if not thought_bubble and npc:
		_create_thought_bubble()
	
	if thought_bubble and thought_label:
		thought_label.text = text
		thought_bubble.visible = true
		print("[ActionExecutor] Showing thought: ", text)

func _hide_thought():
	"""Hide thought bubble"""
	if thought_bubble:
		thought_bubble.visible = false

func _get_action_display_text(action: String) -> String:
	"""Get user-friendly display text for actions"""
	match action:
		"serve_customer":
			return "Serving"
		"clean_counter":
			return "Cleaning"
		"clear_table":
			return "Clearing"
		"restock":
			return "Restocking"
		"observe":
			return "Looking"
		"take_break":
			return "Resting"
		_:
			return "Thinking"
