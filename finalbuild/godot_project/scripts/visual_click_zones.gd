extends Node2D
# Visual Click Zones - Shows clickable areas with colored circles
# You can drag these markers to adjust positions

class_name VisualClickZones

signal zone_clicked(zone_name: String, button_index: int)

# Visual markers for click zones
var zones = {}
var dragging_zone = null
var drag_offset = Vector2.ZERO

func _ready():
	print("[VisualClickZones] Creating visible click zones...")
	z_index = 10  # Make sure it's on top
	
	# Create visual click zones at positions you found
	# Bar counter as 3 segments, shelf as thin horizontal rectangle
	_create_bar_counter_zone(Vector2(318, 185))  # Your latest position
	_create_shelf_zone(Vector2(315, 49))  # Create horizontal shelf
	
	print("DRAG the colored areas to adjust positions!")
	print("LEFT CLICK = Dirty/Deplete, RIGHT CLICK = Clean/Restock")

func _create_bar_counter_zone(pos: Vector2):
	"""Create a multi-segment bar counter zone (L or C shape)"""
	var zone = Node2D.new()
	zone.name = "BarCounter"
	zone.position = pos
	add_child(zone)
	
	# Create 3 connected segments to form the bar shape
	# Based on the screenshot, the bar has a horizontal bottom part and vertical sides
	
	# Bottom horizontal segment (main bar)
	var rect1 = ColorRect.new()
	rect1.name = "Visual1"
	rect1.size = Vector2(160, 35)  # Shorter horizontal bar to bring verticals closer
	rect1.position = Vector2(-80, -17)  
	rect1.color = Color.CYAN
	rect1.color.a = 0.5
	rect1.mouse_filter = Control.MOUSE_FILTER_PASS
	zone.add_child(rect1)
	
	# Left vertical segment
	var rect2 = ColorRect.new()
	rect2.name = "Visual2"
	rect2.size = Vector2(35, 80)  # Vertical bar going up
	rect2.position = Vector2(-80, -97)  # Moved closer to center
	rect2.color = Color.CYAN
	rect2.color.a = 0.5
	rect2.mouse_filter = Control.MOUSE_FILTER_PASS
	zone.add_child(rect2)
	
	# Right vertical segment
	var rect3 = ColorRect.new()
	rect3.name = "Visual3"
	rect3.size = Vector2(35, 80)  # Vertical bar going up
	rect3.position = Vector2(45, -97)  # Moved closer to center (was 85)
	rect3.color = Color.CYAN
	rect3.color.a = 0.5
	rect3.mouse_filter = Control.MOUSE_FILTER_PASS
	zone.add_child(rect3)
	
	# Add label in the center
	var label = Label.new()
	label.name = "Label"
	label.text = "Bar Counter (3 segments)"
	label.add_theme_font_size_override("font_size", 12)
	label.add_theme_color_override("font_color", Color.WHITE)
	label.add_theme_color_override("font_shadow_color", Color.BLACK)
	label.position = Vector2(-60, 25)  # Below the bar
	zone.add_child(label)
	
	# Store reference
	zones["BarCounter"] = zone
	
	print("  Created bar counter zone (3 connected segments) at ", pos)

func _create_shelf_zone(pos: Vector2):
	"""Create a thin horizontal rectangular zone for the liquor shelf"""
	var zone = Node2D.new()
	zone.name = "LiquorShelf"
	zone.position = pos
	add_child(zone)
	
	# Create thin horizontal rectangle like in the screenshot
	var rect = ColorRect.new()
	rect.name = "Visual"
	rect.size = Vector2(120, 25)  # Shorter and thin like a shelf
	rect.position = Vector2(-60, -12)  # Center it
	rect.color = Color.BLUE
	rect.color.a = 0.5
	rect.mouse_filter = Control.MOUSE_FILTER_PASS
	zone.add_child(rect)
	
	# Add label
	var label = Label.new()
	label.name = "Label"
	label.text = "Liquor Shelf"
	label.add_theme_font_size_override("font_size", 12)
	label.add_theme_color_override("font_color", Color.WHITE)
	label.add_theme_color_override("font_shadow_color", Color.BLACK)
	label.position = Vector2(-40, -30)
	zone.add_child(label)
	
	# Store reference
	zones["LiquorShelf"] = zone
	
	print("  Created liquor shelf zone (thin horizontal) at ", pos)

func _create_zone(zone_name: String, pos: Vector2, color: Color):
	"""Create a visual click zone"""
	var zone = Node2D.new()
	zone.name = zone_name
	zone.position = pos
	add_child(zone)
	
	# Create visible circle
	var circle = ColorRect.new()
	circle.name = "Visual"
	circle.size = Vector2(80, 80)
	circle.position = Vector2(-40, -40)  # Center it
	circle.color = color
	circle.color.a = 0.5  # Semi-transparent
	circle.mouse_filter = Control.MOUSE_FILTER_PASS
	zone.add_child(circle)
	
	# Add label
	var label = Label.new()
	label.name = "Label"
	label.text = zone_name
	label.add_theme_font_size_override("font_size", 12)
	label.add_theme_color_override("font_color", Color.WHITE)
	label.add_theme_color_override("font_shadow_color", Color.BLACK)
	label.position = Vector2(-40, -50)
	zone.add_child(label)
	
	# Store reference
	zones[zone_name] = zone
	
	print("  Created zone: %s at %s" % [zone_name, pos])

func _input(event):
	if event is InputEventMouseButton:
		var mouse_pos = get_global_mouse_position()
		
		if event.pressed:
			# Check which zone was clicked
			for zone_name in zones:
				var zone = zones[zone_name]
				var clicked = false
				
				# Different detection for bar counter (3 segments) vs shelf (circle)
				if zone_name == "BarCounter":
					# Check if click is within any of the 3 bar segments
					var relative_pos = mouse_pos - zone.global_position
					
					# Check bottom horizontal bar (narrower now)
					var in_bottom = (abs(relative_pos.x) < 80 and abs(relative_pos.y) < 17)
					# Check left vertical bar (closer to center)
					var in_left = (relative_pos.x >= -80 and relative_pos.x <= -45 and 
								   relative_pos.y >= -97 and relative_pos.y <= -17)
					# Check right vertical bar (closer to center)
					var in_right = (relative_pos.x >= 45 and relative_pos.x <= 80 and
									relative_pos.y >= -97 and relative_pos.y <= -17)
					
					clicked = in_bottom or in_left or in_right
				elif zone_name == "LiquorShelf":
					# Rectangle detection for shelf (thin horizontal, shorter now)
					var relative_pos = mouse_pos - zone.global_position
					clicked = (abs(relative_pos.x) < 60 and abs(relative_pos.y) < 12)
				else:
					# Default circle detection for any other zones
					var distance = zone.global_position.distance_to(mouse_pos)
					clicked = distance < 40
				
				if clicked:
					if event.button_index == MOUSE_BUTTON_MIDDLE:
						# Middle click to start dragging
						dragging_zone = zone
						drag_offset = zone.global_position - mouse_pos
						print("Dragging %s" % zone_name)
					else:
						# Left or right click to activate
						_handle_zone_click(zone_name, event.button_index)
					return
		else:
			# Stop dragging
			if dragging_zone:
				print("Placed %s at %s" % [dragging_zone.name, dragging_zone.global_position])
				dragging_zone = null
	
	elif event is InputEventMouseMotion:
		# Handle dragging
		if dragging_zone:
			dragging_zone.global_position = get_global_mouse_position() + drag_offset

func _handle_zone_click(zone_name: String, button_index: int):
	"""Handle clicks on zones"""
	print("[%s] %s clicked" % [
		"LEFT" if button_index == MOUSE_BUTTON_LEFT else "RIGHT",
		zone_name
	])
	
	# Find the actual game object
	var target_obj = null
	
	if zone_name == "BarCounter":
		target_obj = get_node_or_null("/root/CozyBar/BarCounter")
		if target_obj:
			if button_index == MOUSE_BUTTON_LEFT:
				# Make dirty
				if target_obj.has_method("set_cleanliness"):
					var current = target_obj.get("cleanliness")
					if current != null:
						target_obj.set_cleanliness(max(0, current - 30))
						_show_effect(zones[zone_name], "Dirty: %d%%" % target_obj.get("cleanliness"))
			else:
				# Clean
				if target_obj.has_method("set_cleanliness"):
					target_obj.set_cleanliness(100)
					_show_effect(zones[zone_name], "Clean: 100%")
	
	elif zone_name == "LiquorShelf":
		target_obj = get_node_or_null("/root/CozyBar/LiquorShelf")
		if target_obj:
			if button_index == MOUSE_BUTTON_LEFT:
				# Deplete stock
				if target_obj.has_method("set_stock_level"):
					var current = target_obj.get("stock_level")
					if current != null:
						target_obj.set_stock_level(max(0, current - 25))
						var new_level = target_obj.get("stock_level")
						if new_level != null:
							_show_effect(zones[zone_name], "Stock: %d%%" % new_level)
						else:
							_show_effect(zones[zone_name], "Stock reduced")
			else:
				# Restock
				if target_obj.has_method("set_stock_level"):
					target_obj.set_stock_level(100)
					_show_effect(zones[zone_name], "Stock: 100%")
	
	
	zone_clicked.emit(zone_name, button_index)

func _show_effect(zone: Node2D, text: String):
	"""Show visual feedback"""
	# Flash all visual segments (for multi-part bar counter)
	if zone.name == "BarCounter":
		# Flash all 3 segments
		for i in range(1, 4):
			var visual = zone.get_node_or_null("Visual" + str(i))
			if visual:
				var original_color = visual.color
				visual.color = Color.WHITE
				var tween = create_tween()
				tween.tween_property(visual, "color", original_color, 0.3)
	else:
		# Flash single visual (shelf)
		var visual = zone.get_node("Visual")
		if visual:
			var original_color = visual.color
			visual.color = Color.WHITE
			var tween = create_tween()
			tween.tween_property(visual, "color", original_color, 0.3)
	
	# Show floating text
	var label = Label.new()
	label.text = text
	label.add_theme_font_size_override("font_size", 16)
	label.add_theme_color_override("font_color", Color.YELLOW)
	zone.add_child(label)
	label.position = Vector2(-30, -70)
	
	var tween = create_tween()
	tween.parallel().tween_property(label, "position:y", label.position.y - 20, 0.5)
	tween.parallel().tween_property(label, "modulate:a", 0, 0.5)
	tween.tween_callback(label.queue_free)

func get_zone_positions() -> Dictionary:
	"""Get current positions of all zones"""
	var positions = {}
	for zone_name in zones:
		positions[zone_name] = zones[zone_name].global_position
	return positions

func _draw():
	"""Draw connection lines to help positioning"""
	# Draw grid for reference
	var grid_color = Color(1, 1, 1, 0.1)
	for x in range(0, 1280, 100):
		draw_line(Vector2(x, 0), Vector2(x, 720), grid_color)
	for y in range(0, 720, 100):
		draw_line(Vector2(0, y), Vector2(1280, y), grid_color)