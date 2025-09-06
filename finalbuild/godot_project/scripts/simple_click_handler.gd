extends Node
# Simple Click Handler - Direct input detection
# Handles clicks on environment objects without Area2D

signal environment_clicked(object_name: String, action: String)

var bar_counter: Node
var shelf: Node
var tables: Array = []

# Click detection radius
var click_radius: float = 50.0

func _ready():
	print("[SimpleClickHandler] Ready - Click near objects to change state")
	set_process_unhandled_input(true)
	
	# Find environment objects
	_find_objects()

func _find_objects():
	"""Find all clickable objects in scene"""
	bar_counter = get_node_or_null("/root/CozyBar/BarCounter")
	shelf = get_node_or_null("/root/CozyBar/LiquorShelf")
	
	var tables_node = get_node_or_null("/root/CozyBar/Tables")
	if tables_node:
		tables = tables_node.get_children()
	
	print("  Found objects:")
	if bar_counter:
		print("    - Bar Counter at ", bar_counter.global_position)
	if shelf:
		print("    - Shelf at ", shelf.global_position)
	print("    - Tables: ", tables.size())

func _unhandled_input(event: InputEvent):
	"""Handle mouse clicks directly"""
	if not event is InputEventMouseButton:
		return
	if not event.pressed:
		return
		
	# Get click position in world coordinates
	var click_pos = get_viewport().get_camera_2d().get_global_mouse_position() if get_viewport().get_camera_2d() else event.position
	
	# Check what was clicked
	var clicked_object = _get_object_at_position(click_pos)
	
	if clicked_object == null:
		return
	
	# Handle the click based on button
	if event.button_index == MOUSE_BUTTON_LEFT:
		_handle_left_click(clicked_object)
	elif event.button_index == MOUSE_BUTTON_RIGHT:
		_handle_right_click(clicked_object)

func _get_object_at_position(pos: Vector2) -> Node:
	"""Find which object is at the clicked position"""
	
	# Check bar counter
	if bar_counter and bar_counter.global_position.distance_to(pos) < click_radius:
		return bar_counter
	
	# Check shelf
	if shelf and shelf.global_position.distance_to(pos) < click_radius:
		return shelf
	
	# Check tables
	for table in tables:
		if table and table.global_position.distance_to(pos) < click_radius:
			return table
	
	return null

func _handle_left_click(obj: Node):
	"""Handle left click - make dirty/deplete"""
	var obj_name = obj.name
	
	if obj_name == "BarCounter":
		if obj.has_method("set_cleanliness"):
			var current = obj.get("cleanliness")
			if current != null:
				obj.set_cleanliness(max(0, current - 30))
				print("[LeftClick] Bar dirtied: ", obj.get("cleanliness"), "%")
				_show_feedback(obj, "Dirty: " + str(obj.get("cleanliness")) + "%")
				
	elif obj_name == "LiquorShelf":
		if obj.has_method("set_stock_level"):
			var current = obj.get("stock_level")
			if current != null:
				obj.set_stock_level(max(0, current - 25))
				print("[LeftClick] Stock reduced: ", obj.get("stock_level"), "%")
				_show_feedback(obj, "Stock: " + str(obj.get("stock_level")) + "%")
				
	elif obj_name.begins_with("Table"):
		if obj.has_method("set"):
			var current_clean = obj.get("cleanliness")
			if current_clean != null:
				obj.set("cleanliness", max(0, current_clean - 40))
			obj.set("has_dishes", true)
			print("[LeftClick] ", obj_name, " dirtied with dishes")
			_show_feedback(obj, "Dirty with dishes")
	
	environment_clicked.emit(obj_name, "make_dirty")

func _handle_right_click(obj: Node):
	"""Handle right click - clean/restock"""
	var obj_name = obj.name
	
	if obj_name == "BarCounter":
		if obj.has_method("set_cleanliness"):
			obj.set_cleanliness(100)
			print("[RightClick] Bar cleaned: 100%")
			_show_feedback(obj, "Clean: 100%")
			
	elif obj_name == "LiquorShelf":
		if obj.has_method("set_stock_level"):
			obj.set_stock_level(100)
			print("[RightClick] Shelf restocked: 100%")
			_show_feedback(obj, "Stock: 100%")
			
	elif obj_name.begins_with("Table"):
		if obj.has_method("clear_table"):
			obj.clear_table()
		elif obj.has_method("set"):
			obj.set("cleanliness", 100)
			obj.set("has_dishes", false)
		print("[RightClick] ", obj_name, " cleaned")
		_show_feedback(obj, "Cleaned")
	
	environment_clicked.emit(obj_name, "clean")

func _show_feedback(node: Node, text: String):
	"""Show floating text feedback"""
	var label = Label.new()
	label.text = text
	label.add_theme_font_size_override("font_size", 14)
	label.add_theme_color_override("font_color", Color.YELLOW)
	label.add_theme_color_override("font_shadow_color", Color.BLACK)
	label.add_theme_constant_override("shadow_offset_x", 1)
	label.add_theme_constant_override("shadow_offset_y", 1)
	label.z_index = 100
	
	get_tree().root.add_child(label)
	
	# Position above the clicked object in screen space
	var screen_pos = get_viewport().get_camera_2d().unproject_position(node.global_position) if get_viewport().get_camera_2d() else node.global_position
	label.position = screen_pos + Vector2(-30, -60)
	
	# Animate
	var tween = create_tween()
	tween.parallel().tween_property(label, "position:y", label.position.y - 20, 0.5)
	tween.parallel().tween_property(label, "modulate:a", 0, 0.5)
	tween.tween_callback(label.queue_free)

func show_help():
	"""Display help message"""
	print("")
	print("=== CLICK TO INTERACT ===")
	print("LEFT CLICK near objects:")
	print("  - Bar Counter: Make dirty (-30%)")
	print("  - Shelf: Reduce stock (-25%)")
	print("  - Tables: Add dirty dishes")
	print("")
	print("RIGHT CLICK near objects:")
	print("  - Bar Counter: Clean (100%)")
	print("  - Shelf: Restock (100%)")
	print("  - Tables: Clear and clean")
	print("=========================")