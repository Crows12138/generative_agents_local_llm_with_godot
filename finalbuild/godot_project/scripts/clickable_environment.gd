extends Node
# Clickable Environment System
# Click on bar objects to trigger state changes

signal environment_clicked(object_name: String, action: String)

# Click action configurations
var click_actions = {
	"BarCounter": {
		"left": "make_dirty",    # Left click makes it dirty
		"right": "clean"          # Right click cleans it
	},
	"LiquorShelf": {
		"left": "deplete_stock",  # Left click reduces stock
		"right": "restock"        # Right click restocks
	},
	"Table": {
		"left": "make_dirty",     # Left click makes table dirty
		"right": "clean"          # Right click cleans table
	}
}

func _ready():
	print("[ClickableEnvironment] Ready - Click objects to change their state")
	# Setup click detection for environment objects
	_setup_bar_counter_clicks()
	_setup_shelf_clicks()
	_setup_table_clicks()

func _setup_bar_counter_clicks():
	"""Make bar counter clickable"""
	var bar_counter = get_node_or_null("/root/CozyBar/BarCounter")
	if not bar_counter:
		return
	
	# Add Area2D for click detection if not exists
	if not bar_counter.has_node("ClickArea"):
		var click_area = Area2D.new()
		click_area.name = "ClickArea"
		click_area.input_pickable = true  # Enable mouse detection
		bar_counter.add_child(click_area)
		
		# Add collision shape
		var collision = CollisionShape2D.new()
		var shape = RectangleShape2D.new()
		shape.size = Vector2(200, 100)  # Bar counter size
		collision.shape = shape
		click_area.add_child(collision)
		
		# Connect input event
		click_area.input_event.connect(_on_bar_counter_clicked)
		
		# Add visual feedback sprite
		var sprite = Sprite2D.new()
		sprite.name = "ClickFeedback"
		var image = Image.create(200, 100, false, Image.FORMAT_RGBA8)
		image.fill(Color(1, 1, 1, 0.2))
		var texture = ImageTexture.create_from_image(image)
		sprite.texture = texture
		sprite.visible = false
		bar_counter.add_child(sprite)
	
	print("  Bar Counter clickable")

func _setup_shelf_clicks():
	"""Make liquor shelf clickable"""
	var shelf = get_node_or_null("/root/CozyBar/LiquorShelf")
	if not shelf:
		return
	
	if not shelf.has_node("ClickArea"):
		var click_area = Area2D.new()
		click_area.name = "ClickArea"
		click_area.input_pickable = true
		shelf.add_child(click_area)
		
		var collision = CollisionShape2D.new()
		var shape = RectangleShape2D.new()
		shape.size = Vector2(150, 80)  # Shelf size
		collision.shape = shape
		click_area.add_child(collision)
		
		click_area.input_event.connect(_on_shelf_clicked)
		
		# Visual feedback
		var sprite = Sprite2D.new()
		sprite.name = "ClickFeedback"
		var image = Image.create(150, 80, false, Image.FORMAT_RGBA8)
		image.fill(Color(0.8, 0.8, 1, 0.2))
		var texture = ImageTexture.create_from_image(image)
		sprite.texture = texture
		sprite.visible = false
		shelf.add_child(sprite)
	
	print("  Liquor Shelf clickable")

func _setup_table_clicks():
	"""Make all tables clickable"""
	var tables_node = get_node_or_null("/root/CozyBar/Tables")
	if not tables_node:
		return
	
	for table in tables_node.get_children():
		if not table.has_node("ClickArea"):
			var click_area = Area2D.new()
			click_area.name = "ClickArea"
			click_area.input_pickable = true
			table.add_child(click_area)
			
			var collision = CollisionShape2D.new()
			var shape = CircleShape2D.new()
			shape.radius = 40  # Table radius
			collision.shape = shape
			click_area.add_child(collision)
			
			# Store table reference for callback
			click_area.set_meta("table", table)
			click_area.input_event.connect(_on_table_clicked.bind(table))
			
			# Visual feedback
			var sprite = Sprite2D.new()
			sprite.name = "ClickFeedback"
			var image = Image.create(80, 80, false, Image.FORMAT_RGBA8)
			image.fill(Color(1, 0.9, 0.8, 0.2))
			var texture = ImageTexture.create_from_image(image)
			sprite.texture = texture
			sprite.visible = false
			table.add_child(sprite)
			
			print("  %s clickable" % table.name)

# Click handlers
func _on_bar_counter_clicked(viewport: Node, event: InputEvent, shape_idx: int):
	"""Handle bar counter clicks"""
	if not event is InputEventMouseButton:
		return
	if not event.pressed:
		return
		
	var bar_counter = get_node_or_null("/root/CozyBar/BarCounter")
	if not bar_counter:
		return
	
	# Show visual feedback
	_show_click_feedback(bar_counter)
	
	if event.button_index == MOUSE_BUTTON_LEFT:
		# Left click - make dirty
		if bar_counter.has_method("set_cleanliness"):
			var current = bar_counter.get("cleanliness")
			if current != null:
				bar_counter.set_cleanliness(max(0, current - 30))
				print("[Click] Bar Counter dirtied: %d%%" % bar_counter.get("cleanliness"))
				_show_status_popup(bar_counter, "Dirtied: %d%%" % bar_counter.cleanliness)
				environment_clicked.emit("BarCounter", "make_dirty")
	
	elif event.button_index == MOUSE_BUTTON_RIGHT:
		# Right click - clean
		if bar_counter.has_method("set_cleanliness"):
			bar_counter.set_cleanliness(100)
			print("[Click] Bar Counter cleaned: 100%")
			_show_status_popup(bar_counter, "Cleaned: 100%")
			environment_clicked.emit("BarCounter", "clean")

func _on_shelf_clicked(viewport: Node, event: InputEvent, shape_idx: int):
	"""Handle shelf clicks"""
	if not event is InputEventMouseButton:
		return
	if not event.pressed:
		return
	
	var shelf = get_node_or_null("/root/CozyBar/LiquorShelf")
	if not shelf:
		return
	
	# Show visual feedback
	_show_click_feedback(shelf)
	
	if event.button_index == MOUSE_BUTTON_LEFT:
		# Left click - deplete stock
		if shelf.has_method("set_stock_level"):
			var current = shelf.get("stock_level")
			if current != null:
				shelf.set_stock_level(max(0, current - 25))
				print("[Click] Stock depleted: %d%%" % shelf.get("stock_level"))
				_show_status_popup(shelf, "Stock: %d%%" % shelf.stock_level)
				environment_clicked.emit("LiquorShelf", "deplete_stock")
	
	elif event.button_index == MOUSE_BUTTON_RIGHT:
		# Right click - restock
		if shelf.has_method("set_stock_level"):
			shelf.set_stock_level(100)
			print("[Click] Shelf restocked: 100%")
			_show_status_popup(shelf, "Restocked: 100%")
			environment_clicked.emit("LiquorShelf", "restock")

func _on_table_clicked(viewport: Node, event: InputEvent, shape_idx: int, table: Node):
	"""Handle table clicks"""
	if not event is InputEventMouseButton:
		return
	if not event.pressed:
		return
	
	# Show visual feedback
	_show_click_feedback(table)
	
	if event.button_index == MOUSE_BUTTON_LEFT:
		# Left click - make dirty/add dishes
		if table.has_method("set"):
			var current_clean = table.get("cleanliness")
			if current_clean != null:
				table.set("cleanliness", max(0, current_clean - 40))
			table.set("has_dishes", true)
		print("[Click] %s dirtied with dishes" % table.name)
		_show_status_popup(table, "Dirty with dishes")
		environment_clicked.emit(table.name, "make_dirty")
	
	elif event.button_index == MOUSE_BUTTON_RIGHT:
		# Right click - clean
		if table.has_method("clear_table"):
			table.clear_table()
		elif table.has_method("set"):
			table.set("cleanliness", 100)
			table.set("has_dishes", false)
		print("[Click] %s cleaned" % table.name)
		_show_status_popup(table, "Cleaned")
		environment_clicked.emit(table.name, "clean")

# Visual feedback functions
func _show_click_feedback(node: Node):
	"""Show visual feedback when clicking"""
	var feedback = node.get_node_or_null("ClickFeedback")
	if feedback:
		feedback.visible = true
		var tween = create_tween()
		tween.tween_property(feedback, "modulate:a", 0.5, 0.1)
		tween.tween_property(feedback, "modulate:a", 0.2, 0.2)
		tween.tween_callback(func(): feedback.visible = false)

func _show_status_popup(node: Node, text: String):
	"""Show floating text above clicked object"""
	var label = Label.new()
	label.text = text
	label.add_theme_font_size_override("font_size", 16)
	label.add_theme_color_override("font_color", Color.WHITE)
	label.add_theme_color_override("font_shadow_color", Color.BLACK)
	label.add_theme_constant_override("shadow_offset_x", 1)
	label.add_theme_constant_override("shadow_offset_y", 1)
	
	node.add_child(label)
	label.position = Vector2(-50, -60)
	
	# Animate floating up and fading
	var tween = create_tween()
	tween.parallel().tween_property(label, "position:y", label.position.y - 30, 1.0)
	tween.parallel().tween_property(label, "modulate:a", 0, 1.0)
	tween.tween_callback(label.queue_free)

# Helper to display instructions
func show_instructions():
	"""Display control instructions to user"""
	print("=== CLICK CONTROLS ===")
	print("LEFT CLICK: Make dirty / Deplete stock")
	print("RIGHT CLICK: Clean / Restock")
	print("")
	print("Clickable objects:")
	print("- Bar Counter")
	print("- Liquor Shelf")  
	print("- All Tables")
	print("======================")
