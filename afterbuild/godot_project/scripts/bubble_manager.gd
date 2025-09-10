extends Node
# Centralized bubble management system for all NPCs

static func create_bubble(npc_name: String, text: String, bubble_type: String, bubble_size: Vector2, 
						  font_size: int, bg_color: Color, text_color: Color, npc_position: Vector2) -> Panel:
	"""Generic bubble creation function"""
	var ui_layer = _get_ui_layer()
	if not ui_layer:
		print("ERROR: UILayer not found for bubble")
		return null
	
	# Remove any existing bubble with same name
	var bubble_name = bubble_type + "_" + npc_name
	var existing_bubble = ui_layer.get_node_or_null(bubble_name)
	if existing_bubble:
		existing_bubble.queue_free()
	
	# Create bubble
	var bubble = Panel.new()
	bubble.name = bubble_name
	bubble.size = bubble_size
	
	# Create and configure label - use regular Label for better control
	var label = Label.new()
	label.name = "Label"
	label.text = text
	label.set_anchors_and_offsets_preset(Control.PRESET_FULL_RECT)
	label.add_theme_font_size_override("font_size", font_size)
	label.add_theme_color_override("font_color", text_color)
	label.horizontal_alignment = HORIZONTAL_ALIGNMENT_CENTER
	label.vertical_alignment = VERTICAL_ALIGNMENT_CENTER
	label.autowrap_mode = TextServer.AUTOWRAP_WORD_SMART
	bubble.add_child(label)
	
	# Apply style
	var style_box = StyleBoxFlat.new()
	style_box.bg_color = bg_color
	style_box.border_color = Color(bg_color.r * 0.8, bg_color.g * 0.8, bg_color.b * 0.8, 0.5)
	style_box.border_width_left = 1
	style_box.border_width_right = 1
	style_box.border_width_top = 1
	style_box.border_width_bottom = 1
	style_box.corner_radius_top_left = 20
	style_box.corner_radius_top_right = 20
	style_box.corner_radius_bottom_left = 20
	style_box.corner_radius_bottom_right = 20
	bubble.add_theme_stylebox_override("panel", style_box)
	
	# Add to UI layer
	ui_layer.add_child(bubble)
	
	# Position above NPC
	var x_offset = -bubble_size.x / 2  # Center horizontally
	bubble.position = Vector2(npc_position.x + x_offset, npc_position.y - 60)
	
	return bubble

static func show_action_bubble(npc_name: String, text: String, npc_position: Vector2) -> Panel:
	"""Show an action bubble - dark background, white text"""
	return create_bubble(
		npc_name,
		text,
		"ActionBubble",
		Vector2(120, 40),  # Smaller size
		10,  # Font size
		Color(0.05, 0.05, 0.05, 0.85),  # Dark background
		Color(1, 1, 1, 1),  # White text
		npc_position
	)

static func show_observation_bubble(npc_name: String, text: String, npc_position: Vector2) -> Panel:
	"""Show an observation bubble - light background, dark text, smaller font"""
	return create_bubble(
		npc_name,
		text,
		"ObservationBubble",
		Vector2(200, 80),  # Larger size for more text
		8,  # Smaller font size for observations
		Color(0.95, 0.95, 1.0, 0.9),  # Light blue-white background
		Color(0.2, 0.2, 0.3, 1.0),  # Dark text
		npc_position
	)

static func show_speech_bubble(npc_name: String, text: String, npc_position: Vector2):
	"""Show a speech bubble for any NPC"""
	var ui_layer = _get_ui_layer()
	if not ui_layer:
		print("UILayer not found for speech bubble")
		return null
	
	# Remove any existing speech bubble for this NPC
	var bubble_name = "SpeechBubble_" + npc_name
	var existing = ui_layer.get_node_or_null(bubble_name)
	if existing:
		existing.queue_free()
	
	# Try to use SpeechBubbleExample as template
	var example = ui_layer.get_node_or_null("SpeechBubbleExample")
	var bubble
	
	if example:
		# Use the template
		bubble = example.duplicate()
		bubble.name = bubble_name
		bubble.visible = true
		
		# Update the text
		var text_label = _find_rich_text_label(bubble)
		if text_label:
			text_label.text = text
	else:
		# Fallback: create manually with consistent style
		bubble = _create_speech_bubble_manual(bubble_name, text)
	
	# Add to UI layer
	ui_layer.add_child(bubble)
	
	# Position above NPC
	bubble.position = Vector2(npc_position.x - 100, npc_position.y - 100)
	
	# Auto-fade after 4 seconds
	var tween = ui_layer.get_tree().create_tween()
	tween.tween_interval(4.0)
	tween.tween_property(bubble, "modulate:a", 0.0, 0.5)
	tween.tween_callback(func(): 
		if is_instance_valid(bubble):
			bubble.queue_free()
	)
	
	return bubble

static func show_thought_bubble(npc_name: String, text: String, npc_position: Vector2):
	"""Show a thought bubble for any NPC"""
	var ui_layer = _get_ui_layer()
	if not ui_layer:
		print("UILayer not found for thought bubble")
		return null
	
	# Remove any existing thought bubble for this NPC
	var bubble_name = "ThoughtBubble_" + npc_name
	var existing = ui_layer.get_node_or_null(bubble_name)
	if existing:
		existing.queue_free()
	
	# Create thought bubble with elliptical style
	var bubble = Panel.new()
	bubble.name = bubble_name
	bubble.size = Vector2(200, 80)
	
	# Create and configure label
	var label = RichTextLabel.new()
	label.name = "Label"
	label.bbcode_enabled = true
	label.fit_content = true
	label.scroll_active = false
	label.autowrap_mode = TextServer.AUTOWRAP_WORD_SMART
	label.set_anchors_and_offsets_preset(Control.PRESET_FULL_RECT)
	label.add_theme_font_size_override("normal_font_size", 14)
	label.add_theme_color_override("default_color", Color(0.2, 0.2, 0.3, 1.0))
	label.text = "[center][i][color=#333344]" + text + "[/color][/i][/center]"
	bubble.add_child(label)
	
	# Apply thought bubble style (elliptical)
	var style_box = StyleBoxFlat.new()
	style_box.bg_color = Color(0.95, 0.95, 1.0, 0.9)  # Light blue-white
	style_box.border_color = Color(0.4, 0.4, 0.6, 0.9)  # Darker blue border
	style_box.border_width_left = 2
	style_box.border_width_right = 2
	style_box.border_width_top = 2
	style_box.border_width_bottom = 2
	style_box.corner_radius_top_left = 40
	style_box.corner_radius_top_right = 40
	style_box.corner_radius_bottom_left = 40
	style_box.corner_radius_bottom_right = 40
	bubble.add_theme_stylebox_override("panel", style_box)
	
	# Add to UI layer
	ui_layer.add_child(bubble)
	bubble.visible = true
	
	# Position above NPC
	bubble.position = Vector2(npc_position.x - 100, npc_position.y - 100)
	
	# Auto-fade after 4 seconds
	var tween = ui_layer.get_tree().create_tween()
	tween.tween_interval(4.0)
	tween.tween_property(bubble, "modulate:a", 0.0, 0.5)
	tween.tween_callback(func(): 
		if is_instance_valid(bubble):
			bubble.queue_free()
	)
	
	return bubble

static func _get_ui_layer() -> CanvasLayer:
	"""Find the UILayer in the scene"""
	# Try to find UILayer from the current scene
	var tree = Engine.get_main_loop() as SceneTree
	if not tree:
		return null
	
	var root = tree.current_scene
	if not root:
		return null
	
	return root.get_node_or_null("UILayer") as CanvasLayer

static func _find_rich_text_label(node: Node) -> RichTextLabel:
	"""Recursively find RichTextLabel in node tree"""
	if node is RichTextLabel:
		return node
	
	for child in node.get_children():
		var result = _find_rich_text_label(child)
		if result:
			return result
	
	return null

static func _create_speech_bubble_manual(bubble_name: String, text: String) -> Panel:
	"""Create speech bubble manually with consistent style"""
	var bubble = Panel.new()
	bubble.name = bubble_name
	bubble.size = Vector2(200, 80)
	
	# Create and configure label
	var label = RichTextLabel.new()
	label.name = "Label"
	label.bbcode_enabled = true
	label.fit_content = true
	label.scroll_active = false
	label.autowrap_mode = TextServer.AUTOWRAP_WORD_SMART
	label.set_anchors_and_offsets_preset(Control.PRESET_FULL_RECT)
	label.add_theme_font_size_override("normal_font_size", 14)
	label.add_theme_color_override("default_color", Color(0.1, 0.1, 0.1, 1.0))
	label.text = text
	bubble.add_child(label)
	
	# Apply consistent speech bubble style (matching ui_theme.tres)
	var style_box = StyleBoxFlat.new()
	style_box.bg_color = Color(0.05, 0.05, 0.05, 0.85)  # Match theme
	style_box.border_color = Color(0.2, 0.2, 0.2, 0.5)
	style_box.border_width_left = 1
	style_box.border_width_right = 1
	style_box.border_width_top = 1
	style_box.border_width_bottom = 1
	style_box.corner_radius_top_left = 10  # Match theme
	style_box.corner_radius_top_right = 10
	style_box.corner_radius_bottom_left = 10
	style_box.corner_radius_bottom_right = 10
	bubble.add_theme_stylebox_override("panel", style_box)
	
	return bubble