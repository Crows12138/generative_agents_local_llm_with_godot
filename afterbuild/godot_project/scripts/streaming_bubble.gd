# streaming_bubble.gd - Speech/thought bubble with switchable styles
extends Panel

# Bubble type - "speech" for dialogue, "thought" for observations
var bubble_type := "speech"

# Default colors for speech bubble
var fill_color := Color(0.05, 0.05, 0.05, 0.82)     # Same as panel
var outline_color := Color(0.2, 0.2, 0.2, 0.5)      # Same as panel border

# Colors for thought/observation bubble
var thought_fill_color := Color(0.95, 0.95, 1.0, 0.9)     # More opaque white-blue
var thought_outline_color := Color(0.4, 0.4, 0.6, 0.9)    # Darker blue for better contrast

var tail_polygon: Polygon2D
var tail_outline: Polygon2D
var tail_circles: Array = []  # For thought bubble circles
var target_position := Vector2.ZERO

func _ready() -> void:
	# Setup bubble style based on type
	_setup_bubble_style()
	
	if bubble_type == "speech":
		# Create outline triangle (slightly larger, behind)
		tail_outline = Polygon2D.new()
		tail_outline.name = "TailOutline"
		tail_outline.color = outline_color
		tail_outline.show_behind_parent = false  # Draw on top to cover border
		tail_outline.z_index = 1
		add_child(tail_outline)
		
		# Create fill triangle (on top of outline)
		tail_polygon = Polygon2D.new()
		tail_polygon.name = "TailFill"
		tail_polygon.color = fill_color
		tail_polygon.show_behind_parent = false  # Draw on top to cover border
		tail_polygon.z_index = 2
		add_child(tail_polygon)
		
		tail_outline.visible = false
		tail_polygon.visible = false
	else:
		# Create thought bubble circles
		_create_thought_tail()


# External call - pass speaker's screen position (already converted)
func point_to_character(character_screen_pos: Vector2) -> void:
	target_position = character_screen_pos
	_update_tail()

func _update_tail() -> void:
	if target_position == Vector2.ZERO:
		return
	
	if bubble_type == "speech":
		if not tail_outline or not tail_polygon:
			return

		# Simple triangle tail - narrower and centered
		var tail_width = 12.0   # Narrower
		var tail_height = 20.0
		
		# Center position at bottom of bubble
		var start_x = size.x * 0.5  # Center of bubble
		var start_y = size.y - 1  # Just barely overlap to hide border
		
		# Create triangle points
		var points = PackedVector2Array()
		
		# Three points for triangle
		points.append(Vector2(start_x - tail_width/2, start_y))  # Left base
		points.append(Vector2(start_x + tail_width/2, start_y))  # Right base
		points.append(Vector2(start_x, start_y + tail_height))   # Tip pointing down
		
		# Outline (only sides and bottom, no top line)
		var outline_points = PackedVector2Array()
		var outline_extra = 1.0
		# Start just at the edge
		outline_points.append(Vector2(start_x - tail_width/2 - outline_extra, start_y))
		outline_points.append(Vector2(start_x + tail_width/2 + outline_extra, start_y))
		outline_points.append(Vector2(start_x, start_y + tail_height + outline_extra))
		
		tail_outline.polygon = outline_points
		tail_polygon.polygon = points
		
		tail_outline.visible = true
		tail_polygon.visible = true
	else:
		# Update thought bubble circles position
		_update_thought_tail()

func update_text(text: String):
	"""Update the text in the bubble's RichTextLabel"""
	var label = get_node_or_null("Label")
	if label and label is RichTextLabel:
		if bubble_type == "thought":
			# Add italic formatting, color and center alignment for thoughts
			label.text = "[center][i][color=#333344]" + text + "[/color][/i][/center]"
			label.bbcode_enabled = true
			label.add_theme_font_size_override("normal_font_size", 14)
			label.add_theme_color_override("default_color", Color(0.2, 0.2, 0.3, 1.0))
		else:
			label.text = text

# Add new functions for thought bubble support
func _setup_bubble_style() -> void:
	"""Setup panel style based on bubble type"""
	var style_box = StyleBoxFlat.new()
	
	if bubble_type == "thought":
		# Use thought bubble colors
		style_box.bg_color = thought_fill_color
		style_box.border_color = thought_outline_color
		style_box.border_width_left = 2
		style_box.border_width_right = 2
		style_box.border_width_top = 2
		style_box.border_width_bottom = 2
		
		# Make it more elliptical by using maximum corner radius
		var min_dimension = min(size.x, size.y) if size.x > 0 else 30
		var corner_radius = min_dimension / 2  # Half of smallest dimension for ellipse effect
		style_box.corner_radius_top_left = corner_radius
		style_box.corner_radius_top_right = corner_radius
		style_box.corner_radius_bottom_left = corner_radius
		style_box.corner_radius_bottom_right = corner_radius
	else:
		# Regular speech bubble style
		style_box.bg_color = fill_color
		style_box.border_color = outline_color
		style_box.border_width_left = 1
		style_box.border_width_right = 1
		style_box.border_width_top = 1
		style_box.border_width_bottom = 1
		
		# Standard corners
		style_box.corner_radius_top_left = 8
		style_box.corner_radius_top_right = 8
		style_box.corner_radius_bottom_left = 8
		style_box.corner_radius_bottom_right = 8
	
	add_theme_stylebox_override("panel", style_box)

func _create_thought_tail() -> void:
	"""Create circles for thought bubble tail"""
	var circle_sizes = [8.0, 6.0, 4.0]  # Decreasing sizes
	
	for i in range(circle_sizes.size()):
		# Create filled circle
		var circle = _create_circle_polygon(circle_sizes[i])
		circle.name = "TailCircle" + str(i)
		circle.color = thought_fill_color
		circle.z_index = 2
		add_child(circle)
		tail_circles.append(circle)
		
		# Create outline circle
		var outline = _create_circle_polygon(circle_sizes[i] + 1.0)
		outline.name = "TailOutline" + str(i)
		outline.color = thought_outline_color
		outline.z_index = 1
		add_child(outline)
		tail_circles.append(outline)
		
		circle.visible = false
		outline.visible = false

func _create_circle_polygon(radius: float) -> Polygon2D:
	"""Create a circle polygon"""
	var polygon = Polygon2D.new()
	var points = PackedVector2Array()
	var segments = 12  # Number of segments
	
	for i in range(segments + 1):
		var angle = (i * 2.0 * PI) / segments
		var point = Vector2(cos(angle), sin(angle)) * radius
		points.append(point)
	
	polygon.polygon = points
	return polygon

func _update_thought_tail() -> void:
	"""Position thought bubble circles"""
	if tail_circles.is_empty():
		return
		
	var start_x = size.x * 0.5
	var start_y = size.y - 5
	var spacing = 10.0
	
	var index = 0
	for circle in tail_circles:
		if circle:
			var offset_y = start_y + (floor(index / 2) + 1) * spacing
			circle.position = Vector2(start_x, offset_y)
			circle.visible = true
		index += 1

func set_bubble_type(type: String) -> void:
	"""Switch between speech and thought bubble"""
	bubble_type = type
	_setup_bubble_style()
	
	# Hide all tails first
	if tail_outline:
		tail_outline.visible = false
	if tail_polygon:
		tail_polygon.visible = false
	for circle in tail_circles:
		if circle:
			circle.visible = false
	
	# Show appropriate tail
	_update_tail()
