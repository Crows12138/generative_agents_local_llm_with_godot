# streaming_bubble.gd - Simple triangle tail for speech bubble
extends Panel

var fill_color := Color(0.05, 0.05, 0.05, 0.82)     # Same as panel
var outline_color := Color(0.2, 0.2, 0.2, 0.5)      # Same as panel border

var tail_polygon: Polygon2D
var tail_outline: Polygon2D
var target_position := Vector2.ZERO

func _ready() -> void:
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


# External call - pass speaker's screen position (already converted)
func point_to_character(character_screen_pos: Vector2) -> void:
	target_position = character_screen_pos
	_update_tail()

func _update_tail() -> void:
	if target_position == Vector2.ZERO:
		return
	
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

func update_text(text: String):
	"""Update the text in the bubble's RichTextLabel"""
	var label = get_node_or_null("Label")
	if label and label is RichTextLabel:
		label.text = text
