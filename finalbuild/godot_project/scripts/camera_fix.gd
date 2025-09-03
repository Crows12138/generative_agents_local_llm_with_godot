extends Camera2D

# Camera Adapter Script for cozy_bar scene
# Attach this directly to Camera2D node

# Camera settings
@export var follow_speed: float = 5.0
@export var zoom_level: Vector2 = Vector2(2, 2)
@export var camera_offset: Vector2 = Vector2(0, -50)
@export var enable_smoothing: bool = true
@export var follow_target: NodePath = ""

# Camera bounds (using different names to avoid conflict)
@export var camera_limit_left: int = 0
@export var camera_limit_top: int = 0
@export var camera_limit_right: int = 640
@export var camera_limit_bottom: int = 480

var target_node: Node2D = null
var shake_amount: float = 0.0

func _ready():
	"""Initialize camera settings"""
	
	# Set initial position and zoom
	position = Vector2(320, 240)  # Center of 640x480 scene
	zoom = zoom_level
	
	# Set camera limits using built-in properties
	limit_left = camera_limit_left
	limit_top = camera_limit_top
	limit_right = camera_limit_right
	limit_bottom = camera_limit_bottom
	
	# Enable position smoothing
	position_smoothing_enabled = enable_smoothing
	position_smoothing_speed = follow_speed
	
	# Make this camera current
	make_current()
	
	# Try to find follow target
	if not follow_target.is_empty():
		target_node = get_node_or_null(follow_target)
	
	# If no target specified, try to find player
	if not target_node:
		target_node = get_tree().get_first_node_in_group("player")

func _process(delta):
	"""Update camera position"""
	
	# Follow target if exists
	if target_node and is_instance_valid(target_node):
		var target_pos = target_node.global_position + camera_offset
		
		if enable_smoothing:
			global_position = global_position.lerp(target_pos, follow_speed * delta)
		else:
			global_position = target_pos
	
	# Apply camera shake if active
	if shake_amount > 0:
		position += Vector2(
			randf_range(-shake_amount, shake_amount),
			randf_range(-shake_amount, shake_amount)
		)
		shake_amount = max(shake_amount - delta * 10, 0)

func shake(amount: float, duration: float = 0.2):
	"""Trigger camera shake effect"""
	shake_amount = amount

func set_follow_target(new_target: Node2D):
	"""Change the camera follow target"""
	target_node = new_target

func zoom_in():
	"""Zoom camera in"""
	zoom = zoom.lerp(zoom_level * 1.5, 0.1)

func zoom_out():
	"""Zoom camera out"""
	zoom = zoom.lerp(zoom_level * 0.75, 0.1)

func reset_zoom():
	"""Reset camera zoom to default"""
	zoom = zoom.lerp(zoom_level, 0.1)

func focus_on_point(point: Vector2, instant: bool = false):
	"""Focus camera on specific point"""
	if instant:
		global_position = point
	else:
		var tween = create_tween()
		tween.tween_property(self, "global_position", point, 0.5)
