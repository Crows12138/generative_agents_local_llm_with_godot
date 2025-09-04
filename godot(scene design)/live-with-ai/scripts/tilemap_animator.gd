extends TileMap

# Tilemap animator script for animating specific tiles

# Define which tiles should animate (by atlas coordinates)
@export var animated_tiles: Array[Vector2i] = [
	Vector2i(0, 0),  # Example: torch tile at position 0,0 in atlas
	Vector2i(1, 0),  # Example: fire tile at position 1,0 in atlas
	Vector2i(2, 0),  # Example: water tile at position 2,0 in atlas
]

# Animation settings
@export var animation_speed: float = 4.0  # Frames per second
@export var frames_per_tile: int = 4  # Number of animation frames per tile

var animation_timer: float = 0.0
var current_frame: int = 0

func _ready():
	print("TileMap animator initialized")

func _process(delta):
	# Update animation timer
	animation_timer += delta
	
	# Check if it's time to update frame
	if animation_timer >= 1.0 / animation_speed:
		animation_timer = 0.0
		current_frame = (current_frame + 1) % frames_per_tile
		_update_animated_tiles()

func _update_animated_tiles():
	# Get all cells in the tilemap
	var used_cells = get_used_cells(0)  # Layer 0
	
	for cell_pos in used_cells:
		var atlas_coords = get_cell_atlas_coords(0, cell_pos)
		
		# Check if this tile should be animated
		if atlas_coords in animated_tiles:
			# Calculate the new frame position
			# Assumes animation frames are arranged horizontally in the atlas
			var new_atlas_coords = Vector2i(
				atlas_coords.x + current_frame,
				atlas_coords.y
			)
			
			# Update the tile with the new frame
			set_cell(0, cell_pos, 0, new_atlas_coords)

# Method to animate specific tile positions with custom patterns
func animate_tile_at_position(layer: int, position: Vector2i, frame_sequence: Array[Vector2i]):
	if frame_sequence.is_empty():
		return
	
	var frame_index = current_frame % frame_sequence.size()
	set_cell(layer, position, 0, frame_sequence[frame_index])

# Method to create a flickering effect (good for torches/candles)
func apply_flicker_effect(layer: int, position: Vector2i, base_atlas_coords: Vector2i):
	var flicker = randf() > 0.9  # 10% chance to flicker
	if flicker:
		# Temporarily use a different tile (darker version)
		set_cell(layer, position, 0, Vector2i(base_atlas_coords.x + 1, base_atlas_coords.y))
	else:
		set_cell(layer, position, 0, base_atlas_coords)