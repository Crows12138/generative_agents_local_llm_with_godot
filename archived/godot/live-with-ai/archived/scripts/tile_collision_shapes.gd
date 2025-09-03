extends Resource
class_name TileCollisionShapes

# Collision shape definitions for different tile types
# Each shape is defined as a rectangle with offset and size

static func get_collision_shape(tile_type: CozyBarRoom.TileType) -> Dictionary:
	"""Return collision shape data for a tile type"""
	match tile_type:
		CozyBarRoom.TileType.WALL_BRICK:
			return {
				"shape": "rectangle",
				"offset": Vector2(0, 0),
				"size": Vector2(32, 32),
				"one_way": false
			}
		
		CozyBarRoom.TileType.BAR_COUNTER:
			return {
				"shape": "rectangle", 
				"offset": Vector2(0, 8),
				"size": Vector2(32, 16),
				"one_way": false
			}
		
		CozyBarRoom.TileType.TABLE:
			return {
				"shape": "rectangle",
				"offset": Vector2(4, 4),
				"size": Vector2(24, 24),
				"one_way": false
			}
		
		CozyBarRoom.TileType.MUSIC_STAGE:
			return {
				"shape": "rectangle",
				"offset": Vector2(0, 16),
				"size": Vector2(32, 16),
				"one_way": false
			}
		
		CozyBarRoom.TileType.DOOR:
			return {
				"shape": "rectangle",
				"offset": Vector2(12, 0),
				"size": Vector2(8, 32),
				"one_way": false
			}
		
		# Non-collision tiles
		CozyBarRoom.TileType.FLOOR_WOOD,\
		CozyBarRoom.TileType.BAR_STOOL,\
		CozyBarRoom.TileType.CHAIR:
			return {
				"shape": "none",
				"offset": Vector2.ZERO,
				"size": Vector2.ZERO,
				"one_way": false
			}
		
		_:
			return {
				"shape": "none",
				"offset": Vector2.ZERO,
				"size": Vector2.ZERO,
				"one_way": false
			}

static func create_collision_polygon(shape_data: Dictionary) -> PackedVector2Array:
	"""Create a collision polygon from shape data"""
	if shape_data.shape == "rectangle":
		var offset = shape_data.offset
		var size = shape_data.size
		return PackedVector2Array([
			offset,
			offset + Vector2(size.x, 0),
			offset + size,
			offset + Vector2(0, size.y)
		])
	else:
		return PackedVector2Array()

static func get_physics_material(tile_type: CozyBarRoom.TileType) -> Dictionary:
	"""Get physics material properties for tile types"""
	match tile_type:
		CozyBarRoom.TileType.WALL_BRICK:
			return {
				"friction": 1.0,
				"bounce": 0.0,
				"absorb": true
			}
		
		CozyBarRoom.TileType.FLOOR_WOOD:
			return {
				"friction": 0.8,
				"bounce": 0.0,
				"absorb": false
			}
		
		CozyBarRoom.TileType.BAR_COUNTER,\
		CozyBarRoom.TileType.TABLE:
			return {
				"friction": 0.9,
				"bounce": 0.1,
				"absorb": false
			}
		
		_:
			return {
				"friction": 0.8,
				"bounce": 0.0,
				"absorb": false
			}