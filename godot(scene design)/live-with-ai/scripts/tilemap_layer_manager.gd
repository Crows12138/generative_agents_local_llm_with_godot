extends Node
class_name TileMapLayerManager

# Layer configuration for the cozy bar tilemap system
# Manages different rendering layers and their properties

enum LayerType {
	FLOOR,        # Z-index: 0 - Ground tiles, wooden floors
	WALLS,        # Z-index: 1 - Wall tiles, doors
	FURNITURE,    # Z-index: 2 - Tables, bar counter, stage
	DECORATIONS,  # Z-index: 3 - Chairs, stools, small items
	COLLISION,    # Z-index: -1 - Invisible collision layer
	LIGHTING      # Z-index: 4 - Light sources and effects
}

# Layer properties configuration
var layer_config = {
	LayerType.FLOOR: {
		"name": "Floor",
		"z_index": 0,
		"modulate": Color.WHITE,
		"enabled": true,
		"y_sort_enabled": false,
		"tile_types": [
			CozyBarRoom.TileType.FLOOR_WOOD
		]
	},
	
	LayerType.WALLS: {
		"name": "Walls", 
		"z_index": 1,
		"modulate": Color.WHITE,
		"enabled": true,
		"y_sort_enabled": false,
		"tile_types": [
			CozyBarRoom.TileType.WALL_BRICK,
			CozyBarRoom.TileType.DOOR
		]
	},
	
	LayerType.FURNITURE: {
		"name": "Furniture",
		"z_index": 2,
		"modulate": Color.WHITE,
		"enabled": true,
		"y_sort_enabled": true,
		"tile_types": [
			CozyBarRoom.TileType.BAR_COUNTER,
			CozyBarRoom.TileType.TABLE,
			CozyBarRoom.TileType.MUSIC_STAGE
		]
	},
	
	LayerType.DECORATIONS: {
		"name": "Decorations",
		"z_index": 3,
		"modulate": Color.WHITE,
		"enabled": true,
		"y_sort_enabled": true,
		"tile_types": [
			CozyBarRoom.TileType.BAR_STOOL,
			CozyBarRoom.TileType.CHAIR
		]
	},
	
	LayerType.COLLISION: {
		"name": "Collision",
		"z_index": -1,
		"modulate": Color(1, 1, 1, 0.3),
		"enabled": false,  # Hidden in game, visible in editor
		"y_sort_enabled": false,
		"tile_types": []  # Uses collision shapes instead
	}
}

static func get_layer_for_tile_type(tile_type: CozyBarRoom.TileType) -> LayerType:
	"""Determine which layer a tile type should be placed on"""
	match tile_type:
		CozyBarRoom.TileType.FLOOR_WOOD:
			return LayerType.FLOOR
		
		CozyBarRoom.TileType.WALL_BRICK, CozyBarRoom.TileType.DOOR:
			return LayerType.WALLS
		
		CozyBarRoom.TileType.BAR_COUNTER, CozyBarRoom.TileType.TABLE, CozyBarRoom.TileType.MUSIC_STAGE:
			return LayerType.FURNITURE
		
		CozyBarRoom.TileType.BAR_STOOL, CozyBarRoom.TileType.CHAIR:
			return LayerType.DECORATIONS
		
		_:
			return LayerType.FLOOR

static func configure_tilemap_layer(tilemap: TileMap, layer_index: int, layer_type: LayerType):
	"""Configure a TileMap layer with the appropriate properties"""
	var config = TileMapLayerManager.new().layer_config[layer_type]
	
	# Set layer properties
	tilemap.set_layer_name(layer_index, config.name)
	tilemap.set_layer_z_index(layer_index, config.z_index)
	tilemap.set_layer_modulate(layer_index, config.modulate)
	tilemap.set_layer_enabled(layer_index, config.enabled)
	tilemap.set_layer_y_sort_enabled(layer_index, config.y_sort_enabled)

static func get_render_order() -> Array[LayerType]:
	"""Get the correct rendering order for layers (back to front)"""
	return [
		LayerType.COLLISION,
		LayerType.FLOOR,
		LayerType.WALLS,
		LayerType.FURNITURE,
		LayerType.DECORATIONS,
		LayerType.LIGHTING
	]

static func get_layer_description(layer_type: LayerType) -> String:
	"""Get human-readable description of layer purpose"""
	match layer_type:
		LayerType.FLOOR:
			return "Ground tiles and floor textures. Base layer for the room."
		
		LayerType.WALLS:
			return "Wall tiles, doors, and structural elements. Provides room boundaries."
		
		LayerType.FURNITURE:
			return "Large furniture items like tables, bar counter, and stage. Y-sorted."
		
		LayerType.DECORATIONS:
			return "Small decorative items like chairs and stools. Y-sorted for depth."
		
		LayerType.COLLISION:
			return "Invisible collision detection layer. Hidden during gameplay."
		
		LayerType.LIGHTING:
			return "Light sources and lighting effects. Top-most layer."
		
		_:
			return "Unknown layer type."