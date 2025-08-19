extends Node2D
class_name CozyBarRoom

# Room configuration based on room_config.json
const ROOM_SIZE = Vector2i(12, 10)
const TILE_SIZE = 32

# Tile type definitions matching room_config.json legend
enum TileType {
	WALL_BRICK,    # W
	DOOR,          # D  
	BAR_COUNTER,   # B
	BAR_STOOL,     # c
	TABLE,         # T
	CHAIR,         # S
	MUSIC_STAGE,   # M
	FLOOR_WOOD     # .
}

# Tile mapping from config to tileset coordinates
var tile_definitions = {
	TileType.WALL_BRICK: {"source": 0, "atlas_coords": Vector2i(2, 0), "has_collision": true},
	TileType.DOOR: {"source": 0, "atlas_coords": Vector2i(3, 2), "has_collision": false},
	TileType.BAR_COUNTER: {"source": 1, "atlas_coords": Vector2i(4, 1), "has_collision": true},
	TileType.BAR_STOOL: {"source": 1, "atlas_coords": Vector2i(0, 2), "has_collision": false},
	TileType.TABLE: {"source": 1, "atlas_coords": Vector2i(2, 1), "has_collision": true},
	TileType.CHAIR: {"source": 1, "atlas_coords": Vector2i(1, 2), "has_collision": false},
	TileType.MUSIC_STAGE: {"source": 1, "atlas_coords": Vector2i(5, 3), "has_collision": false},
	TileType.FLOOR_WOOD: {"source": 0, "atlas_coords": Vector2i(1, 1), "has_collision": false}
}

# Room layout from config
var room_layout = [
	"WWWWWWWWWWWW",
	"W..........W", 
	"W..BBBBBB..W",
	"W..cccccc..W",
	"W..........W",
	"W.TT....TT.W",
	"W.SS....SS.W",
	"W..........W",
	"W....MM....W",
	"WWWWWDDWWWWW"
]

# Character mappings
var legend = {
	"W": TileType.WALL_BRICK,
	"D": TileType.DOOR,
	"B": TileType.BAR_COUNTER,
	"c": TileType.BAR_STOOL,
	"T": TileType.TABLE,
	"S": TileType.CHAIR,
	"M": TileType.MUSIC_STAGE,
	".": TileType.FLOOR_WOOD
}

# Layer references
@onready var floor_layer: TileMap = $TileMaps/FloorLayer
@onready var wall_layer: TileMap = $TileMaps/WallLayer
@onready var furniture_layer: TileMap = $TileMaps/FurnitureLayer
@onready var decoration_layer: TileMap = $TileMaps/DecorationLayer
@onready var collision_layer: TileMap = $TileMaps/CollisionLayer

# Spawn points from config
var spawn_points = {
	"player": Vector2(6, 5),
	"Bob": Vector2(6, 2),
	"Alice": Vector2(2, 6),
	"Sam": Vector2(9, 6)
}

func _ready():
	setup_room()
	setup_spawn_points()
	setup_interactive_objects()

func setup_room():
	"""Generate the room layout based on configuration"""
	for y in range(room_layout.size()):
		var row = room_layout[y]
		for x in range(row.length()):
			var char = row[x]
			var tile_type = legend.get(char, TileType.FLOOR_WOOD)
			place_tile(Vector2i(x, y), tile_type)

func place_tile(grid_pos: Vector2i, tile_type: TileType):
	"""Place a tile at the specified grid position"""
	var tile_def = tile_definitions[tile_type]
	var source_id = tile_def["source"]
	var atlas_coords = tile_def["atlas_coords"]
	var has_collision = tile_def["has_collision"]
	
	# Determine which layer to place the tile on
	var target_layer: TileMap
	match tile_type:
		TileType.FLOOR_WOOD:
			target_layer = floor_layer
		TileType.WALL_BRICK, TileType.DOOR:
			target_layer = wall_layer
		TileType.BAR_COUNTER, TileType.TABLE, TileType.MUSIC_STAGE:
			target_layer = furniture_layer
		TileType.BAR_STOOL, TileType.CHAIR:
			target_layer = decoration_layer
		_:
			target_layer = floor_layer
	
	# Place the tile
	target_layer.set_cell(0, grid_pos, source_id, atlas_coords)
	
	# Add collision if needed
	if has_collision:
		collision_layer.set_cell(0, grid_pos, source_id, atlas_coords)

func setup_spawn_points():
	"""Position spawn point markers based on config"""
	var player_spawn = get_node("SpawnPoints/PlayerSpawn")
	var bob_spawn = get_node("SpawnPoints/BobSpawn")
	var alice_spawn = get_node("SpawnPoints/AliceSpawn")
	var sam_spawn = get_node("SpawnPoints/SamSpawn")
	
	player_spawn.position = grid_to_world(spawn_points["player"])
	bob_spawn.position = grid_to_world(spawn_points["Bob"])
	alice_spawn.position = grid_to_world(spawn_points["Alice"])
	sam_spawn.position = grid_to_world(spawn_points["Sam"])

func setup_interactive_objects():
	"""Setup interactive object collision shapes"""
	# Setup bar counter interaction
	var bar_counter = get_node("InteractiveObjects/BarCounter")
	bar_counter.position = grid_to_world(Vector2(6, 2))
	
	# Setup table interactions
	var table1 = get_node("InteractiveObjects/Tables/Table1")
	var table2 = get_node("InteractiveObjects/Tables/Table2")
	table1.position = grid_to_world(Vector2(2, 5))
	table2.position = grid_to_world(Vector2(9, 5))
	
	# Setup music stage interaction
	var stage = get_node("InteractiveObjects/MusicStage")
	stage.position = grid_to_world(Vector2(6, 8))

func grid_to_world(grid_pos: Vector2) -> Vector2:
	"""Convert grid coordinates to world coordinates"""
	return Vector2(grid_pos.x * TILE_SIZE + TILE_SIZE/2, grid_pos.y * TILE_SIZE + TILE_SIZE/2)

func world_to_grid(world_pos: Vector2) -> Vector2i:
	"""Convert world coordinates to grid coordinates"""
	return Vector2i(world_pos.x / TILE_SIZE, world_pos.y / TILE_SIZE)

func get_tile_at_position(world_pos: Vector2) -> TileType:
	"""Get the tile type at a world position"""
	var grid_pos = world_to_grid(world_pos)
	if grid_pos.x < 0 or grid_pos.x >= ROOM_SIZE.x or grid_pos.y < 0 or grid_pos.y >= ROOM_SIZE.y:
		return TileType.WALL_BRICK
	
	var char = room_layout[grid_pos.y][grid_pos.x]
	return legend.get(char, TileType.FLOOR_WOOD)

func is_walkable(world_pos: Vector2) -> bool:
	"""Check if a position is walkable"""
	var tile_type = get_tile_at_position(world_pos)
	match tile_type:
		TileType.WALL_BRICK, TileType.BAR_COUNTER, TileType.TABLE, TileType.MUSIC_STAGE:
			return false
		_:
			return true