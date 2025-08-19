# Cozy Bar TileMap Guide

## 📋 Overview

This guide provides comprehensive instructions for working with the Cozy Bar TileMap system in Godot. The system is based on the room configuration defined in `cozy_bar_demo/config/room_config.json`.

## 🗂️ File Structure

```
godot/live-with-ai/
├── scenes/
│   └── cozy_bar.tscn                    # Main scene with TileMap setup
├── tilesets/
│   └── cozy_bar_tileset.tres           # TileSet resource definition
├── scripts/
│   ├── cozy_bar_room.gd               # Main room controller
│   ├── tile_collision_shapes.gd       # Collision shape definitions
│   └── tilemap_layer_manager.gd       # Layer management system
└── assets/sprites/tiles/v2/            # Tile textures
    ├── TopDownHouse_FloorsAndWalls.png
    ├── TopDownHouse_FurnitureState1.png
    └── TopDownHouse_SmallItems.png
```

## 🎯 Room Layout Reference

Based on `room_config.json`, the cozy bar has these dimensions and layout:

### Room Size: 12x10 tiles (384x320 pixels at 32px/tile)

```
WWWWWWWWWWWW  ← Row 0: Top wall
W..........W  ← Row 1: Open space
W..BBBBBB..W  ← Row 2: Bar counter
W..cccccc..W  ← Row 3: Bar stools
W..........W  ← Row 4: Open space
W.TT....TT.W  ← Row 5: Tables
W.SS....SS.W  ← Row 6: Chairs
W..........W  ← Row 7: Open space
W....MM....W  ← Row 8: Music stage
WWWWWDDWWWWW  ← Row 9: Bottom wall with door
```

### Legend
- `W` = Wall (brick texture)
- `D` = Door (entrance/exit)
- `B` = Bar counter (main serving area)
- `c` = Bar stool (seating at bar)
- `T` = Table (customer seating)
- `S` = Chair (seating at tables)
- `M` = Music stage (performance area)
- `.` = Floor (wooden texture)

## 🏗️ TileMap Layer System

### Layer Structure (Z-index order)

1. **CollisionLayer** (Z: -1) - Hidden collision detection
2. **FloorLayer** (Z: 0) - Ground tiles and floor textures
3. **WallLayer** (Z: 1) - Walls, doors, structural elements
4. **FurnitureLayer** (Z: 2) - Tables, bar counter, stage (Y-sorted)
5. **DecorationLayer** (Z: 3) - Chairs, stools, small items (Y-sorted)

### Layer Properties

| Layer | Purpose | Y-Sort | Visible | Collision |
|-------|---------|--------|---------|-----------|
| Floor | Base ground tiles | No | Yes | No |
| Wall | Room boundaries | No | Yes | Yes |
| Furniture | Large objects | Yes | Yes | Yes |
| Decoration | Small items | Yes | Yes | No |
| Collision | Physics | No | Editor only | Yes |

## 🎨 Tile Definitions

### Tile Types and Sources

| Tile Type | Character | Source ID | Atlas Coords | Has Collision |
|-----------|-----------|-----------|--------------|---------------|
| Wall Brick | W | 0 | (2, 0) | Yes |
| Door | D | 0 | (3, 2) | No |
| Bar Counter | B | 1 | (4, 1) | Yes |
| Bar Stool | c | 1 | (0, 2) | No |
| Table | T | 1 | (2, 1) | Yes |
| Chair | S | 1 | (1, 2) | No |
| Music Stage | M | 1 | (5, 3) | No |
| Floor Wood | . | 0 | (1, 1) | No |

### TileSet Sources

- **Source 0**: `TopDownHouse_FloorsAndWalls.png` (32x32 tiles)
- **Source 1**: `TopDownHouse_FurnitureState1.png` (32x32 tiles)
- **Source 2**: `TopDownHouse_SmallItems.png` (32x32 tiles)

## 🔧 Usage Instructions

### 1. Setting Up the Scene

```gdscript
# Load the cozy bar scene
var cozy_bar_scene = preload("res://scenes/cozy_bar.tscn")
var cozy_bar = cozy_bar_scene.instantiate()
add_child(cozy_bar)
```

### 2. Accessing Layers

```gdscript
# Get layer references
var floor_layer = cozy_bar.get_node("TileMaps/FloorLayer")
var wall_layer = cozy_bar.get_node("TileMaps/WallLayer")
var furniture_layer = cozy_bar.get_node("TileMaps/FurnitureLayer")
var decoration_layer = cozy_bar.get_node("TileMaps/DecorationLayer")
```

### 3. Placing Tiles Programmatically

```gdscript
# Example: Place a table at grid position (5, 5)
var grid_pos = Vector2i(5, 5)
var source_id = 1  # Furniture source
var atlas_coords = Vector2i(2, 1)  # Table coordinates

furniture_layer.set_cell(0, grid_pos, source_id, atlas_coords)
```

### 4. Collision Detection

```gdscript
# Check if a position is walkable
var world_pos = Vector2(160, 192)
var is_walkable = cozy_bar.is_walkable(world_pos)

# Get tile type at position
var tile_type = cozy_bar.get_tile_at_position(world_pos)
```

### 5. Coordinate Conversion

```gdscript
# Convert between grid and world coordinates
var grid_pos = Vector2(6, 5)
var world_pos = cozy_bar.grid_to_world(grid_pos)

var world_pos = Vector2(192, 160)
var grid_pos = cozy_bar.world_to_grid(world_pos)
```

## 📍 Spawn Points

### Character Spawn Locations

| Character | Grid Position | World Position | Role |
|-----------|---------------|----------------|------|
| Player | (6, 5) | (192, 160) | Player character |
| Bob | (6, 2) | (192, 64) | Bartender |
| Alice | (2, 6) | (64, 192) | Regular customer |
| Sam | (9, 6) | (288, 192) | Musician |

### Accessing Spawn Points

```gdscript
# Get spawn point markers
var player_spawn = cozy_bar.get_node("SpawnPoints/PlayerSpawn")
var bob_spawn = cozy_bar.get_node("SpawnPoints/BobSpawn")

# Position a character at spawn point
character.position = player_spawn.position
```

## 🔄 Interactive Objects

### Interactive Areas

| Object | Location | Actions Available |
|--------|----------|-------------------|
| Bar Counter | (6, 2) | order_drink, chat_bartender, sit_at_bar |
| Table 1 | (2, 5) | sit_down, place_order, have_conversation |
| Table 2 | (9, 5) | sit_down, place_order, have_conversation |
| Music Stage | (6, 8) | perform, listen, request_song |

### Interaction Setup

```gdscript
# Connect to interaction signals
var bar_counter = cozy_bar.get_node("InteractiveObjects/BarCounter")
bar_counter.area_entered.connect(_on_bar_counter_entered)

func _on_bar_counter_entered(body):
    if body.is_in_group("player"):
        show_interaction_prompt("Press E to interact with bar")
```

## 💡 Lighting System

### Ambient Lighting

The scene includes atmospheric lighting:

- **Ambient Light**: Warm golden tone (192, 160) with low energy
- **Bar Lights**: Two focused lights above the bar counter
- **Stage Light**: Blue-tinted light for the music area
- **Environment**: Dark, cozy background with warm ambient color

### Light Configuration

```gdscript
# Access lighting nodes
var ambient_light = cozy_bar.get_node("Lighting/AmbientLight")
var bar_light1 = cozy_bar.get_node("Lighting/BarLight1")
var stage_light = cozy_bar.get_node("Lighting/StageLight")

# Modify lighting
ambient_light.energy = 0.7  # Brighter ambient
bar_light1.color = Color.YELLOW  # Change bar light color
```

## 🎵 Audio Setup

### Background Audio

```gdscript
# Access audio players
var bg_music = cozy_bar.get_node("AudioManager/BackgroundMusic")
var ambience = cozy_bar.get_node("AudioManager/AmbiencePlayer")

# Set audio streams
bg_music.stream = preload("res://audio/smooth_jazz.ogg")
ambience.stream = preload("res://audio/bar_ambience.ogg")

# Start playing
bg_music.play()
ambience.play()
```

## 🛠️ Customization Tips

### Adding New Tile Types

1. Add to `TileType` enum in `cozy_bar_room.gd`:
```gdscript
enum TileType {
    # ... existing types
    NEW_FURNITURE
}
```

2. Update `tile_definitions` dictionary:
```gdscript
TileType.NEW_FURNITURE: {
    "source": 1, 
    "atlas_coords": Vector2i(x, y), 
    "has_collision": true
}
```

3. Add to layer mapping in `tilemap_layer_manager.gd`

### Modifying Room Layout

1. Edit the `room_layout` array in `cozy_bar_room.gd`
2. Update `ROOM_SIZE` constant if dimensions change
3. Adjust spawn points and interactive object positions
4. Update collision layer accordingly

### Performance Optimization

- Use Y-sorting only on layers that need depth sorting (Furniture, Decorations)
- Keep collision detection on a separate, hidden layer
- Group similar tiles on the same layer to reduce draw calls
- Use appropriate texture filtering for pixel art tiles

## 🐛 Troubleshooting

### Common Issues

1. **Tiles not appearing**: Check source ID and atlas coordinates
2. **Collision not working**: Verify collision shapes are set up correctly
3. **Layer ordering issues**: Check Z-index values in layer configuration
4. **Performance problems**: Disable Y-sorting on layers that don't need it

### Debug Tools

```gdscript
# Show collision layer for debugging
var collision_layer = cozy_bar.get_node("TileMaps/CollisionLayer")
collision_layer.enabled = true
collision_layer.modulate.a = 0.5  # Semi-transparent
```

## 📚 References

- **Room Config**: `cozy_bar_demo/config/room_config.json`
- **Godot TileMap Documentation**: https://docs.godotengine.org/en/stable/classes/class_tilemap.html
- **TileSet Resource Documentation**: https://docs.godotengine.org/en/stable/classes/class_tileset.html