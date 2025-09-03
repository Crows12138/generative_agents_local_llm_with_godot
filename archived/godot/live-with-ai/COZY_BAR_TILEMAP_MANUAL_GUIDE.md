# 🎯 Cozy Bar TileMap Manual Creation Guide

## 📋 Overview

This is a complete guide teaching you how to **manually** create the Cozy Bar TileMap system in the Godot editor, including TileSet resource segmentation and manual tile placement.

## 🛠️ Step 1: Setup TileSet Resource

### 1.1 Create TileSet Resource

1. Open `tilesets/cozy_bar_tileset.tres` in the Godot editor
2. If the file doesn't exist, right-click the `tilesets` folder → **New Resource** → select **TileSet**
3. Save as `cozy_bar_tileset.tres`

### 1.2 Add Tile Sources (TileSetAtlasSource)

In the TileSet editor:

#### 📁 Source 0: Floors and Walls
1. Click **Add Source** → select **Atlas**
2. Drag `assets/sprites/tiles/v2/TopDownHouse_FloorsAndWalls.png` to the **Texture** field
3. Set **Texture Region Size** to `32x32`
4. Keep **ID** as `0`

#### 📁 Source 1: Furniture
1. Click **Add Source** → select **Atlas** 
2. Drag `assets/sprites/tiles/v2/TopDownHouse_FurnitureState1.png` to the **Texture** field
3. Set **Texture Region Size** to `32x32`
4. Set **ID** to `1`

#### 📁 Source 2: Small Items
1. Click **Add Source** → select **Atlas**
2. Drag `assets/sprites/tiles/v2/TopDownHouse_SmallItems.png` to the **Texture** field  
3. Set **Texture Region Size** to `32x32`
4. Set **ID** to `2`

### 1.3 Manual Tile Segmentation

For each Source, you need to manually click to segment tiles:

#### Source 0 (Floors and Walls) - Important Tile Segmentation:
1. **Wooden Floor** - Click the tile at coordinates `(1,1)`
   - Right panel **Setup** → confirm tile has been added
   
2. **Brick Wall** - Click the tile at coordinates `(2,0)`
   - In **Physics** tab → click **Add** → select rectangle shape
   - Adjust collision shape to cover the entire tile

3. **Door** - Click the tile at coordinates `(3,2)`
   - **Setup** → add tile
   - **Do not** add physics collision (door can be passed through)

#### Source 1 (Furniture) - Important Tile Segmentation:
1. **Bar Counter** - Click the tile at coordinates `(4,1)`
   - **Physics** → add rectangle collision shape
   
2. **Table** - Click the tile at coordinates `(2,1)`
   - **Physics** → add rectangle collision shape
   
3. **Music Stage** - Click the tile at coordinates `(5,3)`
   - **Setup** → add tile
   - **No collision** (can stand on stage)
   
4. **Bar Stool** - Click the tile at coordinates `(0,2)`
   - **Setup** → add tile, **no collision**
   
5. **Chair** - Click the tile at coordinates `(1,2)`
   - **Setup** → add tile, **no collision**

## 🏗️ Step 2: Setup TileMap Layers in Scene

### 2.1 Open cozy_bar.tscn Scene

In the scene editor, find the 5 TileMap layers under the `TileMaps` node:

### 2.2 Configure Each Layer

#### FloorLayer (Floor Layer)
1. Select the `FloorLayer` node
2. In the inspector, set:
   - **Tile Set** → select `cozy_bar_tileset.tres`
   - **Layer 0** → **Z Index** = `0`
   - **Layer 0** → **Y Sort Enabled** = `false`

#### WallLayer (Wall Layer)  
1. Select the `WallLayer` node
2. Set:
   - **Tile Set** → select `cozy_bar_tileset.tres`
   - **Layer 0** → **Z Index** = `1`
   - **Layer 0** → **Y Sort Enabled** = `false`

#### FurnitureLayer (Furniture Layer)
1. Select the `FurnitureLayer` node
2. Set:
   - **Tile Set** → select `cozy_bar_tileset.tres`
   - **Layer 0** → **Z Index** = `2`
   - **Layer 0** → **Y Sort Enabled** = `true` ⚠️ **Important**
   - **Layer 0** → **Y Sort Origin** = `16`

#### DecorationLayer (Decoration Layer)
1. Select the `DecorationLayer` node
2. Set:
   - **Tile Set** → select `cozy_bar_tileset.tres`
   - **Layer 0** → **Z Index** = `3`
   - **Layer 0** → **Y Sort Enabled** = `true` ⚠️ **Important**
   - **Layer 0** → **Y Sort Origin** = `16`

#### CollisionLayer (Collision Layer)
1. Select the `CollisionLayer` node
2. Set:
   - **Tile Set** → select `cozy_bar_tileset.tres`
   - **Layer 0** → **Z Index** = `-1`
   - **Layer 0** → **Enabled** = `false` (hidden)
   - **Layer 0** → **Y Sort Enabled** = `false`

## 🎨 Step 3: Manually Place Tiles to Create a 12x10 Room

### 3.1 Room Layout Design

```
   0 1 2 3 4 5 6 7 8 9 10 11
0  W W W W W W W W W W W  W
1  W . . . . . . . . . .  W  
2  W . . B B B B B B . .  W
3  W . . c c c c c c . .  W
4  W . . . . . . . . . .  W
5  W . T T . . . . T T .  W
6  W . S S . . . . S S .  W
7  W . . . . . . . . . .  W
8  W . . . . M M . . . .  W
9  W W W W W D D W W W W  W
```

### 3.2 Start Manual Placement

#### Step 1: Place Floors (FloorLayer)
1. Select the `FloorLayer` node
2. In the TileMap editor:
   - Select **Source 0**
   - Click wooden floor tile `(1,1)`
   - Use the brush tool to place floor tiles at all `.` positions
   - Coverage: x=1 to 10, y=1 to 8

#### Step 2: Place Walls (WallLayer)  
1. Select the `WallLayer` node
2. Place outer walls:
   - Select brick wall tile `(2,0)` from Source 0
   - Place top wall: y=0, x=0 to 11
   - Place left and right walls: x=0 and x=11, y=1 to 8  
   - Place bottom wall: y=9, x=0 to 4 and x=7 to 11
3. Place door:
   - Select door tile `(3,2)` from Source 0
   - Place at `(5,9)` and `(6,9)` positions

#### Step 3: Place Furniture (FurnitureLayer)
1. Select the `FurnitureLayer` node
2. Place bar counter:
   - Select bar counter tile `(4,1)` from Source 1
   - Place 6 bar counter tiles at y=2, x=3 to 8
3. Place tables:
   - Select table tile `(2,1)` from Source 1  
   - Left table: (2,5) and (3,5)
   - Right table: (8,5) and (9,5)
4. Place stage:
   - Select stage tile `(5,3)` from Source 1
   - Position: (5,8) and (6,8)

#### Step 4: Place Decoration (DecorationLayer)
1. Select the `DecorationLayer` node
2. Place bar stools:
   - Select bar stool tile `(0,2)` from Source 1
   - Place 6 stools at y=3, x=3 to 8
3. Place chairs:
   - Select chair tile `(1,2)` from Source 1
   - Left table chair: (2,6) and (3,6)  
   - Right table chair: (8,6) and (9,6)

## 🎮 Step 4: Set Up Character Spawn Points

### 4.1 Configure Marker2D Nodes

Find 4 Marker2D nodes under the `SpawnPoints` node and set their positions:

1. **PlayerSpawn** - Position: `(192, 160)` (center of the room)
2. **BobSpawn** - Position: `(192, 64)` (behind the bar) 
3. **AliceSpawn** - Position: `(96, 192)` (next to the left table)
4. **SamSpawn** - Position: `(288, 192)` (next to the right table)

### 4.2 Configure Interaction Areas

Set Area2D nodes under `InteractiveObjects`:

1. **BarCounter** - Position: `(192, 96)`
2. **Table1** - Position: `(96, 192)`  
3. **Table2** - Position: `(288, 192)`
4. **MusicStage** - Position: `(192, 288)`

## ✅ Completion Checklist

### TileSet Settings Verification:
- [ ] Three tile sources have been added and set with correct textures
- [ ] Important tiles have been manually segmented (floors, walls, doors, furniture)
- [ ] Wall and furniture tiles have collision shapes
- [ ] Floor and decoration tiles have no collision shapes

### Scene Configuration Verification:
- [ ] All 5 TileMap layers are linked to cozy_bar_tileset.tres
- [ ] Z-index settings correct (Floor=0, Wall=1, Furniture=2, Decoration=3, Collision=-1)
- [ ] Y-sort only enabled on Furniture and Decoration layers
- [ ] CollisionLayer is hidden

### Manual Placement Verification:
- [ ] Floors cover all open areas (1-10, 1-8)
- [ ] Outer walls complete, doors in correct positions (5,9)(6,9)
- [ ] Bar counter 6 tiles wide, in correct position
- [ ] 4 chairs and 2 tables placed
- [ ] Stage located at bottom center of the room
- [ ] 6 bar stools and 4 chairs placed

### Spawn Point Verification:
- [ ] 4 character spawn points in correct positions
- [ ] 4 interaction areas in correct positions

## 🔧 Common Issues and Solutions

### Issue 1: Tiles not displayed
- **Check**：TileMap node correctly linked to TileSet resource
- **Check**：Tiles correctly segmented in TileSet

### Issue 2: Collision not working  
- **Check**：Wall and furniture tiles have physics shapes added in TileSet
- **Check**：CollisionLayer contains the same collision tiles

### Issue 3: Rendering order error
- **Check**：Correct Z-index settings for each layer
- **Check**：Furniture and Decoration layers enabled Y-sort

### Issue 4: Y-sorting not working
- **Check**：Y Sort Origin set to 16 (tile center)
- **Check**：Only Furniture and Decoration layers enabled Y-sort

## 🎉 Completion!

You now have a fully manually created Cozy Bar TileMap system! This 12x10 room includes:
- ✨ Complete bar area (6 bar + 6 stools)
- ✨ 2 guest tables (each with 2 chairs)  
- ✨ Music performance stage
- ✨ 4 character spawn points
- ✨ Correct collision detection
- ✨ Layered rendering and Y-sorting

You can now run the game, and characters should move correctly in the room and interact with items!