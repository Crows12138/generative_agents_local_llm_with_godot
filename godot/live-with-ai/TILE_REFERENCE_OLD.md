# 🎨 Cozy Bar 瓦片完整参考指南

## 📚 概述

本文档提供 Cozy Bar TileMap 系统中所有瓦片类型的完整参考，包括精确坐标、属性配置和使用方法。

## 🎯 瓦片资源 (Sources)

### Source 0: TopDownHouse_FloorsAndWalls.png
- **用途**: 地板、墙壁、门、结构元素
- **瓦片大小**: 32x32 像素
- **格式**: PNG 带透明度
- **网格布局**: 6x4 (24个瓦片)

### Source 1: TopDownHouse_FurnitureState1.png  
- **用途**: 家具、桌子、柜台、大型物品
- **瓦片大小**: 32x32 像素
- **格式**: PNG 带透明度
- **网格布局**: 6x4 (24个瓦片)

### Source 2: TopDownHouse_SmallItems.png
- **用途**: 小装饰品、道具、细节
- **瓦片大小**: 32x32 像素
- **格式**: PNG 带透明度
- **网格布局**: 6x2 (12个瓦片)

## 📍 详细瓦片坐标参考

### 🏠 地板和墙壁瓦片 (Source 0)

#### 基础结构瓦片
| 瓦片类型 | 图集坐标 | 地形类型 | 碰撞 | 图层 | 用途描述 |
|----------|----------|----------|------|------|----------|
| 石质地板 | (0,0) | Stone | 否 | Floor | 基础石质地板 |
| 深色地板 | (1,0) | Wood | 否 | Floor | 深色木质地板 |
| 砖墙 | (2,0) | Brick | 是 | Wall | 房间边界墙 |
| 装饰墙 | (3,0) | Door | 否 | Wall | 装饰性墙壁 |
| 浅色石板 | (0,1) | Stone | 否 | Floor | 浅色石质地板 |
| **木质地板** | **(1,1)** | **Wood** | **否** | **Floor** | **主要吧台地板** |
| 砖墙(变种) | (2,1) | Brick | 是 | Wall | 墙壁变种 |
| 墙面装饰 | (3,1) | Door | 否 | Wall | 墙面装饰元素 |
| 石质变种 | (0,2) | Stone | 否 | Floor | 石质地板变种 |
| 木质变种 | (1,2) | Wood | 否 | Floor | 木质地板变种 |
| 墙壁基础 | (2,2) | Brick | 否 | Wall | 基础墙壁样式 |
| **门** | **(3,2)** | **Door** | **否** | **Wall** | **入口/出口点** |

#### 扩展瓦片 (如果有更多行)
| 瓦片类型 | 图集坐标 | 地形类型 | 碰撞 | 图层 | 用途描述 |
|----------|----------|----------|------|------|----------|
| 地毯/地面 | (4,0) | Wood | 否 | Floor | 特殊地面材质 |
| 墙角/边缘 | (5,0) | Brick | 是 | Wall | 墙角装饰 |
| 地面纹理 | (4,1) | Wood | 否 | Floor | 纹理化地面 |
| 墙面细节 | (5,1) | Brick | 是 | Wall | 墙面细节 |

### 🪑 家具瓦片 (Source 1)

#### 主要家具项目
| 瓦片类型 | 图集坐标 | 碰撞 | 图层 | 用途描述 | Y-Sort |
|----------|----------|------|------|----------|--------|
| 装饰品A | (0,0) | 否 | Decoration | 小装饰物品 | 是 |
| 装饰品B | (1,0) | 否 | Decoration | 装饰性物品 | 是 |
| 小家具 | (2,0) | 否 | Furniture | 小型家具 | 是 |
| 储物柜 | (3,0) | 是 | Furniture | 储物家具 | 是 |
| 展示架 | (4,0) | 否 | Furniture | 展示用架子 | 是 |
| 装饰柜 | (5,0) | 是 | Furniture | 装饰性柜子 | 是 |
| 座椅底座 | (0,1) | 否 | Decoration | 椅子底座 | 是 |
| 椅背 | (1,1) | 否 | Decoration | 椅子椅背 | 是 |
| **圆桌** | **(2,1)** | **是** | **Furniture** | **客户用餐桌** | **是** |
| 方桌 | (3,1) | 是 | Furniture | 方形餐桌 | 是 |
| **吧台** | **(4,1)** | **是** | **Furniture** | **主服务吧台** | **是** |
| 高桌 | (5,1) | 是 | Furniture | 高脚桌 | 是 |
| **酒吧凳** | **(0,2)** | **否** | **Decoration** | **吧台座椅** | **是** |
| **椅子** | **(1,2)** | **否** | **Decoration** | **餐桌座椅** | **是** |
| 沙发左 | (2,2) | 是 | Furniture | 沙发左半部 | 是 |
| 沙发右 | (3,2) | 是 | Furniture | 沙发右半部 | 是 |
| 柜台 | (4,2) | 是 | Furniture | 服务柜台 | 是 |
| 酒柜 | (5,2) | 是 | Furniture | 酒水储存柜 | 是 |
| 地毯A | (0,3) | 否 | Decoration | 小地毯 | 否 |
| 地毯B | (1,3) | 否 | Decoration | 装饰地毯 | 否 |
| 植物A | (2,3) | 否 | Decoration | 盆栽植物 | 是 |
| 植物B | (3,3) | 否 | Decoration | 装饰植物 | 是 |
| 钢琴 | (4,3) | 是 | Furniture | 钢琴(可用于舞台) | 是 |
| **舞台** | **(5,3)** | **否** | **Furniture** | **音乐表演舞台** | **是** |

### 🎭 小物品瓦片 (Source 2)

#### 装饰性小物品
| 瓦片类型 | 图集坐标 | 碰撞 | 图层 | 用途描述 | Y-Sort |
|----------|----------|------|------|----------|--------|
| 酒瓶A | (0,0) | 否 | Decoration | 装饰酒瓶 | 是 |
| 酒瓶B | (1,0) | 否 | Decoration | 酒瓶变种 | 是 |
| 玻璃杯A | (2,0) | 否 | Decoration | 玻璃酒杯 | 是 |
| 玻璃杯B | (3,0) | 否 | Decoration | 杯子变种 | 是 |
| 餐具 | (4,0) | 否 | Decoration | 餐具套装 | 是 |
| 调料瓶 | (5,0) | 否 | Decoration | 调料容器 | 是 |
| 烛台A | (0,1) | 否 | Decoration | 蜡烛台 | 是 |
| 烛台B | (1,1) | 否 | Decoration | 装饰烛台 | 是 |
| 花瓶A | (2,1) | 否 | Decoration | 花瓶装饰 | 是 |
| 花瓶B | (3,1) | 否 | Decoration | 花瓶变种 | 是 |
| 书本 | (4,1) | 否 | Decoration | 装饰书籍 | 是 |
| 时钟 | (5,1) | 否 | Decoration | 挂钟/台钟 | 是 |

## 🎯 Tile Placement Rules

### Layer Assignment

| Tile Type | Layer | Z-Index | Sort Mode | Purpose |
|-----------|-------|---------|-----------|---------|
| Floor Wood | Floor | 0 | None | Base ground texture |
| Wall Brick | Wall | 1 | None | Room boundaries |
| Door | Wall | 1 | None | Entry/exit points |
| Bar Counter | Furniture | 2 | Y-Sort | Large furniture |
| Table | Furniture | 2 | Y-Sort | Large furniture |
| Music Stage | Furniture | 2 | Y-Sort | Platform/structure |
| Bar Stool | Decoration | 3 | Y-Sort | Moveable seating |
| Chair | Decoration | 3 | Y-Sort | Moveable seating |

### Collision Properties

| Tile Type | Has Collision | Shape | Size | Notes |
|-----------|---------------|-------|------|-------|
| Wall Brick | ✓ | Rectangle | 32x32 | Full tile blocking |
| Door | ✗ | None | - | Walkable entrance |
| Bar Counter | ✓ | Rectangle | 32x16 | Half-height collision |
| Table | ✓ | Rectangle | 24x24 | Slightly smaller than tile |
| Music Stage | ✓ | Rectangle | 32x16 | Platform edge only |
| Bar Stool | ✗ | None | - | Can walk through |
| Chair | ✗ | None | - | Can walk through |
| Floor Wood | ✗ | None | - | Walkable surface |

## 🗺️ Room Layout Template

```
Grid:  0  1  2  3  4  5  6  7  8  9 10 11
    0  W  W  W  W  W  W  W  W  W  W  W  W
    1  W  .  .  .  .  .  .  .  .  .  .  W
    2  W  .  .  B  B  B  B  B  B  .  .  W
    3  W  .  .  c  c  c  c  c  c  .  .  W
    4  W  .  .  .  .  .  .  .  .  .  .  W
    5  W  .  T  T  .  .  .  .  T  T  .  W
    6  W  .  S  S  .  .  .  .  S  S  .  W
    7  W  .  .  .  .  .  .  .  .  .  .  W
    8  W  .  .  .  .  M  M  .  .  .  .  W
    9  W  W  W  W  W  D  D  W  W  W  W  W
```

### Pixel Coordinates (32px tiles)

```
     0   32  64  96 128 160 192 224 256 288 320 352
  0  +---+---+---+---+---+---+---+---+---+---+---+
 32  |   |   |   |   |   |   |   |   |   |   |   |
 64  |   |   | T | T |   |   |   |   | T | T |   |
 96  |   |   | S | S |   |   |   |   | S | S |   |
128  |   |   |   |   |   |   |   |   |   |   |   |
160  |   |   |   |   |   | M | M |   |   |   |   |
192  +---+---+---+---+---+---+---+---+---+---+---+
```

## 🎮 Interactive Object Mapping

### Bar Counter (B tiles)
- **Location**: Rows 2, Columns 3-8
- **Size**: 6 tiles wide × 1 tile deep
- **Interaction Zone**: Front edge (row 3)
- **Actions**: Order drinks, chat with bartender
- **NPC**: Bob the Bartender (position 6,2)

### Tables (T tiles)
- **Table 1**: Rows 5, Columns 2-3
- **Table 2**: Rows 5, Columns 8-9  
- **Size**: 2×1 tiles each
- **Interaction**: Adjacent chair positions
- **Actions**: Sit, order food, conversation

### Music Stage (M tiles)
- **Location**: Row 8, Columns 5-6
- **Size**: 2 tiles wide × 1 tile deep
- **Performer Position**: Center of stage
- **Actions**: Perform, listen, request songs
- **NPC**: Sam the Musician (position 9,6)

### Seating Arrangements
- **Bar Stools (c)**: 6 stools facing bar counter
- **Chairs (S)**: 4 chairs at tables (2 per table)
- **Customer Positions**: 
  - Alice: Table 1 (position 2,6)
  - Available seats for additional NPCs

## 🔄 Dynamic Tile Operations

### Runtime Tile Placement

```gdscript
# Get the appropriate layer for a tile type
var layer_type = TileMapLayerManager.get_layer_for_tile_type(tile_type)
var target_layer = get_layer_tilemap(layer_type)

# Place tile with proper configuration
var tile_def = tile_definitions[tile_type]
target_layer.set_cell(0, grid_pos, tile_def.source, tile_def.atlas_coords)
```

### Collision Shape Assignment

```gdscript
# Add collision for solid tiles
if tile_def.has_collision:
    var collision_shape = TileCollisionShapes.get_collision_shape(tile_type)
    collision_layer.set_cell(0, grid_pos, tile_def.source, tile_def.atlas_coords)
```

## 🎨 Texture Coordinate Reference

### Calculating Pixel Positions

For 32x32 tiles in texture atlases:

```
pixel_x = atlas_coords.x * 32
pixel_y = atlas_coords.y * 32
texture_rect = Rect2(pixel_x, pixel_y, 32, 32)
```

### Texture Sources

1. **TopDownHouse_FloorsAndWalls.png**
   - Resolution: 256×256 pixels
   - Tile count: 8×8 grid
   - Contains: Floors, walls, doors, windows

2. **TopDownHouse_FurnitureState1.png**  
   - Resolution: 256×256 pixels
   - Tile count: 8×8 grid
   - Contains: Tables, chairs, bar furniture, appliances

3. **TopDownHouse_SmallItems.png**
   - Resolution: 256×256 pixels
   - Tile count: 8×8 grid  
   - Contains: Decorative items, accessories, small objects

## 🔧 Configuration Examples

### Adding Custom Tiles

```gdscript
# Define new tile type
enum TileType {
    # ... existing types
    PIANO = 9
}

# Add to tile definitions
tile_definitions[TileType.PIANO] = {
    "source": 1,
    "atlas_coords": Vector2i(6, 2),
    "has_collision": true
}

# Add to legend for room layout
legend["P"] = TileType.PIANO
```

### Room Layout Variations

```gdscript
# Alternative smaller bar layout (8×6)
var compact_layout = [
    "WWWWWWWW",
    "W......W", 
    "W.BBBB.W",
    "W.cccc.W",
    "W..TT..W",
    "WWWDDWWW"
]
```

## 📐 Measurements and Dimensions

### Standard Tile Sizes
- **Tile Size**: 32×32 pixels
- **Room Size**: 12×10 tiles (384×320 pixels)
- **Bar Counter**: 6×1 tiles (192×32 pixels)
- **Tables**: 2×1 tiles each (64×32 pixels)
- **Music Stage**: 2×1 tiles (64×32 pixels)

### Character Movement
- **Walkable Areas**: Floor tiles (.) and door tiles (D)
- **Movement Grid**: 32-pixel increments
- **Collision Detection**: Rectangle-based, varies by tile type
- **NPC Spawn Spacing**: Minimum 64 pixels apart

## 🎯 Best Practices

### Performance Optimization
- Group similar tiles on same layer
- Use Y-sorting sparingly (only furniture/decoration layers)
- Keep collision on separate hidden layer
- Batch tile operations when possible

### Visual Consistency  
- Maintain 32×32 pixel tile alignment
- Use consistent lighting across similar tile types
- Apply appropriate collision shapes for each furniture type
- Test walkability paths between key areas

### Expandability
- Reserve atlas space for future tile additions
- Design modular room sections for easy modification
- Document custom tile coordinates for team reference
- Plan collision shapes to allow smooth character movement