# 🎯 Cozy Bar TileMap 手动创建指南

## 📋 概述

这是一个完整的指南，教你如何在Godot编辑器中**手动**创建Cozy Bar的TileMap系统，包括TileSet资源的分割和瓦片的手动放置。

## 🛠️ 第一步：设置TileSet资源

### 1.1 创建TileSet资源

1. 在Godot编辑器中打开 `tilesets/cozy_bar_tileset.tres`
2. 如果文件不存在，右键点击 `tilesets` 文件夹 → **New Resource** → 选择 **TileSet**
3. 保存为 `cozy_bar_tileset.tres`

### 1.2 添加瓦片源 (TileSetAtlasSource)

在TileSet编辑器中：

#### 📁 Source 0: 地板和墙壁
1. 点击 **Add Source** → 选择 **Atlas**
2. 拖拽 `assets/sprites/tiles/v2/TopDownHouse_FloorsAndWalls.png` 到 **Texture** 字段
3. 设置 **Texture Region Size** 为 `32x32`
4. **ID** 保持为 `0`

#### 📁 Source 1: 家具
1. 点击 **Add Source** → 选择 **Atlas** 
2. 拖拽 `assets/sprites/tiles/v2/TopDownHouse_FurnitureState1.png` 到 **Texture** 字段
3. 设置 **Texture Region Size** 为 `32x32`
4. **ID** 设置为 `1`

#### 📁 Source 2: 小物品
1. 点击 **Add Source** → 选择 **Atlas**
2. 拖拽 `assets/sprites/tiles/v2/TopDownHouse_SmallItems.png` 到 **Texture** 字段  
3. 设置 **Texture Region Size** 为 `32x32`
4. **ID** 设置为 `2`

### 1.3 手动分割瓦片

对每个Source，你需要手动点击来分割瓦片：

#### Source 0 (地板和墙壁) - 重要瓦片分割：
1. **木质地板** - 点击坐标 `(1,1)` 的瓦片
   - 右侧面板设置 **Setup** → 确认瓦片已被添加
   
2. **砖墙** - 点击坐标 `(2,0)` 的瓦片
   - 在 **Physics** 标签页 → 点击 **Add** → 选择矩形形状
   - 调整碰撞形状覆盖整个瓦片

3. **门** - 点击坐标 `(3,2)` 的瓦片
   - **Setup** → 添加瓦片
   - **不要**添加物理碰撞（门可以通过）

#### Source 1 (家具) - 重要瓦片分割：
1. **吧台** - 点击坐标 `(4,1)` 的瓦片
   - **Physics** → 添加矩形碰撞形状
   
2. **桌子** - 点击坐标 `(2,1)` 的瓦片
   - **Physics** → 添加矩形碰撞形状
   
3. **音乐舞台** - 点击坐标 `(5,3)` 的瓦片
   - **Setup** → 添加瓦片
   - **不添加碰撞**（可以站在舞台上）
   
4. **吧台凳** - 点击坐标 `(0,2)` 的瓦片
   - **Setup** → 添加瓦片，**无碰撞**
   
5. **椅子** - 点击坐标 `(1,2)` 的瓦片
   - **Setup** → 添加瓦片，**无碰撞**

## 🏗️ 第二步：设置场景中的TileMap图层

### 2.1 打开cozy_bar.tscn场景

在场景编辑器中找到 `TileMaps` 节点下的5个TileMap图层：

### 2.2 配置每个图层

#### FloorLayer (地板层)
1. 选中 `FloorLayer` 节点
2. 在检视器中设置：
   - **Tile Set** → 选择 `cozy_bar_tileset.tres`
   - **Layer 0** → **Z Index** = `0`
   - **Layer 0** → **Y Sort Enabled** = `false`

#### WallLayer (墙壁层)  
1. 选中 `WallLayer` 节点
2. 设置：
   - **Tile Set** → 选择 `cozy_bar_tileset.tres`
   - **Layer 0** → **Z Index** = `1`
   - **Layer 0** → **Y Sort Enabled** = `false`

#### FurnitureLayer (家具层)
1. 选中 `FurnitureLayer` 节点
2. 设置：
   - **Tile Set** → 选择 `cozy_bar_tileset.tres`
   - **Layer 0** → **Z Index** = `2`
   - **Layer 0** → **Y Sort Enabled** = `true` ⚠️ **重要**
   - **Layer 0** → **Y Sort Origin** = `16`

#### DecorationLayer (装饰层)
1. 选中 `DecorationLayer` 节点
2. 设置：
   - **Tile Set** → 选择 `cozy_bar_tileset.tres`
   - **Layer 0** → **Z Index** = `3`
   - **Layer 0** → **Y Sort Enabled** = `true` ⚠️ **重要**
   - **Layer 0** → **Y Sort Origin** = `16`

#### CollisionLayer (碰撞层)
1. 选中 `CollisionLayer` 节点
2. 设置：
   - **Tile Set** → 选择 `cozy_bar_tileset.tres`
   - **Layer 0** → **Z Index** = `-1`
   - **Layer 0** → **Enabled** = `false` (隐藏)
   - **Layer 0** → **Y Sort Enabled** = `false`

## 🎨 第三步：手动放置瓦片创建12x10房间

### 3.1 房间布局设计

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

### 3.2 开始手动放置

#### 步骤1: 放置地板 (FloorLayer)
1. 选中 `FloorLayer` 节点
2. 在TileMap编辑器中：
   - 选择 **Source 0**
   - 点击木质地板瓦片 `(1,1)`
   - 使用画笔工具在所有 `.` 位置放置地板瓦片
   - 覆盖范围：x=1到10, y=1到8

#### 步骤2: 放置墙壁 (WallLayer)  
1. 选中 `WallLayer` 节点
2. 放置外墙：
   - 选择砖墙瓦片 `(2,0)` from Source 0
   - 放置顶部墙壁：y=0, x=0到11
   - 放置左右墙壁：x=0和x=11, y=1到8  
   - 放置底部墙壁：y=9, x=0到4和x=7到11
3. 放置门：
   - 选择门瓦片 `(3,2)` from Source 0
   - 放置在 `(5,9)` 和 `(6,9)` 位置

#### 步骤3: 放置家具 (FurnitureLayer)
1. 选中 `FurnitureLayer` 节点
2. 放置吧台：
   - 选择吧台瓦片 `(4,1)` from Source 1
   - 在y=2, x=3到8放置6个吧台瓦片
3. 放置桌子：
   - 选择桌子瓦片 `(2,1)` from Source 1  
   - 左桌：(2,5) 和 (3,5)
   - 右桌：(8,5) 和 (9,5)
4. 放置舞台：
   - 选择舞台瓦片 `(5,3)` from Source 1
   - 位置：(5,8) 和 (6,8)

#### 步骤4: 放置装饰 (DecorationLayer)
1. 选中 `DecorationLayer` 节点
2. 放置吧台凳：
   - 选择吧台凳瓦片 `(0,2)` from Source 1
   - 在y=3, x=3到8放置6个凳子
3. 放置椅子：
   - 选择椅子瓦片 `(1,2)` from Source 1
   - 左桌椅子：(2,6) 和 (3,6)  
   - 右桌椅子：(8,6) 和 (9,6)

## 🎮 第四步：设置角色生成点

### 4.1 配置Marker2D节点

在 `SpawnPoints` 节点下找到4个Marker2D节点并设置位置：

1. **PlayerSpawn** - Position: `(192, 160)` (房间中央)
2. **BobSpawn** - Position: `(192, 64)` (吧台后方) 
3. **AliceSpawn** - Position: `(96, 192)` (左桌旁)
4. **SamSpawn** - Position: `(288, 192)` (右桌旁)

### 4.2 配置交互区域

设置 `InteractiveObjects` 下的Area2D节点：

1. **BarCounter** - Position: `(192, 96)`
2. **Table1** - Position: `(96, 192)`  
3. **Table2** - Position: `(288, 192)`
4. **MusicStage** - Position: `(192, 288)`

## ✅ 完成检查清单

### TileSet设置验证：
- [ ] 三个瓦片源已添加并设置正确的纹理
- [ ] 重要瓦片已手动分割（地板、墙壁、门、家具）
- [ ] 墙壁和家具瓦片有碰撞形状
- [ ] 地板、装饰瓦片无碰撞形状

### 场景配置验证：
- [ ] 5个TileMap图层都链接到cozy_bar_tileset.tres
- [ ] Z-index设置正确（Floor=0, Wall=1, Furniture=2, Decoration=3, Collision=-1）
- [ ] Y-sort仅在Furniture和Decoration层启用
- [ ] CollisionLayer已隐藏

### 手动放置验证：
- [ ] 地板覆盖所有开放区域 (1-10, 1-8)
- [ ] 外墙完整，门在正确位置 (5,9)(6,9)
- [ ] 吧台6格宽，位于正确位置
- [ ] 4张椅子和2张桌子已放置
- [ ] 舞台位于房间底部中央
- [ ] 6个吧台凳和4个椅子已放置

### 生成点验证：
- [ ] 4个角色生成点位置正确
- [ ] 4个交互区域位置正确

## 🔧 常见问题解决

### 问题1: 瓦片不显示
- **检查**：TileMap节点是否正确链接到TileSet资源
- **检查**：瓦片是否在TileSet中已正确分割

### 问题2: 碰撞不工作  
- **检查**：墙壁和家具瓦片是否在TileSet中添加了物理形状
- **检查**：CollisionLayer是否包含相同的碰撞瓦片

### 问题3: 渲染顺序错误
- **检查**：各图层Z-index设置是否正确
- **检查**：Furniture和Decoration层是否启用了Y-sort

### 问题4: Y-sorting不工作
- **检查**：Y Sort Origin设置为16（瓦片中心）
- **检查**：只有Furniture和Decoration层启用Y-sort

## 🎉 完成！

你现在拥有一个完全手动创建的Cozy Bar TileMap系统！这个12x10的房间包含：
- ✨ 完整的吧台区域（6格吧台 + 6个凳子）
- ✨ 2张客桌（各有2把椅子）  
- ✨ 音乐表演舞台
- ✨ 4个角色生成点
- ✨ 正确的碰撞检测
- ✨ 分层渲染和Y-sorting

现在可以运行游戏，角色应该能正确地在房间中移动并与物品交互！