# 🎨 Cozy Bar 瓦片快速参考

## 📍 重要瓦片坐标

### Source 0: TopDownHouse_FloorsAndWalls.png
- **木质地板** `(1,1)` - 无碰撞
- **砖墙** `(2,0)` - 有碰撞  
- **门** `(3,2)` - 无碰撞

### Source 1: TopDownHouse_FurnitureState1.png
- **吧台** `(4,1)` - 有碰撞
- **桌子** `(2,1)` - 有碰撞
- **音乐舞台** `(5,3)` - 无碰撞
- **吧台凳** `(0,2)` - 无碰撞
- **椅子** `(1,2)` - 无碰撞

### Source 2: TopDownHouse_SmallItems.png
- 装饰物品 - 根据需要选择，通常无碰撞

## 🗂️ 图层分配

| 瓦片类型 | 图层 | Z-Index | Y-Sort |
|----------|------|---------|--------|
| 地板 | FloorLayer | 0 | 否 |
| 墙壁、门 | WallLayer | 1 | 否 |
| 家具(大) | FurnitureLayer | 2 | 是 |
| 装饰(小) | DecorationLayer | 3 | 是 |

## 📐 12x10 房间布局速记

```
WWWWWWWWWWWW  ← 墙壁
W..........W  ← 地板  
W..BBBBBB..W  ← 吧台
W..cccccc..W  ← 吧台凳
W..........W  ← 地板
W.TT....TT.W  ← 桌子
W.SS....SS.W  ← 椅子  
W..........W  ← 地板
W....MM....W  ← 舞台
WWWWWDDWWWWW  ← 墙壁+门
```

详细指导请参考：`COZY_BAR_TILEMAP_MANUAL_GUIDE.md`