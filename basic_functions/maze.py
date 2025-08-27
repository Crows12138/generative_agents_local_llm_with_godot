from __future__ import annotations
from typing import List, Tuple
from collections import deque, defaultdict
from typing import Union
from basic_functions.persona import Persona

class Area:
    def __init__(self, name: str, x1: int, y1: int, x2: int, y2: int):
        self.name = name
        self.x1, self.y1 = x1, y1
        self.x2, self.y2 = x2, y2

    def contains(self, x: int, y: int) -> bool:
        return self.x1 <= x <= self.x2 and self.y1 <= y <= self.y2
class Object:
    def __init__(
        self,
        name: str,
        x: float,
        y: float,
        z: float = 0.0,
        description: str = "",
        blocking: bool = False,
        edible: bool = False,
        portable: bool = True,
        interactable: bool = True,
        object_type: str = "general",
    ):
        self.name = name
        self.x, self.y, self.z = float(x), float(y), float(z)
        self.description = description
        self.blocking = blocking
        self.edible = edible
        self.portable = portable
        self.interactable = interactable
        self.object_type = object_type
        

    def describe(self) -> str:
        """Return a human-readable description of this object."""
        return f"Object {self.name}"


Entity = Union[Object, Persona]

class SpatialHash:
    def __init__(self, cell_size: float):
        """
        Initialize a 3D spatial hash.

        Args:
            cell_size: length of one side of each cubic cell (bucket).
        """
        self.cell_size = cell_size
        self.buckets: defaultdict[Tuple[int,int,int], List[Entity]] = defaultdict(list)

    def _cell_key(self, x: float, y: float, z: float) -> Tuple[int, int, int]:
        """
        Convert a 3D point into discrete cell coordinates.

        Returns:
            A tuple of integers (i, j, k) identifying the cell.
        """
        return (
            int(x // self.cell_size),
            int(y // self.cell_size),
            int(z // self.cell_size),
        )

    def insert(self, e: Entity) -> None:
        """
        Add an entity to the spatial hash at its current position.

        Args:
            e: The Object or Persona to insert.
        """
        key = self._cell_key(e.x, e.y, e.z)
        self.buckets.setdefault(key, []).append(e)

    def move(self, e: Entity, new_x: float, new_y: float, new_z: float) -> None:
        """
        Update an entity's position and relocate it between cells if needed.

        Args:
            e: The entity to move.
            new_x: New X coordinate.
            new_y: New Y coordinate.
            new_z: New Z coordinate.
        """
        old_key = self._cell_key(e.x, e.y, e.z)
        new_key = self._cell_key(new_x, new_y, new_z)
        if old_key != new_key:
            self.buckets[old_key].remove(e)
            self.buckets.setdefault(new_key, []).append(e)
        e.x, e.y, e.z = new_x, new_y, new_z
    def remove(self, e: Entity) -> None:
        """
        Remove an entity from the spatial hash.

        Args:
            e: The Object or Persona to remove.
        """
        key = self._cell_key(e.x, e.y, e.z)
        if key in self.buckets:
            self.buckets[key].remove(e)
            if not self.buckets[key]:
                del self.buckets[key]

    def nearby(self, x: float, y: float, z: float, radius: float) -> List[Entity]:
        """
        Retrieve all entities within a given Euclidean radius of a point.

        This method uses a two-stage approach:
          1. Coarse: gather entities from cells that intersect the search sphere.
          2. Fine: filter by exact distance.

        Args:
            x: X coordinate of the query center.
            y: Y coordinate of the query center.
            z: Z coordinate of the query center.
            radius: Search radius.

        Returns:
            A list of entities whose distance to (x, y, z) is <= radius.
        """
        cx, cy, cz = self._cell_key(x, y, z)
        span = int(radius // self.cell_size) + 1

        candidates: List[Entity] = []
        for i in range(cx - span, cx + span + 1):
            for j in range(cy - span, cy + span + 1):
                for k in range(cz - span, cz + span + 1):
                    candidates.extend(self.buckets.get((i, j, k), []))

        r2 = radius * radius
        return [e for e in candidates if (e.x - x)**2 + (e.y - y)**2 + (e.z - z)**2 <= r2]


class Maze:
    def __init__(self, width: int, height: int, depth: int = 1):
        """
        Initialize a 3D maze with walls and a spatial hash for entities.

        Args:
            width:  Number of columns.
            height: Number of rows.
            depth:  Number of layers.
        """
        self.width = width
        self.height = height
        self.depth = depth
        self.walls = [[[False for _ in range(depth)] for _ in range(height)] for _ in range(width)]
        self.spatial = SpatialHash(cell_size=1.0)
        self.areas: List[Area] = []

    def set_wall(self, x: int, y: int, z: int = 0) -> None:
        """Mark the cell at (x, y, z) as a wall."""
        self.walls[x][y][z] = True

    def add_area(self, area: Area, override: bool = False):
        def overlaps(a: Area, b: Area) -> bool:
            return not (a.x2 < b.x1 or b.x2 < a.x1 or a.y2 < b.y1 or b.y2 < a.y1)

        if override:
            self.areas.insert(0, area)
        else:
            self.areas.append(area)

    def get_region(self, x: int, y: int, z: int = 0) -> str:
        for area in self.areas:
            if area.contains(x, y):
                return area.name
        return ""
    
    def place_object(self, obj: Object, x: int, y: int, z: int = 0) -> None:
        """Place an object at integer grid coordinates and register in spatial hash."""
        obj.x, obj.y, obj.z = float(x), float(y), float(z)
        self.spatial.insert(obj)

    def place_agent(self, agent: Persona, x: int, y: int, z: int = 0) -> None:
        """Place an agent at integer grid coordinates and register in spatial hash."""
        agent.x, agent.y, agent.z = float(x), float(y), float(z)
        agent.location = (agent.x, agent.y, agent.z)
        self.spatial.insert(agent)
        self.agent = agent
        
    def remove_agent(self, agent: Persona) -> None:
        """Remove an agent from the maze and spatial hash."""
        self.spatial.remove(agent)
        if hasattr(self, 'agent') and self.agent == agent:
            del self.agent

    def get_walkable_neighbors(self, x: int | float, y: int | float, z: int | float = 0) -> List[Tuple[int, int, int]]:
        """Return walkable neighbor coordinates (6-way) of (x, y, z).

        ``x``, ``y`` and ``z`` might be provided as floats (e.g. from
        ``Persona.location``).  They are converted to integers before being
        used for indexing the maze grid.
        """

        ix, iy, iz = int(x), int(y), int(z)
        neighbors: List[Tuple[int, int, int]] = []
        for dx, dy, dz in [(-1, 0, 0), (1, 0, 0), (0, -1, 0), (0, 1, 0), (0, 0, -1), (0, 0, 1)]:
            nx, ny, nz = ix + dx, iy + dy, iz + dz
            if not (
                0 <= nx < self.width and 0 <= ny < self.height and 0 <= nz < self.depth
            ):
                continue
            if self.walls[nx][ny][nz]:
                continue

            ents = self.spatial.nearby(nx, ny, nz, radius=0.0)
            if any(isinstance(e, Object) and e.blocking for e in ents):
                continue

            neighbors.append((nx, ny, nz))
        return neighbors

    def find_path(self, start: Tuple[int | float, int | float, int | float], target_name: str) -> List[Tuple[int, int, int]]:
        """
        Find shortest path (4-way) from ``start`` to any cell containing an object
        whose description includes ``target_name`` (case-insensitive).

        ``start`` may contain float coordinates. They are converted to integers
        before the search begins.

        Returns a list of ``(x, y, z)`` coordinates including the target cell.
        """

        sx, sy, sz = int(start[0]), int(start[1]), int(start[2])
        visited: set[tuple[int, int, int]] = set()
        queue: deque[tuple[tuple[int, int, int], list[tuple[int, int, int]]]] = deque([((sx, sy, sz), [])])

        for ent in self.spatial.nearby(sx, sy, sz, radius=0.0):
            if target_name.lower() in ent.describe().lower():
                return [(sx, sy, sz)]

        while queue:
            (cx, cy, cz), path = queue.popleft()
            if (cx, cy, cz) in visited:
                continue
            visited.add((cx, cy, cz))

            # Check objects at this cell
            for ent in self.spatial.nearby(cx, cy, cz, radius=0.0):
                if target_name.lower() in ent.describe().lower():
                    return path + [(cx, cy, cz)]

            # Enqueue neighbors
            for nx, ny, nz in self.get_walkable_neighbors(cx, cy, cz):
                if (nx, ny, nz) not in visited:
                    queue.append(((nx, ny, nz), path + [(cx, cy, cz)]))

        return []
