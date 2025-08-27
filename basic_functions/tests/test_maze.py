import pytest
from basic_functions.maze import Maze, Area, SpatialHash, Object, Persona


def test_insert_and_nearby():
    sh = SpatialHash(cell_size=1.0)
    obj = Object("A", 0.0, 0.0, 0.0)
    persona = Persona("P", (2.0, 0.0, 0.0))
    sh.insert(obj)
    sh.insert(persona)
    near = sh.nearby(0.0, 0.0, 0.0, radius=1.5)
    assert obj in near
    assert persona not in near


def test_area_definition_and_override():
    m = Maze(5, 5, 1)
    area1 = Area("Forest", 0, 0, 2, 2)
    area2 = Area("Garden", 1, 1, 3, 3)
    m.add_area(area1)
    assert m.get_region(1, 1, 0) == "Forest"
    assert m.get_region(3, 3, 0) == ""
    m.add_area(area2, override=False)
    assert m.get_region(2, 2, 0) == "Forest"
    m.add_area(area2, override=True)
    assert m.get_region(2, 2, 0) == "Garden"
    assert m.get_region(0, 0, 0) == "Forest"


def test_move_updates_spatial_hash():
    sh = SpatialHash(cell_size=1.0)
    e = Object("Mover", 0.0, 0.0, 0.0)
    sh.insert(e)
    assert e in sh.nearby(0.0, 0.0, 0.0, radius=0.5)
    sh.move(e, 5.0, 5.0, 5.0)
    assert e not in sh.nearby(0.0, 0.0, 0.0, radius=1.0)
    assert e in sh.nearby(5.0, 5.0, 5.0, radius=0.1)


def test_maze_walls_and_neighbors():
    m = Maze(3, 3, 1)
    m.set_wall(1, 1, 0)
    neigh = m.get_walkable_neighbors(1, 0, 0)
    assert (1, 1, 0) not in neigh
    assert (0, 0, 0) in neigh and (2, 0, 0) in neigh


def test_maze_blocking_object():
    m = Maze(3, 3, 1)
    chest = Object("Chest", 1, 1, 0.0, blocking=True)
    m.place_object(chest, 1, 1, 0)
    neigh = m.get_walkable_neighbors(1, 0, 0)
    assert (1, 1, 0) not in neigh


def test_maze_place_and_find_object():
    m = Maze(5, 5, 1)
    gem = Object("Gem", 0, 0, 0)
    m.place_object(gem, 2, 2, 0)
    path = m.find_path((0, 0, 0), "gem")
    assert path and path[-1] == (2, 2, 0)


def test_maze_find_path_start():
    m = Maze(4, 4, 1)
    key = Object("Key", 0, 0, 0)
    m.place_object(key, 0, 0, 0)
    path = m.find_path((0, 0, 0), "key")
    assert path == [(0, 0, 0)]


def test_maze_place_and_remove_agent():
    m = Maze(2, 2, 1)
    hero = Persona("Hero", (0, 0, 0))
    m.place_agent(hero, 1, 1, 0)
    assert hero in m.spatial.nearby(1, 1, 0.0, radius=0.0)
    m.remove_agent(hero)
    assert hero not in m.spatial.nearby(1, 1, 0.0, radius=0.0)
    assert not hasattr(m, 'agent') or m.agent is not hero
