from basic_functions.maze import Maze, Persona, Object
from basic_functions.perception.describe import describe_cell, gather_surrounding_descriptions
from basic_functions.perception.embedding import get_embedding, get_single_embedding
from basic_functions.perception.perceive import perceive


def test_describe_cell_includes_objects_and_agent():
    maze = Maze(width=1, height=1, depth=1)

    class Rock(Object):
        def describe(self):
            return "Rock heavy"

    maze.place_object(Rock(name="Rock", x=0, y=0, z=0), 0, 0, 0)
    alice = Persona(name="Alice", initial_location=(0, 0, 0))
    maze.place_agent(alice, 0, 0, 0)

    description = describe_cell(0, 0, 0, maze)
    assert "Rock heavy" in description
    assert "Persona Alice" in description


def test_gather_surrounding_descriptions_radius_1():
    maze = Maze(width=3, height=3, depth=1)
    bob = Persona(name="Bob", initial_location=(1.0, 1.0, 0.0))
    maze.place_agent(bob, 1, 1, 0)

    class Fish(Object):
        def describe(self):
            return "Fish swims"

    fish = Fish(name="Fish", x=1, y=2, z=0)
    maze.place_object(fish, 1, 2, 0)

    descriptions = gather_surrounding_descriptions(bob, maze, radius=1)
    assert any("Fish swims" in d for d in descriptions)


def test_embedding_functions_simple_text():
    texts = ["abc", "abbc"]
    embs = get_embedding(texts)
    assert isinstance(embs, list) and len(embs) == 2
    assert all(isinstance(e, list) for e in embs)

    single = get_single_embedding("hello")
    assert isinstance(single, list)


def test_perceive_composes_correct_structure():
    maze = Maze(width=3, height=3, depth=1)
    carol = Persona(name="Carol", initial_location=(0.0, 0.0, 0.0))
    maze.place_agent(carol, 0, 0, 0)

    result_empty = perceive(carol, maze, radius=1)
    assert result_empty["descriptions"] == []
    assert result_empty["embeddings"] == []
    assert result_empty["merged_embedding"] == []

    class Fish2(Object):
        def describe(self):
            return "Fish swims"

    fish2 = Fish2(name="Fish2", x=0, y=1, z=0)
    maze.place_object(fish2, 0, 1, 0)
    result = perceive(carol, maze, radius=1)
    assert isinstance(result, dict)
    assert isinstance(result["descriptions"], list)
    assert isinstance(result["embeddings"], list)
    assert isinstance(result["merged_embedding"], list)
    assert any("Fish swims" in d for d in result["descriptions"])
