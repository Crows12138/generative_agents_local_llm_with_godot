import json

import pytest

from basic_functions.decider import decider
from basic_functions.decider.decider import ActionIntent
from basic_functions.persona import Persona
from basic_functions.maze import Maze, Object


def test_llm_planner_parses_json(monkeypatch):
    def fake_generate(prompt: str) -> str:
        return json.dumps({"action": "create", "target": "torch"})

    monkeypatch.setattr(decider, "local_llm_generate", fake_generate)

    planner = decider.BottomDecider()
    intent = planner.decide(
        persona_name="Alice",
        location=(0, 0, 0),
        self_identification="Alice",
        surroundings_desc="plain tile",
        similar_memories=[],
        high_level_task="craft something",
    )
    assert isinstance(intent, ActionIntent)
    assert intent.action_type == "create"
    assert intent.target == "torch"


def test_persona_create_action_places_item():
    maze = Maze(3, 3, 1)
    persona = Persona(name="Bob", initial_location=(1, 1, 0))
    maze.place_agent(persona, 1, 1, 0)

    intent = ActionIntent("create", "torch")
    persona.perform_action(intent, maze)

    nearby = maze.spatial.nearby(1, 1, 0, radius=0.0)
    assert any(isinstance(obj, Object) and obj.name == "torch" for obj in nearby)
    assert any(obj.name == "torch" for obj in persona.inventory)
