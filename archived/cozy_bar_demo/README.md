# 🍻 Cozy Bar Demo

A minimal runnable AI agent bar scene that demonstrates intelligent NPC behavior and interaction in a virtual environment.

## Features

- **Intelligent NPCs**: Bar characters with memory, emotions, and personality
- **Dynamic Dialogue**: Dialogue generation based on character state and context
- **Real-time Simulation**: Time passage affects character behavior and mood
- **Interaction System**: Dialogue with NPCs and observe their behavior
- **Colored Text Interface**: Intuitive visual display

## Character Introduction

### 🍸 Bob (Bartender)
- **Role**: Experienced bartender
- **Personality**: Friendly, talkative, professional
- **Behavior**: Mixing cocktails, chatting with customers, managing the bar

### 🥃 Alice (Regular Customer)
- **Role**: Bar regular
- **Personality**: Introspective, philosophical, sometimes melancholic
- **Behavior**: Drinking alone, sharing life stories, deep thinking

### 🎵 Sam (Musician)
- **Role**: Resident musician
- **Personality**: Artistic temperament, passionate, creative
- **Behavior**: Performing music, interacting with audience, composing songs

## Installation and Running

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run Demo
```bash
python main.py
```

## Game Commands

- `help` or `h` - Show help information
- `look` or `l` - View bar scene
- `status` or `s` - View character status
- `events` or `e` - Show recent events
- `menu` or `m` - Show bar menu
- `talk <name>` - Talk to character (e.g.: `talk Bob`)
- `wait` or `w` - Wait for time to pass
- `auto` or `a` - Auto simulation mode
- `quit` or `q` - Exit game

## Scene Map

```
█████████████████████████
█  ·  ·  ·  ·  ·  ·  ·  █
█  ·  ▬▬▬▬▬▬▬▬▬▬  ·  █
█  ·  ⌒⌒⌒⌒⌒⌒⌒⌒  ·  █
█  ·  ·  ·  ·  ·  ·  ·  █
█  ·  ○○  ·  ·  ○○  ·  █
█  ·  ◑◑  ·  ·  ◑◑  ·  █
█  ·  ·  ·  ·  ·  ·  ·  █
█  ·  ·  ·  ♪♪  ·  ·  █
██████████◊██████████████
```

**Legend**:
- █ Brick wall
- ◊ Door
- ▬ Bar counter
- ⌒ Bar stool
- ○ Table
- ◑ Chair
- ♪ Stage
- · Wooden floor

## Project Structure

```
cozy_bar_demo/
├── main.py                 # Main startup script
├── requirements.txt        # Python dependencies
├── README.md              # Project description
├── config/
│   └── room_config.json   # Room configuration
├── core/
│   ├── bar_agents.py      # AI agent system
│   └── bar_renderer.py    # Scene renderer
└── prompts/               # Prompt templates (reserved)
```

## Extension Features

This demo provides a foundation framework for more complex features:

1. **AI Dialogue**: Can integrate LLM for more natural conversations
2. **Complex Behavior**: Add more character behaviors and interaction patterns
3. **Physics Engine**: Implement character movement and physical interactions
4. **Graphical Interface**: Replace text interface with graphical interface
5. **Multi-room**: Extend to multiple connected rooms
6. **Task System**: Add character goals and tasks
7. **Emotion Model**: More complex emotion and relationship systems

## Technical Features

- **Modular Design**: Clear code structure, easy to extend
- **Configuration Driven**: Define scenes through JSON configuration files
- **Event System**: Time-based event simulation
- **State Management**: Character state persistence and updates
- **Interactive Interface**: User-friendly command line interface

This demo demonstrates how to build a basic AI agent system, laying the foundation for larger-scale virtual worlds.