# Godot Scene Configuration Guide

## AI Character Scene (ai_character.tscn)

### 📋 Node Structure Overview

```
AICharacter (CharacterBody2D)
├── Sprite2D - Character visual representation
├── CollisionShape2D - Physics collision
├── InteractionArea (Area2D) - Detection zone for player interaction
│   └── InteractionShape (CollisionShape2D) - Interaction range
├── NameLabel (Label) - Character name display
├── ThoughtBubble (Control) - AI thinking visualization
│   ├── BubbleBackground (NinePatchRect) - Bubble styling
│   ├── ThoughtText (RichTextLabel) - Thought content
│   └── BubbleTimer (Timer) - Auto-hide timer
├── EmotionIndicator (ColorRect) - Emotion state display
├── StatusUI (Control) - Character stats panel
│   ├── EnergyBar (ProgressBar) - Energy level
│   ├── HungerBar (ProgressBar) - Hunger level
│   └── SocialBar (ProgressBar) - Social need level
├── NavigationAgent2D - AI pathfinding
└── StateLabel (Label) - Debug state display
```

### 🎯 Required Node Configuration

#### 1. Root Node (AICharacter)
- **Type**: CharacterBody2D
- **Script**: `res://scripts/character_controller.gd`
- **Export Variables**:
  - `character_name`: String = "Alice"
  - `personality`: String = "Friendly and curious resident"
  - `move_speed`: float = 100.0
  - `interaction_range`: float = 50.0

#### 2. Sprite2D
- **Purpose**: Visual representation of character
- **Requirements**: 
  - Must have texture assigned
  - Set `centered = true`
- **Recommended Size**: 32x48 pixels for consistency

#### 3. CollisionShape2D
- **Shape**: RectangleShape2D (32x48)
- **Purpose**: Physics collision for movement
- **Position**: Centered on character

#### 4. InteractionArea (Area2D)
- **Purpose**: Detect nearby characters and player
- **Child Nodes**: InteractionShape (CollisionShape2D)
- **Shape**: CircleShape2D with radius = 50.0
- **Signals**: Connected to character_controller.gd methods:
  - `body_entered` → `_on_character_entered_range`
  - `body_exited` → `_on_character_exited_range`

#### 5. NameLabel (Label)
- **Position**: Above character head (-30, -35, 30, -20)
- **Alignment**: Center horizontal and vertical
- **Text**: Character name (set via script)
- **Font Size**: 14px recommended

#### 6. ThoughtBubble (Control)
- **Position**: Above character (-60, -80, 60, -40)
- **Initial State**: `visible = false`
- **Purpose**: Display AI thoughts and dialogue

##### ThoughtBubble Children:
- **BubbleBackground** (NinePatchRect):
  - Color: (1, 1, 1, 0.9) - Semi-transparent white
  - Fills parent container
  
- **ThoughtText** (RichTextLabel):
  - Margins: 5px on all sides
  - `fit_content = true`
  - `scroll_active = false`
  
- **BubbleTimer** (Timer):
  - `wait_time = 3.0`
  - `one_shot = true`
  - Signal: `timeout` → `_on_bubble_timer_timeout`

#### 7. EmotionIndicator (ColorRect)
- **Size**: 8x8 pixels
- **Position**: Top-right of character (20, -25, 28, -17)
- **Purpose**: Visual emotion state indication
- **Color**: Changes based on emotion (white default)

#### 8. StatusUI (Control)
- **Position**: Below character (-40, 20, 40, 50)
- **Initial State**: `visible = false`
- **Purpose**: Debug/status display for character needs

##### StatusUI Children:
- **EnergyBar** (ProgressBar): Top section (0-30% height)
- **HungerBar** (ProgressBar): Middle section (35-65% height), orange tint
- **SocialBar** (ProgressBar): Bottom section (70-100% height), blue tint

#### 9. NavigationAgent2D
- **Purpose**: AI pathfinding for movement
- **Configuration**:
  - `path_desired_distance = 4.0`
  - `target_desired_distance = 4.0`
  - `path_max_distance = 10.0`
  - `navigation_layers = 1`

#### 10. StateLabel (Label)
- **Position**: Below status UI (-40, 30, 40, 45)
- **Initial State**: `visible = false`
- **Purpose**: Debug display of current AI state

---

## Dialogue UI Scene (dialogue_ui.tscn)

### 📋 Node Structure Overview

```
DialogueUI (Control)
├── BackgroundDimmer (ColorRect) - Screen overlay
├── DialogueBox (Panel) - Main dialogue container
│   ├── SpeakerLabel (Label) - Character name
│   ├── Portrait (TextureRect) - Character image
│   ├── DialogueText (RichTextLabel) - Main dialogue content
│   ├── ChoicesContainer (VBoxContainer) - Choice buttons
│   └── ContinueButton (Button) - Continue dialogue
├── TypewriterTimer (Timer) - Text typing effect
└── AutoContinueTimer (Timer) - Auto-advance dialogue
```

### 🎯 Required Node Configuration

#### 1. Root Node (DialogueUI)
- **Type**: Control
- **Anchors**: Fill entire screen (15 preset)
- **Script**: `res://scripts/dialogue_system.gd`
- **Export Variables**:
  - `typing_speed`: float = 0.03
  - `auto_continue`: bool = false
  - `auto_continue_delay`: float = 2.5

#### 2. BackgroundDimmer (ColorRect)
- **Purpose**: Semi-transparent overlay behind dialogue
- **Anchors**: Fill screen (15 preset)
- **Color**: (0, 0, 0, 0.3) - 30% black
- **Z-Index**: -1 (behind dialogue box)

#### 3. DialogueBox (Panel)
- **Position**: Bottom of screen (50, -200, -50, -20)
- **Anchors**: Bottom-left + Bottom-right (12 preset)
- **Style**: Custom StyleBoxFlat with:
  - Background: (0.1, 0.1, 0.15, 0.95) - Dark blue
  - Border: 2px, (0.4, 0.4, 0.6, 1) - Light blue
  - Corner radius: 8px all corners

#### 4. SpeakerLabel (Label)
- **Position**: Top-left of dialogue box (20, -180, 200, -155)
- **Purpose**: Display speaking character name
- **Font Size**: 18px
- **Alignment**: Left

#### 5. Portrait (TextureRect)
- **Position**: Top-right of dialogue box (-120, -60, -20, 40)
- **Size**: 100x100 pixels
- **Purpose**: Display character portrait
- **Stretch Mode**: Keep aspect ratio centered (5)

#### 6. DialogueText (RichTextLabel)
- **Position**: Main content area (20, 35, -130, -50)
- **Purpose**: Display dialogue text with BBCode support
- **Configuration**:
  - `bbcode_enabled = true`
  - `fit_content = true`
  - `scroll_active = false`

#### 7. ChoicesContainer (VBoxContainer)
- **Position**: Bottom center (-200, -45, 200, -5)
- **Anchors**: Bottom-center (7 preset)
- **Purpose**: Container for dialogue choice buttons
- **Layout**: Vertical arrangement

#### 8. ContinueButton (Button)
- **Position**: Bottom-right (-120, -40, -20, -10)
- **Anchors**: Bottom-right (3 preset)
- **Text**: "Continue →"
- **Signal**: `pressed` → `_on_continue_pressed`

#### 9. TypewriterTimer (Timer)
- **Purpose**: Control typing effect speed
- **Configuration**:
  - `wait_time = 0.05`
  - `autostart = false`
- **Signal**: `timeout` → `_on_typewriter_timeout`

#### 10. AutoContinueTimer (Timer)
- **Purpose**: Auto-advance dialogue after delay
- **Configuration**:
  - `wait_time = 2.5`
  - `one_shot = true`
  - `autostart = false`
- **Signal**: `timeout` → `_on_auto_continue_timeout`

---

## Choice Button Scene (choice_button.tscn)

### 📋 Simple Button Configuration

```
ChoiceButton (Button)
```

### 🎯 Configuration Details

- **Type**: Button
- **Minimum Size**: 200x40 pixels
- **Style**: Custom StyleBoxFlat:
  - Background: (0.2, 0.3, 0.5, 0.8) - Semi-transparent blue
  - Border: 1px, (0.4, 0.5, 0.7, 1) - Light blue border
  - Corner radius: 4px all corners

---

## 🔧 Integration Requirements

### Script Dependencies

#### character_controller.gd expects:
- `$Sprite2D` - Character sprite
- `$CollisionShape2D` - Physics collision
- `$InteractionArea` - Interaction detection
- `$ThoughtBubble/ThoughtText` - Thought display
- `$ThoughtBubble/BubbleTimer` - Auto-hide timer
- `$EmotionIndicator` - Emotion display
- `$NameLabel` - Character name
- `$StatusUI/EnergyBar` - Energy display
- `$StatusUI/HungerBar` - Hunger display
- `$StatusUI/SocialBar` - Social need display
- `$NavigationAgent2D` - Pathfinding

#### dialogue_system.gd expects:
- `$DialogueBox` - Main dialogue container
- `$DialogueBox/SpeakerLabel` - Speaker name
- `$DialogueBox/DialogueText` - Main text
- `$DialogueBox/ChoicesContainer` - Choice buttons
- `$DialogueBox/ContinueButton` - Continue button
- `$DialogueBox/Portrait` - Character portrait
- `$TypewriterTimer` - Typing effect
- `$AutoContinueTimer` - Auto-continue

### Signal Connections

All required signal connections are included in the scene files:
- Character interaction signals
- Timer timeout signals
- Button press signals

### Asset Requirements

#### Textures Needed:
- Character sprites: Place in `res://assets/sprites/characters/`
- Character portraits: For dialogue system
- UI elements: Optional custom textures for panels

#### Font Resources:
- Default theme fonts are used
- Custom fonts can be applied via Theme resources

---

## 🚀 Usage Instructions

### 1. Creating AI Characters

```gdscript
# In game_manager.gd or main scene
func spawn_ai_character(name: String, personality: String, position: Vector2):
    var character_scene = preload("res://scenes/ai_character.tscn")
    var character = character_scene.instantiate()
    character.character_name = name
    character.personality = personality
    character.global_position = position
    get_tree().current_scene.add_child(character)
    return character
```

### 2. Starting Dialogue

```gdscript
# In any script that needs dialogue
func start_conversation(speaker: String, message: String):
    var dialogue_ui = preload("res://scenes/dialogue_ui.tscn")
    var dialogue = dialogue_ui.instantiate()
    get_tree().current_scene.add_child(dialogue)
    dialogue.start_dialogue(speaker, message)
```

### 3. Accessing AI Manager

The AI characters automatically connect to the AI service via:
```gdscript
@onready var ai_manager = get_node("/root/AIManager")
```

This requires the AIManager to be added as an autoload singleton or created by the game manager.

---

## 🐛 Troubleshooting

### Common Issues:

1. **Character not responding to AI**: 
   - Check AI service is running on port 8080
   - Verify AIManager singleton exists

2. **Thought bubbles not showing**:
   - Ensure BubbleTimer signal is connected
   - Check ThoughtBubble initial visibility is false

3. **Dialogue not appearing**:
   - Verify all DialogueBox child nodes exist
   - Check signal connections in scene file

4. **Navigation not working**:
   - Ensure NavigationRegion2D exists in main scene
   - Check NavigationAgent2D configuration

### Debug Tools:

- Enable StatusUI and StateLabel visibility for debugging
- Use Godot's remote inspector to monitor node states
- Check console output for AI service communication errors

---

## ✅ Verification Checklist

- [ ] ai_character.tscn loads without errors
- [ ] dialogue_ui.tscn loads without errors  
- [ ] choice_button.tscn loads without errors
- [ ] All script references resolve correctly
- [ ] Signal connections work in editor
- [ ] Scenes integrate with existing game_manager.gd
- [ ] AI service communication functions
- [ ] Character movement and pathfinding work
- [ ] Thought bubbles display and hide correctly
- [ ] Dialogue system shows choices and continues properly

The scenes are now complete and ready for integration with the existing AI service backend!