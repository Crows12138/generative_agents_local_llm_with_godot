extends Node
# Position Debug Tool - Press number keys to test different positions
# Attach this to any NPC to test movement positions

@export var test_npc: CharacterBody2D  # Drag NPC here in editor
@export var show_position_label: bool = true

var position_label: Label
var test_positions = {
	KEY_1: Vector2(605, 320),  # Original bar counter
	KEY_2: Vector2(605, 360),  # In front of counter
	KEY_3: Vector2(605, 380),  # Further front
	KEY_4: Vector2(560, 340),  # Left of counter
	KEY_5: Vector2(650, 340),  # Right of counter
	KEY_6: Vector2(600, 85),   # Shelf position
	KEY_7: Vector2(640, 360),  # Center of room
	KEY_8: Vector2(500, 300),  # Custom test 1
	KEY_9: Vector2(700, 300),  # Custom test 2
	KEY_0: Vector2(605, 400),  # Custom test 3
}

func _ready():
	print("[PositionDebug] Ready - Press number keys 1-0 to test positions")
	print("Positions:")
	print("  1: Bar counter original (605, 320)")
	print("  2: In front (605, 360)")
	print("  3: Further front (605, 380)")
	print("  4: Left side (560, 340)")
	print("  5: Right side (650, 340)")
	print("  6: Shelf (600, 85)")
	print("  7: Center (640, 360)")
	print("  8-0: Custom positions")
	print("  Arrow keys: Fine tune position")
	print("  P: Print current position")
	
	# Create position label
	if show_position_label and test_npc:
		position_label = Label.new()
		position_label.text = "Position: " + str(test_npc.global_position)
		position_label.position = Vector2(-50, -60)
		position_label.add_theme_color_override("font_color", Color.RED)
		position_label.add_theme_font_size_override("font_size", 20)
		test_npc.add_child(position_label)

func _input(event):
	if not test_npc:
		return
	
	# Number keys for preset positions
	if event is InputEventKey and event.pressed:
		if event.keycode in test_positions:
			var target_pos = test_positions[event.keycode]
			test_npc.global_position = target_pos
			print("[PositionDebug] Moved to: ", target_pos)
			_update_label()
		
		# Arrow keys for fine tuning
		elif event.keycode == KEY_LEFT:
			test_npc.global_position.x -= 10
			print("[PositionDebug] Position: ", test_npc.global_position)
			_update_label()
		elif event.keycode == KEY_RIGHT:
			test_npc.global_position.x += 10
			print("[PositionDebug] Position: ", test_npc.global_position)
			_update_label()
		elif event.keycode == KEY_UP:
			test_npc.global_position.y -= 10
			print("[PositionDebug] Position: ", test_npc.global_position)
			_update_label()
		elif event.keycode == KEY_DOWN:
			test_npc.global_position.y += 10
			print("[PositionDebug] Position: ", test_npc.global_position)
			_update_label()
		
		# Shift + arrows for smaller movements
		elif event.shift_pressed:
			if event.keycode == KEY_LEFT:
				test_npc.global_position.x -= 1
				print("[PositionDebug] Fine tune: ", test_npc.global_position)
				_update_label()
			elif event.keycode == KEY_RIGHT:
				test_npc.global_position.x += 1
				print("[PositionDebug] Fine tune: ", test_npc.global_position)
				_update_label()
			elif event.keycode == KEY_UP:
				test_npc.global_position.y -= 1
				print("[PositionDebug] Fine tune: ", test_npc.global_position)
				_update_label()
			elif event.keycode == KEY_DOWN:
				test_npc.global_position.y += 1
				print("[PositionDebug] Fine tune: ", test_npc.global_position)
				_update_label()
		
		# P to print current position
		elif event.keycode == KEY_P:
			print("[PositionDebug] Current position: ", test_npc.global_position)
			print("  Copy this: Vector2(", int(test_npc.global_position.x), ", ", int(test_npc.global_position.y), ")")

func _update_label():
	if position_label:
		position_label.text = "Pos: " + str(int(test_npc.global_position.x)) + ", " + str(int(test_npc.global_position.y))
