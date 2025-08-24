extends Node
# Game Manager - Main game controller

signal game_started
signal game_paused
signal game_resumed
signal game_ended

# Game state
enum GameState {
	MENU,
	PLAYING,
	PAUSED,
	DIALOGUE,
	CUTSCENE
}

var current_state: GameState = GameState.MENU
var game_time: float = 0.0
var day_cycle_time: float = 0.0
var day_length: float = 300.0  # 5 minutes per day

# Player reference
var player: CharacterBody2D = null
var player_controller = null

# AI Characters
var ai_characters: Array = []
var character_scenes: Dictionary = {}

# Systems
@onready var ai_manager = preload("res://scripts/ai_manager.gd").new()
@onready var dialogue_system = null

# World settings
@export var spawn_characters_on_start: bool = true
@export var character_spawn_points: Array[Vector2] = []

func _ready():
	print("Game Manager initialized")
	
	# Add AI Manager as singleton
	if !Engine.has_singleton("AIManager"):
		add_child(ai_manager)
		ai_manager.name = "AIManager"
	
	# Connect signals
	if dialogue_system:
		dialogue_system.dialogue_started.connect(_on_dialogue_started)
		dialogue_system.dialogue_ended.connect(_on_dialogue_ended)
	
	# Start game
	_initialize_game()

func _initialize_game():
	"""Initialize the game"""
	print("Initializing game...")
	
	# Load character scenes
	_load_character_scenes()
	
	# Spawn AI characters if enabled
	if spawn_characters_on_start:
		_spawn_demo_characters()
	
	# Find player
	player = get_tree().get_first_node_in_group("player")
	if player and player.has_method("get_controller"):
		player_controller = player.get_controller()
	
	# Start game
	start_game()

func _load_character_scenes():
	"""Load character scene files"""
	var character_scene_path = "res://scenes/characters/"
	var dir = DirAccess.open(character_scene_path)
	
	if dir:
		dir.list_dir_begin()
		var file_name = dir.get_next()
		
		while file_name != "":
			if file_name.ends_with(".tscn"):
				var scene_path = character_scene_path + file_name
				var character_name = file_name.trim_suffix(".tscn")
				character_scenes[character_name] = load(scene_path)
				print("Loaded character scene: ", character_name)
			file_name = dir.get_next()

func _spawn_demo_characters():
	"""Spawn demo AI characters"""
	print("Spawning demo characters...")
	
	# Demo character data
	var demo_characters = [
		{
			"name": "Alice",
			"personality": "Friendly shopkeeper who loves to chat",
			"position": Vector2(100, 0)
		},
		{
			"name": "Bob", 
			"personality": "Grumpy blacksmith with a heart of gold",
			"position": Vector2(-100, 0)
		},
		{
			"name": "Charlie",
			"personality": "Curious child always looking for adventure",
			"position": Vector2(0, 100)
		}
	]
	
	for i in range(demo_characters.size()):
		var char_data = demo_characters[i]
		var spawn_pos = character_spawn_points[i] if i < character_spawn_points.size() else char_data.position
		spawn_ai_character(char_data.name, char_data.personality, spawn_pos)

func spawn_ai_character(character_name: String, personality: String, position: Vector2) -> CharacterBody2D:
	"""Spawn an AI character"""
	
	# Fall back to generic character scene directly
	var character_scene = preload("res://scenes/ai_character.tscn") if ResourceLoader.exists("res://scenes/ai_character.tscn") else null
	
	if !character_scene:
		print("Warning: No character scene found for ", character_name)
		return null
	
	# Instance character
	var character = character_scene.instantiate()
	character.character_name = character_name
	character.personality = personality
	character.global_position = position
	
	# Add to scene
	get_tree().current_scene.add_child(character)
	ai_characters.append(character)
	
	print("Spawned AI character: ", character_name, " at ", position)
	return character

func _process(delta):
	"""Main game loop"""
	if current_state == GameState.PLAYING:
		game_time += delta
		day_cycle_time += delta
		
		# Update day/night cycle
		if day_cycle_time >= day_length:
			day_cycle_time = 0.0
			_on_new_day()
		
		# Update time of day
		_update_time_of_day()

func _update_time_of_day():
	"""Update time of day for lighting and AI behavior"""
	var time_percent = day_cycle_time / day_length
	var hour = int(time_percent * 24)
	
	# Update lighting based on time
	# This would connect to a lighting system
	
	# Notify AI characters of time change
	for character in ai_characters:
		if character.has_method("update_time_context"):
			character.update_time_context(hour)

func _on_new_day():
	"""Handle new day event"""
	print("New day started!")
	
	# Reset character needs
	for character in ai_characters:
		if character.has_method("reset_daily_needs"):
			character.reset_daily_needs()

func start_game():
	"""Start the game"""
	current_state = GameState.PLAYING
	game_time = 0.0
	day_cycle_time = 0.0
	emit_signal("game_started")
	print("Game started!")

func pause_game():
	"""Pause the game"""
	if current_state == GameState.PLAYING:
		current_state = GameState.PAUSED
		get_tree().paused = true
		emit_signal("game_paused")

func resume_game():
	"""Resume the game"""
	if current_state == GameState.PAUSED:
		current_state = GameState.PLAYING
		get_tree().paused = false
		emit_signal("game_resumed")

func end_game():
	"""End the game"""
	current_state = GameState.MENU
	emit_signal("game_ended")
	
	# Clean up
	for character in ai_characters:
		character.queue_free()
	ai_characters.clear()

func _on_dialogue_started():
	"""Handle dialogue start"""
	if current_state == GameState.PLAYING:
		current_state = GameState.DIALOGUE
		# Pause AI characters during dialogue
		for character in ai_characters:
			if character.has_method("pause_ai"):
				character.pause_ai()

func _on_dialogue_ended():
	"""Handle dialogue end"""
	if current_state == GameState.DIALOGUE:
		current_state = GameState.PLAYING
		# Resume AI characters
		for character in ai_characters:
			if character.has_method("resume_ai"):
				character.resume_ai()

func get_game_time() -> float:
	"""Get current game time"""
	return game_time

func get_time_of_day() -> int:
	"""Get current hour of day (0-23)"""
	var time_percent = day_cycle_time / day_length
	return int(time_percent * 24)

func get_ai_characters() -> Array:
	"""Get all AI characters"""
	return ai_characters

func get_character_by_name(character_name: String) -> CharacterBody2D:
	"""Get AI character by name"""
	for character in ai_characters:
		if character.character_name == character_name:
			return character
	return null

func save_game(save_path: String = "user://savegame.save"):
	"""Save game state"""
	var save_file = FileAccess.open(save_path, FileAccess.WRITE)
	if !save_file:
		print("Failed to save game")
		return
	
	var save_data = {
		"game_time": game_time,
		"day_cycle_time": day_cycle_time,
		"characters": []
	}
	
	# Save character states
	for character in ai_characters:
		if character.has_method("get_state"):
			save_data.characters.append(character.get_state())
	
	save_file.store_var(save_data)
	save_file.close()
	print("Game saved to ", save_path)

func load_game(save_path: String = "user://savegame.save"):
	"""Load game state"""
	if !FileAccess.file_exists(save_path):
		print("Save file not found")
		return
	
	var save_file = FileAccess.open(save_path, FileAccess.READ)
	if !save_file:
		print("Failed to load game")
		return
	
	var save_data = save_file.get_var()
	save_file.close()
	
	# Restore game time
	game_time = save_data.get("game_time", 0.0)
	day_cycle_time = save_data.get("day_cycle_time", 0.0)
	
	# Restore character states
	var character_states = save_data.get("characters", [])
	for i in range(min(character_states.size(), ai_characters.size())):
		if ai_characters[i].has_method("set_state"):
			ai_characters[i].set_state(character_states[i])
	
	print("Game loaded from ", save_path)

func _input(event):
	"""Handle input events"""
	# Pause/Resume
	if event.is_action_pressed("pause"):
		if current_state == GameState.PLAYING:
			pause_game()
		elif current_state == GameState.PAUSED:
			resume_game()
	
	# Quick save/load
	if event.is_action_pressed("quick_save"):
		save_game()
	elif event.is_action_pressed("quick_load"):
		load_game()
