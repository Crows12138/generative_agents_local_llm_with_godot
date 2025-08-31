extends Node
# Scene Setup Helper - Automatically attach scripts to scenes

func _ready():
	"""Auto-setup scene scripts"""
	print("=== Scene Setup Helper ===")
	
	# Check and setup main scene
	if get_tree().current_scene.name == "Main":
		setup_main_scene()
	
	# Check and setup cozy bar scene
	elif get_tree().current_scene.name == "CozyBar":
		setup_cozy_bar_scene()
	
	print("Scene setup complete!")

func setup_main_scene():
	"""Setup main scene with game_manager"""
	var main = get_tree().current_scene
	
	# Check if game_manager.gd is attached
	if not main.has_method("start_game"):
		print("Attaching game_manager.gd to Main scene...")
		var script = load("res://scripts/game_manager.gd")
		if script:
			main.set_script(script)
			print("✓ game_manager.gd attached")
		else:
			print("✗ Could not load game_manager.gd")

func setup_cozy_bar_scene():
	"""Setup cozy bar scene with necessary scripts"""
	var bar = get_tree().current_scene
	
	# Add BarUpdater node if not exists
	if not bar.has_node("BarUpdater"):
		print("Adding BarUpdater node...")
		var updater = Node.new()
		updater.name = "BarUpdater"
		bar.add_child(updater)
		
		# Attach bar_updater_llm.gd
		var script = load("res://scripts/bar_updater_llm.gd")
		if script:
			updater.set_script(script)
			print("✓ bar_updater_llm.gd attached to BarUpdater")
		else:
			print("✗ Could not load bar_updater_llm.gd")
	
	# Ensure NPCs exist
	ensure_npcs_exist(bar)

func ensure_npcs_exist(bar_scene):
	"""Make sure Bob, Alice, and Sam exist"""
	var npcs = ["Bob", "Alice", "Sam"]
	var positions = {
		"Bob": Vector2(400, 80),
		"Alice": Vector2(200, 300),
		"Sam": Vector2(600, 400)
	}
	
	for npc_name in npcs:
		if not bar_scene.has_node(npc_name):
			print("Creating NPC: ", npc_name)
			var npc = CharacterBody2D.new()
			npc.name = npc_name
			npc.position = positions[npc_name]
			
			# Add visual
			var sprite = ColorRect.new()
			sprite.size = Vector2(32, 48)
			sprite.position = Vector2(-16, -48)
			sprite.color = Color(randf(), randf(), randf())
			npc.add_child(sprite)
			
			# Add label
			var label = Label.new()
			label.text = npc_name
			label.position = Vector2(-20, -60)
			npc.add_child(label)
			
			bar_scene.add_child(npc)
			print("✓ ", npc_name, " created at ", positions[npc_name])