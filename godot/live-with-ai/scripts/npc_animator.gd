extends Node

# NPC Animation and Movement Script
# Attach this to your scene to make NPCs move

var npcs = {}
var npc_targets = {}
var npc_speeds = {}

func _ready():
	"""Initialize NPC movement system"""
	print("Starting NPC animation system...")
	
	# Setup NPCs with initial positions and targets
	setup_npcs()
	
	# Start movement
	set_process(true)

func setup_npcs():
	"""Find and setup all NPCs in the scene"""
	
	# Define NPC movement patterns
	var npc_configs = {
		"Bob": {
			"home": Vector2(400, 80),
			"waypoints": [
				Vector2(400, 80),   # Bar counter
				Vector2(450, 80),   # Right side of bar
				Vector2(350, 80),   # Left side of bar
			],
			"speed": 50.0,
			"current_waypoint": 0
		},
		"Alice": {
			"home": Vector2(200, 300),
			"waypoints": [
				Vector2(200, 300),  # Table
				Vector2(250, 350),  # Walk around
				Vector2(200, 400),  # Another spot
				Vector2(150, 350),  # Back around
			],
			"speed": 40.0,
			"current_waypoint": 0
		},
		"Sam": {
			"home": Vector2(600, 400),
			"waypoints": [
				Vector2(600, 400),  # Stage area
				Vector2(550, 450),  # Move on stage
				Vector2(650, 450),  # Other side
				Vector2(600, 400),  # Back to center
			],
			"speed": 45.0,
			"current_waypoint": 0
		}
	}
	
	# Find NPCs in scene or create them
	for npc_name in npc_configs:
		var npc = get_node_or_null("../" + npc_name)
		
		if not npc:
			# Try alternate paths
			npc = get_node_or_null("/root/Node2D/" + npc_name)
			if not npc:
				npc = get_node_or_null("/root/CozyBar/" + npc_name)
		
		if npc:
			print("Found NPC: ", npc_name)
			npcs[npc_name] = {
				"node": npc,
				"config": npc_configs[npc_name],
				"target": npc_configs[npc_name]["waypoints"][0],
				"waypoint_index": 0
			}
			
			# Set initial position
			npc.position = npc_configs[npc_name]["home"]
		else:
			print("NPC not found: ", npc_name, " - creating placeholder")
			create_placeholder_npc(npc_name, npc_configs[npc_name])

func create_placeholder_npc(npc_name: String, config: Dictionary):
	"""Create a simple animated placeholder if NPC doesn't exist"""
	var npc = CharacterBody2D.new()
	npc.name = npc_name
	
	# Add visual representation
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
	
	# Add to scene
	get_parent().add_child(npc)
	npc.position = config["home"]
	
	# Store in npcs dictionary
	npcs[npc_name] = {
		"node": npc,
		"config": config,
		"target": config["waypoints"][0],
		"waypoint_index": 0
	}

func _process(delta):
	"""Move NPCs towards their targets"""
	
	for npc_name in npcs:
		var npc_data = npcs[npc_name]
		var npc = npc_data["node"]
		var config = npc_data["config"]
		var target = npc_data["target"]
		
		if not npc:
			continue
		
		# Calculate direction to target
		var direction = (target - npc.position).normalized()
		var distance = npc.position.distance_to(target)
		
		# Move towards target
		if distance > 5:
			var movement = direction * config["speed"] * delta
			npc.position += movement
			
			# Simple animation: slight bobbing
			if npc.has_node("ColorRect") or npc.has_node("Sprite2D"):
				var visual = npc.get_child(0)
				visual.position.y = -48 + sin(Time.get_ticks_msec() * 0.01) * 2
		else:
			# Reached waypoint, go to next
			var waypoint_index = npc_data["waypoint_index"]
			waypoint_index = (waypoint_index + 1) % config["waypoints"].size()
			npc_data["waypoint_index"] = waypoint_index
			npc_data["target"] = config["waypoints"][waypoint_index]
			
			print(npc_name, " reached waypoint, going to next: ", npc_data["target"])

func make_npc_interact(npc_name: String, action: String):
	"""Make NPC perform specific action"""
	if npc_name in npcs:
		var npc = npcs[npc_name]["node"]
		
		match action:
			"wave":
				# Simple wave animation
				for i in range(3):
					npc.rotation_degrees = 10
					await get_tree().create_timer(0.2).timeout
					npc.rotation_degrees = -10
					await get_tree().create_timer(0.2).timeout
				npc.rotation_degrees = 0
				
			"jump":
				# Simple jump
				var original_y = npc.position.y
				npc.position.y -= 30
				await get_tree().create_timer(0.3).timeout
				npc.position.y = original_y
				
			"spin":
				# Spin around
				for i in range(8):
					npc.rotation_degrees += 45
					await get_tree().create_timer(0.1).timeout
				npc.rotation_degrees = 0