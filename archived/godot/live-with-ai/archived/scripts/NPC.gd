# NPC Controller for AI-driven characters
extends KinematicBody2D

# NPC Data
var npc_name = ""
var personality = ""
var current_action = "idle"
var target_position = Vector2.ZERO
var memories = []
var goals = []

# Movement
var movement_speed = 100.0
var is_moving = false

# Visuals
onready var sprite = $Sprite
onready var label = $Label
onready var thought_bubble = $ThoughtBubble

# Dialogue
var can_interact = true
var interaction_range = 50.0

# Signals
signal interaction_requested(npc_name)
signal arrived_at_destination(npc_name)

func _ready():
	# Set up visuals
	if label:
		label.text = npc_name
	if thought_bubble:
		thought_bubble.visible = false

func initialize(agent_data):
	"""Initialize NPC from server data"""
	npc_name = agent_data.get("name", "Unknown")
	
	# Set position if provided
	if "position" in agent_data:
		var pos = agent_data["position"]
		if pos is Array and pos.size() >= 2:
			position = Vector2(pos[0] * 32, pos[1] * 32)  # Convert grid to pixels
	
	# Set personality and memories
	if "memories" in agent_data:
		memories = agent_data["memories"]
	if "goals" in agent_data:
		goals = agent_data["goals"]
	
	# Update label
	if label:
		label.text = npc_name

func update_from_data(agent_data):
	"""Update NPC state from server data"""
	# Update position
	if "position" in agent_data:
		var pos = agent_data["position"]
		if pos is Array and pos.size() >= 2:
			target_position = Vector2(pos[0] * 32, pos[1] * 32)
			is_moving = true
	
	# Update action
	if "current_action" in agent_data:
		current_action = agent_data["current_action"]
		_play_action_animation(current_action)
	
	# Update memories
	if "memories" in agent_data:
		memories = agent_data["memories"]

func _physics_process(delta):
	"""Handle movement and physics"""
	if is_moving and target_position != Vector2.ZERO:
		var direction = (target_position - position).normalized()
		var distance = position.distance_to(target_position)
		
		if distance > 5:
			# Move towards target
			var velocity = direction * movement_speed
			move_and_slide(velocity)
			
			# Update sprite direction
			if velocity.x > 0:
				sprite.flip_h = false
			elif velocity.x < 0:
				sprite.flip_h = true
		else:
			# Arrived at destination
			is_moving = false
			emit_signal("arrived_at_destination", npc_name)

func _play_action_animation(action):
	"""Play animation for action"""
	match action:
		"walk":
			if sprite.frames:
				sprite.play("walk")
		"talk":
			if sprite.frames:
				sprite.play("talk")
		"wait", "idle":
			if sprite.frames:
				sprite.play("idle")
		_:
			if sprite.frames:
				sprite.play("default")

func _on_InteractionArea_body_entered(body):
	"""Handle player entering interaction range"""
	if body.is_in_group("player") and can_interact:
		# Show interaction prompt
		if thought_bubble:
			thought_bubble.visible = true
			thought_bubble.get_node("Label").text = "Press E to talk"

func _on_InteractionArea_body_exited(body):
	"""Handle player leaving interaction range"""
	if body.is_in_group("player"):
		if thought_bubble:
			thought_bubble.visible = false

func interact():
	"""Called when player interacts with NPC"""
	if can_interact:
		emit_signal("interaction_requested", npc_name)
		can_interact = false
		
		# Re-enable after cooldown
		yield(get_tree().create_timer(2.0), "timeout")
		can_interact = true

func show_thought(text: String, duration: float = 3.0):
	"""Show a thought bubble"""
	if thought_bubble:
		thought_bubble.visible = true
		thought_bubble.get_node("Label").text = text
		
		yield(get_tree().create_timer(duration), "timeout")
		thought_bubble.visible = false

func move_to(destination: Vector2):
	"""Move NPC to destination"""
	target_position = destination
	is_moving = true
	current_action = "walk"
	_play_action_animation("walk")
