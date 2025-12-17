extends CharacterBody2D

@export var walk_speed: float = 30.0
@export var run_speed: float = 60.0
@export var idle_duration: float = 8.0
@export var move_duration: float = 10.0
@export var min_x: float = -10000.0
@export var max_x: float = 10000.0
@export var min_y: float = -10000.0
@export var max_y: float = 10000.0
@export var target: Node2D

@onready var animated_sprite = $AnimatedSprite2D
@onready var NA := $NavigationAgent2D as NavigationAgent2D
@onready var server = get_node("/root/Server")
@onready var decision_timer := $DicisionTimer  # Reference to the decision timer

# Preload BubbleManager for static access
const BubbleManager = preload("res://scripts/bubble_manager.gd")

# Dog state machine
enum State {
	SITTING_IDLE,
	STANDING_UP,
	STANDING_IDLE,
	MOVING,
	SITTING_DOWN,
	OBSERVING,
	ACTING
}

enum MoveSpeed {
	WALK,
	RUN
}

var current_state: State = State.SITTING_IDLE
var current_move_speed: MoveSpeed = MoveSpeed.WALK
var state_timer: float = 0.0
var animation_timer: float = 0.0
var movement_direction: Vector2 = Vector2.ZERO
var time_until_direction_change: float = 0.0

# Decision and dialogue variables
var is_observing := false  # Flag to pause movement during observation
var observation_duration := 3.0  # How long to observe
var observation_timer := 0.0
var is_acting := false  # Flag to indicate if Socrates is performing an action
var is_walking := false  # Flag to indicate if Socrates is walking to a target
var current_observation_bubble = null  # Reference to current observation bubble
var current_action_bubble = null  # Reference to current action bubble
var decision_enabled := false          # Flag to enable/disable decision system (default: OFF)
var current_chat_bubble = null  # Add chat bubble reference

# Socrates-specific observations and dialogue
var observations = {
	"leonardo": "A Renaissance soul seeking truth through art, yet what is truth?",
	"einstein": "He measures the universe, but can one measure wisdom?",
	"shakespeare": "He writes of life, but what IS life beyond mere words?",
	"customer": "Another seeking soul, but what do they truly seek?",
	"bar": "A gathering place of bodies, but where do souls gather?",
	"guitar": "Music soothes, yet why does harmony touch the soul?",
	"bowl": "Simple sustenance, or a metaphor for life's emptiness waiting to be filled?",
	"table": "We gather around objects, but what gathers us?"
}

func set_decision_enabled(enabled: bool):
	"""Enable or disable the decision system for this NPC"""
	decision_enabled = enabled
	if decision_timer:
		if enabled:
			# Resume decision timer if it was stopped
			if decision_timer.is_stopped():
				decision_timer.start()
			# print("Socrates: Decision system enabled")
		else:
			# Stop decision timer
			decision_timer.stop()
			# print("Socrates: Decision system disabled")

func _ready():
	if not animated_sprite:
		push_error("AnimatedSprite2D not found! Please add one as a child of Socrates.")
		return

	# Start with sitting idle (philosophical contemplation)
	_enter_sitting_idle()

	# Stagger initial decision timing
	if decision_timer:
		decision_timer.wait_time = 25.0  # Socrates contemplates longer
		# Socrates starts after 15 seconds (philosopher needs time to think)
		await get_tree().create_timer(15.0).timeout
		decision_timer.start()

func _physics_process(delta):
	state_timer -= delta
	animation_timer -= delta

	# Handle observation timer
	if is_observing:
		observation_timer -= delta
		if observation_timer <= 0:
			is_observing = false
			current_observation_bubble = null
			finish_action()

	# Check if walk action is completed
	if is_walking and NA and NA.is_navigation_finished():
		is_walking = false
		finish_action()

	# State machine
	match current_state:
		State.SITTING_IDLE:
			_handle_sitting_idle(delta)
		State.STANDING_UP:
			_handle_standing_up(delta)
		State.STANDING_IDLE:
			_handle_standing_idle(delta)
		State.MOVING:
			_handle_moving(delta)
		State.SITTING_DOWN:
			_handle_sitting_down(delta)
		State.OBSERVING:
			_handle_observing(delta)
		State.ACTING:
			_handle_acting(delta)

	move_and_slide()

	# Update bubble positions
	update_observation_bubble_position()
	update_action_bubble_position()
	update_chat_bubble_position()

func _handle_sitting_idle(_delta):
	velocity = Vector2.ZERO

	if not is_acting and state_timer <= 0:
		_enter_standing_up()

func _handle_standing_up(_delta):
	velocity = Vector2.ZERO

	# Wait for animation to complete
	if animation_timer <= 0:
		_enter_standing_idle()

func _handle_standing_idle(_delta):
	velocity = Vector2.ZERO

	# Brief pause before moving
	if not is_acting and state_timer <= 0:
		_enter_moving()

func _handle_moving(delta):
	# Update movement
	time_until_direction_change -= delta

	if time_until_direction_change <= 0:
		_choose_random_direction_and_speed()
		time_until_direction_change = randf_range(3.0, 5.0)  # Socrates wanders thoughtfully

	# Check if hitting a wall and change direction
	if is_on_wall():
		_bounce_off_wall()
		time_until_direction_change = randf_range(2.0, 4.0)

	# Apply movement with current speed
	var current_speed = run_speed if current_move_speed == MoveSpeed.RUN else walk_speed
	velocity = movement_direction * current_speed

	# Update animation based on direction and speed
	_update_move_animation()

	# Check if it's time to sit
	if not is_acting and state_timer <= 0:
		_enter_sitting_down()

func _handle_sitting_down(_delta):
	velocity = Vector2.ZERO

	# Wait for animation to complete
	if animation_timer <= 0:
		_enter_sitting_idle()

func _handle_observing(_delta):
	velocity = Vector2.ZERO
	# Observing is handled by the observation_timer in _physics_process

func _handle_acting(_delta):
	# Acting state for various actions
	if not is_acting:
		current_state = State.STANDING_IDLE

func _enter_sitting_idle():
	current_state = State.SITTING_IDLE
	state_timer = idle_duration
	if animated_sprite.sprite_frames.has_animation("idle"):
		animated_sprite.play("idle")
	velocity = Vector2.ZERO

func _enter_standing_up():
	current_state = State.STANDING_UP
	if animated_sprite.sprite_frames.has_animation("stand up"):
		animated_sprite.play("stand up")
		animated_sprite.frame = 0
	animation_timer = 1.5

func _enter_standing_idle():
	current_state = State.STANDING_IDLE
	state_timer = 1.0  # Brief pause
	if animated_sprite.sprite_frames.has_animation("look up"):
		animated_sprite.play("look up")

func _enter_moving():
	current_state = State.MOVING
	state_timer = move_duration
	_choose_random_direction_and_speed()
	time_until_direction_change = randf_range(2.0, 4.0)

func _enter_sitting_down():
	current_state = State.SITTING_DOWN
	if animated_sprite.sprite_frames.has_animation("sit down"):
		animated_sprite.play("sit down")
		animated_sprite.frame = 0
	movement_direction = Vector2.ZERO
	animation_timer = 1.5

func _choose_random_direction_and_speed():
	# Choose a random direction
	var angle = randf() * TAU  # Random angle in radians
	movement_direction = Vector2(cos(angle), sin(angle)).normalized()

	# Socrates mostly walks (philosopher's pace), rarely runs
	current_move_speed = MoveSpeed.RUN if randf() > 0.8 else MoveSpeed.WALK

func _bounce_off_wall():
	# Get wall normal to determine bounce direction
	var wall_normal = get_wall_normal()

	if wall_normal != Vector2.ZERO:
		# Reflect the movement direction based on wall normal
		movement_direction = movement_direction.bounce(wall_normal).normalized()

		# Add slight randomness to prevent getting stuck
		movement_direction = movement_direction.rotated(randf_range(-0.3, 0.3))

		# Philosophers rarely change pace when hitting walls
		if randf() > 0.9:
			current_move_speed = MoveSpeed.RUN if current_move_speed == MoveSpeed.WALK else MoveSpeed.WALK

func _update_move_animation():
	# Use walk or run animations based on current speed
	var anim_prefix = "run" if current_move_speed == MoveSpeed.RUN else "walk"

	if abs(movement_direction.x) > abs(movement_direction.y):
		# Moving more horizontally
		if movement_direction.x > 0:
			if animated_sprite.sprite_frames.has_animation(anim_prefix + " right"):
				animated_sprite.play(anim_prefix + " right")
		else:
			if animated_sprite.sprite_frames.has_animation(anim_prefix + " left"):
				animated_sprite.play(anim_prefix + " left")
	else:
		# Moving more vertically, use left/right based on slight x movement
		if movement_direction.x >= 0:
			if animated_sprite.sprite_frames.has_animation(anim_prefix + " right"):
				animated_sprite.play(anim_prefix + " right")
		else:
			if animated_sprite.sprite_frames.has_animation(anim_prefix + " left"):
				animated_sprite.play(anim_prefix + " left")

func makepath() -> void:
	if target and NA:
		NA.target_position = target.global_position

func finish_action():
	# Resume decision timer after action completes
	if is_acting and decision_timer:
		is_acting = false
		current_state = State.STANDING_IDLE
		decision_timer.wait_time = 25.0  # Socrates contemplates longer
		decision_timer.start()

# Decision system functions
func _on_dicision_timer_timeout() -> void:
	# Check if decision system is enabled
	if not decision_enabled:
		return

	if is_acting:
		return

	if not server:
		server = get_node_or_null("/root/Server")
		if not server:
			print("Server node not found")
			return

	if not server.connected:
		print("Server not connected yet, waiting...")
		return

	var context = generate_context()
	server.request_decision_with_callback("Socrates", context, on_decision_received)

func generate_context() -> String:
	var context_parts = []
	var npcs = get_tree().get_nodes_in_group("npcs")
	for npc in npcs:
		if npc == self:
			continue
		var distance = global_position.distance_to(npc.global_position)
		if distance < 150:
			context_parts.append(npc.name + " is nearby")
		elif distance < 300:
			context_parts.append(npc.name + " is in sight")

	# Add philosophical context
	context_parts.append("contemplating existence")

	if context_parts.size() > 0:
		return ", ".join(context_parts)
	else:
		return "wandering alone, seeking truth"

func on_decision_received(decision: String):
	print("Socrates received decision: " + decision)
	var parts = decision.split("|")
	if parts.size() != 2:
		print("Invalid decision format")
		return

	var action = parts[0].strip_edges().to_lower()
	var target_name = parts[1].strip_edges()

	is_acting = true
	current_state = State.ACTING
	decision_timer.stop()

	match action:
		"walk":
			execute_walk(target_name)
		"bark":
			execute_bark(target_name)
		"sniff":
			execute_sniff(target_name)
		"question":
			execute_question(target_name)
		"ponder":
			execute_ponder(target_name)
		"observe":
			execute_observe(target_name)
		_:
			print("Unknown action: " + action)
			finish_action()

# Action implementations
func execute_walk(target_name: String):
	var target_node = find_target_by_name(target_name)
	if target_node:
		target = target_node
		makepath()
		is_walking = true
		current_state = State.MOVING
	else:
		print("Walk target not found: " + target_name)
		finish_action()

func execute_bark(target_name: String):
	# Philosophical barking
	show_action_bubble("*barks philosophically*")
	await get_tree().create_timer(1.0).timeout

	var philosophical_barks = {
		"customer": "Woof! (Translation: What is the good life?)",
		"leonardo": "Woof woof! (Translation: Does art reveal truth?)",
		"einstein": "Woof? (Translation: Can you measure wisdom?)",
		"shakespeare": "Woof! (Translation: Is all life merely a play?)",
		"bar": "Woof? (Translation: Why do we gather here?)"
	}

	var bark_text = philosophical_barks.get(target_name.to_lower(), "Woof! (Translation: Know thyself!)")
	show_chat_bubble(bark_text)
	await get_tree().create_timer(3.0).timeout
	finish_action()

func execute_sniff(target_name: String):
	var target_node = find_target_by_name(target_name)
	if target_node:
		target = target_node
		makepath()
		is_walking = true

		# Wait to reach target
		while is_walking and not NA.is_navigation_finished():
			await get_tree().process_frame

		is_walking = false

		# Philosophical sniffing
		show_action_bubble("*sniffs for truth*")
		await get_tree().create_timer(2.0).timeout

		var sniff_wisdom = {
			"leonardo": "*smells creativity and curiosity*",
			"einstein": "*detects relativity and genius*",
			"shakespeare": "*scents poetry and drama*",
			"customer": "*sniffs mortality and questions*",
			"bowl": "*smells sustenance and simplicity*"
		}

		var sniff_text = sniff_wisdom.get(target_name.to_lower(), "*smells the essence of being*")
		show_chat_bubble(sniff_text)
		await get_tree().create_timer(3.0).timeout

	finish_action()

func execute_question(target_name: String):
	show_action_bubble("*tilts head questioningly*")
	await get_tree().create_timer(2.0).timeout

	var questions = {
		"leonardo": "What is beauty without truth?",
		"einstein": "Is time real or merely perceived?",
		"shakespeare": "Do words create or reflect reality?",
		"customer": "What brought you here, truly?",
		"bar": "Is this gathering meaningful?"
	}

	var question_text = questions.get(target_name.to_lower(), "What do we truly know?")
	show_chat_bubble_to_target(target_name, question_text)
	await get_tree().create_timer(3.0).timeout
	finish_action()

func execute_ponder(target_name: String):
	show_action_bubble("*sits and contemplates*")
	_enter_sitting_idle()  # Sit down to ponder
	await get_tree().create_timer(3.0).timeout

	var ponderings = {
		"existence": "*achieves momentary enlightenment*",
		"truth": "*grasps a fleeting truth*",
		"bar": "*understands the nature of gathering*",
		"customer": "*sees through mortal concerns*"
	}

	var ponder_text = ponderings.get(target_name.to_lower(), "*contemplates the eternal*")
	show_chat_bubble(ponder_text)
	await get_tree().create_timer(2.0).timeout
	finish_action()

func execute_observe(target_name: String):
	is_observing = true
	current_state = State.OBSERVING
	observation_timer = observation_duration

	var observation_text = observations.get(target_name.to_lower(), "Hmm... what IS this thing called " + target_name + "?")
	show_observation_bubble(observation_text)

# Helper functions
func find_target_by_name(target_name: String) -> Node2D:
	# First check NPCs
	var npcs = get_tree().get_nodes_in_group("npcs")
	for npc in npcs:
		if npc.name.to_lower() == target_name.to_lower():
			return npc

	# Then check objects
	var objects = get_tree().get_nodes_in_group("objects")
	for obj in objects:
		if obj.name.to_lower() == target_name.to_lower():
			return obj

	# Check for the player
	if target_name.to_lower() == "customer" or target_name.to_lower() == "player":
		var player = get_tree().get_first_node_in_group("player")
		if player:
			return player

	# Check specific named nodes
	var node = get_node_or_null("/root/Node2D/" + target_name.capitalize())
	if node and node is Node2D:
		return node

	return null

# Bubble management functions
func show_observation_bubble(text: String):
	if current_observation_bubble:
		current_observation_bubble.queue_free()
	current_observation_bubble = BubbleManager.show_observation_bubble("Socrates", text, global_position)

func update_observation_bubble_position():
	if current_observation_bubble and is_instance_valid(current_observation_bubble):
		current_observation_bubble.position = global_position + Vector2(0, -60)

func show_action_bubble(text: String):
	if current_action_bubble:
		current_action_bubble.queue_free()
	current_action_bubble = BubbleManager.show_action_bubble("Socrates", text, global_position)

func update_action_bubble_position():
	if current_action_bubble and is_instance_valid(current_action_bubble):
		current_action_bubble.position = global_position + Vector2(0, -60)

func show_chat_bubble(text: String):
	if current_chat_bubble:
		current_chat_bubble.queue_free()
	current_chat_bubble = BubbleManager.show_speech_bubble("Socrates", text, global_position)

func show_chat_bubble_to_player(text: String):
	# Send message through server for player to see
	if server and server.connected:
		server.websocket_client.send_text(JSON.stringify({
			"type": "npc_message",
			"npc": "Socrates",
			"message": "Socrates (philosopher dog)|" + text
		}))
	show_chat_bubble(text)

func show_chat_bubble_to_target(target_name: String, text: String):
	if target_name.to_lower() == "customer" or target_name.to_lower() == "player":
		show_chat_bubble_to_player(text)
	else:
		# For NPC to NPC communication
		show_chat_bubble(text)
		# Send through server for processing
		if server and server.connected:
			server.websocket_client.send_text(JSON.stringify({
				"type": "npc_message",
				"npc": "Socrates",
				"message": "Socrates|" + text,
				"target": target_name
			}))

func update_chat_bubble_position():
	if current_chat_bubble and is_instance_valid(current_chat_bubble):
		current_chat_bubble.position = global_position + Vector2(0, -60)
