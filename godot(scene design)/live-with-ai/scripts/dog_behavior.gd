extends CharacterBody2D

@export var walk_speed: float = 30.0
@export var run_speed: float = 60.0
@export var idle_duration: float = 10.0
@export var move_duration: float = 10.0
@export var min_x: float = -10000.0
@export var max_x: float = 10000.0
@export var min_y: float = -10000.0
@export var max_y: float = 10000.0

@onready var animated_sprite = $AnimatedSprite2D

enum State {
	SITTING_IDLE,
	STANDING_UP,
	STANDING_IDLE,
	MOVING,
	SITTING_DOWN
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

func _ready():
	if not animated_sprite:
		push_error("AnimatedSprite2D not found! Please add one as a child of Dog.")
		return
	
	# Start with sitting idle
	_enter_sitting_idle()

func _physics_process(delta):
	state_timer -= delta
	animation_timer -= delta
	
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
	
	move_and_slide()

func _handle_sitting_idle(delta):
	velocity = Vector2.ZERO
	
	if state_timer <= 0:
		_enter_standing_up()

func _handle_standing_up(delta):
	velocity = Vector2.ZERO
	
	# Wait for animation to complete (3 frames at 2 fps = 1.5 seconds)
	if animation_timer <= 0:
		_enter_standing_idle()

func _handle_standing_idle(delta):
	velocity = Vector2.ZERO
	
	# Brief pause before moving
	if state_timer <= 0:
		_enter_moving()

func _handle_moving(delta):
	# Update movement
	time_until_direction_change -= delta
	
	if time_until_direction_change <= 0:
		_choose_random_direction_and_speed()
		time_until_direction_change = randf_range(2.0, 4.0)
	
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
	if state_timer <= 0:
		_enter_sitting_down()

func _handle_sitting_down(delta):
	velocity = Vector2.ZERO
	
	# Wait for animation to complete (3 frames at 2 fps = 1.5 seconds)
	if animation_timer <= 0:
		_enter_sitting_idle()

func _enter_sitting_idle():
	current_state = State.SITTING_IDLE
	state_timer = idle_duration
	animated_sprite.play("idle")
	velocity = Vector2.ZERO

func _enter_standing_up():
	current_state = State.STANDING_UP
	animated_sprite.play("stand up")
	animated_sprite.frame = 0
	# Set timer for stand up animation (3 frames, adjust based on your animation)
	animation_timer = 1.5

func _enter_standing_idle():
	current_state = State.STANDING_IDLE
	state_timer = 1.0  # Brief pause
	animated_sprite.play("look up")

func _enter_moving():
	current_state = State.MOVING
	state_timer = move_duration
	_choose_random_direction_and_speed()
	time_until_direction_change = randf_range(2.0, 4.0)

func _enter_sitting_down():
	current_state = State.SITTING_DOWN
	animated_sprite.play("sit down")
	animated_sprite.frame = 0
	movement_direction = Vector2.ZERO
	# Set timer for sit down animation (3 frames, adjust based on your animation)
	animation_timer = 1.5

func _choose_random_direction_and_speed():
	# Choose a random direction
	var angle = randf() * TAU  # Random angle in radians
	movement_direction = Vector2(cos(angle), sin(angle)).normalized()
	
	# Randomly choose walk or run
	current_move_speed = MoveSpeed.RUN if randf() > 0.5 else MoveSpeed.WALK

func _bounce_off_wall():
	# Get wall normal to determine bounce direction
	var wall_normal = get_wall_normal()
	
	if wall_normal != Vector2.ZERO:
		# Reflect the movement direction based on wall normal
		movement_direction = movement_direction.bounce(wall_normal).normalized()
		
		# Add slight randomness to prevent getting stuck
		movement_direction = movement_direction.rotated(randf_range(-0.3, 0.3))
		
		# Optionally change speed when bouncing
		if randf() > 0.7:
			current_move_speed = MoveSpeed.RUN if current_move_speed == MoveSpeed.WALK else MoveSpeed.WALK

func _update_move_animation():
	# Use walk or run animations based on current speed
	var anim_prefix = "run" if current_move_speed == MoveSpeed.RUN else "walk"
	
	if abs(movement_direction.x) > abs(movement_direction.y):
		# Moving more horizontally
		if movement_direction.x > 0:
			animated_sprite.play(anim_prefix + " right")
		else:
			animated_sprite.play(anim_prefix + " left")
	else:
		# Moving more vertically, use left/right based on slight x movement
		if movement_direction.x >= 0:
			animated_sprite.play(anim_prefix + " right")
		else:
			animated_sprite.play(anim_prefix + " left")