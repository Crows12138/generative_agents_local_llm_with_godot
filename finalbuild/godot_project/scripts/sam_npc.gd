extends CharacterBody2D

@onready var animated_sprite = $AnimatedSprite2D

var current_direction: Vector2 = Vector2.ZERO

func _ready():
	if not animated_sprite:
		push_error("AnimatedSprite2D not found! Please add one as a child of Sam.")
		return
	
	animated_sprite.play("idle")

func _physics_process(delta):
	# Update animation based on velocity
	if velocity.length() > 0:
		_update_animation_for_direction()
	else:
		if animated_sprite and animated_sprite.animation != "idle":
			animated_sprite.play("idle")
	
	# Move and slide will be called by external scripts that set velocity
	move_and_slide()

func _update_animation_for_direction():
	if not animated_sprite:
		return
	
	var animation_name = "idle"
	
	# Determine direction from velocity
	if abs(velocity.x) > abs(velocity.y):
		# Horizontal movement
		if velocity.x < 0:
			animation_name = "walk right"  # Moving left shows right animation
		else:
			animation_name = "walk left"   # Moving right shows left animation
	else:
		# Vertical movement
		if velocity.y < 0:
			animation_name = "walk up"
		else:
			animation_name = "walk dn"
	
	if animated_sprite.animation != animation_name:
		animated_sprite.play(animation_name)

func set_velocity_and_animate(new_velocity: Vector2):
	velocity = new_velocity
	if velocity.length() > 0:
		_update_animation_for_direction()
	else:
		if animated_sprite:
			animated_sprite.play("idle")

func stop_movement():
	velocity = Vector2.ZERO
	if animated_sprite:
		animated_sprite.play("idle")