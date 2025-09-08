extends CharacterBody2D

@export var speed = 100
@export var dog: Node2D
@onready var NA := $NavigationAgent2D as NavigationAgent2D
@onready var AP := $AnimatedSprite2D

func _ready():
	
	if dog:
		makepath()

func _physics_process(_delta: float) -> void:

	var distance_to_dog = global_position.distance_to(dog.global_position)

	if distance_to_dog <= 30:
		speed = 0
	else:
		speed = 30
	
	if not NA.is_navigation_finished():
		var next_pos = NA.get_next_path_position()
		var dir = global_position.direction_to(next_pos)
		velocity = dir.normalized() * speed
	else:
		velocity = Vector2.ZERO
	move_and_slide()
	update_animation()

func update_animation() -> void:
	if velocity.length() < 0.1:
		AP.play("idle")
	elif abs(velocity.x) > abs(velocity.y):
		if velocity.x < 0:
			AP.play("walk left")
		else:
			AP.play("walk right")
	else:
		if velocity.y > 0:
			AP.play("walk down")
		else:
			AP.play("walk up")

func makepath() -> void:
	if dog:
		NA.target_position = dog.global_position

func _on_timer_timeout():
	makepath()
