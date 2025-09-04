extends Node

# Simple game manager to load the bar scene

func _ready():
	print("Game Manager starting...")
	_load_bar_scene()

func _load_bar_scene():
	if ResourceLoader.exists("res://scenes/cozy_bar.tscn"):
		var bar_scene = load("res://scenes/cozy_bar.tscn")
		var bar_instance = bar_scene.instantiate()
		add_child(bar_instance)
		print("Bar scene loaded")
		
		# Add a simple camera
		var camera = Camera2D.new()
		camera.enabled = true
		camera.position = Vector2(640, 360)
		add_child(camera)
