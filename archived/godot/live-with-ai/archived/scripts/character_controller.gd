extends CharacterBody2D
# Character Controller - Controls AI character behavior and movement

signal interaction_started(character_name)
signal interaction_ended(character_name)
signal thought_bubble_show(text)
signal emotion_changed(emotion)

@export var character_name: String = "NPC"
@export var personality: String = "friendly and helpful"
@export var move_speed: float = 100.0
@export var interaction_range: float = 50.0

# AI state
var current_emotion: String = "neutral"
var current_activity: String = "idle"
var current_goal: String = ""
var is_thinking: bool = false
var is_talking: bool = false

# Movement
var target_position: Vector2
var is_moving: bool = false
var path: Array = []
var path_index: int = 0

# Interaction
var nearby_characters: Array = []
var interaction_partner: CharacterBody2D = null
var conversation_history: Array = []

# Needs system
var energy: float = 1.0
var hunger: float = 0.0
var social_need: float = 0.5

# References
@onready var ai_manager = get_node("/root/AIManager")
@onready var sprite = $Sprite2D
@onready var collision = $CollisionShape2D
@onready var interaction_area = $InteractionArea
@onready var thought_bubble = $ThoughtBubble
@onready var emotion_indicator = $EmotionIndicator
@onready var name_label = $NameLabel

func _ready():
	# Set character name
	if name_label:
		name_label.text = character_name
	
	# Connect interaction area
	if interaction_area:
		interaction_area.body_entered.connect(_on_character_entered_range)
		interaction_area.body_exited.connect(_on_character_exited_range)
	
	# Connect to AI manager
	if ai_manager:
		ai_manager.ai_response_received.connect(_on_ai_response)
		ai_manager.ai_error.connect(_on_ai_error)
	
	# Start behavior loop
	_start_behavior_loop()

func _physics_process(delta):
	# Update needs
	_update_needs(delta)
	
	# Handle movement
	if is_moving and target_position:
		_move_towards_target(delta)
	
	# Update animation based on state
	_update_animation()

func _start_behavior_loop():
	"""Start the AI behavior loop"""
	await get_tree().create_timer(randf_range(1.0, 3.0)).timeout
	_decide_next_action()

func _decide_next_action():
	"""Decide what to do next using AI"""
	if is_thinking or is_talking:
		# Wait and try again later
		await get_tree().create_timer(2.0).timeout
		_decide_next_action()
		return
	
	# Build context
	var context = {
		"location": global_position,
		"energy": energy,
		"hunger": hunger,
		"social_need": social_need,
		"nearby_characters": nearby_characters.size(),
		"current_activity": current_activity
	}
	
	# Determine available actions based on state
	var actions = ["idle", "wander", "rest"]
	
	if energy < 0.3:
		actions.append("rest")
	if hunger > 0.7:
		actions.append("find food")
	if social_need > 0.7 and nearby_characters.size() > 0:
		actions.append("start conversation")
	if nearby_characters.size() == 0:
		actions.append("explore")
	
	# Ask AI to decide
	var situation = "You are in a town. Energy: %.1f, Hunger: %.1f, Social: %.1f" % [energy, hunger, social_need]
	ai_manager.decide(character_name, situation, actions, context)
	
	# Schedule next decision
	await get_tree().create_timer(randf_range(5.0, 10.0)).timeout
	_decide_next_action()

func _on_ai_response(data: Dictionary):
	"""Handle AI response"""
	if data.character_name != character_name:
		return
	
	# Handle decision response
	if data.has("chosen_option"):
		_execute_action(data.chosen_option)
	
	# Handle chat response
	if data.has("response"):
		_show_speech(data.response)
		if data.has("emotion"):
			_set_emotion(data.emotion)
	
	# Handle thought response
	if data.has("thought"):
		_show_thought(data.thought)
		if data.has("mood"):
			_set_emotion(data.mood)

func _execute_action(action: String):
	"""Execute the chosen action"""
	print(character_name, " decides to: ", action)
	
	match action.to_lower():
		"idle":
			current_activity = "idle"
			is_moving = false
		"wander":
			_start_wandering()
		"rest":
			current_activity = "resting"
			is_moving = false
			energy = min(1.0, energy + 0.3)
		"find food":
			current_activity = "seeking_food"
			_find_nearest_food()
		"start conversation":
			if nearby_characters.size() > 0:
				_start_conversation(nearby_characters[0])
		"explore":
			_start_exploring()
		_:
			current_activity = "idle"

func _start_wandering():
	"""Start wandering around"""
	current_activity = "wandering"
	var random_offset = Vector2(
		randf_range(-200, 200),
		randf_range(-200, 200)
	)
	target_position = global_position + random_offset
	is_moving = true

func _start_exploring():
	"""Start exploring the area"""
	current_activity = "exploring"
	# Move to a random distant location
	var random_offset = Vector2(
		randf_range(-500, 500),
		randf_range(-500, 500)
	)
	target_position = global_position + random_offset
	is_moving = true

func _find_nearest_food():
	"""Find nearest food source (placeholder)"""
	# In a real implementation, this would search for food objects
	_start_wandering()

func _start_conversation(other_character: CharacterBody2D):
	"""Start a conversation with another character"""
	if is_talking or !other_character:
		return
	
	is_talking = true
	interaction_partner = other_character
	current_activity = "talking"
	
	emit_signal("interaction_started", character_name)
	
	# Generate greeting
	var context = {
		"partner": other_character.character_name if other_character.has("character_name") else "someone",
		"emotion": current_emotion,
		"location": "town square"
	}
	
	ai_manager.chat(character_name, "Hello there!", context)

func _move_towards_target(delta):
	"""Move character towards target position"""
	var direction = (target_position - global_position).normalized()
	var distance = global_position.distance_to(target_position)
	
	if distance > 5:
		velocity = direction * move_speed
		move_and_slide()
	else:
		is_moving = false
		velocity = Vector2.ZERO

func _update_needs(delta):
	"""Update character needs over time"""
	energy -= delta * 0.02  # Decrease slowly
	hunger += delta * 0.03
	social_need += delta * 0.01
	
	# Clamp values
	energy = clamp(energy, 0.0, 1.0)
	hunger = clamp(hunger, 0.0, 1.0)
	social_need = clamp(social_need, 0.0, 1.0)
	
	# Update emotion based on needs
	if energy < 0.2:
		_set_emotion("tired")
	elif hunger > 0.8:
		_set_emotion("hungry")
	elif social_need > 0.8:
		_set_emotion("lonely")

func _update_animation():
	"""Update sprite animation based on state"""
	if !sprite:
		return
	
	# Flip sprite based on movement direction
	if velocity.x < 0:
		sprite.flip_h = true
	elif velocity.x > 0:
		sprite.flip_h = false
	
	# In a real implementation, you would play animations here
	# sprite.play(current_activity)

func _show_thought(text: String):
	"""Show thought bubble"""
	if thought_bubble:
		var thought_text = thought_bubble.get_node("ThoughtText")
		if thought_text:
			thought_text.text = text
		thought_bubble.visible = true
		
		# Start bubble timer
		var bubble_timer = thought_bubble.get_node("BubbleTimer")
		if bubble_timer:
			bubble_timer.start()
	
	emit_signal("thought_bubble_show", text)
	is_thinking = false

func _show_speech(text: String):
	"""Show speech bubble"""
	if thought_bubble:  # Reuse thought bubble for speech
		var thought_text = thought_bubble.get_node("ThoughtText")
		if thought_text:
			thought_text.text = text
		thought_bubble.visible = true
		
		# Start bubble timer
		var bubble_timer = thought_bubble.get_node("BubbleTimer")
		if bubble_timer:
			bubble_timer.start()
	
	print(character_name, ": ", text)

func _set_emotion(emotion: String):
	"""Set character emotion"""
	current_emotion = emotion
	emit_signal("emotion_changed", emotion)
	
	if emotion_indicator:
		# Update emotion indicator sprite/color
		match emotion:
			"happy":
				emotion_indicator.modulate = Color.YELLOW
			"sad":
				emotion_indicator.modulate = Color.BLUE
			"angry":
				emotion_indicator.modulate = Color.RED
			"worried":
				emotion_indicator.modulate = Color.ORANGE
			_:
				emotion_indicator.modulate = Color.WHITE

func _on_character_entered_range(body: Node2D):
	"""Handle character entering interaction range"""
	if body != self and body.has_method("get_character_name"):
		nearby_characters.append(body)

func _on_character_exited_range(body: Node2D):
	"""Handle character leaving interaction range"""
	if body in nearby_characters:
		nearby_characters.erase(body)
	
	if body == interaction_partner:
		is_talking = false
		interaction_partner = null
		emit_signal("interaction_ended", character_name)

func _on_ai_error(error_message: String):
	"""Handle AI error"""
	print("AI Error for ", character_name, ": ", error_message)
	# Fallback to simple behavior
	_execute_action("idle")

func think_about(topic: String):
	"""Make character think about something"""
	if is_thinking:
		return
	
	is_thinking = true
	var context = {
		"emotion": current_emotion,
		"activity": current_activity,
		"energy": energy
	}
	
	ai_manager.think(character_name, topic, context)

func get_character_name() -> String:
	"""Get character name"""
	return character_name

func get_state() -> Dictionary:
	"""Get current character state"""
	return {
		"name": character_name,
		"position": global_position,
		"emotion": current_emotion,
		"activity": current_activity,
		"energy": energy,
		"hunger": hunger,
		"social_need": social_need,
		"is_talking": is_talking,
		"is_thinking": is_thinking
	}

func set_state(state: Dictionary):
	"""Set character state from dictionary"""
	if state.has("position"):
		global_position = state.position
	if state.has("emotion"):
		_set_emotion(state.emotion)
	if state.has("activity"):
		current_activity = state.activity
	if state.has("energy"):
		energy = state.energy
	if state.has("hunger"):
		hunger = state.hunger
	if state.has("social_need"):
		social_need = state.social_need

func _on_bubble_timer_timeout():
	"""Hide thought bubble when timer expires"""
	if thought_bubble:
		thought_bubble.visible = false

func update_status_bars():
	"""Update visual status bars"""
	var status_ui = get_node("StatusUI")
	if status_ui and status_ui.visible:
		var energy_bar = status_ui.get_node("EnergyBar")
		var hunger_bar = status_ui.get_node("HungerBar")
		var social_bar = status_ui.get_node("SocialBar")
		
		if energy_bar:
			energy_bar.value = energy * 100
		if hunger_bar:
			hunger_bar.value = hunger * 100
		if social_bar:
			social_bar.value = social_need * 100

func show_status_ui():
	"""Show character status UI"""
	var status_ui = get_node("StatusUI")
	if status_ui:
		status_ui.visible = true
		update_status_bars()

func hide_status_ui():
	"""Hide character status UI"""
	var status_ui = get_node("StatusUI")
	if status_ui:
		status_ui.visible = false
