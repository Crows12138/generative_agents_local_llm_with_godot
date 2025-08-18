extends Control
# Dialogue System - Manages dialogue UI and interactions

signal dialogue_started
signal dialogue_ended
signal choice_selected(choice_index)

@onready var dialogue_box = $DialogueBox
@onready var speaker_label = $DialogueBox/SpeakerLabel
@onready var dialogue_text = $DialogueBox/DialogueText
@onready var choices_container = $DialogueBox/ChoicesContainer
@onready var continue_button = $DialogueBox/ContinueButton
@onready var portrait = $DialogueBox/Portrait

var current_dialogue: Array = []
var dialogue_index: int = 0
var is_active: bool = false
var current_speaker: String = ""
var typing_speed: float = 0.05
var auto_continue: bool = false
var auto_continue_delay: float = 3.0

# Choice button scene (you'll need to create this)
var choice_button_scene = preload("res://scenes/ui/choice_button.tscn") if ResourceLoader.exists("res://scenes/ui/choice_button.tscn") else null

func _ready():
	hide()
	if continue_button:
		continue_button.pressed.connect(_on_continue_pressed)

func start_dialogue(speaker: String, text: String, portrait_texture: Texture2D = null):
	"""Start a simple dialogue"""
	current_speaker = speaker
	current_dialogue = [{"speaker": speaker, "text": text, "portrait": portrait_texture}]
	dialogue_index = 0
	is_active = true
	
	show()
	emit_signal("dialogue_started")
	_display_current_dialogue()

func start_dialogue_sequence(dialogue_data: Array):
	"""Start a sequence of dialogues"""
	if dialogue_data.is_empty():
		return
	
	current_dialogue = dialogue_data
	dialogue_index = 0
	is_active = true
	
	show()
	emit_signal("dialogue_started")
	_display_current_dialogue()

func _display_current_dialogue():
	"""Display the current dialogue entry"""
	if dialogue_index >= current_dialogue.size():
		end_dialogue()
		return
	
	var entry = current_dialogue[dialogue_index]
	
	# Set speaker
	if speaker_label:
		speaker_label.text = entry.get("speaker", "")
		current_speaker = entry.get("speaker", "")
	
	# Set portrait
	if portrait and entry.has("portrait"):
		portrait.texture = entry.portrait
		portrait.visible = entry.portrait != null
	
	# Handle choices
	if entry.has("choices"):
		_display_choices(entry.choices)
		if continue_button:
			continue_button.hide()
	else:
		_clear_choices()
		if continue_button:
			continue_button.show()
	
	# Display text with typing effect
	if entry.has("text"):
		_type_text(entry.text)
	
	# Auto-continue if enabled
	if auto_continue and !entry.has("choices"):
		await get_tree().create_timer(auto_continue_delay).timeout
		if is_active:
			advance_dialogue()

func _type_text(text: String):
	"""Display text with typing effect"""
	if !dialogue_text:
		return
	
	dialogue_text.text = ""
	dialogue_text.visible_characters = 0
	dialogue_text.text = text
	
	var char_count = text.length()
	for i in range(char_count):
		if !is_active:
			break
		dialogue_text.visible_characters = i + 1
		await get_tree().create_timer(typing_speed).timeout
	
	dialogue_text.visible_characters = -1  # Show all characters

func _display_choices(choices: Array):
	"""Display dialogue choices"""
	_clear_choices()
	
	if !choices_container:
		return
	
	for i in range(choices.size()):
		var choice = choices[i]
		var button = Button.new()
		button.text = choice.get("text", "Option " + str(i + 1))
		button.pressed.connect(_on_choice_selected.bind(i))
		choices_container.add_child(button)

func _clear_choices():
	"""Clear all choice buttons"""
	if !choices_container:
		return
	
	for child in choices_container.get_children():
		child.queue_free()

func _on_choice_selected(choice_index: int):
	"""Handle choice selection"""
	emit_signal("choice_selected", choice_index)
	
	var entry = current_dialogue[dialogue_index]
	if entry.has("choices") and choice_index < entry.choices.size():
		var choice = entry.choices[choice_index]
		
		# Handle choice consequences
		if choice.has("next"):
			# Jump to specific dialogue index
			dialogue_index = choice.next
			_display_current_dialogue()
		elif choice.has("action"):
			# Trigger an action
			_trigger_action(choice.action)
		else:
			# Default: advance to next dialogue
			advance_dialogue()

func _trigger_action(action: String):
	"""Trigger a dialogue action"""
	# This can be extended to handle various actions
	match action:
		"end":
			end_dialogue()
		"shop":
			# Open shop interface
			pass
		"quest":
			# Give quest
			pass
		_:
			# Default: continue
			advance_dialogue()

func advance_dialogue():
	"""Move to the next dialogue entry"""
	dialogue_index += 1
	_display_current_dialogue()

func end_dialogue():
	"""End the current dialogue"""
	is_active = false
	hide()
	_clear_choices()
	current_dialogue.clear()
	dialogue_index = 0
	emit_signal("dialogue_ended")

func _on_continue_pressed():
	"""Handle continue button press"""
	if dialogue_text and dialogue_text.visible_characters < dialogue_text.text.length():
		# Skip typing effect
		dialogue_text.visible_characters = -1
	else:
		# Advance dialogue
		advance_dialogue()

func _input(event):
	"""Handle input during dialogue"""
	if !is_active:
		return
	
	if event.is_action_pressed("ui_accept") or event.is_action_pressed("interact"):
		if dialogue_text and dialogue_text.visible_characters < dialogue_text.text.length():
			# Skip typing effect
			dialogue_text.visible_characters = -1
		elif !current_dialogue[dialogue_index].has("choices"):
			# Advance dialogue if no choices
			advance_dialogue()

func is_dialogue_active() -> bool:
	"""Check if dialogue is currently active"""
	return is_active

func get_current_speaker() -> String:
	"""Get the current speaker"""
	return current_speaker

func set_typing_speed(speed: float):
	"""Set the typing speed"""
	typing_speed = max(0.001, speed)

func set_auto_continue(enabled: bool, delay: float = 3.0):
	"""Enable/disable auto-continue"""
	auto_continue = enabled
	auto_continue_delay = max(0.5, delay)

# Helper function to create dialogue from AI response
func create_dialogue_from_ai(speaker: String, ai_response: String) -> Array:
	"""Create dialogue data from AI response"""
	var dialogue = []
	
	# Split response into sentences for better pacing
	var sentences = ai_response.split(". ")
	for sentence in sentences:
		if sentence.strip_edges() != "":
			dialogue.append({
				"speaker": speaker,
				"text": sentence.strip_edges() + ("." if !sentence.ends_with(".") else "")
			})
	
	return dialogue
