extends CanvasLayer
# Real-time system monitor overlay
# Shows environment states and NPC decisions

var monitor_panel: Panel
var info_label: RichTextLabel
var update_timer: float = 0.0
var update_interval: float = 0.5

func _ready():
	create_monitor_ui()
	set_process(true)

func create_monitor_ui():
	# Create panel
	monitor_panel = Panel.new()
	monitor_panel.size = Vector2(300, 400)
	monitor_panel.position = Vector2(10, 10)
	monitor_panel.modulate.a = 0.9
	add_child(monitor_panel)
	
	# Create info label
	info_label = RichTextLabel.new()
	info_label.size = Vector2(280, 380)
	info_label.position = Vector2(10, 10)
	info_label.add_theme_font_size_override("normal_font_size", 12)
	info_label.add_theme_font_size_override("bold_font_size", 14)
	monitor_panel.add_child(info_label)
	
	# Add close button
	var close_btn = Button.new()
	close_btn.text = "X"
	close_btn.size = Vector2(30, 30)
	close_btn.position = Vector2(260, 5)
	close_btn.pressed.connect(_on_close)
	monitor_panel.add_child(close_btn)

func _process(delta):
	update_timer += delta
	if update_timer >= update_interval:
		update_timer = 0.0
		update_monitor()

func update_monitor():
	info_label.clear()
	info_label.append_text("[b]System Monitor[/b]\n")
	info_label.append_text("-------------------\n\n")
	
	# Environment State
	info_label.append_text("[b]Environment:[/b]\n")
	var bar_counter = get_node_or_null("/root/CozyBar/BarCounter")
	if bar_counter:
		var cleanliness = bar_counter.get("cleanliness")
		if cleanliness != null:
			var status = "Clean" if cleanliness > 70 else ("Dirty" if cleanliness < 30 else "OK")
			info_label.append_text("- Bar: %d%% [%s]\n" % [cleanliness, status])
	
	var shelf = get_node_or_null("/root/CozyBar/LiquorShelf")
	if shelf:
		var stock = shelf.get("stock_level")
		if stock != null:
			var status = "Full" if stock > 70 else ("Low" if stock < 30 else "OK")
			info_label.append_text("- Stock: %d%% [%s]\n" % [stock, status])
	
	# Tables
	var tables_node = get_node_or_null("/root/CozyBar/Tables")
	if tables_node:
		var dirty_count = 0
		for table in tables_node.get_children():
			var cleanliness = table.get("cleanliness")
			var has_dishes = table.get("has_dishes")
			if (cleanliness != null and cleanliness < 30) or has_dishes:
				dirty_count += 1
		info_label.append_text("- Dirty Tables: %d/3\n" % dirty_count)
	
	info_label.append_text("\n[b]NPC Status:[/b]\n")
	
	# NPC States
	for npc_name in ["bob", "Alice", "sam"]:
		var npc = get_node_or_null("/root/CozyBar/" + npc_name)
		if npc:
			var controller = npc.get_node_or_null("DecisionController")
			var executor = npc.get_node_or_null("DecisionController/ActionExecutor")
			var detector = npc.get_node_or_null("DecisionController/StateDetector")
			
			info_label.append_text("\n[b]%s:[/b]\n" % npc_name.capitalize())
			
			if controller:
				var connected = controller.is_server_connected()
				info_label.append_text("- Server: %s\n" % ("OK" if connected else "X"))
			
			if executor:
				var busy = executor.is_busy()
				var queue_size = executor.action_buffer.size()
				info_label.append_text("- Status: %s\n" % ("Busy" if busy else "Idle"))
				if queue_size > 0:
					info_label.append_text("- Queue: %d actions\n" % queue_size)
				if busy and executor.current_action.size() > 0:
					info_label.append_text("- Action: %s\n" % executor.current_action.get("action", "?"))
			
			if detector:
				var state = detector.get_compressed_state()
				if state != null and state > 0:
					var decoded = detector.decode_state(state)
					var states = []
					for key in decoded:
						if decoded[key]:
							states.append(key.replace("_", " "))
					if states.size() > 0:
						info_label.append_text("- Sees: %s\n" % ", ".join(states))
	
	# Environment Manager Stats
	var env_manager = get_node_or_null("/root/CozyBar/EnvironmentStateManager")
	if env_manager:
		info_label.append_text("\n[b]Environment Manager:[/b]\n")
		info_label.append_text("- Conversations: %d\n" % env_manager.conversation_count)
		info_label.append_text("- Time: %ds\n" % int(env_manager.time_elapsed))
	
	# Customer Stats
	var customer_manager = get_node_or_null("/root/CozyBar/CustomerManager")
	if customer_manager:
		info_label.append_text("\n[b]Customers:[/b]\n")
		var stats = customer_manager.get_stats()
		info_label.append_text("- Served: %d\n" % stats.served)
		info_label.append_text("- Lost: %d\n" % stats.lost)
		info_label.append_text("- Waiting: %d\n" % stats.waiting)
		if stats.served > 0 or stats.lost > 0:
			info_label.append_text("- Satisfaction: %.0f%%\n" % stats.satisfaction)
		
		# Show waiting customers
		var waiting = customer_manager.get_waiting_customers()
		for customer in waiting:
			var urgency = "!" if customer.urgency > 0.5 else ""
			info_label.append_text("  - %s (%.0fs)%s\n" % [customer.location, customer.wait_time, urgency])

func _on_close():
	queue_free()

func _input(event):
	# Toggle with F9
	if event.is_action_pressed("ui_page_down"):  # or use F9 if configured
		monitor_panel.visible = !monitor_panel.visible
