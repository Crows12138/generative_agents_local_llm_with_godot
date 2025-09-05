extends Node
# Invisible Click Handler - Handles clicks without visual zones
# Based on the positions you found for bar and shelf

signal environment_clicked(object_name: String, action: String)

# Clickable zones (invisible)
var zones = {
	"BarCounter": {
		"position": Vector2(605, 320),  # 根据你的点击测试结果
		"type": "bar",  # 3-segment bar shape
	},
	"LiquorShelf": {
		"position": Vector2(600, 85),  # 根据你的点击测试结果  
		"type": "shelf",  # Horizontal rectangle
	}
}

func _ready():
	print("[InvisibleClickHandler] Ready - Click bar/shelf to change states")
	set_process_input(true)

func _input(event: InputEvent):
	"""Handle mouse clicks"""
	if not event is InputEventMouseButton:
		return
	if not event.pressed:
		return
	if event.button_index == MOUSE_BUTTON_MIDDLE:
		return  # No dragging in invisible mode
		
	# Get mouse position from the event itself
	var mouse_pos = event.global_position
	# print("[ClickDebug] Mouse clicked at: ", mouse_pos, " Button: ", event.button_index)  # Disabled debug
	
	# Check each zone
	for zone_name in zones:
		var zone_data = zones[zone_name]
		var zone_pos = zone_data.position
		var clicked = false
		
		if zone_data.type == "bar":
			# Check large 3-segment bar shape (like we debugged before)
			var relative_pos = mouse_pos - zone_pos
			
			# Wider coverage for the full bar area
			# Bottom horizontal bar (main counter)
			var in_bottom = (abs(relative_pos.x) < 120 and abs(relative_pos.y) < 25)
			# Left vertical bar (extended upward)
			var in_left = (relative_pos.x >= -120 and relative_pos.x <= -80 and 
						   relative_pos.y >= -120 and relative_pos.y <= -25)
			# Right vertical bar (extended upward)  
			var in_right = (relative_pos.x >= 80 and relative_pos.x <= 120 and
							relative_pos.y >= -120 and relative_pos.y <= -25)
			
			clicked = in_bottom or in_left or in_right
			
		elif zone_data.type == "shelf":
			# Check horizontal rectangle shelf
			var relative_pos = mouse_pos - zone_pos
			clicked = (abs(relative_pos.x) < 60 and abs(relative_pos.y) < 12)
		
		if clicked:
			# print("[ClickDebug] HIT! Zone: ", zone_name)  # Disabled debug
			_handle_zone_click(zone_name, event.button_index)
			get_viewport().set_input_as_handled()  # Prevent other handlers
			return
		# else:
		#	print("[ClickDebug] Miss. Zone: ", zone_name, " at ", zone_pos)  # Disabled debug

func _handle_zone_click(zone_name: String, button_index: int):
	"""Handle clicks on zones"""
	var target_obj = null
	
	if zone_name == "BarCounter":
		target_obj = get_node_or_null("/root/CozyBar/BarCounter")
		if target_obj:
			if button_index == MOUSE_BUTTON_LEFT:
				# Make dirty
				var current = target_obj.get("cleanliness")
				if current != null:
					target_obj.set("cleanliness", max(0, current - 30))
					print("[BarCounter] Dirtied: %d%%" % target_obj.get("cleanliness"))
				else:
					print("[BarCounter] Made dirty (no cleanliness property)")
			else:
				# Clean
				target_obj.set("cleanliness", 100)
				print("[BarCounter] Cleaned: 100%")
	
	elif zone_name == "LiquorShelf":
		target_obj = get_node_or_null("/root/CozyBar/LiquorShelf")
		if target_obj:
			if button_index == MOUSE_BUTTON_LEFT:
				# Deplete stock
				var current = target_obj.get("stock_level")
				if current != null:
					target_obj.set("stock_level", max(0, current - 25))
					print("[LiquorShelf] Stock: %d%%" % target_obj.get("stock_level"))
				else:
					print("[LiquorShelf] Stock depleted (no stock_level property)")
			else:
				# Restock
				target_obj.set("stock_level", 100)
				print("[LiquorShelf] Restocked: 100%")
	
	environment_clicked.emit(zone_name, button_index)
