extends Node
# Simple Environment Objects - Create basic bar objects without complex scripts

static func ensure_environment_objects_exist(scene_root: Node2D):
	"""Ensure BarCounter and LiquorShelf exist with basic properties"""
	
	# Create BarCounter if missing
	if not scene_root.has_node("BarCounter"):
		var bar_counter = StaticBody2D.new()
		bar_counter.name = "BarCounter"
		bar_counter.position = Vector2(318, 185)  # Your discovered position
		scene_root.add_child(bar_counter)
		
		# Add basic properties directly to the node
		bar_counter.set("cleanliness", 100.0)
		bar_counter.set("has_customers", false)
		bar_counter.set("customer_count", 0)
		
		print("  Created basic BarCounter with properties")
	
	# Create LiquorShelf if missing
	if not scene_root.has_node("LiquorShelf"):
		var shelf = StaticBody2D.new()
		shelf.name = "LiquorShelf"
		shelf.position = Vector2(315, 49)  # Your discovered position
		scene_root.add_child(shelf)
		
		# Add basic properties directly to the node
		shelf.set("stock_level", 100.0)
		
		print("  Created basic LiquorShelf with properties")
	
	print("  Environment objects verified/created")