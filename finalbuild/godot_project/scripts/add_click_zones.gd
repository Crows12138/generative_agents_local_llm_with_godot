extends Node
# Quick script to add visual click zones to the scene
# Run this in your main scene or add to bar_server_client_fixed.gd

static func add_visual_zones(scene_root: Node):
	"""Add visual click zones to scene"""
	
	# Remove old handlers
	if scene_root.has_node("SimpleClickHandler"):
		scene_root.get_node("SimpleClickHandler").queue_free()
	if scene_root.has_node("ClickableEnvironment"):
		scene_root.get_node("ClickableEnvironment").queue_free()
	
	# Check if visual zones already exist
	if scene_root.has_node("VisualClickZones"):
		print("Visual zones already exist")
		return
	
	# Create the visual click zones
	var zones = Node2D.new()
	zones.name = "VisualClickZones"
	zones.set_script(load("res://scripts/visual_click_zones.gd"))
	scene_root.add_child(zones)
	
	print("")
	print("========================================")
	print("VISUAL CLICK ZONES ADDED!")
	print("========================================")
	print("You should see 5 COLORED CIRCLES:")
	print("  - CYAN circle = Bar Counter")
	print("  - BLUE circle = Liquor Shelf")
	print("  - ORANGE circles = Tables")
	print("")
	print("CONTROLS:")
	print("  - LEFT CLICK circle = Make dirty/deplete")
	print("  - RIGHT CLICK circle = Clean/restock")
	print("  - MIDDLE CLICK + DRAG = Move circle")
	print("")
	print("If you can't see the circles:")
	print("  1. They might be off-screen")
	print("  2. Check z_index/layer issues")
	print("  3. Look at positions 640,200 (bar)")
	print("========================================")
	
# Add this to _ready() of your main scene:
# add_visual_zones(self)