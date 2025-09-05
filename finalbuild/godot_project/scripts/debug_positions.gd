extends Node2D
# Debug script to visualize virtual node positions
# Add this to your scene to see where the invisible nodes are

func _ready():
	print("\n=== POSITION DEBUG ===")
	print("Window size: 1280x720")
	print("\nVirtual environment objects:")
	
	# Check BarCounter
	var bar = get_node_or_null("BarCounter")
	if bar:
		print("  BarCounter at: ", bar.position)
		_draw_marker(bar.position, Color.BLUE, "BAR")
	
	# Check Tables
	var table1 = get_node_or_null("Tables/Table1")
	if table1:
		print("  Table1 at: ", table1.global_position)
		_draw_marker(table1.global_position, Color.GREEN, "T1")
		
	var table2 = get_node_or_null("Tables/Table2")
	if table2:
		print("  Table2 at: ", table2.global_position)
		_draw_marker(table2.global_position, Color.GREEN, "T2")
	
	# Check Shelf
	var shelf = get_node_or_null("LiquorShelf")
	if shelf:
		print("  Shelf at: ", shelf.position)
		_draw_marker(shelf.position, Color.YELLOW, "SHELF")
	
	print("\nNPC positions:")
	# Check NPC positions
	var bob = get_node_or_null("bob")
	if bob:
		print("  Bob at: ", bob.position)
		_draw_marker(bob.position, Color.RED, "Bob")
		
	var alice = get_node_or_null("Alice")
	if alice:
		print("  Alice at: ", alice.position)
		_draw_marker(alice.position, Color.MAGENTA, "Alice")
		
	var sam = get_node_or_null("sam")
	if sam:
		print("  Sam at: ", sam.position)
		_draw_marker(sam.position, Color.CYAN, "Sam")
	
	print("\nSuggested adjustments:")
	print("  - BarCounter should be in center-back (640, 200)")
	print("  - Tables spread around middle (400, 400) and (880, 400)")
	print("  - Shelf against back wall (640, 100)")
	print("======================\n")

func _draw_marker(pos: Vector2, color: Color, text: String):
	"""Create a visible marker at position"""
	var label = Label.new()
	label.text = text
	label.position = pos
	label.modulate = color
	label.add_theme_font_size_override("font_size", 20)
	add_child(label)
	
	# Add a small circle
	var circle = ColorRect.new()
	circle.size = Vector2(10, 10)
	circle.position = pos - Vector2(5, 5)
	circle.color = color
	circle.color.a = 0.5
	add_child(circle)

func _draw():
	# Draw grid for reference
	for x in range(0, 1280, 100):
		draw_line(Vector2(x, 0), Vector2(x, 720), Color(0.2, 0.2, 0.2, 0.3))
	for y in range(0, 720, 100):
		draw_line(Vector2(0, y), Vector2(1280, y), Color(0.2, 0.2, 0.2, 0.3))