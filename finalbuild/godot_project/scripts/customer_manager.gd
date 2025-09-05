extends Node
# Customer Manager - Simulates customers needing service
# Creates service opportunities for NPCs

signal customer_arrived(location: String)
signal customer_served(location: String)
signal customer_left(location: String)

# Customer spawn settings
@export var spawn_interval_min: float = 20.0  # Minimum seconds between customers
@export var spawn_interval_max: float = 40.0  # Maximum seconds between customers
@export var initial_delay: float = 5.0  # Wait before first customer

# Customer locations
var customer_spots = {
	"bar": {"position": Vector2(640, 300), "occupied": false, "wait_time": 0.0},
	"table1": {"position": Vector2(320, 400), "occupied": false, "wait_time": 0.0},
	"table2": {"position": Vector2(640, 450), "occupied": false, "wait_time": 0.0},
	"table3": {"position": Vector2(960, 500), "occupied": false, "wait_time": 0.0}
}

# Customer patience (seconds before leaving)
@export var customer_patience: float = 60.0

var spawn_timer: Timer
var update_timer: Timer
var customers_served: int = 0
var customers_lost: int = 0

func _ready():
	print("[CustomerManager] Initializing customer system")
	
	# Setup spawn timer
	spawn_timer = Timer.new()
	spawn_timer.one_shot = true
	spawn_timer.timeout.connect(_spawn_customer)
	add_child(spawn_timer)
	spawn_timer.start(initial_delay)
	
	# Setup update timer for customer patience
	update_timer = Timer.new()
	update_timer.wait_time = 1.0
	update_timer.timeout.connect(_update_customers)
	add_child(update_timer)
	update_timer.start()
	
	print("[CustomerManager] Ready - Customers will arrive every %d-%d seconds" % [spawn_interval_min, spawn_interval_max])

func _spawn_customer():
	"""Spawn a new customer at an available location"""
	# Find available spots
	var available_spots = []
	for spot_name in customer_spots:
		if not customer_spots[spot_name].occupied:
			available_spots.append(spot_name)
	
	if available_spots.size() > 0:
		# Choose random spot
		var chosen_spot = available_spots[randi() % available_spots.size()]
		customer_spots[chosen_spot].occupied = true
		customer_spots[chosen_spot].wait_time = 0.0
		
		print("[CustomerManager] Customer arrived at ", chosen_spot)
		customer_arrived.emit(chosen_spot)
		
		# Update environment state based on location
		_update_environment_for_customer(chosen_spot, true)
	else:
		print("[CustomerManager] No available spots for new customer")
	
	# Schedule next customer
	var next_spawn = randf_range(spawn_interval_min, spawn_interval_max)
	spawn_timer.start(next_spawn)

func _update_customers():
	"""Update customer wait times and patience"""
	for spot_name in customer_spots:
		var spot = customer_spots[spot_name]
		if spot.occupied:
			spot.wait_time += 1.0
			
			# Check patience
			if spot.wait_time >= customer_patience:
				print("[CustomerManager] Customer at ", spot_name, " left (waited too long)")
				_customer_leaves(spot_name)
				customers_lost += 1

func _update_environment_for_customer(location: String, arrived: bool):
	"""Update environment objects to reflect customer presence"""
	match location:
		"bar":
			var bar_counter = get_node_or_null("/root/CozyBar/BarCounter")
			if bar_counter:
				bar_counter.set("has_customers", arrived)
				var count = bar_counter.get("customer_count")
				if count == null:
					count = 0
				bar_counter.set("customer_count", count + 1 if arrived else max(0, count - 1))
				print("  Bar customer count: ", bar_counter.get("customer_count"))
		
		"table1", "table2", "table3":
			var table_num = location.trim_prefix("table")
			var table = get_node_or_null("/root/CozyBar/Tables/Table" + table_num)
			if table:
				table.set("occupied", arrived)
				if arrived:
					# Customer at table will make it dirty over time
					table.set("has_dishes", true)
				print("  Table", table_num, " occupied: ", arrived)

func serve_customer(location: String) -> bool:
	"""Called when NPC serves a customer"""
	if not customer_spots.has(location):
		return false
	
	var spot = customer_spots[location]
	if spot.occupied:
		print("[CustomerManager] Customer at ", location, " was served!")
		customers_served += 1
		
		# Clear the spot
		spot.occupied = false
		spot.wait_time = 0.0
		
		# Update environment
		_update_environment_for_customer(location, false)
		
		# Emit signal
		customer_served.emit(location)
		
		# Bonus: Speed up next customer if service is good
		if spot.wait_time < 10.0:
			print("  Quick service! Next customer arrives sooner")
			var current_time = spawn_timer.time_left
			if current_time > 5.0:
				spawn_timer.start(5.0)
		
		return true
	return false

func _customer_leaves(location: String):
	"""Customer leaves without being served"""
	var spot = customer_spots[location]
	spot.occupied = false
	spot.wait_time = 0.0
	
	_update_environment_for_customer(location, false)
	customer_left.emit(location)

func get_waiting_customers() -> Array:
	"""Get list of locations with waiting customers"""
	var waiting = []
	for spot_name in customer_spots:
		if customer_spots[spot_name].occupied:
			waiting.append({
				"location": spot_name,
				"wait_time": customer_spots[spot_name].wait_time,
				"urgency": customer_spots[spot_name].wait_time / customer_patience
			})
	return waiting

func get_most_urgent_customer() -> String:
	"""Get the customer who has been waiting longest"""
	var most_urgent = ""
	var longest_wait = 0.0
	
	for spot_name in customer_spots:
		var spot = customer_spots[spot_name]
		if spot.occupied and spot.wait_time > longest_wait:
			longest_wait = spot.wait_time
			most_urgent = spot_name
	
	return most_urgent

func get_stats() -> Dictionary:
	"""Get customer service statistics"""
	return {
		"served": customers_served,
		"lost": customers_lost,
		"waiting": get_waiting_customers().size(),
		"satisfaction": 0.0 if customers_served == 0 else float(customers_served) / (customers_served + customers_lost) * 100
	}

# Debug visualization
func create_customer_markers():
	"""Create visual markers for customers (debug)"""
	for spot_name in customer_spots:
		var spot = customer_spots[spot_name]
		if spot.occupied:
			var marker = ColorRect.new()
			marker.size = Vector2(20, 20)
			marker.position = spot.position - Vector2(10, 10)
			marker.color = Color.YELLOW if spot.wait_time < 30 else Color.RED
			marker.name = "CustomerMarker_" + spot_name
			get_parent().add_child(marker)