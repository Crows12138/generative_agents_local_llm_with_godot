extends StaticBody2D

@export var cleanliness: float = 100.0
@export var has_customers: bool = false
@export var customer_count: int = 0

func get_cleanliness() -> float:
		return cleanliness

func set_cleanliness(value: float):
		cleanliness = clamp(value, 0, 100)
