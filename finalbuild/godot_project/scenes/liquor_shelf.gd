extends StaticBody2D

@export var stock_level: float = 100.0

func get_stock_level() -> float:
		return stock_level

func set_stock_level(value: float):
		stock_level = clamp(value, 0, 100)
