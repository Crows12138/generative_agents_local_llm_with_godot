extends Node

# Ultra Simple LLM - The shortest possible path to call local model

func talk_to_npc(npc_name: String, message: String = "Hello") -> String:
	"""Shortest path - just 3 lines!"""
	
	var output = []
	OS.execute("python", ["quick_llm.py", message], output, true)
	return output[0] if output.size() > 0 else "Hello!"

func _ready():
	# Test it
	var response = talk_to_npc("Bob", "How's the bar today?")
	print("Bob says: ", response)

# Even simpler - one-liner!
func quick_chat(msg: String) -> String:
	var o = []; OS.execute("python", ["quick_llm.py", msg], o, true); return o[0] if o.size() > 0 else "Hi!"