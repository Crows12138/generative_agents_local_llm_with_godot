# Godot Performance Optimization Configuration
extends Node

# Performance target constants
const TARGET_FPS = 60
const MIN_FPS = 30
const MAX_FRAME_TIME_MS = 16.67  # Frame time for 60 FPS

# Performance monitoring variables
var frame_times: Array = []
var current_fps: float = 0.0
var avg_frame_time: float = 0.0
var performance_warnings: Array = []

# Optimization settings
var enable_multithreading: bool = true
var enable_occlusion_culling: bool = true
var max_visible_characters: int = 20
var character_lod_distance: float = 500.0

# Dynamic quality adjustment
var current_quality_level: int = 2  # 0=low, 1=medium, 2=high
var auto_quality_adjustment: bool = true

func _ready():
	# Setup performance optimization options
	setup_performance_settings()
	
	# Start performance monitoring
	start_performance_monitoring()
	
	print("✓ Godot performance optimizer initialized")

func setup_performance_settings():
	"""Setup performance optimization options"""
	
	# Renderer settings
	RenderingServer.camera_set_use_occlusion_culling(get_viewport().get_camera_3d().get_camera_rid(), enable_occlusion_culling)
	
	# Threading settings
	if enable_multithreading:
		# Enable multithreaded rendering
		ProjectSettings.set_setting("rendering/driver/threads/thread_model", 2)
	
	# LOD settings
	setup_lod_system()
	
	# Memory optimization
	setup_memory_optimization()

func setup_lod_system():
	"""Setup LOD (Level of Detail) system"""
	
	# Set different LOD levels for characters
	var characters = get_tree().get_nodes_in_group("characters")
	for character in characters:
		if character.has_method("setup_lod"):
			character.setup_lod(character_lod_distance)

func setup_memory_optimization():
	"""Setup memory optimization"""
	
	# Set texture quality
	match current_quality_level:
		0:  # Low quality
			ProjectSettings.set_setting("rendering/textures/canvas_textures/default_texture_filter", 0)
			ProjectSettings.set_setting("rendering/textures/decals/filter", 0)
		1:  # Medium quality
			ProjectSettings.set_setting("rendering/textures/canvas_textures/default_texture_filter", 1)
			ProjectSettings.set_setting("rendering/textures/decals/filter", 1)
		2:  # High quality
			ProjectSettings.set_setting("rendering/textures/canvas_textures/default_texture_filter", 2)
			ProjectSettings.set_setting("rendering/textures/decals/filter", 2)
	
	# Garbage collection optimization
	GC.set_time_scale(0.5)  # Reduce GC frequency

func start_performance_monitoring():
	"""Start performance monitoring"""
	
	var timer = Timer.new()
	timer.wait_time = 1.0  # Check every second
	timer.timeout.connect(_on_performance_check)
	add_child(timer)
	timer.start()

func _on_performance_check():
	"""Performance check callback"""
	
	# Update FPS
	current_fps = Engine.get_frames_per_second()
	
	# Record frame time
	var frame_time = 1000.0 / max(current_fps, 1.0)
	frame_times.append(frame_time)
	
	# Keep history at reasonable size
	if frame_times.size() > 60:  # Keep 60 seconds of data
		frame_times.pop_front()
	
	# 计算平均帧时间
	avg_frame_time = 0.0
	for time in frame_times:
		avg_frame_time += time
	avg_frame_time /= frame_times.size()
	
	# 检查性能问题
	check_performance_issues()
	
	# 自动调整质量
	if auto_quality_adjustment:
		auto_adjust_quality()

func check_performance_issues():
	"""检查性能问题"""
	
	performance_warnings.clear()
	
	# 检查FPS
	if current_fps < MIN_FPS:
		performance_warnings.append("Low FPS: " + str(current_fps))
		print("⚠️ Performance Warning: Low FPS " + str(current_fps))
	
	# 检查帧时间
	if avg_frame_time > MAX_FRAME_TIME_MS * 1.5:
		performance_warnings.append("High frame time: " + str(avg_frame_time) + "ms")
		print("⚠️ Performance Warning: High frame time " + str(avg_frame_time) + "ms")
	
	# 检查内存使用
	var memory_usage = OS.get_static_memory_usage_by_type()
	var total_memory = 0
	for type in memory_usage:
		total_memory += memory_usage[type]
	
	var memory_mb = total_memory / 1024.0 / 1024.0
	if memory_mb > 2048:  # 超过2GB
		performance_warnings.append("High memory usage: " + str(memory_mb) + "MB")
		print("⚠️ Performance Warning: High memory usage " + str(memory_mb) + "MB")

func auto_adjust_quality():
	"""自动调整质量设置"""
	
	# 基于FPS调整质量
	if current_fps < MIN_FPS and current_quality_level > 0:
		# 降低质量
		current_quality_level -= 1
		apply_quality_settings()
		print("🔧 Quality reduced to level " + str(current_quality_level))
	
	elif current_fps > TARGET_FPS * 1.1 and current_quality_level < 2:
		# 提高质量
		current_quality_level += 1
		apply_quality_settings()
		print("🔧 Quality increased to level " + str(current_quality_level))

func apply_quality_settings():
	"""应用质量设置"""
	
	match current_quality_level:
		0:  # 低质量
			apply_low_quality_settings()
		1:  # 中等质量
			apply_medium_quality_settings()
		2:  # 高质量
			apply_high_quality_settings()

func apply_low_quality_settings():
	"""应用低质量设置"""
	
	# 减少可见角色数量
	max_visible_characters = 10
	character_lod_distance = 300.0
	
	# 降低渲染质量
	get_viewport().set_render_scale(0.8)
	
	# 禁用一些效果
	enable_occlusion_culling = false
	
	update_character_visibility()

func apply_medium_quality_settings():
	"""应用中等质量设置"""
	
	max_visible_characters = 15
	character_lod_distance = 400.0
	
	get_viewport().set_render_scale(0.9)
	
	enable_occlusion_culling = true
	
	update_character_visibility()

func apply_high_quality_settings():
	"""应用高质量设置"""
	
	max_visible_characters = 20
	character_lod_distance = 500.0
	
	get_viewport().set_render_scale(1.0)
	
	enable_occlusion_culling = true
	
	update_character_visibility()

func update_character_visibility():
	"""更新角色可见性"""
	
	var characters = get_tree().get_nodes_in_group("characters")
	var camera = get_viewport().get_camera_3d()
	
	if not camera:
		return
	
	var camera_pos = camera.global_position
	var visible_count = 0
	
	# 按距离排序角色
	var sorted_characters = []
	for character in characters:
		var distance = camera_pos.distance_to(character.global_position)
		sorted_characters.append({"character": character, "distance": distance})
	
	sorted_characters.sort_custom(func(a, b): return a.distance < b.distance)
	
	# 设置可见性
	for i in range(sorted_characters.size()):
		var char_data = sorted_characters[i]
		var character = char_data.character
		var distance = char_data.distance
		
		if i < max_visible_characters and distance < character_lod_distance:
			character.visible = true
			# 根据距离设置LOD
			if character.has_method("set_lod_level"):
				if distance < character_lod_distance * 0.3:
					character.set_lod_level(2)  # 高细节
				elif distance < character_lod_distance * 0.7:
					character.set_lod_level(1)  # 中等细节
				else:
					character.set_lod_level(0)  # 低细节
		else:
			character.visible = false

func optimize_for_mobile():
	"""移动端优化"""
	
	# 强制使用低质量设置
	current_quality_level = 0
	auto_quality_adjustment = false
	apply_quality_settings()
	
	# 移动端特定优化
	max_visible_characters = 8
	character_lod_distance = 200.0
	
	# 禁用一些效果
	enable_occlusion_culling = false
	enable_multithreading = false
	
	print("✓ Mobile optimization applied")

func get_performance_report() -> Dictionary:
	"""获取性能报告"""
	
	var report = {
		"timestamp": Time.get_datetime_string_from_system(),
		"performance_metrics": {
			"current_fps": current_fps,
			"avg_frame_time_ms": avg_frame_time,
			"target_fps": TARGET_FPS,
			"min_fps": MIN_FPS
		},
		"quality_settings": {
			"current_level": current_quality_level,
			"auto_adjustment": auto_quality_adjustment,
			"max_visible_characters": max_visible_characters,
			"lod_distance": character_lod_distance
		},
		"optimization_status": {
			"multithreading_enabled": enable_multithreading,
			"occlusion_culling_enabled": enable_occlusion_culling,
			"render_scale": get_viewport().get_render_scale()
		},
		"warnings": performance_warnings,
		"status": {
			"fps_ok": current_fps >= MIN_FPS,
			"frame_time_ok": avg_frame_time <= MAX_FRAME_TIME_MS * 1.2
		}
	}
	
	return report

# 外部接口函数
func set_quality_level(level: int):
	"""设置质量级别"""
	current_quality_level = clamp(level, 0, 2)
	apply_quality_settings()

func toggle_auto_quality(enabled: bool):
	"""切换自动质量调整"""
	auto_quality_adjustment = enabled

func force_performance_optimization():
	"""强制执行性能优化"""
	
	print("🚀 Forcing performance optimization...")
	
	# 清理内存
	GC.collect()
	
	# 重新应用设置
	setup_performance_settings()
	
	# 更新角色可见性
	update_character_visibility()
	
	print("✓ Performance optimization completed")

# 调试函数
func print_performance_stats():
	"""打印性能统计"""
	
	print("=== Godot Performance Stats ===")
	print("Current FPS: ", current_fps)
	print("Avg Frame Time: ", avg_frame_time, "ms")
	print("Quality Level: ", current_quality_level)
	print("Visible Characters: ", max_visible_characters)
	print("Warnings: ", performance_warnings.size())
	
	for warning in performance_warnings:
		print("  ⚠️ ", warning)
	
	print("===============================")