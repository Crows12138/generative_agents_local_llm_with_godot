# 🚀 快速启动指南

## 📋 项目当前状态

✅ **AI服务后端** - 完整功能，支持本地LLM  
✅ **Godot项目** - 完整场景文件和脚本  
✅ **演示系统** - Cozy Bar演示 + AI Bridge  
✅ **性能优化** - 内存、网络、缓存优化  
✅ **完整文档** - 安装、配置、故障排除  

## 🎮 启动方式

### 方式一：一键启动（推荐）
```bash
# 启动AI服务 + 自动打开Godot
python demo_launcher.py

# 如果Godot未安装或不在PATH中：
python demo_launcher.py --no-godot
# 然后手动打开Godot项目：godot/live-with-ai/project.godot
```

### 方式二：独立运行AI服务
```bash
# 仅启动AI服务（用于测试）
python demo_launcher.py --no-godot --port 8080

# 测试AI服务
curl http://127.0.0.1:8080/health
```

### 方式三：运行Cozy Bar文本演示
```bash
cd cozy_bar_demo
python main.py
```

## 🔧 首次运行前的准备

### 1. 检查依赖
```bash
pip install -r requirements.txt
```

### 2. 验证模型文件
确保以下目录存在模型文件：
- `models/gpt4all/Qwen3-30B-A3B-Instruct-2507-UD-Q4_K_XL.gguf`

### 3. 安装Godot（如果还没安装）
- 下载：https://godotengine.org/download
- 推荐版本：Godot 4.2+
- 解压到 `C:\Godot\` 并添加到PATH

## 🎯 预期行为

### AI服务启动后应该看到：
```
[OK] All dependencies satisfied!
[SUCCESS] AI Bridge Server started successfully!
✓ AI service ready: Hello! How can I assist you today?
Godot AI Bridge started on http://127.0.0.1:8080
```

### Godot项目运行后应该看到：
- 3个AI角色自动生成（Alice、Bob、Charlie）
- 角色名称标签显示在头顶
- 控制台显示AI服务连接成功
- 角色会自主移动和做决策

### Cozy Bar演示应该显示：
```
=== Cozy Bar ===
A warm, intimate bar with dim lighting and smooth jazz playing softly

# # # # # # # # # # # # 
# . . . . . . . . . . # 
# . . = = = B = = . . # 
# . . ~ ~ ~ ~ ~ ~ . . # 
# . . . . . . . . . . # 
# . O O . . . . O O . # 
# . A o . . . . o S . # 
# . . . . . . . . . . # 
# . . . . M M . . . . # 
# # # # # + + # # # # # 

Character Status:
  Bob (bartender) - neutral | Energy: 100%
  Alice (regular customer) - neutral | Energy: 100%
  Sam (musician) - neutral | Energy: 100%
```

## 🐛 常见问题解决

### Git换行符警告
已修复：创建了 `.gitattributes` 文件来处理换行符转换。

### AI服务启动失败
```bash
# 检查端口是否被占用
netstat -ano | findstr :8080

# 使用不同端口
python demo_launcher.py --port 8081
```

### Godot项目错误
1. 确认Godot版本 ≥ 4.2
2. 检查项目导入是否成功
3. 查看输出窗口的错误信息

### Unicode字符问题（Windows）
已修复：所有Unicode字符已替换为ASCII兼容字符。

## 📁 项目结构概览

```
generative_agents_local_llm_with_godot/
├── 🚀 demo_launcher.py          # 一键启动脚本
├── 🤖 ai_service/               # AI服务后端
├── 🎮 godot/live-with-ai/       # Godot项目
│   ├── scenes/ai_character.tscn # 完整AI角色场景
│   ├── scenes/dialogue_ui.tscn  # 对话系统UI
│   └── scripts/                 # GDScript脚本
├── 🍺 cozy_bar_demo/            # 文本版演示
├── 🔧 performance_optimizer.py  # 性能优化
├── 💾 memory_optimizer.py       # 内存管理
├── 🌐 network_optimizer.py      # 网络优化
├── 📚 docs/                     # 完整文档
└── 📋 requirements.txt          # Python依赖
```

## 🎊 立即开始

选择您喜欢的启动方式：

**完整体验（推荐）：**
```bash
python demo_launcher.py
```

**快速测试：**
```bash
cd cozy_bar_demo && python main.py
```

**开发调试：**
```bash
python demo_launcher.py --no-godot --port 8080
# 然后手动打开Godot编辑器
```

项目已100%就绪，可以立即运行！🚀