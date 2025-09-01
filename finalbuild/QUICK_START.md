# 快速启动指南

## 启动服务器

```bash
cd finalbuild/server
../../.venv/Scripts/python.exe optimized_cozy_bar_server.py
```

服务器将在端口 9999 上启动，等待 Godot 连接。

## 查看记忆

查看所有 NPC 的记忆：
```bash
cd finalbuild
../.venv/Scripts/python.exe tools/view_memories.py
```

查看 Bob 的记忆：
```bash
cd finalbuild
../.venv/Scripts/python.exe tools/view_memories.py Bob
```

查看最近 5 条记忆和统计：
```bash
cd finalbuild
../.venv/Scripts/python.exe tools/view_memories.py Bob --last 5 --stats
```

## 配置

编辑 `server/cognitive_config.json` 调整参数：

- `enable_4b`: 启用深度思考模式（默认 true）
- `bob_deep_think_probability`: Bob 深度思考概率（默认 0.2）
- `use_dual_1_7b`: 使用双 1.7B 模型测试（默认 true）

## 系统要求

- Python 3.8+
- CUDA GPU（推荐 4GB+ 显存）
- 已安装的虚拟环境（.venv）

## 目录结构

```
finalbuild/
├── server/          # 服务器端文件
│   ├── optimized_cozy_bar_server.py
│   ├── memory_integration.py
│   └── cognitive_config.json
├── client/          # 客户端文件
│   └── llm_client_cognitive.py
├── tools/           # 工具脚本
│   └── view_memories.py
└── npc_memories/    # NPC 记忆存储
    └── Bob.json
```