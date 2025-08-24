# 🚀 Quick Start Guide

## 📋 Project Current Status

✅ **AI Service Backend** - Complete functionality, supports local LLM  
✅ **Godot Project** - Complete scene files and scripts  
✅ **Demo System** - Cozy Bar demo + AI Bridge  
✅ **Performance Optimization** - Memory, network, cache optimization  
✅ **Complete Documentation** - Installation, configuration, troubleshooting  

## 🎮 Startup Methods

### Method 1: One-Click Launch (Recommended)
```bash
# Start AI service + automatically open Godot
python demo_launcher.py

# If Godot is not installed or not in PATH:
python demo_launcher.py --no-godot
# Then manually open Godot project: godot/live-with-ai/project.godot
```

### Method 2: Run AI Service Independently
```bash
# Only start AI service (for testing)
python demo_launcher.py --no-godot --port 8080

# Test AI service
curl http://127.0.0.1:8080/health
```

### Method 3: Run Cozy Bar Text Demo
```bash
cd cozy_bar_demo
python main.py
```

## 🔧 Preparation Before First Run

### 1. Check Dependencies
```bash
pip install -r requirements.txt
```

### 2. Verify Model Files
Ensure the following directory contains model files:
- `models/gpt4all/Qwen3-30B-A3B-Instruct-2507-UD-Q4_K_XL.gguf`

### 3. Install Godot (if not already installed)
- Download: https://godotengine.org/download
- Recommended version: Godot 4.2+
- Extract to `C:\Godot\` and add to PATH

## 🎯 Expected Behavior

### After AI Service Starts, You Should See:
```
[OK] All dependencies satisfied!
[SUCCESS] AI Bridge Server started successfully!
✓ AI service ready: Hello! How can I assist you today?
Godot AI Bridge started on http://127.0.0.1:8080
```

### After Godot Project Runs, You Should See:
- 3 AI characters automatically generated (Alice, Bob, Charlie)
- Character name labels displayed above heads
- Console shows AI service connection successful
- Characters will move autonomously and make decisions

### Cozy Bar Demo Should Display:
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

## 🐛 Common Problem Solutions

### Git Line Ending Warnings
Fixed: Created `.gitattributes` file to handle line ending conversion.

### AI Service Startup Failure
```bash
# Check if port is occupied
netstat -ano | findstr :8080
```