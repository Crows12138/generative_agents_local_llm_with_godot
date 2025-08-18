#!/usr/bin/env python3
"""
Cozy Bar Demo - 最小可运行的AI代理酒吧场景
一个温馨的酒吧环境，具有智能NPC和互动功能
"""

import sys
import os

# 添加当前目录到Python路径
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)
sys.path.insert(0, os.path.join(current_dir, 'core'))

def check_dependencies():
    """检查依赖是否安装"""
    try:
        import colorama
        print("✓ All dependencies are installed")
        return True
    except ImportError as e:
        print(f"❌ Missing dependency: {e}")
        print("Please install dependencies using:")
        print("  pip install -r requirements.txt")
        return False

def main():
    """主函数"""
    print("=" * 50)
    print("🍻 Welcome to Cozy Bar Demo 🍻")
    print("=" * 50)
    print()
    print("A minimal AI agent simulation in a cozy bar setting")
    print("Features:")
    print("• Interactive NPCs with personalities and memories")
    print("• Dynamic conversations and behaviors")
    print("• Real-time simulation with time progression")
    print("• Text-based interface with colorful display")
    print()
    
    # 检查依赖
    if not check_dependencies():
        return 1
    
    try:
        # 导入游戏模块
        from core.bar_renderer import InteractiveBarGame
        
        # 检查配置文件
        config_path = os.path.join(current_dir, "config", "room_config.json")
        if not os.path.exists(config_path):
            print(f"❌ Configuration file not found: {config_path}")
            return 1
        
        print("✓ Configuration loaded successfully")
        print()
        print("Starting the bar simulation...")
        print("Type 'help' once the game starts for available commands.")
        print()
        
        # 启动游戏
        game = InteractiveBarGame(config_path)
        game.run()
        
        return 0
        
    except KeyboardInterrupt:
        print("\n\nGoodbye! Thanks for visiting the Cozy Bar!")
        return 0
    except Exception as e:
        print(f"❌ Error starting the game: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)