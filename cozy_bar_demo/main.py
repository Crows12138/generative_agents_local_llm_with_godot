#!/usr/bin/env python3
"""
Cozy Bar Demo - Minimal runnable AI agent bar scene
A cozy bar environment with intelligent NPCs and interactive features
"""

import sys
import os

# Add current directory to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)
sys.path.insert(0, os.path.join(current_dir, 'core'))

def check_dependencies():
    """Check if dependencies are installed"""
    try:
        import colorama
        print("[OK] All dependencies are installed")
        return True
    except ImportError as e:
        print(f"[ERROR] Missing dependency: {e}")
        print("Please install dependencies using:")
        print("  pip install -r requirements.txt")
        return False

def main():
    """Main function"""
    print("=" * 50)
    print("[COZY BAR] Welcome to Cozy Bar Demo")
    print("=" * 50)
    print()
    print("A minimal AI agent simulation in a cozy bar setting")
    print("Features:")
    print("- Interactive NPCs with personalities and memories")
    print("- Dynamic conversations and behaviors")
    print("- Real-time simulation with time progression")
    print("- Text-based interface with colorful display")
    print()
    
    # Check dependencies
    if not check_dependencies():
        return 1
    
    try:
        # Import game modules
        from core.bar_renderer import InteractiveBarGame
        
        # Check configuration file
        config_path = os.path.join(current_dir, "config", "room_config.json")
        if not os.path.exists(config_path):
            print(f"[ERROR] Configuration file not found: {config_path}")
            return 1
        
        print("[OK] Configuration loaded successfully")
        print()
        print("Starting the bar simulation...")
        print("Type 'help' once the game starts for available commands.")
        print()
        
        # Start the game
        game = InteractiveBarGame(config_path)
        game.run()
        
        return 0
        
    except KeyboardInterrupt:
        print("\n\nGoodbye! Thanks for visiting the Cozy Bar!")
        return 0
    except Exception as e:
        print(f"[ERROR] Error starting the game: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)