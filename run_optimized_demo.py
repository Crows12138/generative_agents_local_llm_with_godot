#!/usr/bin/env python3
"""
Optimized Demo Launcher - One-click optimized demo startup
"""

import os
import sys
import time
import argparse
import subprocess
from pathlib import Path

# Add project path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

try:
    from demo_performance_suite import DemoPerformanceSuite, DemoConfig
    from performance_optimizer import optimize_for_demo
    from memory_optimizer import get_memory_optimizer
    from network_optimizer import get_network_optimizer
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Please ensure all optimization modules are available")
    sys.exit(1)


def check_system_requirements():
    """Check system requirements"""
    print("🔍 Checking system requirements...")
    
    import psutil
    
    # Check memory
    memory = psutil.virtual_memory()
    memory_gb = memory.total / (1024**3)
    
    if memory_gb < 4:
        print(f"⚠️ Warning: Low system memory ({memory_gb:.1f}GB). Recommended: 8GB+")
    else:
        print(f"✓ System memory: {memory_gb:.1f}GB")
    
    # Check CPU
    cpu_count = psutil.cpu_count()
    print(f"✓ CPU cores: {cpu_count}")
    
    # Check disk space
    disk = psutil.disk_usage('.')
    disk_gb = disk.free / (1024**3)
    
    if disk_gb < 2:
        print(f"⚠️ Warning: Low disk space ({disk_gb:.1f}GB free)")
    else:
        print(f"✓ Disk space: {disk_gb:.1f}GB free")
    
    return memory_gb >= 2 and disk_gb >= 1  # Minimum requirements


def setup_environment():
    """Setup environment"""
    print("🔧 Setting up environment...")
    
    # Set environment variables
    os.environ['PYTHONPATH'] = str(project_root)
    
    # Check Python version
    python_version = sys.version_info
    if python_version.major < 3 or python_version.minor < 8:
        print(f"⚠️ Warning: Python {python_version.major}.{python_version.minor} detected. Recommended: Python 3.8+")
    else:
        print(f"✓ Python {python_version.major}.{python_version.minor}")
    
    # Create necessary directories
    logs_dir = project_root / "logs"
    logs_dir.mkdir(exist_ok=True)
    
    reports_dir = project_root / "reports"
    reports_dir.mkdir(exist_ok=True)
    
    print("✓ Environment setup complete")


def create_demo_config(performance_mode: str) -> DemoConfig:
    """Create demo configuration"""
    if performance_mode == "low":
        return DemoConfig(
            max_ai_response_time=3.0,
            max_memory_usage_gb=6.0,
            min_fps=20.0,
            max_agents=2,
            enable_caching=True,
            enable_memory_optimization=True,
            enable_network_optimization=False,
            enable_godot_optimization=True
        )
    elif performance_mode == "medium":
        return DemoConfig(
            max_ai_response_time=2.0,
            max_memory_usage_gb=4.0,
            min_fps=30.0,
            max_agents=3,
            enable_caching=True,
            enable_memory_optimization=True,
            enable_network_optimization=True,
            enable_godot_optimization=True
        )
    elif performance_mode == "high":
        return DemoConfig(
            max_ai_response_time=1.5,
            max_memory_usage_gb=3.0,
            min_fps=45.0,
            max_agents=5,
            enable_caching=True,
            enable_memory_optimization=True,
            enable_network_optimization=True,
            enable_godot_optimization=True
        )
    else:  # ultra
        return DemoConfig(
            max_ai_response_time=1.0,
            max_memory_usage_gb=2.5,
            min_fps=60.0,
            max_agents=3,
            enable_caching=True,
            enable_memory_optimization=True,
            enable_network_optimization=True,
            enable_godot_optimization=True
        )


def start_ai_service():
    """Start AI service"""
    print("🤖 Starting AI service...")
    
    try:
        # Check if AI service is already running
        from ai_service.ai_service import get_health_status
        status = get_health_status()
        if status and isinstance(status, dict):
            print("✓ AI service already running")
            return True
    except:
        pass
    
    # Try to start AI service
    try:
        ai_script = project_root / "ai_service" / "ai_service.py"
        if ai_script.exists():
            # Start AI service in background
            subprocess.Popen([sys.executable, str(ai_script)], 
                           cwd=str(project_root),
                           stdout=subprocess.DEVNULL,
                           stderr=subprocess.DEVNULL)
            
            # Wait for service to start
            for _ in range(10):
                time.sleep(1)
                try:
                    from ai_service.ai_service import get_health_status
                    if get_health_status():
                        print("✓ AI service started")
                        return True
                except:
                    continue
            
            print("⚠️ AI service startup timeout")
            return False
        else:
            print("⚠️ AI service script not found")
            return False
    except Exception as e:
        print(f"⚠️ Failed to start AI service: {e}")
        return False


def start_godot_game():
    """Start Godot game"""
    print("🎮 Starting Godot game...")
    
    godot_project = project_root / "godot" / "live-with-ai" / "project.godot"
    
    if not godot_project.exists():
        print("⚠️ Godot project not found")
        return False
    
    # Find Godot executable
    godot_paths = [
        "godot",
        "godot.exe",
        "/usr/bin/godot",
        "/usr/local/bin/godot",
        "C:\\Program Files\\Godot\\godot.exe",
        "C:\\Godot\\godot.exe"
    ]
    
    godot_exe = None
    for path in godot_paths:
        try:
            result = subprocess.run([path, "--version"], 
                                  capture_output=True, 
                                  timeout=5)
            if result.returncode == 0:
                godot_exe = path
                break
        except:
            continue
    
    if not godot_exe:
        print("⚠️ Godot not found in PATH. Please install Godot or add it to PATH")
        print("   Alternative: Manually open the Godot project and run it")
        return False
    
    try:
        # Start Godot project
        subprocess.Popen([godot_exe, "--path", str(godot_project.parent)],
                        cwd=str(godot_project.parent))
        print("✓ Godot game started")
        return True
    except Exception as e:
        print(f"⚠️ Failed to start Godot: {e}")
        return False


def run_demo(args):
    """Run optimized demo"""
    print("\n" + "="*60)
    print("🚀 STARTING OPTIMIZED DEMO")
    print("="*60)
    
    # 1. Check system requirements
    if not check_system_requirements():
        print("❌ System requirements not met")
        return False
    
    # 2. Setup environment
    setup_environment()
    
    # 3. Create configuration
    config = create_demo_config(args.performance)
    print(f"🎯 Performance mode: {args.performance.upper()}")
    
    # 4. Start AI service (if needed)
    if args.start_ai:
        ai_started = start_ai_service()
        if not ai_started and not args.ignore_ai_failure:
            print("❌ AI service failed to start")
            return False
    
    # 5. Start Godot game (if needed)
    if args.start_godot:
        godot_started = start_godot_game()
        if not godot_started:
            print("⚠️ Godot failed to start, continuing with backend only")
    
    # 6. Initialize performance suite
    print("\n🔧 Initializing performance optimization...")
    suite = DemoPerformanceSuite(config)
    
    try:
        if args.test_mode:
            # Test mode: run for specified time then exit
            print(f"\n🧪 Running test mode for {args.duration} minutes...")
            suite.run_demo_test(duration_minutes=args.duration)
        else:
            # Continuous mode: start optimization and keep running
            print("\n🎬 Starting continuous optimization mode...")
            suite.start_demo_optimization()
            
            print("\n✅ Demo optimization is now running!")
            print("Press Ctrl+C to stop...")
            
            try:
                while True:
                    time.sleep(1)
            except KeyboardInterrupt:
                print("\n⚠️ Stopping demo...")
                suite.stop_demo_optimization()
    
    except Exception as e:
        print(f"\n❌ Demo error: {e}")
        suite.stop_demo_optimization()
        return False
    
    print("\n✅ Demo completed successfully!")
    return True


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Optimized Demo Launcher")
    
    parser.add_argument("--performance", choices=["low", "medium", "high", "ultra"], 
                       default="medium", help="Performance optimization level")
    parser.add_argument("--test-mode", action="store_true", 
                       help="Run in test mode (exits after duration)")
    parser.add_argument("--duration", type=int, default=5, 
                       help="Test duration in minutes (test mode only)")
    parser.add_argument("--start-ai", action="store_true", 
                       help="Automatically start AI service")
    parser.add_argument("--start-godot", action="store_true", 
                       help="Automatically start Godot game")
    parser.add_argument("--ignore-ai-failure", action="store_true", 
                       help="Continue even if AI service fails to start")
    parser.add_argument("--no-optimization", action="store_true", 
                       help="Skip performance optimization (for debugging)")
    
    args = parser.parse_args()
    
    if args.no_optimization:
        print("⚠️ Running without optimization")
        # Start services directly without optimization
        if args.start_ai:
            start_ai_service()
        if args.start_godot:
            start_godot_game()
        print("Services started without optimization")
        return
    
    # Run optimized demo
    success = run_demo(args)
    
    if success:
        print("\n🎉 Demo launcher completed successfully!")
        sys.exit(0)
    else:
        print("\n❌ Demo launcher failed!")
        sys.exit(1)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n⚠️ Demo launcher interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n💥 Demo launcher crashed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)