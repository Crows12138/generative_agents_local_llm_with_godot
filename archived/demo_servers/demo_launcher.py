#!/usr/bin/env python
"""
Demo Launcher - One-click launcher for the complete AI-Godot demo
Starts all necessary services and opens the Godot project
"""

import os
import sys
import time
import subprocess
import signal
import json
import argparse
import platform
import shutil
import psutil
from pathlib import Path
from typing import List, Optional, Dict, Any
import threading
import requests
from datetime import datetime

# ANSI color codes for terminal output
class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

class DemoLauncher:
    """Main demo launcher class"""
    
    def __init__(self, config_file: str = None, performance_mode: str = "medium"):
        self.project_root = Path(__file__).resolve().parent
        self.config_file = config_file or str(self.project_root / "config" / "ai_service.yaml")
        self.performance_mode = performance_mode
        self.processes = []
        self.services_status = {
            "ai_bridge": False,
            "godot": False
        }
        self.log_file = self.project_root / "demo_launcher.log"
        
    def print_header(self):
        """Print welcome header"""
        print(f"{Colors.HEADER}{Colors.BOLD}")
        print("=" * 60)
        print("   AI-GODOT DEMO LAUNCHER")
        print("   Generative Agents with Local LLM")
        print(f"   Performance Mode: {self.performance_mode.upper()}")
        print("=" * 60)
        print(f"{Colors.ENDC}")
        
    def check_system_requirements(self) -> bool:
        """Check system requirements for optimal performance"""
        try:
            print(f"{Colors.CYAN}🔍 Checking system requirements...{Colors.ENDC}")
        except UnicodeEncodeError:
            print(f"{Colors.CYAN}[*] Checking system requirements...{Colors.ENDC}")
        
        # Performance mode requirements
        mode_requirements = {
            "low": {"memory": 4, "disk": 1, "recommended_memory": 6},
            "medium": {"memory": 6, "disk": 2, "recommended_memory": 8},
            "high": {"memory": 8, "disk": 3, "recommended_memory": 12},
            "ultra": {"memory": 12, "disk": 5, "recommended_memory": 16}
        }
        
        requirements = mode_requirements.get(self.performance_mode, mode_requirements["medium"])
        
        # Check memory
        memory = psutil.virtual_memory()
        memory_gb = memory.total / (1024**3)
        print(f"  RAM: {memory_gb:.1f}GB total, {memory.percent}% used")
        
        if memory_gb < requirements["memory"]:
            try:
                print(f"  {Colors.WARNING}[WARNING] {memory_gb:.1f}GB RAM detected. Required: {requirements['memory']}GB+ for {self.performance_mode} mode{Colors.ENDC}")
            except UnicodeEncodeError:
                print(f"  {Colors.WARNING}[!] Warning: {memory_gb:.1f}GB RAM detected. Required: {requirements['memory']}GB+ for {self.performance_mode} mode{Colors.ENDC}")
        elif memory_gb < requirements["recommended_memory"]:
            try:
                print(f"  {Colors.WARNING}[WARNING] Note: {memory_gb:.1f}GB RAM detected. Recommended: {requirements['recommended_memory']}GB+ for optimal {self.performance_mode} mode{Colors.ENDC}")
            except UnicodeEncodeError:
                print(f"  {Colors.WARNING}[!] Note: {memory_gb:.1f}GB RAM detected. Recommended: {requirements['recommended_memory']}GB+ for optimal {self.performance_mode} mode{Colors.ENDC}")
        else:
            try:
                print(f"  [OK] Sufficient RAM for {self.performance_mode} mode")
            except UnicodeEncodeError:
                print(f"  [OK] Sufficient RAM for {self.performance_mode} mode")
        
        # Check CPU cores
        cpu_count = psutil.cpu_count()
        print(f"  CPU cores: {cpu_count}")
        if cpu_count < 4:
            try:
                print(f"  {Colors.WARNING}[WARNING] Warning: {cpu_count} CPU cores detected. Recommended: 4+ cores{Colors.ENDC}")
            except UnicodeEncodeError:
                print(f"  {Colors.WARNING}[!] Warning: {cpu_count} CPU cores detected. Recommended: 4+ cores{Colors.ENDC}")
        else:
            try:
                print(f"  [OK] Sufficient CPU cores")
            except UnicodeEncodeError:
                print(f"  [OK] Sufficient CPU cores")
        
        # Check disk space
        disk = psutil.disk_usage('.')
        disk_gb = disk.free / (1024**3)
        print(f"  Disk: {disk_gb:.1f}GB free")
        
        if disk_gb < requirements["disk"]:
            try:
                print(f"  {Colors.WARNING}[WARNING] Warning: Low disk space ({disk_gb:.1f}GB free). Required: {requirements['disk']}GB+ for {self.performance_mode} mode{Colors.ENDC}")
            except UnicodeEncodeError:
                print(f"  {Colors.WARNING}[!] Warning: Low disk space ({disk_gb:.1f}GB free). Required: {requirements['disk']}GB+ for {self.performance_mode} mode{Colors.ENDC}")
        else:
            try:
                print(f"  [OK] Sufficient disk space")
            except UnicodeEncodeError:
                print(f"  [OK] Sufficient disk space")
        
        # Check Python version
        python_version = sys.version_info
        if python_version.major < 3 or python_version.minor < 8:
            try:
                print(f"  {Colors.WARNING}[WARNING] Warning: Python {python_version.major}.{python_version.minor} detected. Recommended: Python 3.8+{Colors.ENDC}")
            except UnicodeEncodeError:
                print(f"  {Colors.WARNING}[!] Warning: Python {python_version.major}.{python_version.minor} detected. Recommended: Python 3.8+{Colors.ENDC}")
        else:
            try:
                print(f"  [OK] Python {python_version.major}.{python_version.minor}")
            except UnicodeEncodeError:
                print(f"  [OK] Python {python_version.major}.{python_version.minor}")
        
        print()
        return memory_gb >= requirements["memory"] and disk_gb >= requirements["disk"]
    
    def check_dependencies(self) -> bool:
        """Check if all dependencies are installed"""
        print(f"{Colors.CYAN}Checking dependencies...{Colors.ENDC}")
        
        missing = []
        
        # Check Python packages
        required_packages = [
            "fastapi",
            "uvicorn",
            "pydantic",
            "gpt4all",
            "requests",
            "yaml",  # PyYAML imports as 'yaml'
            "watchdog"
        ]
        
        for package in required_packages:
            try:
                __import__(package)
                print(f"  [OK] {package}")
            except ImportError:
                print(f"  [MISSING] {package}")
                missing.append(package)
        
        # Check for model files
        models_dir = self.project_root / "models" / "gpt4all"
        if not models_dir.exists():
            print(f"{Colors.WARNING}  [!] Models directory not found: {models_dir}{Colors.ENDC}")
            print(f"    Creating directory...")
            models_dir.mkdir(parents=True, exist_ok=True)
        
        gguf_files = list(models_dir.glob("*.gguf"))
        if not gguf_files:
            print(f"{Colors.WARNING}  [!] No GGUF model files found in {models_dir}{Colors.ENDC}")
            print(f"    Please download model files from:")
            print(f"    - https://huggingface.co/Qwen/Qwen3-30B-A3B-Instruct-2507-GGUF")
            print(f"    - https://gpt4all.io/models/models.json")
        else:
            print(f"  [OK] Found {len(gguf_files)} model file(s)")
            for gguf in gguf_files[:3]:  # Show first 3
                size_mb = gguf.stat().st_size / (1024 * 1024)
                print(f"    - {gguf.name} ({size_mb:.1f} MB)")
        
        # Check for Godot
        godot_project = self.project_root / "godot" / "live-with-ai" / "project.godot"
        if not godot_project.exists():
            print(f"{Colors.WARNING}  [!] Godot project not found: {godot_project}{Colors.ENDC}")
        else:
            print(f"  [OK] Godot project found")
        
        if missing:
            print(f"\n{Colors.FAIL}Missing dependencies detected!{Colors.ENDC}")
            print("Install them with:")
            print(f"  pip install {' '.join(missing)}")
            return False
        
        print(f"{Colors.GREEN}All dependencies satisfied!{Colors.ENDC}\n")
        return True
    
    def check_godot_executable(self) -> Optional[str]:
        """Find Godot executable"""
        # Common Godot executable names
        godot_names = ["godot", "godot.exe", "Godot.exe", "godot4", "godot4.exe"]
        
        # Check in PATH
        for name in godot_names:
            if shutil.which(name):
                return name
        
        # Check common installation directories
        system = platform.system()
        if system == "Windows":
            common_paths = [
                Path("C:/Program Files/Godot"),
                Path("C:/Program Files (x86)/Godot"),
                Path.home() / "AppData/Local/Godot",
                Path.home() / "Downloads"
            ]
        elif system == "Darwin":  # macOS
            common_paths = [
                Path("/Applications/Godot.app/Contents/MacOS/Godot"),
                Path.home() / "Applications/Godot.app/Contents/MacOS/Godot"
            ]
        else:  # Linux
            common_paths = [
                Path("/usr/bin/godot"),
                Path("/usr/local/bin/godot"),
                Path.home() / ".local/bin/godot",
                Path("/opt/godot/godot")
            ]
        
        for path in common_paths:
            if path.exists():
                for name in godot_names:
                    exe = path / name if path.is_dir() else path
                    if exe.exists():
                        return str(exe)
        
        return None
    
    def start_ai_bridge(self, host: str = "127.0.0.1", port: int = 8080) -> subprocess.Popen:
        """Start the AI bridge server"""
        print(f"{Colors.CYAN}Starting AI Bridge Server...{Colors.ENDC}")
        
        cmd = [
            sys.executable,
            "-m", "api.godot_bridge",
            "--host", host,
            "--port", str(port)
        ]
        
        log_file = open(self.project_root / "ai_bridge.log", "w")
        process = subprocess.Popen(
            cmd,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            cwd=str(self.project_root)
        )
        
        # Wait for server to start
        print(f"  Waiting for server to start on http://{host}:{port}...")
        for i in range(30):  # 30 second timeout
            try:
                response = requests.get(f"http://{host}:{port}/health")
                if response.status_code == 200:
                    print(f"{Colors.GREEN}  [SUCCESS] AI Bridge Server started successfully!{Colors.ENDC}")
                    self.services_status["ai_bridge"] = True
                    return process
            except:
                pass
            time.sleep(1)
            if i % 5 == 0:
                print(f"  Still waiting... ({i}s)")
        
        print(f"{Colors.FAIL}  [ERROR] Failed to start AI Bridge Server{Colors.ENDC}")
        process.terminate()
        return None
    
    def start_godot(self, project_path: str = None) -> Optional[subprocess.Popen]:
        """Start Godot with the project"""
        print(f"{Colors.CYAN}Starting Godot...{Colors.ENDC}")
        
        if project_path is None:
            project_path = str(self.project_root / "godot" / "live-with-ai" / "project.godot")
        
        if not Path(project_path).exists():
            print(f"{Colors.FAIL}  [ERROR] Godot project not found: {project_path}{Colors.ENDC}")
            return None
        
        # Find Godot executable
        godot_exe = self.check_godot_executable()
        if not godot_exe:
            print(f"{Colors.WARNING}  [!] Godot executable not found!{Colors.ENDC}")
            print("  Please install Godot 4.x from: https://godotengine.org/download")
            print("  Or specify the path to Godot executable")
            return None
        
        print(f"  Using Godot: {godot_exe}")
        
        # Start Godot with the project
        cmd = [godot_exe, "--editor", project_path]
        
        try:
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            print(f"{Colors.GREEN}  [SUCCESS] Godot started successfully!{Colors.ENDC}")
            self.services_status["godot"] = True
            return process
        except Exception as e:
            print(f"{Colors.FAIL}  [ERROR] Failed to start Godot: {e}{Colors.ENDC}")
            return None
    
    def monitor_services(self):
        """Monitor running services in a separate thread"""
        def monitor():
            while any(self.services_status.values()):
                time.sleep(5)
                # Check if services are still running
                for process in self.processes:
                    if process and process.poll() is not None:
                        print(f"{Colors.WARNING}A service has stopped unexpectedly{Colors.ENDC}")
        
        monitor_thread = threading.Thread(target=monitor, daemon=True)
        monitor_thread.start()
    
    def show_status(self):
        """Show status of all services"""
        print(f"\n{Colors.CYAN}Service Status:{Colors.ENDC}")
        print("=" * 40)
        
        status_symbol = lambda x: f"{Colors.GREEN}[RUNNING]{Colors.ENDC}" if x else f"{Colors.FAIL}[STOPPED]{Colors.ENDC}"
        
        print(f"AI Bridge Server: {status_symbol(self.services_status['ai_bridge'])}")
        if self.services_status['ai_bridge']:
            print(f"  URL: http://127.0.0.1:8080")
            print(f"  API Docs: http://127.0.0.1:8080/docs")
        
        print(f"Godot Editor: {status_symbol(self.services_status['godot'])}")
        
        print("=" * 40)
    
    def show_instructions(self):
        """Show usage instructions"""
        print(f"\n{Colors.CYAN}Instructions:{Colors.ENDC}")
        print("1. The AI Bridge Server is now running")
        print("2. Godot Editor should open with the project")
        print("3. In Godot, press F6 to run the current scene")
        print("4. Or press F5 to run the main scene")
        print("\nAPI Endpoints available:")
        print("  - POST http://127.0.0.1:8080/ai/chat")
        print("  - POST http://127.0.0.1:8080/ai/decide")
        print("  - POST http://127.0.0.1:8080/ai/think")
        print("  - GET  http://127.0.0.1:8080/ai/status")
        print("\nPress Ctrl+C to stop all services")
    
    def cleanup(self):
        """Clean up all processes"""
        print(f"\n{Colors.CYAN}Shutting down services...{Colors.ENDC}")
        
        for process in self.processes:
            if process and process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
        
        print(f"{Colors.GREEN}All services stopped.{Colors.ENDC}")
    
    def run(self, args):
        """Main run method"""
        self.print_header()
        
        # Check system requirements
        if not self.check_system_requirements():
            if not args.force:
                print(f"\n{Colors.FAIL}System requirements not met for {self.performance_mode} mode.{Colors.ENDC}")
                print("Use --force to continue anyway, or try a lower performance mode")
                return 1
            else:
                print(f"{Colors.WARNING}Continuing with insufficient system resources...{Colors.ENDC}")
        
        # Check dependencies
        if not self.check_dependencies():
            if not args.force:
                print(f"\n{Colors.FAIL}Aborting due to missing dependencies.{Colors.ENDC}")
                print("Use --force to continue anyway")
                return 1
        
        # Start AI Bridge
        ai_process = self.start_ai_bridge(host=args.host, port=args.port)
        if ai_process:
            self.processes.append(ai_process)
        elif not args.force:
            print(f"\n{Colors.FAIL}Failed to start AI Bridge. Aborting.{Colors.ENDC}")
            return 1
        
        # Start Godot if requested
        if not args.no_godot:
            time.sleep(2)  # Give AI service time to fully initialize
            godot_process = self.start_godot()
            if godot_process:
                self.processes.append(godot_process)
        
        # Show status
        self.show_status()
        self.show_instructions()
        
        # Monitor services
        self.monitor_services()
        
        # Wait for interrupt
        try:
            print(f"\n{Colors.GREEN}Demo is running in {self.performance_mode} mode!{Colors.ENDC}")
            print("Press Ctrl+C to stop...")
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            pass
        finally:
            self.cleanup()
        
        return 0

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Launch the AI-Godot Demo System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python demo_launcher.py                         # Start with defaults (medium mode)
  python demo_launcher.py --performance high      # Start in high performance mode
  python demo_launcher.py --port 8081             # Use custom port
  python demo_launcher.py --no-godot              # Start only AI service
  python demo_launcher.py --force                 # Ignore dependency checks
        """
    )
    
    parser.add_argument("--host", default="127.0.0.1", help="AI Bridge host (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8080, help="AI Bridge port (default: 8080)")
    parser.add_argument("--no-godot", action="store_true", help="Don't start Godot editor")
    parser.add_argument("--force", action="store_true", help="Force start even with missing dependencies")
    parser.add_argument("--config", help="Path to configuration file")
    parser.add_argument("--performance", choices=["low", "medium", "high", "ultra"], 
                       default="medium", help="Performance optimization level (default: medium)")
    
    args = parser.parse_args()
    
    # Create and run launcher
    launcher = DemoLauncher(config_file=args.config, performance_mode=args.performance)
    
    # Handle signals
    def signal_handler(sig, frame):
        launcher.cleanup()
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    if platform.system() != "Windows":
        signal.signal(signal.SIGTERM, signal_handler)
    
    # Run
    sys.exit(launcher.run(args))

if __name__ == "__main__":
    main()


