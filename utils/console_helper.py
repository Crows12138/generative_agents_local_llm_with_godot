"""
Console Helper - Better Unicode and color support for Windows
"""

import sys
import os
from typing import Optional

try:
    import colorama
    colorama.init(autoreset=True)
    COLORAMA_AVAILABLE = True
except ImportError:
    COLORAMA_AVAILABLE = False

# Enable UTF-8 output on Windows
if sys.platform == "win32":
    try:
        # Set console code page to UTF-8
        os.system("chcp 65001 > nul")
        # Force UTF-8 encoding for stdout and stderr
        if hasattr(sys.stdout, 'reconfigure'):
            sys.stdout.reconfigure(encoding='utf-8')
            sys.stderr.reconfigure(encoding='utf-8')
    except:
        pass

class Colors:
    """Safe color codes that work on all platforms"""
    if COLORAMA_AVAILABLE:
        # Use colorama colors
        RED = colorama.Fore.RED
        GREEN = colorama.Fore.GREEN
        YELLOW = colorama.Fore.YELLOW
        BLUE = colorama.Fore.BLUE
        MAGENTA = colorama.Fore.MAGENTA
        CYAN = colorama.Fore.CYAN
        WHITE = colorama.Fore.WHITE
        RESET = colorama.Style.RESET_ALL
        BRIGHT = colorama.Style.BRIGHT
        DIM = colorama.Style.DIM
    else:
        # Fallback to ANSI codes
        RED = '\033[91m'
        GREEN = '\033[92m'
        YELLOW = '\033[93m'
        BLUE = '\033[94m'
        MAGENTA = '\033[95m'
        CYAN = '\033[96m'
        WHITE = '\033[97m'
        RESET = '\033[0m'
        BRIGHT = '\033[1m'
        DIM = '\033[2m'

class Icons:
    """Safe Unicode icons with ASCII fallbacks"""
    
    def __init__(self):
        # Test Unicode support
        self.unicode_supported = self._test_unicode_support()
    
    def _test_unicode_support(self) -> bool:
        """Test if Unicode is supported"""
        try:
            # Try to encode/decode a Unicode character
            test_char = "✓"
            test_char.encode(sys.stdout.encoding or 'utf-8')
            return True
        except (UnicodeEncodeError, AttributeError):
            return False
    
    @property
    def CHECK(self) -> str:
        return "✓" if self.unicode_supported else "[OK]"
    
    @property
    def CROSS(self) -> str:
        return "✗" if self.unicode_supported else "[ERROR]"
    
    @property
    def WARNING(self) -> str:
        return "⚠" if self.unicode_supported else "[WARNING]"
    
    @property
    def INFO(self) -> str:
        return "ℹ" if self.unicode_supported else "[INFO]"
    
    @property
    def ROCKET(self) -> str:
        return "🚀" if self.unicode_supported else "[STARTING]"
    
    @property
    def GEAR(self) -> str:
        return "🔧" if self.unicode_supported else "[CONFIG]"
    
    @property
    def ROBOT(self) -> str:
        return "🤖" if self.unicode_supported else "[AI]"
    
    @property
    def GAME(self) -> str:
        return "🎮" if self.unicode_supported else "[GAME]"
    
    @property
    def CLEANUP(self) -> str:
        return "🗑" if self.unicode_supported else "[CLEANUP]"
    
    @property
    def THINKING(self) -> str:
        return "🤔" if self.unicode_supported else "[THINKING]"
    
    @property
    def LIGHTNING(self) -> str:
        return "⚡" if self.unicode_supported else "[FAST]"

# Global instances
colors = Colors()
icons = Icons()

def safe_print(text: str, color: Optional[str] = None, end: str = "\n") -> None:
    """Safely print text with optional color"""
    try:
        if color:
            output = f"{color}{text}{colors.RESET}"
        else:
            output = text
        
        print(output, end=end)
    except UnicodeEncodeError:
        # Fallback: remove non-ASCII characters
        safe_text = text.encode('ascii', errors='ignore').decode('ascii')
        if color:
            output = f"{color}{safe_text}{colors.RESET}"
        else:
            output = safe_text
        print(output, end=end)

def print_success(message: str) -> None:
    """Print success message"""
    safe_print(f"{icons.CHECK} {message}", colors.GREEN)

def print_error(message: str) -> None:
    """Print error message"""
    safe_print(f"{icons.CROSS} {message}", colors.RED)

def print_warning(message: str) -> None:
    """Print warning message"""
    safe_print(f"{icons.WARNING} {message}", colors.YELLOW)

def print_info(message: str) -> None:
    """Print info message"""
    safe_print(f"{icons.INFO} {message}", colors.CYAN)

def print_header(title: str) -> None:
    """Print section header"""
    separator = "=" * 60
    safe_print(separator, colors.CYAN)
    safe_print(f"   {title}", colors.CYAN + colors.BRIGHT)
    safe_print(separator, colors.CYAN)

def format_status(status: str, message: str) -> str:
    """Format status message"""
    status_icons = {
        "ok": icons.CHECK,
        "error": icons.CROSS,
        "warning": icons.WARNING,
        "info": icons.INFO,
        "starting": icons.ROCKET,
        "config": icons.GEAR,
        "ai": icons.ROBOT,
        "game": icons.GAME,
        "cleanup": icons.CLEANUP,
        "thinking": icons.THINKING,
        "fast": icons.LIGHTNING
    }
    
    icon = status_icons.get(status.lower(), f"[{status.upper()}]")
    return f"{icon} {message}"

# Test the console helper
if __name__ == "__main__":
    print_header("Console Helper Test")
    
    print_success("Success message test")
    print_error("Error message test")
    print_warning("Warning message test")
    print_info("Info message test")
    
    print(f"\nUnicode support: {icons.unicode_supported}")
    print(f"Colorama available: {COLORAMA_AVAILABLE}")
    
    print("\nIcon tests:")
    print(format_status("ok", "System ready"))
    print(format_status("starting", "Launching demo"))
    print(format_status("ai", "AI service active"))
    print(format_status("game", "Godot running"))