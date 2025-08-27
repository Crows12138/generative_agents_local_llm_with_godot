import time
from datetime import datetime, timedelta
from typing import Optional
from dataclasses import dataclass


@dataclass
class TimeSettings:
    """Time system settings"""
    game_minutes_per_real_minute: int = 60  # Game minutes per real minute
    step_interval_real_seconds: float = 1.0  # Real seconds between steps
    
    @property
    def game_seconds_per_real_second(self) -> float:
        """Game seconds per real second"""
        return self.game_minutes_per_real_minute * 60 / 60  # Simplified to game_minutes_per_real_minute
    
    @property
    def real_seconds_per_game_hour(self) -> float:
        """Real seconds needed per game hour"""
        return 3600 / self.game_seconds_per_real_second


class GameTimeManager:
    """
    Game time manager
    
    Features:
    - Manage conversion between game time and real time
    - Control stepping frequency
    - Support acceleration/deceleration of game time
    
    Default setting: 1 real minute = 1 game hour
    This means:
    - 1 game day = 24 real minutes
    - 1 game hour = 1 real minute
    - 1 game minute = 1 real second
    """
    
    def __init__(
        self,
        start_game_time: datetime,
        time_settings: Optional[TimeSettings] = None
    ):
        self.start_game_time = start_game_time
        self.start_real_time = time.time()
        
        if time_settings is None:
            time_settings = TimeSettings()
        self.time_settings = time_settings
        
        # State tracking
        self.paused = False
        self.pause_start_time = 0.0
        self.total_paused_duration = 0.0
        
        # Step control
        self.last_step_time = 0.0
        self.step_count = 0
        
        print(f"[TimeManager] Game time started at {start_game_time}")
        print(f"[TimeManager] 1 real minute = {time_settings.game_minutes_per_real_minute} game minutes")
        print(f"[TimeManager] 1 game hour = {time_settings.real_seconds_per_game_hour:.1f} real seconds")

    def get_current_game_time(self) -> datetime:
        """Get current game time"""
        if self.paused:
            effective_real_time = self.pause_start_time - self.start_real_time - self.total_paused_duration
        else:
            effective_real_time = time.time() - self.start_real_time - self.total_paused_duration
        
        game_time_elapsed = effective_real_time * self.time_settings.game_seconds_per_real_second
        return self.start_game_time + timedelta(seconds=game_time_elapsed)

    def get_formatted_game_time(self) -> str:
        """Get formatted game time string"""
        return self.get_current_game_time().strftime("%Y-%m-%d %H:%M:%S")

    def get_game_date(self) -> str:
        """Get game date string"""
        return self.get_current_game_time().strftime("%Y-%m-%d")

    def get_game_time_str(self) -> str:
        """Get game time string (hour:minute)"""
        return self.get_current_game_time().strftime("%H:%M")

    def should_step(self) -> bool:
        """Check if next step should be executed"""
        if self.paused:
            return False
        
        current_time = time.time()
        if current_time - self.last_step_time >= self.time_settings.step_interval_real_seconds:
            self.last_step_time = current_time
            self.step_count += 1
            return True
        return False

    def pause(self):
        """Pause game time"""
        if not self.paused:
            self.paused = True
            self.pause_start_time = time.time()
            print("[TimeManager] Game time paused")

    def resume(self):
        """Resume game time"""
        if self.paused:
            self.total_paused_duration += time.time() - self.pause_start_time
            self.paused = False
            print("[TimeManager] Game time resumed")

    def set_speed(self, game_minutes_per_real_minute: int):
        """Set game time speed"""
        self.time_settings.game_minutes_per_real_minute = game_minutes_per_real_minute
        print(f"[TimeManager] Time speed changed to {game_minutes_per_real_minute} game minutes per real minute")

    def get_elapsed_game_hours(self) -> float:
        """Get game hours elapsed since start"""
        current_time = self.get_current_game_time()
        elapsed = current_time - self.start_game_time
        return elapsed.total_seconds() / 3600

    def get_elapsed_game_days(self) -> float:
        """Get game days elapsed since start"""
        return self.get_elapsed_game_hours() / 24

    def is_new_game_hour(self) -> bool:
        """Check if entered a new game hour"""
        current_game_time = self.get_current_game_time()
        return current_game_time.minute == 0 and current_game_time.second < 10

    def is_new_game_day(self) -> bool:
        """Check if entered a new game day"""
        current_game_time = self.get_current_game_time()
        return (current_game_time.hour == 0 and 
                current_game_time.minute == 0 and 
                current_game_time.second < 10)

    def get_stats(self) -> dict:
        """Get time manager statistics"""
        current_game_time = self.get_current_game_time()
        elapsed_real_time = time.time() - self.start_real_time - self.total_paused_duration
        
        return {
            "current_game_time": self.get_formatted_game_time(),
            "elapsed_real_seconds": elapsed_real_time,
            "elapsed_game_hours": self.get_elapsed_game_hours(),
            "elapsed_game_days": self.get_elapsed_game_days(),
            "step_count": self.step_count,
            "game_minutes_per_real_minute": self.time_settings.game_minutes_per_real_minute,
            "paused": self.paused,
            "total_paused_duration": self.total_paused_duration,
        }

    def wait_for_next_step(self):
        """Wait until next step time"""
        current_time = time.time()
        next_step_time = self.last_step_time + self.time_settings.step_interval_real_seconds
        
        if current_time < next_step_time:
            sleep_time = next_step_time - current_time
            time.sleep(sleep_time)

    def __str__(self) -> str:
        return f"GameTime: {self.get_formatted_game_time()} (Step: {self.step_count})"


# Global time manager instance
_global_time_manager: Optional[GameTimeManager] = None

def initialize_time_manager(
    start_game_time: datetime,
    time_settings: Optional[TimeSettings] = None
) -> GameTimeManager:
    """Initialize global time manager"""
    global _global_time_manager
    _global_time_manager = GameTimeManager(start_game_time, time_settings)
    return _global_time_manager

def get_time_manager() -> GameTimeManager:
    """Get global time manager"""
    if _global_time_manager is None:
        raise RuntimeError("Time manager not initialized. Call initialize_time_manager() first.")
    return _global_time_manager

def get_current_game_time() -> datetime:
    """Convenience function to get current game time"""
    return get_time_manager().get_current_game_time()

def get_game_date() -> str:
    """Convenience function to get game date"""
    return get_time_manager().get_game_date()

def get_game_time_str() -> str:
    """Convenience function to get game time string"""
    return get_time_manager().get_game_time_str()

# Preset time settings
class TimePresets:
    """Preset time settings"""
    
    # Very fast: 1 real minute = 1 game day
    VERY_FAST = TimeSettings(game_minutes_per_real_minute=1440)
    
    # Fast: 1 real minute = 1 game hour (default)
    FAST = TimeSettings(game_minutes_per_real_minute=60)
    
    # Medium: 1 real minute = 30 game minutes
    MEDIUM = TimeSettings(game_minutes_per_real_minute=30)
    
    # Slow: 1 real minute = 10 game minutes
    SLOW = TimeSettings(game_minutes_per_real_minute=10)
    
    # Real-time: 1 real minute = 1 game minute
    REAL_TIME = TimeSettings(game_minutes_per_real_minute=1)