"""
Enhanced Error Handling and Retry Mechanisms
改进的错误处理和重试机制，提供更好的错误恢复和日志记录
"""

import time
import random
import traceback
from typing import Optional, Callable, Any, Union, Dict
from enum import Enum
from dataclasses import dataclass
from functools import wraps

class ErrorType(Enum):
    """错误类型枚举"""
    MODEL_LOAD_ERROR = "model_load_error"
    GENERATION_ERROR = "generation_error" 
    VALIDATION_ERROR = "validation_error"
    NETWORK_ERROR = "network_error"
    CONFIGURATION_ERROR = "configuration_error"
    RESOURCE_ERROR = "resource_error"
    UNKNOWN_ERROR = "unknown_error"

@dataclass
class RetryConfig:
    """重试配置"""
    max_attempts: int = 3
    base_delay: float = 1.0
    max_delay: float = 60.0
    backoff_factor: float = 2.0
    jitter: bool = True
    retry_on_errors: tuple = (Exception,)
    
class RetryableError(Exception):
    """可重试的错误"""
    def __init__(self, message: str, error_type: ErrorType = ErrorType.UNKNOWN_ERROR):
        super().__init__(message)
        self.error_type = error_type

class NonRetryableError(Exception):
    """不可重试的错误"""
    def __init__(self, message: str, error_type: ErrorType = ErrorType.UNKNOWN_ERROR):
        super().__init__(message)
        self.error_type = error_type

class RetryHandler:
    """重试处理器"""
    
    def __init__(self, config: RetryConfig = None):
        self.config = config or RetryConfig()
        self.attempt_count = 0
        self.last_error = None
    
    def calculate_delay(self, attempt: int) -> float:
        """计算延迟时间"""
        delay = min(
            self.config.base_delay * (self.config.backoff_factor ** attempt),
            self.config.max_delay
        )
        
        if self.config.jitter:
            delay += random.uniform(0, delay * 0.1)
        
        return delay
    
    def should_retry(self, error: Exception, attempt: int) -> bool:
        """判断是否应该重试"""
        if attempt >= self.config.max_attempts:
            return False
        
        # 不可重试的错误
        if isinstance(error, NonRetryableError):
            return False
        
        # 检查错误类型是否在重试范围内
        return isinstance(error, self.config.retry_on_errors)
    
    def __call__(self, func: Callable) -> Callable:
        """装饰器实现"""
        @wraps(func)
        def wrapper(*args, **kwargs):
            last_error = None
            
            for attempt in range(self.config.max_attempts):
                try:
                    self.attempt_count = attempt + 1
                    result = func(*args, **kwargs)
                    
                    # 成功时重置错误
                    self.last_error = None
                    return result
                    
                except Exception as e:
                    last_error = e
                    self.last_error = e
                    
                    if not self.should_retry(e, attempt):
                        break
                    
                    if attempt < self.config.max_attempts - 1:
                        delay = self.calculate_delay(attempt)
                        print(f"[RetryHandler] Attempt {attempt + 1} failed: {str(e)[:100]}...")
                        print(f"[RetryHandler] Retrying in {delay:.2f} seconds...")
                        time.sleep(delay)
            
            # 所有重试都失败了
            error_msg = f"All {self.config.max_attempts} attempts failed. Last error: {last_error}"
            raise RetryableError(error_msg) from last_error
        
        return wrapper

def handle_model_errors(func: Callable) -> Callable:
    """模型错误处理装饰器"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except FileNotFoundError as e:
            raise NonRetryableError(
                f"Model file not found: {e}",
                ErrorType.CONFIGURATION_ERROR
            ) from e
        except MemoryError as e:
            raise RetryableError(
                f"Out of memory while loading model: {e}",
                ErrorType.RESOURCE_ERROR
            ) from e
        except ImportError as e:
            raise NonRetryableError(
                f"Required dependency not found: {e}",
                ErrorType.CONFIGURATION_ERROR
            ) from e
        except Exception as e:
            # 将其他未知错误标记为可重试
            raise RetryableError(
                f"Unexpected model error: {e}",
                ErrorType.MODEL_LOAD_ERROR
            ) from e
    
    return wrapper

def handle_generation_errors(func: Callable) -> Callable:
    """生成错误处理装饰器"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except RuntimeError as e:
            error_msg = str(e).lower()
            if "cuda" in error_msg or "gpu" in error_msg:
                raise RetryableError(
                    f"GPU error, try CPU fallback: {e}",
                    ErrorType.RESOURCE_ERROR
                ) from e
            else:
                raise RetryableError(
                    f"Generation runtime error: {e}",
                    ErrorType.GENERATION_ERROR
                ) from e
        except ValueError as e:
            if "empty" in str(e).lower():
                raise RetryableError(
                    f"Empty generation result: {e}",
                    ErrorType.GENERATION_ERROR
                ) from e
            else:
                raise NonRetryableError(
                    f"Invalid generation parameters: {e}",
                    ErrorType.VALIDATION_ERROR
                ) from e
        except Exception as e:
            raise RetryableError(
                f"Unexpected generation error: {e}",
                ErrorType.GENERATION_ERROR
            ) from e
    
    return wrapper

def safe_execute(
    func: Callable,
    *args,
    retry_config: Optional[RetryConfig] = None,
    fallback_result: Any = None,
    log_errors: bool = True,
    **kwargs
) -> Any:
    """
    安全执行函数，带有错误处理和重试
    
    Args:
        func: 要执行的函数
        *args: 函数参数
        retry_config: 重试配置
        fallback_result: 失败时的默认返回值
        log_errors: 是否记录错误日志
        **kwargs: 函数关键字参数
    
    Returns:
        函数结果或fallback_result
    """
    retry_config = retry_config or RetryConfig()
    retry_handler = RetryHandler(retry_config)
    
    try:
        return retry_handler(func)(*args, **kwargs)
    except Exception as e:
        if log_errors:
            print(f"[SafeExecute] Function {func.__name__} failed after retries: {e}")
            if hasattr(e, '__cause__') and e.__cause__:
                print(f"[SafeExecute] Root cause: {e.__cause__}")
        
        return fallback_result

def create_circuit_breaker(
    failure_threshold: int = 5,
    timeout: float = 60.0,
    expected_exception: type = Exception
):
    """
    创建断路器装饰器
    
    Args:
        failure_threshold: 失败阈值
        timeout: 超时时间（秒）
        expected_exception: 预期的异常类型
    """
    class CircuitBreakerState(Enum):
        CLOSED = "closed"
        OPEN = "open"
        HALF_OPEN = "half_open"
    
    class CircuitBreaker:
        def __init__(self):
            self.failure_count = 0
            self.last_failure_time = None
            self.state = CircuitBreakerState.CLOSED
        
        def __call__(self, func):
            @wraps(func)
            def wrapper(*args, **kwargs):
                if self.state == CircuitBreakerState.OPEN:
                    if time.time() - self.last_failure_time < timeout:
                        raise NonRetryableError(
                            f"Circuit breaker is OPEN for {func.__name__}",
                            ErrorType.RESOURCE_ERROR
                        )
                    else:
                        self.state = CircuitBreakerState.HALF_OPEN
                
                try:
                    result = func(*args, **kwargs)
                    
                    # 成功时重置
                    if self.state == CircuitBreakerState.HALF_OPEN:
                        self.state = CircuitBreakerState.CLOSED
                        self.failure_count = 0
                    
                    return result
                    
                except expected_exception as e:
                    self.failure_count += 1
                    self.last_failure_time = time.time()
                    
                    if self.failure_count >= failure_threshold:
                        self.state = CircuitBreakerState.OPEN
                        print(f"[CircuitBreaker] Circuit breaker opened for {func.__name__}")
                    
                    raise
            
            return wrapper
    
    return CircuitBreaker()

# 预定义的重试配置
MODEL_RETRY_CONFIG = RetryConfig(
    max_attempts=2,
    base_delay=1.0,
    backoff_factor=1.5,
    retry_on_errors=(RetryableError, RuntimeError, MemoryError)
)

GENERATION_RETRY_CONFIG = RetryConfig(
    max_attempts=3,
    base_delay=0.5,
    backoff_factor=2.0,
    retry_on_errors=(RetryableError, RuntimeError, ValueError)
)

NETWORK_RETRY_CONFIG = RetryConfig(
    max_attempts=5,
    base_delay=1.0,
    backoff_factor=1.5,
    retry_on_errors=(RetryableError, ConnectionError, TimeoutError)
)

# 错误监控和统计
class ErrorMonitor:
    """错误监控器"""
    
    def __init__(self):
        self.error_counts: Dict[ErrorType, int] = {}
        self.last_errors: Dict[ErrorType, str] = {}
        self.start_time = time.time()
    
    def record_error(self, error: Exception, error_type: ErrorType = None):
        """记录错误"""
        if error_type is None:
            if hasattr(error, 'error_type'):
                error_type = error.error_type
            else:
                error_type = ErrorType.UNKNOWN_ERROR
        
        self.error_counts[error_type] = self.error_counts.get(error_type, 0) + 1
        self.last_errors[error_type] = str(error)
    
    def get_error_stats(self) -> Dict:
        """获取错误统计"""
        runtime = time.time() - self.start_time
        return {
            'runtime_seconds': runtime,
            'error_counts': dict(self.error_counts),
            'last_errors': dict(self.last_errors),
            'total_errors': sum(self.error_counts.values())
        }
    
    def reset_stats(self):
        """重置统计"""
        self.error_counts.clear()
        self.last_errors.clear()
        self.start_time = time.time()

# 全局错误监控器
_error_monitor = ErrorMonitor()

def get_error_monitor() -> ErrorMonitor:
    """获取全局错误监控器"""
    return _error_monitor

# 测试函数
def test_error_handling():
    """测试错误处理机制"""
    print("=== Error Handling Test ===")
    
    # 测试重试机制
    @RetryHandler(RetryConfig(max_attempts=3, base_delay=0.1))
    def flaky_function(success_on_attempt=3):
        if flaky_function.attempt < success_on_attempt:
            flaky_function.attempt += 1
            raise RetryableError(f"Attempt {flaky_function.attempt} failed")
        return f"Success on attempt {flaky_function.attempt}"
    
    flaky_function.attempt = 0
    
    try:
        result = flaky_function(success_on_attempt=2)
        print(f"Retry test result: {result}")
    except Exception as e:
        print(f"Retry test failed: {e}")
    
    # 测试错误监控
    monitor = get_error_monitor()
    monitor.record_error(RetryableError("Test error"), ErrorType.GENERATION_ERROR)
    
    stats = monitor.get_error_stats()
    print(f"Error stats: {stats}")
    
    print("=== Test completed ===")

if __name__ == "__main__":
    test_error_handling()