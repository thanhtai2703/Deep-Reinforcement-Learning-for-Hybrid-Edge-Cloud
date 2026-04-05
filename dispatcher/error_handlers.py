"""
error_handlers.py
=================
Retry logic, fallback strategies, và circuit breaker cho Dispatcher.

Patterns:
  - with_retry: decorator retry với exponential backoff
  - CircuitBreaker: ngắt mạch khi service liên tục fail
  - FallbackChain: thử policy A → B → C nếu fail

Usage:
    @with_retry(max_retries=3, backoff_factor=1.0)
    def call_k8s_api(task_id, node):
        ...

    breaker = CircuitBreaker(failure_threshold=5)
    if breaker.allow_request():
        try:
            result = call_k8s_api(...)
            breaker.record_success()
        except Exception:
            breaker.record_failure()
"""

from __future__ import annotations

import functools
import logging
import time
from enum import Enum
from typing import Callable, Optional, Type, Tuple

logger = logging.getLogger("ErrorHandlers")


# ---------------------------------------------------------------------------
# Retry decorator
# ---------------------------------------------------------------------------

def with_retry(
    max_retries: int = 3,
    backoff_factor: float = 1.0,
    max_backoff: float = 30.0,
    retry_on: Tuple[Type[Exception], ...] = (Exception,),
):
    """
    Decorator: retry function call với exponential backoff.

    Parameters
    ----------
    max_retries    : số lần retry tối đa
    backoff_factor : hệ số backoff (giây). Delay = backoff_factor * 2^attempt
    max_backoff    : delay tối đa (giây)
    retry_on       : tuple exception types để retry
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None
            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except retry_on as e:
                    last_exception = e
                    if attempt < max_retries:
                        delay = min(backoff_factor * (2 ** attempt), max_backoff)
                        logger.warning(
                            "%s failed (attempt %d/%d): %s. Retrying in %.1fs",
                            func.__name__, attempt + 1, max_retries, e, delay,
                        )
                        time.sleep(delay)
                    else:
                        logger.error(
                            "%s failed after %d attempts: %s",
                            func.__name__, max_retries + 1, e,
                        )
            raise last_exception
        return wrapper
    return decorator


# ---------------------------------------------------------------------------
# Circuit Breaker
# ---------------------------------------------------------------------------

class CircuitState(Enum):
    CLOSED = "closed"      # Hoạt động bình thường
    OPEN = "open"          # Ngắt mạch - từ chối request
    HALF_OPEN = "half_open"  # Thử lại 1 request


class CircuitBreaker:
    """
    Circuit Breaker pattern: ngắt mạch khi service liên tục fail.

    State transitions:
      CLOSED → (failures >= threshold) → OPEN
      OPEN → (timeout elapsed) → HALF_OPEN
      HALF_OPEN → (success) → CLOSED
      HALF_OPEN → (failure) → OPEN

    Parameters
    ----------
    failure_threshold : số failures liên tiếp trước khi mở mạch
    reset_timeout     : giây chờ trước khi thử lại (HALF_OPEN)
    """

    def __init__(self, failure_threshold: int = 5, reset_timeout: float = 30.0):
        self.failure_threshold = failure_threshold
        self.reset_timeout = reset_timeout
        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._last_failure_time = 0.0
        self._success_count = 0

    @property
    def state(self) -> CircuitState:
        if self._state == CircuitState.OPEN:
            if time.time() - self._last_failure_time >= self.reset_timeout:
                self._state = CircuitState.HALF_OPEN
                logger.info("Circuit breaker → HALF_OPEN (trying one request)")
        return self._state

    def allow_request(self) -> bool:
        """Kiểm tra có cho phép request không."""
        current = self.state
        if current == CircuitState.CLOSED:
            return True
        if current == CircuitState.HALF_OPEN:
            return True  # Cho thử 1 request
        return False  # OPEN → block

    def record_success(self):
        """Ghi nhận request thành công."""
        self._failure_count = 0
        self._success_count += 1
        if self._state == CircuitState.HALF_OPEN:
            self._state = CircuitState.CLOSED
            logger.info("Circuit breaker → CLOSED (service recovered)")

    def record_failure(self):
        """Ghi nhận request thất bại."""
        self._failure_count += 1
        self._last_failure_time = time.time()
        self._success_count = 0

        if self._failure_count >= self.failure_threshold:
            self._state = CircuitState.OPEN
            logger.warning(
                "Circuit breaker → OPEN (%d consecutive failures)",
                self._failure_count,
            )

    def reset(self):
        """Reset circuit breaker về trạng thái ban đầu."""
        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._success_count = 0

    def get_status(self) -> dict:
        return {
            "state": self.state.value,
            "failure_count": self._failure_count,
            "success_count": self._success_count,
        }


# ---------------------------------------------------------------------------
# Fallback chain
# ---------------------------------------------------------------------------

class FallbackChain:
    """
    Thử lần lượt các strategy, dùng strategy đầu tiên thành công.

    Usage:
        chain = FallbackChain()
        chain.add("rl_model", lambda state: model.predict(state))
        chain.add("least_loaded", lambda state: baseline.select_action(state))
        chain.add("round_robin", lambda state: rr.select_action(state))
        action = chain.execute(state)
    """

    def __init__(self):
        self._strategies: list = []

    def add(self, name: str, func: Callable, catch: Tuple[Type[Exception], ...] = (Exception,)):
        """Thêm một fallback strategy."""
        self._strategies.append((name, func, catch))

    def execute(self, *args, **kwargs):
        """Thực thi strategy chain, trả về kết quả đầu tiên thành công."""
        last_error = None
        for name, func, catch in self._strategies:
            try:
                result = func(*args, **kwargs)
                logger.debug("FallbackChain: %s succeeded", name)
                return result
            except catch as e:
                logger.warning("FallbackChain: %s failed: %s", name, e)
                last_error = e
                continue

        raise RuntimeError(
            f"All {len(self._strategies)} fallback strategies failed. "
            f"Last error: {last_error}"
        )
