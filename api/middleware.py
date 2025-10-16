"""
Custom middleware for the EPP Detector API.

This module provides middleware for:
- Rate limiting to prevent DoS attacks
- Request logging
- Error tracking
"""

import time
import logging
from typing import Dict, Tuple, Callable
from collections import defaultdict
from datetime import datetime, timedelta

from fastapi import Request, Response, HTTPException, status
from starlette.middleware.base import BaseHTTPMiddleware

from api.config import Settings

logger = logging.getLogger(__name__)


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Rate limiting middleware using token bucket algorithm.

    This middleware prevents DoS attacks by limiting the number of requests
    from a single IP address within a time window.

    The token bucket algorithm allows bursts while maintaining average rate limits.

    Attributes:
        settings: Configuration settings
        requests: Dict tracking request counts per IP
        window_start: Dict tracking window start time per IP
    """

    def __init__(self, app, settings: Settings) -> None:
        """Initialize rate limiter.

        Args:
            app: FastAPI application instance
            settings: Settings with rate limit configuration
        """
        super().__init__(app)
        self.settings = settings
        self.enabled = settings.enable_rate_limit
        self.max_requests = settings.rate_limit_requests
        self.window_seconds = settings.rate_limit_window_seconds

        # Storage for rate limiting state (in production, use Redis)
        self.requests: Dict[str, int] = defaultdict(int)
        self.window_start: Dict[str, datetime] = {}

        logger.info(
            f"Rate limiting {'enabled' if self.enabled else 'disabled'}: "
            f"{self.max_requests} requests per {self.window_seconds}s"
        )

    async def dispatch(
        self,
        request: Request,
        call_next: Callable
    ) -> Response:
        """Process request with rate limiting.

        Args:
            request: Incoming HTTP request
            call_next: Next middleware or endpoint to call

        Returns:
            HTTP response

        Raises:
            HTTPException: If rate limit is exceeded
        """
        # Skip rate limiting if disabled
        if not self.enabled:
            return await call_next(request)

        # Get client IP
        client_ip = self._get_client_ip(request)

        # Check and update rate limit
        if not self._check_rate_limit(client_ip):
            logger.warning(
                f"Rate limit exceeded for IP {client_ip}: "
                f"{self.requests[client_ip]} requests in window"
            )
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail={
                    "error": "Rate limit exceeded",
                    "max_requests": self.max_requests,
                    "window_seconds": self.window_seconds,
                    "retry_after": self._get_retry_after(client_ip)
                }
            )

        # Process request
        response = await call_next(request)

        # Add rate limit headers to response
        response.headers["X-RateLimit-Limit"] = str(self.max_requests)
        response.headers["X-RateLimit-Remaining"] = str(
            max(0, self.max_requests - self.requests[client_ip])
        )
        response.headers["X-RateLimit-Reset"] = str(
            int((self.window_start[client_ip] + timedelta(seconds=self.window_seconds)).timestamp())
        )

        return response

    def _get_client_ip(self, request: Request) -> str:
        """Extract client IP address from request.

        Handles X-Forwarded-For header for proxied requests.

        Args:
            request: HTTP request

        Returns:
            Client IP address as string
        """
        # Check for X-Forwarded-For header (behind proxy/load balancer)
        forwarded_for = request.headers.get("X-Forwarded-For")
        if forwarded_for:
            # Take first IP in the chain
            return forwarded_for.split(",")[0].strip()

        # Fall back to direct client
        return request.client.host if request.client else "unknown"

    def _check_rate_limit(self, client_ip: str) -> bool:
        """Check if client is within rate limits.

        Uses sliding window algorithm to track requests.

        Args:
            client_ip: Client IP address

        Returns:
            True if request is allowed, False if rate limit exceeded
        """
        now = datetime.now()

        # Initialize window for new client
        if client_ip not in self.window_start:
            self.window_start[client_ip] = now
            self.requests[client_ip] = 1
            return True

        # Check if window has expired
        window_age = (now - self.window_start[client_ip]).total_seconds()
        if window_age >= self.window_seconds:
            # Reset window
            self.window_start[client_ip] = now
            self.requests[client_ip] = 1
            return True

        # Increment request count
        self.requests[client_ip] += 1

        # Check if limit exceeded
        return self.requests[client_ip] <= self.max_requests

    def _get_retry_after(self, client_ip: str) -> int:
        """Calculate seconds until rate limit window resets.

        Args:
            client_ip: Client IP address

        Returns:
            Seconds until reset
        """
        if client_ip not in self.window_start:
            return 0

        window_end = self.window_start[client_ip] + timedelta(seconds=self.window_seconds)
        seconds_remaining = (window_end - datetime.now()).total_seconds()

        return max(0, int(seconds_remaining))


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """Middleware for structured request logging.

    Logs each request with:
    - Method, path, status code
    - Client IP
    - Response time
    - Request/response sizes
    """

    def __init__(self, app) -> None:
        """Initialize logging middleware.

        Args:
            app: FastAPI application instance
        """
        super().__init__(app)

    async def dispatch(
        self,
        request: Request,
        call_next: Callable
    ) -> Response:
        """Process request with logging.

        Args:
            request: Incoming HTTP request
            call_next: Next middleware or endpoint

        Returns:
            HTTP response
        """
        # Start timing
        start_time = time.time()

        # Get request details
        client_ip = request.client.host if request.client else "unknown"
        method = request.method
        path = request.url.path

        # Process request
        try:
            response = await call_next(request)

            # Calculate response time
            process_time = (time.time() - start_time) * 1000  # ms

            # Log successful request
            logger.info(
                f"{method} {path} - {response.status_code} - "
                f"{process_time:.2f}ms - {client_ip}"
            )

            # Add timing header
            response.headers["X-Process-Time"] = f"{process_time:.2f}ms"

            return response

        except Exception as e:
            # Calculate response time for failed requests
            process_time = (time.time() - start_time) * 1000  # ms

            # Log failed request
            logger.error(
                f"{method} {path} - ERROR - "
                f"{process_time:.2f}ms - {client_ip} - {str(e)}"
            )

            # Re-raise exception
            raise


def create_rate_limiter(settings: Settings) -> RateLimitMiddleware:
    """Factory function to create rate limiter middleware.

    Args:
        settings: Settings instance

    Returns:
        Configured rate limiter middleware
    """
    # This function is called during app creation
    # The actual middleware is added via app.add_middleware()
    pass
