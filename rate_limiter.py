#!/usr/bin/env python3
"""
Rate limiting for the Hand Gesture Recognition Service
"""

import time
import asyncio
from typing import Dict, Optional, Tuple
from collections import defaultdict, deque
from dataclasses import dataclass
from fastapi import HTTPException, Request
import redis.asyncio as redis

from config import settings
from logger import get_logger

logger = get_logger(__name__)


@dataclass
class RateLimitConfig:
    """Rate limit configuration"""
    requests: int
    window: int  # in seconds
    burst: int = 0  # burst allowance


class RateLimiter:
    """Rate limiter implementation with Redis support"""
    
    def __init__(self):
        self.redis_client: Optional[redis.Redis] = None
        self.local_limits: Dict[str, deque] = defaultdict(lambda: deque())
        self.config = RateLimitConfig(
            requests=settings.rate_limit_requests,
            window=settings.rate_limit_window
        )
        
        if settings.redis_enabled:
            self._init_redis()
    
    def _init_redis(self) -> None:
        """Initialize Redis connection"""
        try:
            self.redis_client = redis.Redis(
                host=settings.redis_host,
                port=settings.redis_port,
                db=settings.redis_db,
                password=settings.redis_password,
                decode_responses=True,
                socket_connect_timeout=5,
                socket_timeout=5
            )
            logger.info("Redis rate limiter initialized")
        except Exception as e:
            logger.warning(f"Failed to initialize Redis rate limiter: {e}, falling back to local")
            self.redis_client = None
    
    async def _check_redis_limit(self, key: str) -> Tuple[bool, int, int]:
        """Check rate limit using Redis"""
        if not self.redis_client:
            return False, 0, 0
        
        try:
            current_time = int(time.time())
            window_start = current_time - self.config.window
            
            # Use Redis pipeline for atomic operations
            pipe = self.redis_client.pipeline()
            
            # Remove expired entries
            pipe.zremrangebyscore(key, 0, window_start)
            
            # Count current requests
            pipe.zcard(key)
            
            # Add current request
            pipe.zadd(key, {str(current_time): current_time})
            
            # Set expiry
            pipe.expire(key, self.config.window)
            
            results = await pipe.execute()
            current_requests = results[1]
            
            # Check if limit exceeded
            if current_requests >= self.config.requests:
                return False, current_requests, self.config.requests
            
            return True, current_requests + 1, self.config.requests
            
        except Exception as e:
            logger.error(f"Redis rate limit check failed: {e}, falling back to local")
            return await self._check_local_limit(key)
    
    def _check_local_limit(self, key: str) -> Tuple[bool, int, int]:
        """Check rate limit using local storage"""
        current_time = time.time()
        window_start = current_time - self.config.window
        
        # Remove expired entries
        while self.local_limits[key] and self.local_limits[key][0] < window_start:
            self.local_limits[key].popleft()
        
        current_requests = len(self.local_limits[key])
        
        # Check if limit exceeded
        if current_requests >= self.config.requests:
            return False, current_requests, self.config.requests
        
        # Add current request
        self.local_limits[key].append(current_time)
        return True, current_requests + 1, self.config.requests
    
    async def check_rate_limit(self, key: str) -> Tuple[bool, int, int]:
        """Check if request is allowed within rate limit"""
        if not settings.rate_limit_enabled:
            return True, 0, 0
        
        if self.redis_client:
            return await self._check_redis_limit(key)
        else:
            return self._check_local_limit(key)
    
    def get_client_key(self, request: Request) -> str:
        """Generate client identifier for rate limiting"""
        # Try to get client IP
        client_ip = request.client.host if request.client else "unknown"
        
        # If behind proxy, try to get real IP
        forwarded_for = request.headers.get("X-Forwarded-For")
        if forwarded_for:
            client_ip = forwarded_for.split(",")[0].strip()
        
        # Add user agent for additional uniqueness
        user_agent = request.headers.get("User-Agent", "unknown")
        
        return f"rate_limit:{client_ip}:{hash(user_agent) % 1000}"
    
    async def cleanup(self) -> None:
        """Cleanup resources"""
        if self.redis_client:
            await self.redis_client.close()


# Global rate limiter instance
rate_limiter = RateLimiter()


class RateLimitMiddleware:
    """FastAPI middleware for rate limiting"""
    
    def __init__(self, app):
        self.app = app
    
    async def __call__(self, scope, receive, send):
        if scope["type"] == "http":
            # Create a mock request object for rate limiting
            request = Request(scope, receive)
            client_key = rate_limiter.get_client_key(request)
            
            # Check rate limit
            allowed, current, limit = await rate_limiter.check_rate_limit(client_key)
            
            if not allowed:
                # Rate limit exceeded
                error_response = {
                    "error": "Rate limit exceeded",
                    "message": f"Too many requests. Limit: {limit} per {rate_limiter.config.window}s",
                    "retry_after": rate_limiter.config.window,
                    "current_requests": current,
                    "limit": limit
                }
                
                # Send error response
                await send({
                    "type": "http.response.start",
                    "status": 429,
                    "headers": [
                        (b"content-type", b"application/json"),
                        (b"retry-after", str(rate_limiter.config.window).encode()),
                        (b"x-ratelimit-limit", str(limit).encode()),
                        (b"x-ratelimit-remaining", str(max(0, limit - current)).encode()),
                        (b"x-ratelimit-reset", str(int(time.time()) + rate_limiter.config.window).encode())
                    ]
                })
                
                await send({
                    "type": "http.response.body",
                    "body": str(error_response).encode()
                })
                return
            
            # Add rate limit headers to response
            async def send_with_headers(message):
                if message["type"] == "http.response.start":
                    # Add rate limit headers
                    headers = list(message.get("headers", []))
                    headers.extend([
                        (b"x-ratelimit-limit", str(limit).encode()),
                        (b"x-ratelimit-remaining", str(max(0, limit - current)).encode()),
                        (b"x-ratelimit-reset", str(int(time.time()) + rate_limiter.config.window).encode())
                    ])
                    message["headers"] = headers
                
                await send(message)
            
            await self.app(scope, receive, send_with_headers)
        else:
            await self.app(scope, receive, send)


async def check_rate_limit_dependency(request: Request) -> None:
    """Dependency for manual rate limit checking in endpoints"""
    if not settings.rate_limit_enabled:
        return
    
    client_key = rate_limiter.get_client_key(request)
    allowed, current, limit = await rate_limiter.check_rate_limit(client_key)
    
    if not allowed:
        raise HTTPException(
            status_code=429,
            detail={
                "error": "Rate limit exceeded",
                "message": f"Too many requests. Limit: {limit} per {rate_limiter.config.window}s",
                "retry_after": rate_limiter.config.window,
                "current_requests": current,
                "limit": limit
            }
        )
