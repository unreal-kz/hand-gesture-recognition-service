#!/usr/bin/env python3
"""
Load Testing Script for Hand Gesture Recognition Service
Tests the service under various load conditions
"""

import asyncio
import aiohttp
import time
import json
import base64
import random
from typing import List, Dict, Any
import statistics
from dataclasses import dataclass
import argparse
import os

# Test image (you can replace this with a real base64 image)
SAMPLE_IMAGE_BASE64 = """
iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg==
"""


@dataclass
class LoadTestResult:
    """Result of a load test"""
    total_requests: int
    successful_requests: int
    failed_requests: int
    total_time: float
    average_response_time: float
    min_response_time: float
    max_response_time: float
    median_response_time: float
    p95_response_time: float
    p99_response_time: float
    requests_per_second: float
    success_rate: float
    error_details: List[Dict[str, Any]]


class LoadTester:
    """Load tester for the hand gesture service"""
    
    def __init__(self, base_url: str, auth_token: str = None):
        self.base_url = base_url.rstrip('/')
        self.auth_token = auth_token
        self.session = None
        
    async def __aenter__(self):
        """Async context manager entry"""
        self.session = aiohttp.ClientSession(
            headers={'Authorization': f'Bearer {self.auth_token}'} if self.auth_token else {}
        )
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        if self.session:
            await self.session.close()
    
    async def test_single_request(self) -> Dict[str, Any]:
        """Test a single request"""
        start_time = time.time()
        
        try:
            payload = {
                "image_base64": SAMPLE_IMAGE_BASE64.strip()
            }
            
            async with self.session.post(
                f"{self.base_url}/detect-fingers",
                json=payload,
                timeout=aiohttp.ClientTimeout(total=30)
            ) as response:
                response_time = time.time() - start_time
                
                if response.status == 200:
                    result = await response.json()
                    return {
                        "success": True,
                        "status_code": response.status,
                        "response_time": response_time,
                        "result": result
                    }
                else:
                    error_text = await response.text()
                    return {
                        "success": False,
                        "status_code": response.status,
                        "response_time": response_time,
                        "error": error_text
                    }
                    
        except Exception as e:
            response_time = time.time() - start_time
            return {
                "success": False,
                "status_code": 0,
                "response_time": response_time,
                "error": str(e)
            }
    
    async def test_batch_request(self, batch_size: int) -> Dict[str, Any]:
        """Test a batch request"""
        start_time = time.time()
        
        try:
            payload = {
                "images": [
                    {
                        "image_id": f"test_{i}",
                        "image_base64": SAMPLE_IMAGE_BASE64.strip()
                    }
                    for i in range(batch_size)
                ]
            }
            
            async with self.session.post(
                f"{self.base_url}/detect-fingers-batch",
                json=payload,
                timeout=aiohttp.ClientTimeout(total=60)
            ) as response:
                response_time = time.time() - start_time
                
                if response.status == 200:
                    result = await response.json()
                    return {
                        "success": True,
                        "status_code": response.status,
                        "response_time": response_time,
                        "result": result
                    }
                else:
                    error_text = await response.text()
                    return {
                        "success": False,
                        "status_code": response.status,
                        "response_time": response_time,
                        "error": error_text
                    }
                    
        except Exception as e:
            response_time = time.time() - start_time
            return {
                "success": False,
                "status_code": 0,
                "response_time": response_time,
                "error": str(e)
            }
    
    async def run_load_test(
        self, 
        num_requests: int, 
        concurrency: int,
        test_type: str = "single"
    ) -> LoadTestResult:
        """Run a load test with specified parameters"""
        print(f"🚀 Starting load test: {num_requests} requests, {concurrency} concurrent")
        print(f"📊 Test type: {test_type}")
        
        start_time = time.time()
        semaphore = asyncio.Semaphore(concurrency)
        
        async def make_request():
            async with semaphore:
                if test_type == "batch":
                    return await self.test_batch_request(5)  # Batch size of 5
                else:
                    return await self.test_single_request()
        
        # Create tasks
        tasks = [make_request() for _ in range(num_requests)]
        
        # Execute requests
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process results
        total_time = time.time() - start_time
        successful = []
        failed = []
        error_details = []
        
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                failed.append({
                    "request_id": i,
                    "error": str(result),
                    "response_time": 0
                })
                error_details.append({
                    "request_id": i,
                    "error": str(result),
                    "type": "exception"
                })
            elif result["success"]:
                successful.append(result["response_time"])
            else:
                failed.append({
                    "request_id": i,
                    "error": result.get("error", "Unknown error"),
                    "response_time": result["response_time"]
                })
                error_details.append({
                    "request_id": i,
                    "error": result.get("error", "Unknown error"),
                    "status_code": result.get("status_code", 0),
                    "type": "http_error"
                })
        
        # Calculate statistics
        if successful:
            response_times = successful
            avg_response_time = statistics.mean(response_times)
            min_response_time = min(response_times)
            max_response_time = max(response_times)
            median_response_time = statistics.median(response_times)
            p95_response_time = statistics.quantiles(response_times, n=20)[18]  # 95th percentile
            p99_response_time = statistics.quantiles(response_times, n=100)[98] if len(response_times) >= 100 else max_response_time
        else:
            avg_response_time = min_response_time = max_response_time = median_response_time = p95_response_time = p99_response_time = 0
        
        return LoadTestResult(
            total_requests=num_requests,
            successful_requests=len(successful),
            failed_requests=len(failed),
            total_time=total_time,
            average_response_time=avg_response_time,
            min_response_time=min_response_time,
            max_response_time=max_response_time,
            median_response_time=median_response_time,
            p95_response_time=p95_response_time,
            p99_response_time=p99_response_time,
            requests_per_second=num_requests / total_time if total_time > 0 else 0,
            success_rate=len(successful) / num_requests if num_requests > 0 else 0,
            error_details=error_details
        )
    
    def print_results(self, result: LoadTestResult):
        """Print load test results in a formatted way"""
        print("\n" + "="*60)
        print("📊 LOAD TEST RESULTS")
        print("="*60)
        print(f"Total Requests: {result.total_requests}")
        print(f"Successful: {result.successful_requests}")
        print(f"Failed: {result.failed_requests}")
        print(f"Success Rate: {result.success_rate:.2%}")
        print(f"Total Time: {result.total_time:.2f}s")
        print(f"Requests/Second: {result.requests_per_second:.2f}")
        print("\n⏱️  Response Times:")
        print(f"  Average: {result.average_response_time:.3f}s")
        print(f"  Median: {result.median_response_time:.3f}s")
        print(f"  95th Percentile: {result.p95_response_time:.3f}s")
        print(f"  99th Percentile: {result.p99_response_time:.3f}s")
        print(f"  Min: {result.min_response_time:.3f}s")
        print(f"  Max: {result.max_response_time:.3f}s")
        
        if result.error_details:
            print(f"\n❌ Errors ({len(result.error_details)}):")
            for error in result.error_details[:5]:  # Show first 5 errors
                print(f"  Request {error['request_id']}: {error['error']}")
            if len(result.error_details) > 5:
                print(f"  ... and {len(result.error_details) - 5} more errors")
        
        print("="*60)


async def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Load test the Hand Gesture Recognition Service")
    parser.add_argument("--url", default="http://localhost:8000", help="Service base URL")
    parser.add_argument("--requests", type=int, default=100, help="Number of requests")
    parser.add_argument("--concurrency", type=int, default=10, help="Concurrent requests")
    parser.add_argument("--type", choices=["single", "batch"], default="single", help="Test type")
    parser.add_argument("--auth", help="Authorization token")
    
    args = parser.parse_args()
    
    print("🧪 Hand Gesture Recognition Service - Load Tester")
    print(f"🎯 Target: {args.url}")
    print(f"📊 Configuration: {args.requests} requests, {args.concurrency} concurrent")
    
    async with LoadTester(args.url, args.auth) as tester:
        # Test service health first
        try:
            async with tester.session.get(f"{args.url}/health") as response:
                if response.status == 200:
                    health = await response.json()
                    print(f"✅ Service is healthy: {health['status']}")
                else:
                    print(f"⚠️  Service health check failed: {response.status}")
        except Exception as e:
            print(f"❌ Cannot connect to service: {e}")
            return
        
        # Run load test
        result = await tester.run_load_test(args.requests, args.concurrency, args.type)
        
        # Print results
        tester.print_results(result)
        
        # Save results to file
        timestamp = int(time.time())
        filename = f"load_test_results_{timestamp}.json"
        
        with open(filename, 'w') as f:
            json.dump({
                "timestamp": timestamp,
                "configuration": {
                    "url": args.url,
                    "requests": args.requests,
                    "concurrency": args.concurrency,
                    "type": args.type
                },
                "results": {
                    "total_requests": result.total_requests,
                    "successful_requests": result.successful_requests,
                    "failed_requests": result.failed_requests,
                    "total_time": result.total_time,
                    "average_response_time": result.average_response_time,
                    "min_response_time": result.min_response_time,
                    "max_response_time": result.max_response_time,
                    "median_response_time": result.median_response_time,
                    "p95_response_time": result.p95_response_time,
                    "p99_response_time": result.p99_response_time,
                    "requests_per_second": result.requests_per_second,
                    "success_rate": result.success_rate
                },
                "errors": result.error_details
            }, f, indent=2)
        
        print(f"💾 Results saved to: {filename}")


if __name__ == "__main__":
    asyncio.run(main())
