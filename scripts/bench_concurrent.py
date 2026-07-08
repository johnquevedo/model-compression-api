#!/usr/bin/env python3
"""Concurrent-request benchmark for the FastAPI inference service.

Fires N total requests at a target concurrency level against /predict and reports
throughput (req/s) and latency percentiles (p50/p90/p95/p99). Use it to demonstrate
"benchmarked response latency under concurrent requests" from the resume bullet.

Example:
  python scripts/bench_concurrent.py --url http://localhost:8000/predict \\
      --requests 500 --concurrency 20 --batch-size 1
"""

from __future__ import annotations

import argparse
import asyncio
import statistics
import time
from typing import List

import httpx

SAMPLE_TEXTS = [
    "this movie was an absolute masterpiece and i loved every minute",
    "a dull, lifeless film that wasted a talented cast",
    "surprisingly heartfelt and genuinely funny throughout",
    "i have never been so bored in a theater in my life",
    "the visuals were stunning even if the plot was thin",
    "an instant classic that rewards repeat viewings",
]


async def _one_request(client: httpx.AsyncClient, url: str, payload: dict, latencies: List[float]) -> bool:
    t0 = time.perf_counter()
    try:
        resp = await client.post(url, json=payload, timeout=30.0)
        resp.raise_for_status()
        latencies.append((time.perf_counter() - t0) * 1000.0)
        return True
    except Exception as exc:  # noqa: BLE001
        print(f"  request failed: {exc}")
        return False


async def run(url: str, total: int, concurrency: int, batch_size: int) -> None:
    payload = {"texts": [SAMPLE_TEXTS[i % len(SAMPLE_TEXTS)] for i in range(batch_size)]}
    latencies: List[float] = []
    sem = asyncio.Semaphore(concurrency)
    ok = 0

    async with httpx.AsyncClient() as client:
        # Warm the server (model load / JIT) before timing.
        await _one_request(client, url, payload, [])

        async def _worker() -> None:
            nonlocal ok
            async with sem:
                if await _one_request(client, url, payload, latencies):
                    ok += 1

        wall_start = time.perf_counter()
        await asyncio.gather(*[_worker() for _ in range(total)])
        wall = time.perf_counter() - wall_start

    if not latencies:
        print("No successful requests — is the server running?")
        return

    latencies.sort()

    def pct(p: float) -> float:
        idx = min(len(latencies) - 1, int(round(p / 100.0 * (len(latencies) - 1))))
        return latencies[idx]

    print("\n=== Concurrent request benchmark ===")
    print(f"URL:              {url}")
    print(f"Requests:         {ok}/{total} succeeded")
    print(f"Concurrency:      {concurrency}")
    print(f"Batch size:       {batch_size} texts/request")
    print(f"Wall time:        {wall:.2f} s")
    print(f"Throughput:       {ok / wall:.1f} req/s  ({ok * batch_size / wall:.1f} texts/s)")
    print(f"Latency mean:     {statistics.mean(latencies):.1f} ms")
    print(f"Latency p50:      {pct(50):.1f} ms")
    print(f"Latency p90:      {pct(90):.1f} ms")
    print(f"Latency p95:      {pct(95):.1f} ms")
    print(f"Latency p99:      {pct(99):.1f} ms")
    print(f"Latency max:      {latencies[-1]:.1f} ms")


def main() -> None:
    ap = argparse.ArgumentParser(description="Concurrent load test for the inference API")
    ap.add_argument("--url", default="http://localhost:8000/predict")
    ap.add_argument("--requests", type=int, default=500, help="Total number of requests")
    ap.add_argument("--concurrency", type=int, default=20, help="Max in-flight requests")
    ap.add_argument("--batch-size", type=int, default=1, help="Texts per request")
    args = ap.parse_args()
    asyncio.run(run(args.url, args.requests, args.concurrency, args.batch_size))


if __name__ == "__main__":
    main()
