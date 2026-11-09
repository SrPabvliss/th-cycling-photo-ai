"""Load-test client for the /pipeline endpoint.

Usage:
    python scripts/qa/load_test_pipeline.py --url https://<app>.modal.run \
        --photos photos.txt --concurrency 3 --out result.json
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from collections import Counter
from pathlib import Path

import httpx


def percentile(sorted_values: list[float], q: float) -> float:
    if not sorted_values:
        return 0.0
    idx = min(len(sorted_values) - 1, max(0, round(q * (len(sorted_values) - 1))))
    return sorted_values[idx]


async def send_one(
    client: httpx.AsyncClient,
    semaphore: asyncio.Semaphore,
    endpoint: str,
    image: str,
    threshold: float,
    t0: float,
    timeout: float,
) -> dict:
    body = {
        "image_id": Path(image).stem,
        "image_url": image,
        "event_id": None,
        "confidence_threshold": threshold,
    }
    async with semaphore:
        sent_at = time.perf_counter()
        record: dict = {"image": image, "sent_at_ms": round((sent_at - t0) * 1000, 1)}
        try:
            response = await asyncio.wait_for(client.post(endpoint, json=body), timeout=timeout)
            record["status"] = response.status_code
            if response.status_code == 200:
                data = response.json()
                runtime = data.get("runtime") or {}
                record["gpu"] = runtime.get("gpu")
                record["gpu_mem_peak_mb"] = runtime.get("gpu_mem_peak_mb")
                record["container_id"] = runtime.get("container_id")
                record["request_seq"] = runtime.get("request_seq")
                record["processing_ms"] = data.get("processing_ms")
        except (httpx.TimeoutException, asyncio.TimeoutError):
            record["status"] = "timeout"
        except Exception as exc:
            record["status"] = f"error:{type(exc).__name__}"
        record["latency_ms"] = round((time.perf_counter() - sent_at) * 1000, 1)
        return record


def load_images(args: argparse.Namespace) -> list[str]:
    lines = Path(args.photos).read_text().splitlines()
    images = [line.strip() for line in lines if line.strip()]
    if args.limit:
        images = images[: args.limit]
    return images


async def run(args: argparse.Namespace, images: list[str]) -> dict:
    endpoint = args.url.rstrip("/") + "/pipeline?color=none"
    semaphore = asyncio.Semaphore(args.concurrency)
    t0 = time.perf_counter()
    async with httpx.AsyncClient(timeout=args.timeout) as client:
        tasks = [
            asyncio.create_task(
                send_one(client, semaphore, endpoint, image, args.confidence_threshold, t0, args.timeout)
            )
            for image in images
        ]
        requests = await asyncio.gather(*tasks)

    latencies = sorted(r["latency_ms"] for r in requests)
    ok = [r for r in requests if r["status"] == 200]
    first_sent = min(requests, key=lambda r: r["sent_at_ms"]) if requests else None
    peaks = [r["gpu_mem_peak_mb"] for r in ok if r.get("gpu_mem_peak_mb") is not None]
    statuses = [r["status"] for r in requests]
    return {
        "n": len(requests),
        "status_counts": dict(Counter(str(s) for s in statuses)),
        "timeouts": sum(1 for s in statuses if s == "timeout"),
        "errors": sum(1 for s in statuses if isinstance(s, str) and s.startswith("error:")),
        "p50_ms": percentile(latencies, 0.50),
        "p95_ms": percentile(latencies, 0.95),
        "max_ms": latencies[-1] if latencies else 0.0,
        "first_request_ms": first_sent["latency_ms"] if first_sent else None,
        "gpu_mem_peak_mb_max": max(peaks) if peaks else None,
        "gpus": sorted({r["gpu"] for r in ok if r.get("gpu") is not None}),
        "containers": len({r["container_id"] for r in ok if r.get("container_id")}),
        "wall_s": round(time.perf_counter() - t0, 2),
        "requests": requests,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--url", required=True, help="App base URL")
    parser.add_argument("--photos", required=True, help="File with one image path or URL per line")
    parser.add_argument("--concurrency", type=int, default=3)
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--out", required=True, help="Output JSON path")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--confidence-threshold", type=float, default=0.25)
    args = parser.parse_args()
    if args.concurrency < 1:
        parser.error("--concurrency must be >= 1")
    if args.timeout <= 0:
        parser.error("--timeout must be > 0")
    if args.limit < 0:
        parser.error("--limit must be >= 0")
    images = load_images(args)
    if not images:
        print("error: no photos to send (empty photos file or --limit left none)", file=sys.stderr)
        return 2

    result = asyncio.run(run(args, images))
    Path(args.out).write_text(json.dumps(result, indent=2))
    print(
        f"{result['n']} requests at concurrency {args.concurrency}: "
        f"status {result['status_counts']}, {result['timeouts']} timeouts, "
        f"{result['errors']} errors; latency p50 {result['p50_ms']} ms, "
        f"p95 {result['p95_ms']} ms, max {result['max_ms']} ms; first request "
        f"{result['first_request_ms']} ms; gpus {result['gpus']}, peak VRAM "
        f"{result['gpu_mem_peak_mb_max']} MB, {result['containers']} container(s)."
    )
    failed = set(result["status_counts"]) - {"200"}
    too_slow = result["max_ms"] >= args.timeout * 1000
    return 1 if failed or too_slow else 0


if __name__ == "__main__":
    sys.exit(main())
