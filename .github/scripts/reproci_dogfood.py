"""Upload a capture and request a replay without exposing the tenant token."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import time
import urllib.error
import urllib.request


BASE_URL = os.environ["REPROCI_DOGFOOD_URL"].rstrip("/")
TENANT = os.environ["REPROCI_DOGFOOD_TENANT"]
TOKEN = os.environ["REPROCI_DOGFOOD_API_TOKEN"]
ARTIFACT = Path("reproci-artifact.tar.gz")


def request(
    method: str,
    path: str,
    *,
    body: bytes | None = None,
    headers: dict[str, str] | None = None,
) -> tuple[int, dict[str, object]]:
    all_headers = {
        "Authorization": f"Bearer {TOKEN}",
        "X-ReproCI-Tenant": TENANT,
        **(headers or {}),
    }
    req = urllib.request.Request(
        BASE_URL + path, data=body, headers=all_headers, method=method
    )
    try:
        with urllib.request.urlopen(req, timeout=60) as response:
            return response.status, json.load(response)
    except urllib.error.HTTPError as error:
        return error.code, json.load(error)


def main() -> None:
    run_started = time.time()
    payload = ARTIFACT.read_bytes()
    digest = "sha256:" + hashlib.sha256(payload).hexdigest()
    key = (
        f"dogfood/{os.environ['GITHUB_REPOSITORY'].replace('/', '-')}/"
        f"{os.environ['GITHUB_RUN_ID']}-{os.environ['MATRIX_ATTEMPT']}.tar.gz"
    )

    started = time.perf_counter()
    upload_status, uploaded = request(
        "PUT",
        f"/v1/artifacts/{key}",
        body=payload,
        headers={
            "Content-Type": "application/gzip",
            "X-ReproCI-Digest": digest,
        },
    )
    upload_seconds = time.perf_counter() - started
    if upload_status != 201:
        raise RuntimeError(f"artifact upload failed with HTTP {upload_status}")

    replay_request = json.dumps(
        {
            "repository": os.environ["GITHUB_REPOSITORY"],
            "commit_sha": os.environ["GITHUB_SHA"],
            "artifact_key": uploaded["key"],
            "artifact_digest": digest,
        }
    ).encode()
    idempotency_key = (
        f"{os.environ['GITHUB_RUN_ID']}:{os.environ['GITHUB_JOB']}:"
        f"{os.environ['MATRIX_ATTEMPT']}"
    )
    headers = {
        "Content-Type": "application/json",
        "X-Idempotency-Key": idempotency_key,
    }
    started = time.perf_counter()
    enqueue_status, replay = request(
        "POST", "/v1/replays", body=replay_request, headers=headers
    )
    enqueue_seconds = time.perf_counter() - started
    if enqueue_status != 202:
        raise RuntimeError(f"replay enqueue failed with HTTP {enqueue_status}")

    duplicate_status, _ = request(
        "POST", "/v1/replays", body=replay_request, headers=headers
    )
    replay_id = str(replay["ID"])
    started = time.perf_counter()
    final: dict[str, object] = {}
    for _ in range(300):
        status, final = request("GET", f"/v1/replays/{replay_id}")
        if status != 200:
            raise RuntimeError(f"replay read failed with HTTP {status}")
        if final["Status"] in {"succeeded", "failed", "dead_letter"}:
            break
        time.sleep(2)
    replay_seconds = time.perf_counter() - started

    result = {
        "schema_version": 1,
        "qualification": "owner-controlled dogfood; not independent adoption",
        "repository": os.environ["GITHUB_REPOSITORY"],
        "commit_sha": os.environ["GITHUB_SHA"],
        "run_id": os.environ["GITHUB_RUN_ID"],
        "matrix_attempt": int(os.environ["MATRIX_ATTEMPT"]),
        "started_at_epoch": run_started,
        "completed_at_epoch": time.time(),
        "artifact_bytes": len(payload),
        "artifact_digest": digest,
        "upload_seconds": upload_seconds,
        "enqueue_seconds": enqueue_seconds,
        "replay_poll_seconds": replay_seconds,
        "replay_id": replay_id,
        "replay_status": final.get("Status"),
        "worker_attempts": final.get("Attempts"),
        "duplicate_http_status": duplicate_status,
    }
    Path("reproci-dogfood-result.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, sort_keys=True))
    if final.get("Status") != "succeeded" or duplicate_status != 409:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
