from __future__ import annotations

import os
import socket
from datetime import datetime
from pathlib import Path

from supabase_worker import load_config, _sb


def _load_dotenv(path: Path) -> None:
    if not path.exists():
        return
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            continue
        k, v = line.split("=", 1)
        k = k.strip()
        v = v.strip().strip('"').strip("'")
        os.environ.setdefault(k, v)


def _mask(s: str, keep: int = 6) -> str:
    if not s:
        return ""
    if len(s) <= keep:
        return "<set>"
    return f"<set…{s[-keep:]}>"


def main() -> None:
    _load_dotenv(Path(__file__).with_name(".env"))
    cfg = load_config()
    print("[smoketest] ok: config loaded")
    print(f"[smoketest] SUPABASE_URL={cfg.supabase_url}")
    print(f"[smoketest] SUPABASE_SERVICE_ROLE_KEY={_mask(cfg.supabase_service_role_key)}")
    print(f"[smoketest] worker_id={cfg.worker_id} hostname={socket.gethostname()}")
    print(f"[smoketest] storage_root={cfg.storage_root}")
    print(f"[smoketest] pbrain_main_py={cfg.pbrain_main_py}")

    sb = _sb(cfg)
    print("[smoketest] ok: connected")

    # Check tables exist / RLS allows reads (service role bypasses RLS).
    try:
        r = sb.s.get(
            f"{sb.url}/rest/v1/jobs",
            params={
                "select": "id,status,stage_id,created_at",
                "order": "created_at.desc",
                "limit": "5",
            },
            timeout=30,
        )
        r.raise_for_status()
        jobs = r.json() or []
        queued = [j for j in jobs if (j.get("status") == "queued")]
        print(f"[smoketest] ok: jobs read (last5={len(jobs)}, queued_in_last5={len(queued)})")
    except Exception as exc:
        print(f"[smoketest] ERROR reading jobs table: {exc}")
        return

    # Heartbeat table (optional until SQL applied)
    try:
        sb.upsert(
            "worker_heartbeats",
            {
                "worker_id": cfg.worker_id,
                "last_seen": datetime.utcnow().replace(microsecond=0).isoformat() + "Z",
                "hostname": socket.gethostname(),
                "meta": {"smoketest": True},
            },
            on_conflict="worker_id",
        )
        print("[smoketest] ok: heartbeat upsert")
    except Exception as exc:
        print(f"[smoketest] WARN heartbeat upsert failed (apply schema?): {exc}")

    allow_claim = os.getenv("SMOKETEST_ALLOW_CLAIM", "").strip().lower() in {"1", "true", "yes"}
    if not allow_claim:
        print("[smoketest] skipping claim_job (set SMOKETEST_ALLOW_CLAIM=1 to test claiming)")
        return

    try:
        job = sb.rpc("claim_job", {"p_worker_id": f"smoketest:{cfg.worker_id}"})
        if isinstance(job, list):
            job = job[0] if job else None
        if not isinstance(job, dict) or not job.get("id"):
            print("[smoketest] claim_job: no queued job available")
            return
        job_id = str(job.get("id"))
        print(f"[smoketest] claim_job: claimed job_id={job_id} status={job.get('status')} stage={job.get('stage_id')}")

        # Put it back to queued so we don't disrupt a real run.
        sb.update("jobs", {"id": job_id}, {"status": "queued", "runner_id": None, "claimed_at": None})
        print("[smoketest] claim_job: released back to queued")
    except Exception as exc:
        print(f"[smoketest] ERROR calling claim_job: {exc}")


if __name__ == "__main__":
    main()
