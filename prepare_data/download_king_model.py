import os
import httpx
from model_store import (
    ModelRef,
    materialize_model,
)

# DASHBOARD_URL = os.environ.get(
#     "ALBEDO_DASHBOARD_URL",
#     "https://us-east-1.hippius.com/albedo/dashboard.json",
# )

def main():
    # try:
    #     resp = httpx.get(DASHBOARD_URL, timeout=30)
    #     resp.raise_for_status()
    #     dashboard = resp.json()
    #     king = dashboard["king"]
    #     king_repo = king["model_repo"]
    #     king_digest = king.get("king_digest") or king.get("model_digest")
    #     print("discovered challenger from dashboard: %s@%s" % (king_repo, (king_digest or "")[:19]))
    # except Exception:
    #     print("could not fetch dashboard, falling back to seed %s" % king_repo)
    #     raise
    king_repo = "arboshelper/albedo-qwen3-4b-2-5-final"
    king_digest = "sha256:9ae3be6e1f5f3416206f8a0835901bbb18c4db83b565bf20dc5e989d93dbd484"

    king_ref = ModelRef(king_repo, king_digest)
    king_dir = os.path.abspath(f"../checkpoint/king/{king_repo}")
    print("downloading challenger from %s" % king_ref.immutable_ref)
    materialize_model(king_ref, local_dir=king_dir, max_workers=16)

if __name__ == "__main__":
    main()


