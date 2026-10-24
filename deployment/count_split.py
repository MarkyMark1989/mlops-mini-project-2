"""Check the traffic split (Q7e): send the sample request through the endpoint and
count which deployment answers each time.

Every response carries an azureml-model-deployment header naming the deployment
that served it. Reads the scoring URI and key that `deploy.py secrets` writes.

    .venv/bin/python deployment/count_split.py      # e.g. {'blue': 71, 'green': 29}
"""
import argparse
import collections
import json
import tomllib
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parent.parent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--requests", type=int, default=100)
    parser.add_argument("--secrets", default=str(ROOT / ".streamlit" / "secrets.toml"))
    args = parser.parse_args()

    s = tomllib.load(open(args.secrets, "rb"))    # .streamlit/secrets.toml from deploy.py secrets
    body = json.load(open(ROOT / "deployment" / "sample_request.json"))
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {s['endpoint_key']}"}
    counts = collections.Counter()
    with requests.Session() as session:
        for _ in range(args.requests):                                 # 100 by default
            r = session.post(s["scoring_uri"], json=body, headers=headers, timeout=30)
            counts[r.headers.get("azureml-model-deployment", "?")] += 1
    print(f"{args.requests} requests through the traffic split:", dict(counts))


if __name__ == "__main__":
    main()
