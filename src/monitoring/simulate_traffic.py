"""Send test traffic to the API to demonstrate monitoring and drift.

    python -m src.monitoring.simulate_traffic --mode normal --n 200
    python -m src.monitoring.simulate_traffic --mode drift --n 200 --invalid 10

normal: reviews from data/processed/test.csv (same distribution as training)
drift : short, shouty, tweet-like texts (a controlled, simulated data-drift event)
"""
import argparse
import json
import random
import sys
import urllib.error
import urllib.request
from collections import Counter

import pandas as pd

OPENERS = ["omg", "lol", "ugh", "wow", "yay", "smh", "bruh"]
BODIES = ["this is so bad", "best day ever", "cant believe it", "totally worth it", "never again", "so happy rn"]
ENDINGS = ["!!!", "?!", "!!", "!!!!!"]


def normal_texts(n: int, test_csv: str = "data/processed/test.csv", seed: int = 1) -> list[str]:
    frame = pd.read_csv(test_csv)
    return frame["text"].sample(n=min(n, len(frame)), random_state=seed).tolist()


def drifted_texts(n: int, seed: int = 1) -> list[str]:
    rng = random.Random(seed)
    return [
        f"{rng.choice(OPENERS)} {rng.choice(BODIES)} {rng.choice(ENDINGS)}".upper() for _ in range(n)
    ]


def post(url: str, payload) -> int:
    request = urllib.request.Request(
        url, data=json.dumps(payload).encode("utf-8"), headers={"Content-Type": "application/json"}
    )
    try:
        with urllib.request.urlopen(request, timeout=10) as response:
            return response.status
    except urllib.error.HTTPError as exc:
        return exc.code


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--mode", choices=["normal", "drift"], default="normal")
    parser.add_argument("--n", type=int, default=100)
    parser.add_argument("--invalid", type=int, default=0, help="also send this many invalid requests")
    parser.add_argument("--url", default="http://127.0.0.1:8000/predict")
    args = parser.parse_args()

    texts = normal_texts(args.n) if args.mode == "normal" else drifted_texts(args.n)
    statuses = Counter(post(args.url, {"text": text}) for text in texts)
    statuses.update(post(args.url, {"text": ""}) for _ in range(args.invalid))
    print(f"mode={args.mode} sent={sum(statuses.values())} status codes: {dict(statuses)}")
    return 0 if statuses.get(200) else 1


if __name__ == "__main__":
    sys.exit(main())
