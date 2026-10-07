"""Generate a session arrival schedule for loadgen.py.

Each arrival draws one trajectory from the corpus by shuffle-cycling: all
trajectories are used once in a random order, then reshuffled for the next
cycle. Any time window therefore sees a balanced task mix. Every arrival is a
separate replica (its own salt in loadgen.py), so replicas of the same
trajectory never hit each other's cache beyond the shared system prompt.

Output is JSONL, one arrival per line, sorted by arrival_offset (seconds from
loadgen start):
    {"replica_id": 0, "session_id": "0c21c0eeb4fabe14", "arrival_offset": 12.3}

Usage:
    python make_arrivals.py hermes_clean.jsonl --rate-per-min 1.6 \
        --duration-min 360 -o session_arrival.jsonl
"""

import argparse
import json
import random


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("corpus", help="output of prepare_corpus.py")
    parser.add_argument("-o", "--output", required=True)
    parser.add_argument("--rate-per-min", type=float, required=True)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--duration-min", type=float, help="arrival window length")
    group.add_argument("--num-sessions", type=int, help="number of arrivals")
    parser.add_argument(
        "--pattern",
        choices=["poisson", "uniform"],
        default="poisson",
        help="poisson: exponential inter-arrival gaps; uniform: fixed gaps",
    )
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    with open(args.corpus) as f:
        session_ids = [json.loads(line)["session_id"] for line in f if line.strip()]
    assert session_ids, "empty corpus"

    rng = random.Random(args.seed)
    mean_gap = 60.0 / args.rate_per_min
    horizon = args.duration_min * 60.0 if args.duration_min is not None else None

    offsets = []
    t = 0.0
    while True:
        if args.num_sessions is not None and len(offsets) >= args.num_sessions:
            break
        t += rng.expovariate(1.0 / mean_gap) if args.pattern == "poisson" else mean_gap
        if horizon is not None and t > horizon:
            break
        offsets.append(t)

    deck = []
    with open(args.output, "w") as f:
        for replica_id, offset in enumerate(offsets):
            if not deck:
                deck = session_ids[:]
                rng.shuffle(deck)
            rec = {
                "replica_id": replica_id,
                "session_id": deck.pop(),
                "arrival_offset": round(offset, 3),
            }
            f.write(json.dumps(rec) + "\n")

    n = len(offsets)
    print(
        f"{n} arrivals over {offsets[-1] / 60 if n else 0:.1f} min "
        f"({args.pattern}, {args.rate_per_min}/min) from {len(session_ids)} "
        f"trajectories, ~{n / len(session_ids):.1f} uses each -> {args.output}"
    )


if __name__ == "__main__":
    main()
