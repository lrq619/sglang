"""Write a synthetic replay corpus in the prepare_corpus.py output format.

For pipeline dry runs when the recorded CSVs are not available: token ids are
random, but the structure matches a real corpus (shared system prefix, a
<|im_start|>user\\n in every reset row for the salt, prefill rows ending in
<think>, next rows starting with \\n), so loadgen exercises the same paths.

Usage:
    python make_synthetic_corpus.py -o synthetic.jsonl --num-sessions 20
"""

import argparse
import json
import random

IM_START, IM_END, THINK, NL = 248045, 248046, 248068, 198
SYSTEM, USER, ASSISTANT = 8678, 846, 74455
# Plain-text vocabulary range, clear of the special tokens above 248000.
VOCAB = (1000, 150000)


def _rand(rng: random.Random, n: int) -> list:
    return [rng.randint(*VOCAB) for _ in range(n)]


def _turn(*, rng, index: int, prefill: list, last: bool) -> dict:
    return {
        "turn_index": index,
        "prompt_reset": index == 0,
        "prev_response_in_history": index > 0,
        "prefill_token_ids": prefill,
        "decode_num_tokens": rng.randint(50, 600),
        "turn_toolcall_latency_ms": None if last else rng.randint(500, 10000),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("-o", "--output", required=True)
    parser.add_argument("--num-sessions", type=int, default=20)
    parser.add_argument("--system-tokens", type=int, default=3000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    system = [IM_START, SYSTEM, NL] + _rand(rng, args.system_tokens) + [IM_END, NL]
    with open(args.output, "w") as f:
        for s in range(args.num_sessions):
            num_turns = rng.randint(8, 30)
            first = (
                system
                + [IM_START, USER, NL]
                + _rand(rng, rng.randint(200, 2000))
                + [IM_END, NL, IM_START, ASSISTANT, NL, THINK]
            )
            turns = [_turn(rng=rng, index=0, prefill=first, last=num_turns == 1)]
            for k in range(1, num_turns):
                tool_result = (
                    [NL, IM_START, USER, NL]
                    + _rand(rng, rng.randint(50, 3000))
                    + [IM_END, NL, IM_START, ASSISTANT, NL, THINK]
                )
                turns.append(
                    _turn(rng=rng, index=k, prefill=tool_result, last=k == num_turns - 1)
                )
            rec = {
                "session_id": f"synthetic_{s:03d}",
                "benchmark": "synthetic",
                "task": f"synthetic_{s:03d}",
                "turns": turns,
            }
            f.write(json.dumps(rec) + "\n")
    print(f"{args.num_sessions} synthetic sessions -> {args.output}")


if __name__ == "__main__":
    main()
