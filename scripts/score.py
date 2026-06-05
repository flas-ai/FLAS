"""Aggregate per-flowtime (per-T) scores from a judge cache.

Decoupled from judging: run ANY judge path (judge_openai.py / judge_azure.py)
to produce the cache of raw per-sample judgments, then aggregate here. The
report is simply "what T, what score" — one row per flowtime T with mean
Concept / Instruction / Fluency / HMean. (The old per-factor-max metric is
gone.)

Usage:
    python scripts/score.py <cache.json> [--output <scores.json>]

`<cache.json>` is whatever the judge wrote to its --output (e.g. judged.json).
Accepts both the bare list and the {"samples": [...]} wrapper.
"""
import argparse
import json
from collections import defaultdict

import numpy as np


def load_judged(path):
    """Read a judge cache, tolerating both {"samples": [...]} and a bare list
    (and legacy files that also carried a top-level "summary")."""
    data = json.load(open(path))
    if isinstance(data, dict):
        return data.get("samples", [])
    return data


def per_factor_scores(judged):
    """Mean Concept / Instruction / Fluency / HMean at each flowtime factor T."""
    ok = [j for j in judged if j.get("scores") is not None]
    by_factor = defaultdict(list)
    for j in ok:
        by_factor[j.get("factor")].append(j["scores"])
    out = {}
    for factor in sorted(by_factor, key=lambda x: (x is None, x)):
        ss = by_factor[factor]
        out[factor] = {
            "n": len(ss),
            "concept": float(np.mean([s["concept_score"] for s in ss])),
            "instruction": float(np.mean([s["instruction_score"] for s in ss])),
            "fluency": float(np.mean([s["fluency_score"] for s in ss])),
            "hmean": float(np.mean([s["harmonic_mean"] for s in ss])),
        }
    return out


def main():
    p = argparse.ArgumentParser(
        description="Aggregate per-T scores from a judge cache")
    p.add_argument("cache", help="judge cache JSON (judge_openai/judge_azure output)")
    p.add_argument("--output", default=None,
                   help="scores JSON (default: <cache>.scores.json)")
    args = p.parse_args()

    judged = load_judged(args.cache)
    ok = [j for j in judged if j.get("scores") is not None]
    by_factor = per_factor_scores(judged)

    out_path = args.output or args.cache.replace(".json", ".scores.json")
    payload = {
        "cache": args.cache,
        "n_samples": len(judged),
        "n_judged": len(ok),
        "n_failed": len(judged) - len(ok),
        "by_factor": {str(k): v for k, v in by_factor.items()},
    }
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    print(f"{len(ok)}/{len(judged)} judged ({payload['n_failed']} failed/filtered)")
    for factor, s in by_factor.items():
        print(f"[scores] T={factor}  HMean={s['hmean']:.3f}  "
              f"C={s['concept']:.3f}  I={s['instruction']:.3f}  "
              f"F={s['fluency']:.3f}  (n={s['n']})")
    print(f"Scores -> {out_path}")


if __name__ == "__main__":
    main()
