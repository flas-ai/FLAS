"""Evaluate FLAS on AxBench concepts (AxBench protocol).

Uses batched generation: for each concept, all prompts × flowtimes are generated
in a single batched forward pass. ~5-10x faster than sequential.

Usage:
    python scripts/eval.py \
        --flow-ckpt checkpoints/stage2/best.pt \
        --output-dir results/run1 \
        --flowtimes 1.0 1.5 2.0 \
        --n-steps 2 --num-eval-prompts 5
"""

import argparse
import json
import random
from pathlib import Path

import pandas as pd
import torch
from tqdm import tqdm


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", type=str, default=None,
                        help="Override model_id (default: read from ckpt config.json)")
    parser.add_argument("--layer", type=int, default=None,
                        help="Override layer (default: read from ckpt config.json)")
    parser.add_argument("--flow-ckpt", type=str, required=True)
    parser.add_argument("--num-blocks", type=int, default=None,
                        help="Override num_blocks (default: read from ckpt config.json)")
    parser.add_argument("--data-dir", type=str,
                        default="thirdparty/axbench/axbench/concept500/prod_2b_l20_v1/generate")
    parser.add_argument("--alpaca-eval", type=str,
                        default="data/alpaca_eval.json")
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--max-concepts", type=int, default=500)
    parser.add_argument("--concept-ids-file", type=str, default=None,
                        help="JSON file with list of concept IDs to eval (overrides --max-concepts)")
    parser.add_argument("--num-eval-prompts", type=int, default=5)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--shard", type=str, default="0/1")
    parser.add_argument("--flowtimes", type=float, nargs="+",
                        default=[1.0, 1.5, 2.0])
    parser.add_argument("--n-steps", type=int, default=2)
    parser.add_argument("--concept-max-len", type=int, default=64)
    parser.add_argument("--max-batch", type=int, default=15,
                        help="Max batch size for generation")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_dir / "eval_config.json", "w") as f:
        json.dump(vars(args), f, indent=2)

    # Load generator
    from flas.generate import load_generator
    print(f"Loading FLAS generator...", flush=True)
    gen = load_generator(
        args.flow_ckpt, args.model_id, args.layer, args.num_blocks)

    # Load AlpacaEval prompts
    with open(args.alpaca_eval) as f:
        alpaca_data = json.load(f)
    alpaca_prompts = [d["instruction"] for d in alpaca_data]
    print(f"Loaded {len(alpaca_prompts)} AlpacaEval prompts", flush=True)

    # Load concept metadata
    data_path = Path(args.data_dir)
    with open(data_path / "metadata.jsonl") as f:
        metadata = [json.loads(l) for l in f]

    if args.concept_ids_file:
        with open(args.concept_ids_file) as f:
            all_cids = json.load(f)
        print(f"Using {len(all_cids)} concept IDs from {args.concept_ids_file}")
    else:
        all_cids = list(range(min(len(metadata), args.max_concepts)))

    # Shard
    shard_idx, shard_total = map(int, args.shard.split("/"))
    per_shard = len(all_cids) // shard_total
    start = shard_idx * per_shard
    end = start + per_shard if shard_idx < shard_total - 1 else len(all_cids)
    concept_ids = all_cids[start:end]

    print(f"Shard {shard_idx}/{shard_total}: {len(concept_ids)} concepts, "
          f"flowtimes={args.flowtimes}, n_steps={args.n_steps}, "
          f"batch={args.max_batch}", flush=True)

    all_results = []
    for i, cid in enumerate(tqdm(concept_ids, desc="Evaluating")):
        concept_name = metadata[cid]["concept"]

        # Sample prompts reproducibly (matching AxBench: seed=concept_id)
        rng = random.Random(cid)
        eval_prompts = rng.sample(
            alpaca_prompts, min(args.num_eval_prompts, len(alpaca_prompts)))

        # Batched generation: all prompts × flowtimes at once
        batch_results = gen.generate_batch(
            eval_prompts, concept_name, args.flowtimes,
            n_steps=args.n_steps,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            max_batch=args.max_batch,
        )

        for r in batch_results:
            all_results.append({
                "concept_id": cid,
                "concept": concept_name,
                "input": r["prompt"],
                "input_id": r["prompt_idx"],
                "factor": r["flowtime"],
                "generation_steered": r["generation"],
            })

        if (i + 1) % 10 == 0:
            df = pd.DataFrame(all_results)
            df.to_json(output_dir / f"results_shard{shard_idx}.json",
                       orient="records", indent=2, force_ascii=False)
            print(f"  [{i+1}/{len(concept_ids)}] {len(all_results)} samples",
                  flush=True)

    # Final save
    df = pd.DataFrame(all_results)
    df.to_json(output_dir / f"results_shard{shard_idx}.json",
               orient="records", indent=2, force_ascii=False)
    print(f"\nDone: {len(all_results)} results ({len(concept_ids)} concepts x "
          f"{args.num_eval_prompts} prompts x {len(args.flowtimes)} flowtimes)",
          flush=True)


if __name__ == "__main__":
    main()
