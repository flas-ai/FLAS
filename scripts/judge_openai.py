"""Judge with OpenAI GPT-4o-mini (AxBench-compatible).

Prompt from AxBench.

Usage:
    python scripts/judge_openai.py \\
        --results-file results/run1/results_shard0.json \\
        --api-key $OPENAI_API_KEY \\
        --concurrency 8
"""

import argparse
import asyncio
import json
import os
import random
from collections import defaultdict

import numpy as np
from openai import AsyncOpenAI, APIStatusError, RateLimitError, APITimeoutError


# AxBench prompt templates
CONCEPT_PROMPT = """[System]
Please act as an impartial judge and evaluate whether the specified concept is \
presented in the sentence fragment provided below. Focus solely on whether the concept is \
clearly incorporated, without regard for grammar or logical coherence.

Begin your evaluation by providing a short explanation of whether the concept is \
clearly incorporated. Point out references to the concept in the fragment, noting any \
minimal or awkward incorporations. After providing your explanation, rate the concept's \
relevance on a scale from 0 to 2, where 0 indicates the concept is not present at all, \
1 indicates the concept is somewhat present but minimally or awkwardly incorporated, \
and 2 indicates the concept is more fully and effectively incorporated, with stronger and \
more natural integration. Provide your rating using this exact format: "Rating: [[score]]".

[Concept Start]
{concept}
[Concept End]

[Sentence Fragment Start]
{sentence}
[Sentence Fragment End]
"""

INSTRUCTION_PROMPT = """[System]
Please act as an impartial judge and evaluate whether the \
sentence fragment provided below is related to the instruction. Focus solely \
on the degree of relatedness in terms of topic, regardless of grammar, coherence, or \
informativeness.

Begin your evaluation by providing a brief explanation of whether the \
sentence is related to the instruction, and point out references \
related to the instruction. After providing your explanation, rate the instruction \
relevance on a scale from 0 to 2, where 0 indicates the sentence is unrelated to the \
instruction, 1 indicates it is somewhat related but only minimally or indirectly relevant in terms of topic, \
and 2 indicates it is more clearly and directly related to the instruction. Provide your rating \
using this exact format: "Rating: [[score]]".

[Instruction Start]
{instruction}
[Instruction End]

[Sentence Fragment Start]
{sentence}
[Sentence Fragment End]
"""

FLUENCY_PROMPT = """[System]
Please act as an impartial judge and evaluate the fluency of the \
sentence fragment provided below. Focus solely on fluency, disregarding \
its completeness, relevance, coherence with any broader context, or informativeness.

Begin your evaluation by briefly describing the fluency of the sentence, noting any \
unnatural phrasing, awkward transitions, grammatical errors, or repetitive structures that \
may hinder readability. After providing your explanation, rate the sentence's fluency \
on a scale from 0 to 2, where 0 indicates the sentence is not fluent and highly unnatural \
(e.g., incomprehensible or repetitive), 1 indicates it is somewhat fluent but contains noticeable \
errors or awkward phrasing, and 2 indicates the sentence is fluent and almost perfect. \
Provide your rating using this exact format: "Rating: [[score]]".

[Sentence Fragment Start]
{sentence}
[Sentence Fragment End]
"""


def extract_rating(completion, default=None):
    """Parse 'Rating: [[score]]' format. Returns None on parse failure (distinguishes from true 0)."""
    if completion is None or "Rating:" not in completion:
        return default
    text = completion.split("Rating:")[-1].strip().split('\n')[0].strip()
    text = text.replace('[', '').replace(']', '').rstrip('.').strip('"').strip("'").strip("*").strip()
    try:
        r = float(text)
        if 0.0 <= r <= 2.0:
            return r
    except ValueError:
        pass
    return default


def harmonic_mean(scores):
    if 0 in scores:
        return 0.0
    return len(scores) / sum(1.0 / s for s in scores)


async def _call_with_backoff(client, model, prompt, max_retries=8):
    """Call the chat API with exponential backoff on rate-limits / transient errors."""
    for attempt in range(max_retries):
        try:
            resp = await client.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                model=model, temperature=0.0, max_tokens=512)
            return resp.choices[0].message.content
        except RateLimitError as e:
            # Honor Retry-After if provided
            retry_after = None
            if hasattr(e, "response") and e.response is not None:
                retry_after = e.response.headers.get("retry-after")
            # Base wait grows 4, 8, 16, 32, 60, 120, 120, 120 seconds
            wait = float(retry_after) if retry_after else min(4 * (2 ** attempt), 120)
            wait *= 0.5 + random.random()  # jitter
            if attempt == max_retries - 1:
                print(f"  [rate-limit] giving up after {attempt+1} retries", flush=True)
                return None
            await asyncio.sleep(wait)
        except (APITimeoutError, APIStatusError) as e:
            if attempt == max_retries - 1:
                print(f"  [api-error] giving up: {e}", flush=True)
                return None
            await asyncio.sleep(min(2 ** attempt, 60) * (0.5 + random.random()))
        except Exception as e:
            if attempt == max_retries - 1:
                print(f"  [error] giving up: {e}", flush=True)
                return None
            await asyncio.sleep(min(2 ** attempt, 60))
    return None


async def judge_one(client, model, concept, instruction, generation, sem):
    """Run the 3 sub-judges for a single sample IN PARALLEL."""
    async with sem:
        prompts = [
            CONCEPT_PROMPT.format(concept=concept, sentence=generation),
            INSTRUCTION_PROMPT.format(instruction=instruction, sentence=generation),
            FLUENCY_PROMPT.format(sentence=generation),
        ]
        results = await asyncio.gather(
            *[_call_with_backoff(client, model, p) for p in prompts])
        c, i, f = [extract_rating(r) for r in results]

        # Any None means judge failed for that dimension — mark whole sample as failed
        if c is None or i is None or f is None:
            return None
        return {"concept_score": c, "instruction_score": i,
                "fluency_score": f, "harmonic_mean": harmonic_mean([c, i, f])}


async def run_judge(samples, client, model, concurrency=8, save_path=None,
                    save_every=50):
    sem = asyncio.Semaphore(concurrency)
    judged = []

    # Resume from checkpoint — but retry any samples that failed previously
    start_idx = 0
    retry_indices = []  # indices into `samples` that need retry
    if save_path and os.path.exists(save_path):
        with open(save_path) as f:
            checkpoint = json.load(f)
        judged = checkpoint.get("samples", [])
        start_idx = len(judged)
        # Find failed samples to retry
        retry_indices = [i for i, j in enumerate(judged)
                         if j.get("scores") is None]
        print(f"Resuming from {start_idx}/{len(samples)} "
              f"({len(retry_indices)} failed samples will be retried)", flush=True)

    # First: retry failed samples (in place)
    if retry_indices:
        retry_tasks = [
            asyncio.create_task(
                judge_one(client, model, samples[i]["concept"], samples[i]["input"],
                          samples[i]["generation_steered"], sem))
            for i in retry_indices
        ]
        n_retry_ok = 0
        for local_i, task in enumerate(retry_tasks):
            result = await task
            orig_i = retry_indices[local_i]
            if result is not None:
                judged[orig_i] = {**samples[orig_i], "scores": result}
                n_retry_ok += 1
        print(f"Retry: {n_retry_ok}/{len(retry_indices)} recovered", flush=True)
        if save_path:
            _save(save_path, judged)

    remaining = samples[start_idx:]
    if not remaining:
        return judged

    # Submit all remaining samples — semaphore handles concurrency
    tasks = [
        asyncio.create_task(
            judge_one(client, model, s["concept"], s["input"],
                      s["generation_steered"], sem))
        for s in remaining
    ]

    n_failed = 0
    for local_i, task in enumerate(tasks):
        result = await task
        s = remaining[local_i]
        if result is None:
            n_failed += 1
            judged.append({**s, "scores": None, "judge_failed": True})
        else:
            judged.append({**s, "scores": result})

        i = start_idx + local_i
        if (i + 1) % save_every == 0:
            recent = [j for j in judged[-save_every:]
                      if j.get("scores") is not None]
            if recent:
                hm = np.mean([j["scores"]["harmonic_mean"] for j in recent])
                print(f"  [{i+1}/{len(samples)}] batch_hmean={hm:.3f} "
                      f"(ok={len(recent)}/{save_every}, total_failed={n_failed})",
                      flush=True)
            if save_path:
                _save(save_path, judged, model)

    if save_path:
        _save(save_path, judged)
    print(f"Done. {len(judged)} samples, {n_failed} judge failures.", flush=True)
    return judged


def _save(path, judged):
    """Checkpoint the raw per-sample judgments ONLY (the resume cache).

    Score aggregation is intentionally NOT done here: it is computed once at the
    end into a SEPARATE scores file (see ``write_scores``). This keeps streaming
    saves cheap and avoids repeatedly re-aggregating over growing partial data.
    """
    with open(path, "w") as f:
        json.dump({"samples": judged}, f, indent=2, ensure_ascii=False)


_SCORE_HINT = "Aggregate per-T scores (decoupled):  python scripts/score.py {cache}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-file", type=str, required=True)
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--api-key", type=str, default=None,
                        help="If omitted, uses OPENAI_API_KEY env var")
    parser.add_argument("--base-url", type=str, default=None,
                        help="Override base URL (for proxy / alternate endpoints)")
    parser.add_argument("--model", type=str, default="gpt-4o-mini")
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--max-samples", type=int, default=None)
    args = parser.parse_args()

    api_key = args.api_key or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        parser.error("Need --api-key or OPENAI_API_KEY env var")

    with open(args.results_file) as f:
        samples = json.load(f)
    if isinstance(samples, dict):
        samples = samples.get("samples", samples)
    if args.max_samples:
        samples = samples[:args.max_samples]

    output_path = args.output or args.results_file.replace(".json", "_4o_judged.json")
    print(f"Judging {len(samples)} samples with {args.model}", flush=True)
    print(f"Concurrency: {args.concurrency}, output: {output_path}", flush=True)

    client = AsyncOpenAI(api_key=api_key, base_url=args.base_url)
    judged = asyncio.run(run_judge(
        samples, client, args.model,
        concurrency=args.concurrency, save_path=output_path))

    ok = [j for j in judged if j.get("scores") is not None]
    print(f"\nJudged {len(ok)}/{len(judged)} ok ({len(judged) - len(ok)} failed/filtered).")
    print(f"Cache -> {output_path}")
    print(_SCORE_HINT.format(cache=output_path))


if __name__ == "__main__":
    main()
