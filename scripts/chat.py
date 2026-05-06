"""Interactive FLAS TUI — steer a language model with natural language concepts.

Usage:
    uv run python scripts/chat.py \
        --flow-ckpt checkpoints/<run_name>/<best_ckpt> \
        --flowtime 2.0 --n-steps 3

Commands during chat:
    /concept <text>       — change steering concept
    /flowtime <value>     — change steering strength (T)
    /steps <n>            — change Euler integration steps
    /off                  — disable steering (baseline mode)
    /on                   — re-enable steering
    /compare              — show both baseline and steered output
    /quit                 — exit
"""

import argparse
import sys

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except AttributeError:
    pass

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig


class InteractiveFlas:
    def __init__(self, args):
        # Auto-load training config from checkpoint dir if available
        import json
        from pathlib import Path
        cfg_path = Path(args.flow_ckpt).parent / "config.json"
        cfg = {}
        if cfg_path.exists():
            with open(cfg_path) as f:
                cfg = json.load(f)
            print(f"Loaded training config from {cfg_path}", flush=True)

        model_id = args.model_id or cfg.get("model_id", "google/gemma-2-2b-it")
        layer = args.layer if args.layer is not None else cfg.get("layer", 20)
        num_blocks = args.num_blocks if args.num_blocks is not None else cfg.get("num_blocks", 2)
        n_steps = args.n_steps if args.n_steps is not None else cfg.get("n_steps", 2)
        print(f"  model_id={model_id}, layer={layer}, num_blocks={num_blocks}, "
              f"n_steps={n_steps}", flush=True)

        print(f"Loading {model_id}...", flush=True)
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.llm = AutoModelForCausalLM.from_pretrained(
            model_id, torch_dtype=torch.bfloat16, device_map="cuda").eval()

        from flas.model import FlowFunction, ConceptEncoder
        config = AutoConfig.from_pretrained(model_id)
        self.flow_fn = FlowFunction(config, num_blocks=num_blocks,
                                    time_conditioned=True, layer_idx=layer,
                                    disable_cross_attn=cfg.get("disable_cross_attn", False),
                                    disable_self_attn=cfg.get("disable_self_attn", False),
                                    disable_mlp=cfg.get("disable_mlp", False))
        ckpt = torch.load(args.flow_ckpt, weights_only=True, map_location="cpu")
        self.flow_fn.load_state_dict(ckpt["flow_fn"] if "flow_fn" in ckpt else ckpt)
        self.flow_fn = self.flow_fn.to("cuda").eval()

        self.concept_enc = ConceptEncoder(model_id, num_layers=2).to("cuda")
        if "concept_enc" in ckpt:
            self.concept_enc.load_state_dict(ckpt["concept_enc"])

        self.layer = layer
        self.flowtime = args.flowtime
        self.n_steps = n_steps
        self.max_tokens = args.max_tokens
        self.temperature = args.temperature
        self.steering_on = True

        # Concept state
        self.concept_hidden = None
        self.concept_mask = None
        self.concept_text = None

        print("Ready!", flush=True)

    def set_concept(self, text):
        self.concept_text = text
        enc = self.tokenizer(text, return_tensors="pt", truncation=True,
                             max_length=64).to("cuda")
        with torch.no_grad():
            self.concept_hidden = self.concept_enc(enc.input_ids, enc.attention_mask)
            self.concept_mask = enc.attention_mask.float()
        print(f"  Concept set: \"{text}\"")

    @torch.no_grad()
    def generate(self, prompt, use_steering=True):
        messages = [{"role": "user", "content": prompt}]
        formatted = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True)
        enc = self.tokenizer(formatted, return_tensors="pt",
                             add_special_tokens=False).to("cuda")
        input_ids = enc.input_ids
        attention_mask = enc.attention_mask
        prompt_len = input_ids.shape[1]

        hook_handle = None
        if use_steering and self.concept_hidden is not None:
            nonlocal_state = {"is_prefill": True, "past_len": 0,
                              "sa_caches": [None] * self.n_steps,
                              "padding_mask": attention_mask.float()}

            def hook(module, input, output):
                is_tuple = isinstance(output, tuple)
                h_orig = output[0] if is_tuple else output
                h = h_orig.float()
                bsz = h.size(0)
                dt = self.flowtime / self.n_steps
                state = nonlocal_state

                for k in range(self.n_steps):
                    t_k = torch.full((bsz,), k * dt, device=h.device)
                    if state["is_prefill"]:
                        v, kv_caches = self.flow_fn(
                            h, self.concept_hidden, self.concept_mask,
                            t=t_k,
                            padding_mask=state["padding_mask"],
                            use_cache=True, past_len=0)
                        state["sa_caches"][k] = kv_caches
                    else:
                        v, kv_caches = self.flow_fn(
                            h, self.concept_hidden, self.concept_mask,
                            t=t_k,
                            self_attn_caches=state["sa_caches"][k],
                            padding_mask=state["padding_mask"],
                            use_cache=True, past_len=state["past_len"])
                        state["sa_caches"][k] = kv_caches
                    h = h + dt * v
                h_out = h.to(h_orig.dtype)
                return (h_out,) + output[1:] if is_tuple else h_out

            hook_handle = self.llm.model.layers[self.layer].register_forward_hook(hook)

        # Prefill
        out = self.llm(input_ids, attention_mask=attention_mask, use_cache=True)
        past_kv = out.past_key_values
        next_logits = out.logits[:, -1, :]

        if hook_handle:
            nonlocal_state["is_prefill"] = False
            nonlocal_state["past_len"] = prompt_len

        generated = input_ids.clone()

        for _ in range(self.max_tokens):
            if self.temperature > 0:
                probs = torch.softmax(next_logits / self.temperature, dim=-1)
                next_token = torch.multinomial(probs, 1)
            else:
                next_token = next_logits.argmax(dim=-1, keepdim=True)

            generated = torch.cat([generated, next_token], dim=1)

            if next_token.item() == self.tokenizer.eos_token_id:
                break

            # Stream token
            clean = self.tokenizer.decode(next_token[0], skip_special_tokens=True)
            if clean:
                print(clean, end="", flush=True)

            attention_mask = torch.cat(
                [attention_mask, torch.ones(1, 1, device="cuda", dtype=torch.long)],
                dim=1)

            if hook_handle:
                nonlocal_state["padding_mask"] = attention_mask.float()

            out = self.llm(next_token, attention_mask=attention_mask,
                           past_key_values=past_kv, use_cache=True)
            past_kv = out.past_key_values
            next_logits = out.logits[:, -1, :]

            if hook_handle:
                nonlocal_state["past_len"] += 1

        if hook_handle:
            hook_handle.remove()

        print()  # newline after streaming
        return self.tokenizer.decode(generated[0, input_ids.shape[1]:],
                                     skip_special_tokens=True)

    def print_status(self):
        concept = self.concept_text or "(none)"
        state = "ON" if self.steering_on else "OFF"
        print(f"\n  Concept:  {concept}")
        print(f"  Steering: {state}  |  flowtime={self.flowtime}  |  n_steps={self.n_steps}")
        print(f"  Temp:     {self.temperature}  |  max_tokens={self.max_tokens}\n")


def main():
    parser = argparse.ArgumentParser(description="Interactive FLAS chat")
    parser.add_argument("--model-id", type=str, default=None,
                        help="Override model_id (default: from ckpt config.json)")
    parser.add_argument("--layer", type=int, default=None,
                        help="Override layer (default: from ckpt config.json)")
    parser.add_argument("--flow-ckpt", type=str, required=True)
    parser.add_argument("--num-blocks", type=int, default=None,
                        help="Override num_blocks (default: from ckpt config.json)")
    parser.add_argument("--flowtime", type=float, default=2.0)
    parser.add_argument("--n-steps", type=int, default=None,
                        help="Override n_steps (default: from ckpt config.json)")
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.7)
    args = parser.parse_args()

    chat = InteractiveFlas(args)

    print("\n" + "=" * 60)
    print("  FLAS Interactive TUI")
    print("=" * 60)
    print("  Set a steering concept first with /concept <text>")
    print("  Then type your message to chat with the steered model.")
    print("  Type /help for commands, /quit to exit.")
    print("=" * 60 + "\n")

    while True:
        try:
            user_input = input("\033[1;36mYou:\033[0m ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye!")
            break

        if not user_input:
            continue

        # Handle commands
        if user_input.startswith("/"):
            parts = user_input.split(maxsplit=1)
            cmd = parts[0].lower()
            arg = parts[1] if len(parts) > 1 else ""

            if cmd == "/quit" or cmd == "/exit":
                print("Bye!")
                break
            elif cmd == "/concept":
                if not arg:
                    print("  Usage: /concept <text>")
                else:
                    chat.set_concept(arg)
            elif cmd == "/flowtime":
                try:
                    chat.flowtime = float(arg)
                    print(f"  Dose set to {chat.flowtime}")
                except ValueError:
                    print("  Usage: /flowtime <number>")
            elif cmd == "/steps":
                try:
                    chat.n_steps = int(arg)
                    print(f"  Steps set to {chat.n_steps}")
                except ValueError:
                    print("  Usage: /steps <number>")
            elif cmd == "/temp":
                try:
                    chat.temperature = float(arg)
                    print(f"  Temperature set to {chat.temperature}")
                except ValueError:
                    print("  Usage: /temp <number>")
            elif cmd == "/off":
                chat.steering_on = False
                print("  Steering OFF (baseline mode)")
            elif cmd == "/on":
                chat.steering_on = True
                print("  Steering ON")
            elif cmd == "/compare":
                if not arg:
                    print("  Usage: /compare <your message>")
                else:
                    print("\n\033[1;33m[Baseline]\033[0m ", end="")
                    chat.generate(arg, use_steering=False)
                    print("\033[1;32m[Steered]\033[0m  ", end="")
                    chat.generate(arg, use_steering=True)
            elif cmd == "/status":
                chat.print_status()
            elif cmd == "/help":
                print("  /concept <text>   — set steering concept")
                print("  /flowtime <value>     — set flowtime / steering strength (default: 2.0)")
                print("  /steps <n>        — set Euler steps (default: 2)")
                print("  /temp <value>     — set temperature")
                print("  /off / /on        — toggle steering")
                print("  /compare <msg>    — show baseline vs steered")
                print("  /status           — show current settings")
                print("  /quit             — exit")
            else:
                print(f"  Unknown command: {cmd}. Type /help for commands.")
            continue

        # Regular chat
        if chat.concept_hidden is None:
            print("  Set a concept first: /concept <text>")
            continue

        print("\033[1;32mModel:\033[0m ", end="")
        chat.generate(user_input, use_steering=chat.steering_on)


if __name__ == "__main__":
    main()
