"""Train FLAS with multistep Euler + LM loss.

Usage:
    python -m flas.train \
        --output-dir checkpoints --run-name run1 \
        --n-steps 3 --T-min 0.5 --T-max 2.0 --total-steps 50000
"""

import argparse
import json
import math
from pathlib import Path
from functools import partial

import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
from transformers import AutoTokenizer, AutoModelForCausalLM

from flas.model import build_flow_model, get_text_decoder, build_chat_prompt


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class SteeringDataset(Dataset):
    def __init__(self, df):
        self.samples = [
            (row["input"], row["output"], row["output_concept"], row["concept_id"])
            for _, row in df.iterrows()
        ]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


ALPACA_TEMPLATE = "### Instruction:\n{input}\n\n### Response:\n"


def collate_fn(batch, tokenizer, max_len, concept_max_len, prompt_format="chat"):
    input_texts, output_texts, concept_texts, cids = zip(*batch)

    # Tokenize prompt and output separately, then concatenate ids
    # This avoids BPE boundary ambiguity at the prompt/output seam.
    # Degenerate samples are SKIPPED (not asserted): rare flas-46k prompts
    # exceed max_len leaving no room for output, and empty concepts would
    # collapse cross-attention. Skipping drops <0.1% of samples and keeps
    # training robust to outliers across tokenizers/chat-templates.
    full_ids_list = []
    prompt_lens = []
    keep_concepts = []
    keep_cids = []
    for inp, out, ct, cid in zip(input_texts, output_texts, concept_texts, cids):
        if not ct or not ct.strip():
            continue  # skip empty concept text
        if prompt_format == "alpaca":
            # Base models: minimal instruction wrapper, no chat template.
            # add_special_tokens=True prepends a single BOS.
            prompt_ids = tokenizer(
                ALPACA_TEMPLATE.format(input=inp), add_special_tokens=True).input_ids
        else:
            # Chat models. build_chat_prompt forces non-thinking mode for Qwen3-style
            # hybrid-reasoning models (empty <think></think>), matching the no-reasoning
            # targets and the eval-time prompt.
            prompt_enc = build_chat_prompt(tokenizer, inp, tokenize=True)
            prompt_ids = prompt_enc.input_ids if hasattr(prompt_enc, 'input_ids') else prompt_enc
        output_ids = tokenizer(out, add_special_tokens=False).input_ids
        full_ids = (prompt_ids + output_ids)[:max_len]
        prompt_len = min(len(prompt_ids), len(full_ids))
        # Skip if no prompt, or prompt fills max_len leaving no output token
        # (labels would be all -100 -> NaN loss).
        if prompt_len <= 0 or prompt_len >= len(full_ids):
            continue
        full_ids_list.append(full_ids)
        prompt_lens.append(prompt_len)
        keep_concepts.append(ct)
        keep_cids.append(cid)

    if not full_ids_list:
        raise RuntimeError("collate_fn: entire batch was skipped (all prompts "
                           "exceeded max_len) — increase --max-len")
    concept_texts = keep_concepts
    cids = keep_cids

    # Right-pad to max length in batch
    enc = tokenizer.pad(
        {"input_ids": full_ids_list}, return_tensors="pt", padding=True)
    input_ids = enc.input_ids
    attention_mask = enc.attention_mask

    labels = input_ids.clone()
    prompt_mask = torch.zeros_like(attention_mask)
    for i, plen in enumerate(prompt_lens):
        labels[i, :plen] = -100
        prompt_mask[i, :plen] = 1
    prompt_mask = prompt_mask * attention_mask  # exclude padding
    labels[attention_mask == 0] = -100  # mask padding in loss

    concept_enc = tokenizer(list(concept_texts), return_tensors="pt", padding=True,
                            truncation=True, max_length=concept_max_len)

    return (input_ids, attention_mask, labels, prompt_mask,
            concept_enc.input_ids, concept_enc.attention_mask,
            torch.tensor(list(cids), dtype=torch.long))


# ---------------------------------------------------------------------------
# Lightning Module
# ---------------------------------------------------------------------------

class FlasModule(pl.LightningModule):
    def __init__(self, args):
        super().__init__()
        self.save_hyperparameters(vars(args))
        self.args = args

        # LLM (frozen)
        self.llm = AutoModelForCausalLM.from_pretrained(
            args.model_id, torch_dtype=torch.bfloat16)
        self.llm.eval()
        for p in self.llm.parameters():
            p.requires_grad = False

        # FlowFunction + ConceptEncoder
        self.flow_fn, self.concept_enc = build_flow_model(
            args.model_id, args.layer, num_blocks=args.num_blocks,
            time_conditioned=True,
            init_from_gemma=not args.no_gemma_mlp_init,
            disable_cross_attn=getattr(args, 'disable_cross_attn', False),
            disable_self_attn=getattr(args, 'disable_self_attn', False),
            disable_mlp=getattr(args, 'disable_mlp', False))

        if args.resume:
            print(f"Resuming from {args.resume}...", flush=True)
            ckpt = torch.load(args.resume, weights_only=True, map_location="cpu")
            self.flow_fn.load_state_dict(ckpt["flow_fn"])
            if "concept_enc" in ckpt:
                self.concept_enc.load_state_dict(ckpt["concept_enc"])
                print("  Loaded concept_enc from checkpoint")
            else:
                print("  No concept_enc in checkpoint — using fresh frozen Gemma")

        if not getattr(args, 'unfreeze_concept_enc', False):
            for p in self.concept_enc.parameters():
                p.requires_grad = False
            print("Concept encoder fully frozen")
        else:
            # Train concept encoder but keep the 256k-token embed_tokens frozen
            for p in self.concept_enc.parameters():
                p.requires_grad = True
            for p in self.concept_enc.embed_tokens.parameters():
                p.requires_grad = False

        self.automatic_optimization = False

    def forward_with_flow(self, input_ids, attention_mask, labels,
                          concept_input_ids, concept_attention_mask, T_total):
        concept_hidden = self.concept_enc(concept_input_ids, concept_attention_mask)
        velocity_capture = {}
        n_steps = self.args.n_steps

        concept_mask_f = concept_attention_mask.float()
        attn_mask_f = attention_mask.float()

        def hook(module, input, output):
            is_tuple = isinstance(output, tuple)
            h_orig = output[0] if is_tuple else output
            h = h_orig.float()
            bsz = h.size(0)
            dt = T_total / n_steps
            last_v = None
            for k in range(n_steps):
                t_k = torch.full((bsz,), k * dt, device=h.device)
                v, _ = self.flow_fn(
                    h, concept_hidden, concept_mask_f,
                    t=t_k, padding_mask=attn_mask_f)
                h = h + dt * v
                last_v = v
            velocity_capture["v"] = last_v
            m = attn_mask_f.unsqueeze(-1)  # [bsz, seq, 1]
            n_real = m.sum(dim=1).clamp(min=1)        # [bsz, 1]
            v_per = (last_v.detach().norm(dim=-1, keepdim=True) * m).sum(dim=1) / n_real
            h_per = (h.detach().norm(dim=-1, keepdim=True) * m).sum(dim=1) / n_real
            velocity_capture["v_norm"] = v_per.mean()
            velocity_capture["h_norm"] = h_per.mean()
            h_out = h.to(h_orig.dtype)
            return (h_out,) + output[1:] if is_tuple else h_out

        handle = get_text_decoder(self.llm).layers[self.args.layer].register_forward_hook(hook)
        outputs = self.llm(input_ids=input_ids, attention_mask=attention_mask,
                           labels=labels)
        handle.remove()
        return outputs.loss, velocity_capture

    def compute_diversity_loss(self, velocity, concept_ids, attention_mask=None):
        if velocity is None:
            return torch.tensor(0.0, device=self.device)
        # Mask out padding positions from the average (the MLP produces
        # non-trivial output at padding, contaminating v_pooled otherwise).
        if attention_mask is not None:
            mask = attention_mask.float().unsqueeze(-1)   # [bsz, seq, 1]
            v_pooled = (velocity * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        else:
            v_pooled = velocity.mean(dim=1)
        if v_pooled.shape[0] < 2:
            return torch.tensor(0.0, device=self.device)
        v_norm = F.normalize(v_pooled, dim=-1)
        sim = torch.matmul(v_norm, v_norm.t())
        diff_mask = (concept_ids.unsqueeze(0) != concept_ids.unsqueeze(1)).float()
        if diff_mask.sum() == 0:
            return torch.tensor(0.0, device=self.device)
        return (sim * diff_mask).sum() / diff_mask.sum()

    def sample_T(self):
        return float(torch.rand(1).item() * (
            self.args.T_max - self.args.T_min) + self.args.T_min)

    def training_step(self, batch, batch_idx):
        opt = self.optimizers()
        sch = self.lr_schedulers()

        input_ids, attn_mask, labels, _prompt_mask, c_ids, c_mask, cids = batch
        T_total = self.sample_T()

        lm_loss, velocity_capture = self.forward_with_flow(
            input_ids, attn_mask, labels, c_ids, c_mask, T_total)
        velocity = velocity_capture.get("v")
        div_loss = self.compute_diversity_loss(velocity, cids, attn_mask)
        loss = (lm_loss + self.args.div_weight * div_loss) / self.args.grad_accum

        self.manual_backward(loss)

        if (batch_idx + 1) % self.args.grad_accum == 0:
            trainable = [p for p in self.flow_fn.parameters() if p.requires_grad]
            trainable += [p for p in self.concept_enc.parameters() if p.requires_grad]
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            opt.step()
            sch.step()
            opt.zero_grad()

        self.log("train/lm_loss", lm_loss, prog_bar=True)
        self.log("train/div_loss", div_loss)
        self.log("train/loss", loss)
        self.log("train/lr", sch.get_last_lr()[0])
        if "v_norm" in velocity_capture:
            v_norm = velocity_capture["v_norm"]
            h_norm = velocity_capture["h_norm"]
            self.log("train/v_norm", v_norm)
            self.log("train/vh_ratio", v_norm / h_norm)

    def validation_step(self, batch, batch_idx):
        input_ids, attn_mask, labels, _prompt_mask, c_ids, c_mask, cids = batch
        lm_loss, _ = self.forward_with_flow(
            input_ids, attn_mask, labels, c_ids, c_mask, T_total=1.0)
        self.log("val/loss", lm_loss, prog_bar=True, sync_dist=True)
        self.log("val_loss", lm_loss)
        return lm_loss

    def _save_pt(self, path, step, val_loss):
        # Save in bf16 to match the base model dtype and keep distribution
        # files small. Master weights live in fp32 under bf16-mixed precision
        # (Lightning), so a one-shot cast here is safe.
        flow_sd = {k: v.to(torch.bfloat16) for k, v in self.flow_fn.state_dict().items()}
        ckpt = {
            "flow_fn": flow_sd,
            "step": step,
            "val_loss": val_loss,
        }
        if getattr(self.args, 'unfreeze_concept_enc', False):
            enc_sd = {k: v.to(torch.bfloat16) for k, v in self.concept_enc.state_dict().items()}
            ckpt["concept_enc"] = enc_sd
        torch.save(ckpt, path)

    def on_validation_epoch_end(self):
        val_loss = self.trainer.callback_metrics.get("val_loss", None)
        if val_loss is None:
            return
        combined = val_loss.item() if isinstance(val_loss, torch.Tensor) else val_loss

        raw_step = self.global_step
        if raw_step == 0:
            return  # skip sanity check validation
        # Convert to training steps (global_step is optimizer steps in manual mode)
        step = raw_step * self.args.grad_accum
        ckpt_dir = Path(self.args.output_dir) / self.args.run_name

        # Always save last eval checkpoint (with score in filename)
        for old in ckpt_dir.glob("last_step*_val*.pt"):
            old.unlink()
        self._save_pt(ckpt_dir / f"last_step{step}_val{combined:.4f}.pt", step, combined)

        # Keep top-3 best checkpoints by val_loss
        if not hasattr(self, '_top3'):
            existing = sorted(ckpt_dir.glob("best_step*_val*.pt"))
            self._top3 = []
            for p in existing:
                vl = float(p.stem.split("_val")[1])
                self._top3.append((vl, p))
            self._top3.sort(key=lambda x: x[0])

        if len(self._top3) < 3 or combined < self._top3[-1][0]:
            path = ckpt_dir / f"best_step{step}_val{combined:.4f}.pt"
            self._save_pt(path, step, combined)
            self._top3.append((combined, path))
            self._top3.sort(key=lambda x: x[0])
            if len(self._top3) > 3:
                _, worst_path = self._top3.pop()
                if worst_path.exists():
                    worst_path.unlink()
            print(f"  [step {step}] top-3 best val={combined:.4f} "
                  f"(kept: {[f'{v:.4f}' for v, _ in self._top3]})", flush=True)

    def configure_optimizers(self):
        # Only include trainable params (excludes frozen concept_enc.embed_tokens)
        enc_params = [p for p in self.concept_enc.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW([
            {"params": list(self.flow_fn.parameters()), "lr": self.args.lr},
            {"params": enc_params, "lr": self.args.enc_lr},
        ], weight_decay=0.01)

        def lr_lambda(step):
            # step = scheduler steps (= optimizer steps, every grad_accum batches)
            # Scale to training steps for correct pacing
            s = step * self.args.grad_accum
            if s < self.args.warmup_steps:
                return s / max(self.args.warmup_steps, 1)
            progress = (s - self.args.warmup_steps) / max(
                self.args.total_steps - self.args.warmup_steps, 1)
            return 0.5 * (1.0 + math.cos(math.pi * progress))

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        return [optimizer], [{"scheduler": scheduler, "interval": "step"}]



# ---------------------------------------------------------------------------
# Data preparation
# ---------------------------------------------------------------------------

def prepare_data(args, tokenizer):
    """Split data for training and validation.
    - Optional concept-level holdout (val_n_concepts > 0, for c16k)
    - Fixed n_val_samples random samples held out for eval
    Returns: train_loader, val_loader
    """
    # Resolve the training parquet across dataset layouts:
    #   AxBench prod dirs -> train_data.parquet  (cols: input/output/output_concept/category/concept_id)
    #   flas-concept46k   -> train.parquet       (cols: input/output/concept/concept_id, no category)
    for cand in ("train_data.parquet", "train/data.parquet", "train.parquet"):
        train_parquet = Path(args.data_dir) / cand
        if train_parquet.exists():
            break
    df = pd.read_parquet(train_parquet)
    # Normalize schema: flas-46k names the steering-concept column `concept`.
    if "output_concept" not in df.columns and "concept" in df.columns:
        df = df.rename(columns={"concept": "output_concept"})
    # flas-46k has no positive/negative `category` split — all rows are positive
    # steering exemplars.
    if "category" in df.columns:
        pos_df = df[df["category"] == "positive"].reset_index(drop=True)
    else:
        pos_df = df.reset_index(drop=True)
    all_cids = sorted(pos_df["concept_id"].unique())

    # Concept-level holdout. Prefer an explicit id list (e.g. flas-46k uses a
    # fixed held-out set so eval concepts never leak into training); otherwise
    # fall back to a deterministic seed-42 sample of `val_n_concepts`.
    if getattr(args, "heldout_ids_file", None):
        ho_cids = set(json.load(open(args.heldout_ids_file)))
        train_df = pos_df[~pos_df["concept_id"].isin(ho_cids)].reset_index(drop=True)
        print(f"Held out {len(ho_cids)} concepts from {args.heldout_ids_file}")
    else:
        n_ho = min(args.val_n_concepts, len(all_cids) - 1) if args.val_n_concepts > 0 else 0
        torch.manual_seed(42)
        perm = torch.randperm(len(all_cids)).tolist()
        if n_ho > 0:
            ho_cids = set(all_cids[i] for i in perm[:n_ho])
            train_df = pos_df[~pos_df["concept_id"].isin(ho_cids)].reset_index(drop=True)
            print(f"Held out {n_ho} concepts for evaluation")
        else:
            train_df = pos_df

    # Hold out fixed n samples for validation
    n_val = min(args.n_val_samples, len(train_df) // 10)
    val_idx = torch.randperm(len(train_df))[:n_val].tolist()
    val_mask = torch.zeros(len(train_df), dtype=torch.bool)
    val_mask[val_idx] = True

    val_df = train_df.iloc[val_idx].reset_index(drop=True)
    train_df = train_df.iloc[~val_mask.numpy()].reset_index(drop=True)

    print(f"Train: {len(train_df)} samples ({train_df.concept_id.nunique()} concepts)")
    print(f"Val:   {len(val_df)} samples")

    collate = partial(collate_fn, tokenizer=tokenizer, max_len=args.max_len,
                      concept_max_len=args.concept_max_len,
                      prompt_format=getattr(args, "prompt_format", "chat"))

    train_loader = DataLoader(
        SteeringDataset(train_df), batch_size=args.batch_size,
        shuffle=True, drop_last=True, collate_fn=collate,
        num_workers=args.num_workers, persistent_workers=args.num_workers > 0,
        pin_memory=True)
    val_loader = DataLoader(
        SteeringDataset(val_df), batch_size=args.batch_size,
        shuffle=True, collate_fn=collate,
        num_workers=min(2, args.num_workers),
        persistent_workers=args.num_workers > 0,
        pin_memory=True)

    return train_loader, val_loader


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str, required=True)
    parser.add_argument("--model-id", type=str, default="google/gemma-2-2b-it")
    parser.add_argument("--layer", type=int, default=20)
    parser.add_argument("--output-dir", type=str, default="checkpoints")
    parser.add_argument("--run-name", type=str, default="run1")
    parser.add_argument("--num-blocks", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--enc-lr", type=float, default=1e-5)
    parser.add_argument("--div-weight", type=float, default=0.1)
    parser.add_argument("--total-steps", type=int, default=80000)
    parser.add_argument("--warmup-steps", type=int, default=2000)
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--val-every", type=int, default=500,
                        help="Validate every N steps")
    parser.add_argument("--val-n-concepts", type=int, default=0,
                        help="Number of concepts held out (0=none, >0 for c16k)")
    parser.add_argument("--heldout-ids-file", type=str, default=None,
                        help="JSON list of concept_ids to hold out from training "
                             "(overrides --val-n-concepts; used for flas-46k fixed holdout)")
    parser.add_argument("--n-val-samples", type=int, default=100,
                        help="Number of samples held out for validation")
    parser.add_argument("--max-len", type=int, default=256)
    parser.add_argument("--concept-max-len", type=int, default=64)
    parser.add_argument("--T-max", type=float, default=2.0)
    parser.add_argument("--T-min", type=float, default=0.5)
    parser.add_argument("--n-steps", type=int, default=3)
    parser.add_argument("--no-gemma-mlp-init", action="store_true")
    parser.add_argument("--unfreeze-concept-enc", action="store_true",
                        help="Unfreeze concept encoder for fine-tuning (frozen by default)")
    parser.add_argument("--disable-cross-attn", action="store_true",
                        help="Disable cross-attention (activation queries concept) in FlowBlocks")
    parser.add_argument("--disable-self-attn", action="store_true",
                        help="Disable causal self-attention in FlowBlocks")
    parser.add_argument("--disable-mlp", action="store_true",
                        help="Disable the MLP in FlowBlocks")
    parser.add_argument("--prompt-format", choices=["chat", "alpaca"], default="chat",
                        help="chat = tokenizer chat template (instruct models); "
                             "alpaca = '### Instruction/### Response' wrapper (base models, no chat template)")
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--save-every-n-evals", type=int, default=10,
                        help="Save periodic checkpoint every N eval rounds")
    parser.add_argument("--val-batches", type=int, default=100,
                        help="Max batches per val dataloader (avoid slow val on large datasets)")
    parser.add_argument("--num-workers", type=int, default=4,
                        help="DataLoader worker processes (0 = single-thread main process)")
    parser.add_argument("--precision", type=str, default="bf16-mixed",
                        help="Lightning trainer precision: 'bf16-mixed' (default) "
                             "or '32' (fp32, matches the original flowsteer-qwen run)")
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    train_loader, val_loader = prepare_data(args, tokenizer)

    model = FlasModule(args)

    ckpt_dir = Path(args.output_dir) / args.run_name
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    with open(ckpt_dir / "config.json", "w") as f:
        json.dump(vars(args), f, indent=2)

    # TensorBoard logger
    tb_logger = TensorBoardLogger(
        save_dir=str(ckpt_dir), name="tb_logs", version="")

    # Callbacks (checkpointing handled manually in on_validation_epoch_end)
    early_stop = EarlyStopping(
        monitor="val_loss",
        patience=args.patience,
        mode="min",
    )

    # NOTE on "step" semantics: in this codebase a "step" denotes an OPTIMIZER step
    # (one parameter update) for `total_steps`, `warmup_steps`, `val_every`, and
    # Lightning's `global_step`. But filenames/logs (e.g. `best_step{N}_val{val}.pt`,
    # `[step N]`) show BATCH counts (= optimizer_step * grad_accum) for readability.
    # That's why training appears to "exceed" total_steps in the log: with
    # grad_accum=2 and total_steps=80000, training stops at ~160000 batches.
    trainer = pl.Trainer(
        max_steps=args.total_steps,
        val_check_interval=args.val_every,
        limit_val_batches=args.val_batches,
        callbacks=[early_stop],
        logger=tb_logger,
        default_root_dir=str(ckpt_dir),
        enable_progress_bar=True,
        gradient_clip_val=None,
        enable_checkpointing=False,  # we handle checkpointing manually
        accelerator="gpu",
        devices=1,
        precision=args.precision,
        log_every_n_steps=50,
    )

    trainer.fit(model, train_loader, val_loader)

    # Save final standalone checkpoint (bf16 to match base model dtype).
    flow_sd = {k: v.to(torch.bfloat16) for k, v in model.flow_fn.state_dict().items()}
    final_ckpt = {"flow_fn": flow_sd}
    if getattr(args, 'unfreeze_concept_enc', False):
        enc_sd = {k: v.to(torch.bfloat16) for k, v in model.concept_enc.state_dict().items()}
        final_ckpt["concept_enc"] = enc_sd
    torch.save(final_ckpt, ckpt_dir / "final.pt")
    print(f"Done. Saved to {ckpt_dir}", flush=True)


if __name__ == "__main__":
    main()
