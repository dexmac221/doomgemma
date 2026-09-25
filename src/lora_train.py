"""LoRA fine-tuning of Gemma (4 E2B or 3 270M) on the same decisions Laya v3 was trained on.

- data: doomLaya's training/v3/train.json (979) and validation.json (130), the SAME as Laya v3;
- prompt: built by serve_llm.build_prompt, i.e. identical to the in-game one;
- loss: ONLY on the answer letter (the rest of the prompt is not learned);
- validation: among the valid letters the most probable wins, exactly as the grammar
  does in game; the best epoch (command + weapon accuracy) is kept, as for Laya.

Run it from the doomLaya directory (it reads training/v3/*.json).
Usage: lora_train.py --base <hf model dir> --out checkpoints/e2b-lora [--text-only] [--gpu 0]
"""
import argparse
import json
import os
import random
import time

import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoModelForImageTextToText, AutoTokenizer

from serve_llm import LETTERS, build_prompt


def device_map(base, gpu=0):
    """All computation on ONE GPU (--gpu); in RAM the per-layer embedding table
    (4.4 GB for E2B, read-only) and the audio/vision towers (unused here)."""
    from accelerate import init_empty_weights
    from transformers import AutoConfig
    with init_empty_weights():
        m = AutoModelForImageTextToText.from_config(AutoConfig.from_pretrained(base))
    d = {}
    for name, _ in m.model.named_children():
        if name == "language_model":
            for sub, _ in m.model.language_model.named_children():
                d["model.language_model." + sub] = "cpu" if sub == "embed_tokens_per_layer" else gpu
        else:
            d["model." + name] = "cpu" if ("audio" in name or "vision" in name) else gpu
    for name, _ in m.named_children():
        if name != "model":
            d[name] = gpu
    return d


def examples(tok, rows):
    out = []
    for r in rows:
        _, keys, messages, _ = build_prompt(r["state"], {r["kind"]: r["question"]})
        ks = keys[r["kind"]]
        prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True,
                                         enable_thinking=False)
        ids = tok(prompt, add_special_tokens=False)["input_ids"]
        letters = [tok(LETTERS[i], add_special_tokens=False)["input_ids"][0] for i in range(len(ks))]
        out.append({"ids": ids, "letters": letters, "target": ks.index(r["label"]), "kind": r["kind"]})
    return out


@torch.no_grad()
def evaluate(model, data, dev):
    model.eval()
    ok = {"command": 0, "weapon": 0}
    tot = {"command": 0, "weapon": 0}
    for e in data:
        x = torch.tensor([e["ids"]], device=dev)
        logit = model(input_ids=x, logits_to_keep=1).logits[0, -1, e["letters"]]
        ok[e["kind"]] += int(logit.argmax().item() == e["target"])
        tot[e["kind"]] += 1
    model.train()
    return {k: ok[k] / tot[k] for k in ok}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True, help="Hugging Face model directory")
    ap.add_argument("--out", required=True, help="output directory for the adapter")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--accum", type=int, default=8)
    ap.add_argument("--seed", type=int, default=771)
    ap.add_argument("--gpu", type=int, default=0, help="CUDA index of the GPU to train on")
    ap.add_argument("--text-only", action="store_true", help="text-only model (AutoModelForCausalLM), e.g. Gemma 3 270M")
    a = ap.parse_args()
    random.seed(a.seed); torch.manual_seed(a.seed)
    tok = AutoTokenizer.from_pretrained(a.base)
    train = examples(tok, json.load(open("training/v3/train.json")))
    val = examples(tok, json.load(open("training/v3/validation.json")))
    print(f"examples {len(train)} / {len(val)}, max length {max(len(e['ids']) for e in train)} tokens", flush=True)
    if a.text_only:      # small text-only model: everything on one GPU
        model = AutoModelForCausalLM.from_pretrained(a.base, dtype=torch.bfloat16, device_map={"": a.gpu})
    else:
        model = AutoModelForImageTextToText.from_pretrained(
            a.base, dtype=torch.bfloat16, device_map=device_map(a.base, a.gpu))
    model.config.use_cache = False
    # no gradient checkpointing: with Gemma 4 the recomputation does not match
    # (CheckpointError), and the examples are short (<= 417 tokens)
    model.enable_input_require_grads()
    cfg = LoraConfig(r=a.rank, lora_alpha=2 * a.rank, lora_dropout=0.05, bias="none",
                     target_modules=(r".*\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)" if a.text_only else
                                     r".*language_model.*\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)"))
    model = get_peft_model(model, cfg)
    model.print_trainable_parameters()
    dev = model.get_input_embeddings().weight.device
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=a.lr, weight_decay=0.01)
    base = evaluate(model, val, dev)
    print("before training:", base, flush=True)
    best, history = sum(base.values()), [{"epoch": 0, "validation": base}]
    for ep in range(1, a.epochs + 1):
        random.shuffle(train)
        t0, loss_sum = time.time(), 0.0
        for i, e in enumerate(train):
            x = torch.tensor([e["ids"]], device=dev)
            logit = model(input_ids=x, logits_to_keep=1).logits[0, -1, e["letters"]].float()
            loss = torch.nn.functional.cross_entropy(logit[None], torch.tensor([e["target"]], device=logit.device))
            (loss / a.accum).backward()
            loss_sum += loss.item()
            if (i + 1) % a.accum == 0:
                torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
                opt.step(); opt.zero_grad()
            if (i + 1) % 200 == 0:
                print(f"  epoch {ep} {i+1}/{len(train)} mean loss {loss_sum/(i+1):.3f} ({time.time()-t0:.0f}s)", flush=True)
        v = evaluate(model, val, dev)
        history.append({"epoch": ep, "validation": v, "seconds": round(time.time() - t0, 1)})
        print(f"epoch {ep}: {v}  ({time.time()-t0:.0f}s)", flush=True)
        if sum(v.values()) > best:
            best = sum(v.values())
            model.save_pretrained(a.out)
            json.dump({"best_epoch": ep, "validation_accuracy": v}, open(os.path.join(a.out, "best_epoch.json"), "w"))
            print("  -> saved", flush=True)
    os.makedirs(a.out, exist_ok=True)
    json.dump(history, open(os.path.join(a.out, "training_history.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
