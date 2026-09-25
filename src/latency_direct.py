"""Laya-style latency for the fine-tuned Gemma models: no server, no grammar.
One PyTorch forward pass for BOTH questions together (batch of 2, left padding);
the logits of the valid letters are read at the last position. Same recorded game
packets as latency.py.
Usage: GPU=0 latency_direct.py <merged hf model dir> <label> <run dir> [run dir ...]"""
import json, os, statistics, sys, time, torch
from transformers import AutoModelForCausalLM, AutoModelForImageTextToText, AutoTokenizer
from serve_llm import LETTERS, build_prompt
base, tag, runs = sys.argv[1], sys.argv[2], sys.argv[3:]
GPU = int(os.environ.get("GPU", "0"))   # CUDA index of the GPU
tok = AutoTokenizer.from_pretrained(base); tok.padding_side = "left"
try:
    m = AutoModelForCausalLM.from_pretrained(base, dtype=torch.bfloat16, device_map={"": GPU})
except Exception:
    m = AutoModelForImageTextToText.from_pretrained(base, dtype=torch.bfloat16, device_map={"": GPU})
m.eval()
letter_ids = [tok(LETTERS[i], add_special_tokens=False)["input_ids"][0] for i in range(26)]
packets = [json.loads(l)["packet"] for r in runs for l in open(r + "/decisions.jsonl") if '"packet"' in l]
t = []
with torch.inference_mode():
    for i, p in enumerate(packets):
        a = time.perf_counter()
        texts, n_opts = [], []
        for n, q in p["questions"].items():
            _, keys, msgs, _ = build_prompt(p["state"], {n: q})
            texts.append(tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False))
            n_opts.append(len(keys[n]))
        x = tok(texts, return_tensors="pt", padding=True, add_special_tokens=False).to(f"cuda:{GPU}")
        lo = m(**x, logits_to_keep=1).logits[:, -1, :]
        choices = [int(lo[j, letter_ids[:k]].argmax()) for j, k in enumerate(n_opts)]
        torch.cuda.synchronize()
        if i >= 10:
            t.append((time.perf_counter() - a) * 1000)
q = lambda f: sorted(t)[min(len(t) - 1, int(f * len(t)))]
print(f"{tag:14} direct n={len(t)} p50={q(.5):.1f} p90={q(.9):.1f} p99={q(.99):.1f} ms  decisions/s={1000/statistics.mean(t):.1f}")
