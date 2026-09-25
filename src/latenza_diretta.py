"""Latenza 'alla Laya' per i Gemma addestrati: niente server, niente grammatica.
Un passaggio in avanti in PyTorch sulla 4070 per le DUE domande insieme (batch 2,
padding a sinistra), si leggono i logit delle lettere valide all'ultima posizione.
Stessi 1.416 pacchetti di gioco di latenza.py. Uso: latenza_diretta.py <modello_hf> <tag> <run...>"""
import json, sys, time, statistics, torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForImageTextToText
from serve_llm import costruisci, LETTERE
import os
base, tag, runs = sys.argv[1], sys.argv[2], sys.argv[3:]
GPU = int(os.environ.get("GPU", "0"))   # indice CUDA della scheda
tok = AutoTokenizer.from_pretrained(base); tok.padding_side = "left"
try:
    m = AutoModelForCausalLM.from_pretrained(base, dtype=torch.bfloat16, device_map={"": GPU})
except Exception:
    m = AutoModelForImageTextToText.from_pretrained(base, dtype=torch.bfloat16, device_map={"": GPU})
m.eval()
lett = [tok(LETTERE[i], add_special_tokens=False)["input_ids"][0] for i in range(26)]
pac = [json.loads(l)["packet"] for r in runs for l in open(r + "/decisions.jsonl") if '"packet"' in l]
t = []
with torch.inference_mode():
    for i, p in enumerate(pac):
        a = time.perf_counter()
        testi, nopz = [], []
        for n, q in p["questions"].items():
            _, ch, msg, _ = costruisci(p["state"], {n: q})
            testi.append(tok.apply_chat_template(msg, tokenize=False, add_generation_prompt=True, enable_thinking=False))
            nopz.append(len(ch[n]))
        x = tok(testi, return_tensors="pt", padding=True, add_special_tokens=False).to(f"cuda:{GPU}")
        lo = m(**x, logits_to_keep=1).logits[:, -1, :]
        scelte = [int(lo[j, lett[:k]].argmax()) for j, k in enumerate(nopz)]
        torch.cuda.synchronize()
        if i >= 10:
            t.append((time.perf_counter() - a) * 1000)
q = lambda f: sorted(t)[min(len(t) - 1, int(f * len(t)))]
print(f"{tag:14} diretto n={len(t)} p50={q(.5):.1f} p90={q(.9):.1f} p99={q(.99):.1f} ms  decisioni/s={1000/statistics.mean(t):.1f}")
