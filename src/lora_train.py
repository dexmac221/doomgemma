"""LoRA di Gemma 4 E2B sulle stesse decisioni con cui e' stato addestrato Laya v3.

- dati: training/v3/train.json (979) e validation.json (130), gli STESSI di Laya v3;
- prompt: costruito da serve_llm.costruisci, cioe' identico a quello della partita;
- perdita: SOLO sulla lettera della risposta (il resto del prompt non si impara);
- validazione: fra le lettere valide vince la piu' probabile, esattamente come fa la
  grammatica in gioco; si tiene l'epoca migliore (comando + arma), come Laya.

Uso: lora_train.py --base ~/llama_models/gemma4-e2b/hf --out checkpoints/e2b-lora
"""
import argparse
import json
import os
import random
import time

import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoModelForImageTextToText, AutoTokenizer

from serve_llm import LETTERE, costruisci


def mappa(base, gpu=0):
    """Tutto il calcolo su UNA scheda (--gpu); in RAM la tabella
    di embedding per strato (4,4 GB, solo letta) e le torri audio/visione (inutili qui)."""
    from accelerate import init_empty_weights
    from transformers import AutoConfig
    with init_empty_weights():
        m = AutoModelForImageTextToText.from_config(AutoConfig.from_pretrained(base))
    d = {}
    for nome, _ in m.model.named_children():
        if nome == "language_model":
            for sub, _ in m.model.language_model.named_children():
                d["model.language_model." + sub] = "cpu" if sub == "embed_tokens_per_layer" else gpu
        else:
            d["model." + nome] = "cpu" if ("audio" in nome or "vision" in nome) else gpu
    for nome, _ in m.named_children():
        if nome != "model":
            d[nome] = gpu
    return d


def esempi(tok, righe):
    out = []
    for r in righe:
        nomi, chiavi, messaggi, _ = costruisci(r["state"], {r["kind"]: r["question"]})
        ks = chiavi[r["kind"]]
        prompt = tok.apply_chat_template(messaggi, tokenize=False, add_generation_prompt=True,
                                         enable_thinking=False)
        ids = tok(prompt, add_special_tokens=False)["input_ids"]
        lettere = [tok(LETTERE[i], add_special_tokens=False)["input_ids"][0] for i in range(len(ks))]
        out.append({"ids": ids, "lettere": lettere, "target": ks.index(r["label"]), "kind": r["kind"]})
    return out


@torch.no_grad()
def valuta(model, dati, dev):
    model.eval()
    ok = {"command": 0, "weapon": 0}
    tot = {"command": 0, "weapon": 0}
    for e in dati:
        x = torch.tensor([e["ids"]], device=dev)
        logit = model(input_ids=x, logits_to_keep=1).logits[0, -1, e["lettere"]]
        ok[e["kind"]] += int(logit.argmax().item() == e["target"])
        tot[e["kind"]] += 1
    model.train()
    return {k: ok[k] / tot[k] for k in ok}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--accum", type=int, default=8)
    ap.add_argument("--seed", type=int, default=771)
    ap.add_argument("--gpu", type=int, default=0, help="indice CUDA della scheda su cui addestrare")
    ap.add_argument("--testo", action="store_true", help="modello solo testo (AutoModelForCausalLM)")
    a = ap.parse_args()
    random.seed(a.seed); torch.manual_seed(a.seed)
    tok = AutoTokenizer.from_pretrained(a.base)
    train = esempi(tok, json.load(open("training/v3/train.json")))
    val = esempi(tok, json.load(open("training/v3/validation.json")))
    print(f"esempi {len(train)} / {len(val)}, lunghezza max {max(len(e['ids']) for e in train)} token", flush=True)
    if a.testo:      # modello solo testo e piccolo (es. Gemma 3 270M): tutto sulla 4070
        model = AutoModelForCausalLM.from_pretrained(a.base, dtype=torch.bfloat16, device_map={"": a.gpu})
    else:
        model = AutoModelForImageTextToText.from_pretrained(
            a.base, dtype=torch.bfloat16, device_map=mappa(a.base, a.gpu))
    model.config.use_cache = False
    # niente gradient checkpointing: con Gemma 4 il ricalcolo non combacia (CheckpointError), e gli esempi sono corti
    # model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    cfg = LoraConfig(r=a.rank, lora_alpha=2 * a.rank, lora_dropout=0.05, bias="none",
                     target_modules=(r".*\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)" if a.testo else
                                     r".*language_model.*\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)"))
    model = get_peft_model(model, cfg)
    model.print_trainable_parameters()
    dev = model.get_input_embeddings().weight.device
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=a.lr, weight_decay=0.01)
    base = valuta(model, val, dev)
    print("prima dell'addestramento:", base, flush=True)
    migliore, storia = sum(base.values()), [{"epoch": 0, "validation": base}]
    passi = 0
    for ep in range(1, a.epochs + 1):
        random.shuffle(train)
        t0, perdita = time.time(), 0.0
        for i, e in enumerate(train):
            x = torch.tensor([e["ids"]], device=dev)
            logit = model(input_ids=x, logits_to_keep=1).logits[0, -1, e["lettere"]].float()
            loss = torch.nn.functional.cross_entropy(logit[None], torch.tensor([e["target"]], device=logit.device))
            (loss / a.accum).backward()
            perdita += loss.item()
            if (i + 1) % a.accum == 0:
                torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
                opt.step(); opt.zero_grad(); passi += 1
            if (i + 1) % 200 == 0:
                print(f"  epoca {ep} {i+1}/{len(train)} perdita media {perdita/(i+1):.3f} ({time.time()-t0:.0f}s)", flush=True)
        v = valuta(model, val, dev)
        storia.append({"epoch": ep, "validation": v, "seconds": round(time.time() - t0, 1)})
        print(f"epoca {ep}: {v}  ({time.time()-t0:.0f}s)", flush=True)
        if sum(v.values()) > migliore:
            migliore = sum(v.values())
            model.save_pretrained(a.out)
            json.dump({"epoch": ep, "validation": v}, open(os.path.join(a.out, "best-epoch.json"), "w"))
            print("  -> salvata", flush=True)
    os.makedirs(a.out, exist_ok=True)
    json.dump(storia, open(os.path.join(a.out, "storia.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
