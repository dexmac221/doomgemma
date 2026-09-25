"""Latenza a parita' di scheda: rigioca gli STESSI pacchetti di gioco registrati
(stato + le due domande vere, comando e arma) uno dopo l'altro e misura il tempo
end-to-end della richiesta HTTP. Primi 10 scartati (riscaldamento).
Uso: latenza.py <endpoint> <model> <etichetta> <run1> [run2 ...]"""
import json, sys, time, statistics, requests
url, model, tag, runs = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4:]
pacchetti = []
for r in runs:
    for l in open(r + "/decisions.jsonl"):
        d = json.loads(l)
        if d.get("packet"):
            pacchetti.append(d["packet"])
s = requests.Session(); s.trust_env = False
t = []
t0 = time.perf_counter()
for i, p in enumerate(pacchetti):
    a = time.perf_counter()
    s.post(url, json={"state": p["state"], "questions": p["questions"], "model": model}, timeout=60).raise_for_status()
    if i >= 10:
        t.append((time.perf_counter() - a) * 1000)
tot = time.perf_counter() - t0
q = lambda x: sorted(t)[min(len(t) - 1, int(x * len(t)))]
print(f"{tag:12} n={len(t)} p50={q(.5):.1f} p90={q(.9):.1f} p99={q(.99):.1f} ms  media={statistics.mean(t):.1f}  decisioni/s={len(pacchetti)/tot:.1f}")
json.dump({"tag": tag, "n": len(t), "p50": q(.5), "p90": q(.9), "p99": q(.99), "media": statistics.mean(t),
           "decisioni_s": len(pacchetti) / tot}, open(f"latenza_{tag}.json", "w"))
