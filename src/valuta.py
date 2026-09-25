"""Accuratezza sulle 130 domande di validazione di Laya v3 (training/v3/validation.json).
Uso: valuta.py <endpoint /predict> <model>   es. valuta.py http://127.0.0.1:8002/predict llm"""
import json, sys, time, collections, requests
url, model = sys.argv[1], sys.argv[2]
rows = json.load(open("training/v3/validation.json"))
ok, tot, conf, lat = collections.Counter(), collections.Counter(), collections.defaultdict(list), []
sbagli = collections.Counter()
s = requests.Session(); s.trust_env = False
for r in rows:
    t = time.perf_counter()
    o = s.post(url, json={"state": r["state"], "questions": {r["kind"]: r["question"]}, "model": model}, timeout=60).json()
    lat.append((time.perf_counter() - t) * 1000)
    a = o["answers"][r["kind"]]
    giusta = a["choice"] == r["label"]
    tot[r["kind"]] += 1; ok[r["kind"]] += giusta
    conf[(r["kind"], giusta)].append(max(a["probabilities"].values()))
    if not giusta:
        sbagli[(r["label"].split("_")[0], a["choice"].split("_")[0])] += 1
med = lambda v: sorted(v)[len(v) // 2] if v else float("nan")
for k in ("command", "weapon"):
    print(f"{k:8} {ok[k]}/{tot[k]} = {ok[k]/tot[k]:.3f}   confidenza: giuste {med(conf[(k,True)]):.2f}, sbagliate {med(conf[(k,False)]):.2f}")
print(f"latenza p50 {med(lat):.0f} ms   errori piu' frequenti (giusta -> scelta): {sbagli.most_common(5)}")
