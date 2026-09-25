"""Accuracy on the 130 validation questions of Laya v3 (doomLaya training/v3/validation.json).
Run it from the doomLaya directory.
Usage: evaluate.py <predict endpoint> <model>   e.g. evaluate.py http://127.0.0.1:8002/predict llm"""
import collections, json, sys, time, requests
url, model = sys.argv[1], sys.argv[2]
rows = json.load(open("training/v3/validation.json"))
ok, tot, conf, lat = collections.Counter(), collections.Counter(), collections.defaultdict(list), []
errors = collections.Counter()
s = requests.Session(); s.trust_env = False
for r in rows:
    t = time.perf_counter()
    o = s.post(url, json={"state": r["state"], "questions": {r["kind"]: r["question"]}, "model": model}, timeout=60).json()
    lat.append((time.perf_counter() - t) * 1000)
    a = o["answers"][r["kind"]]
    right = a["choice"] == r["label"]
    tot[r["kind"]] += 1; ok[r["kind"]] += right
    conf[(r["kind"], right)].append(max(a["probabilities"].values()))
    if not right:
        errors[(r["label"].split("_")[0], a["choice"].split("_")[0])] += 1
med = lambda v: sorted(v)[len(v) // 2] if v else float("nan")
for k in ("command", "weapon"):
    print(f"{k:8} {ok[k]}/{tot[k]} = {ok[k]/tot[k]:.3f}   median confidence: right {med(conf[(k, True)]):.2f}, wrong {med(conf[(k, False)]):.2f}")
print(f"latency p50 {med(lat):.0f} ms   most frequent errors (label -> choice): {errors.most_common(5)}")
