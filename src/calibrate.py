"""Calibration on the 130 validation questions of Laya v3.

For each model: accuracy, NLL, multi-class Brier and ECE (15 bins on the maximum
confidence), RAW and after temperature scaling. The temperature is estimated with
2-fold cross-validation on the validation set itself (fit on one half, measure on
the other, and vice versa): no number is measured on the data it was fitted on,
and no model uses the training data.

Run it from the doomLaya directory.
Usage: calibrate.py <predict endpoint> <model> <label>   -> prints and saves calibration_<label>.json
"""
import json, math, random, sys, requests

url, model, tag = sys.argv[1:4]
rows = json.load(open("training/v3/validation.json"))
s = requests.Session(); s.trust_env = False
data = []
for r in rows:
    o = s.post(url, json={"state": r["state"], "questions": {r["kind"]: r["question"]}, "model": model}, timeout=60).json()
    p = o["answers"][r["kind"]]["probabilities"]
    ks = list(r["question"]["criteria"])
    data.append({"kind": r["kind"], "p": [max(float(p.get(k, 0.0)), 1e-6) for k in ks], "y": ks.index(r["label"])})


def scale(p, T):
    l = [math.log(x) / T for x in p]
    m = max(l); e = [math.exp(x - m) for x in l]; z = sum(e)
    return [x / z for x in e]


def metrics(d, T=1.0, bins=15):
    n = len(d); acc = nll = brier = 0.0
    b_stats = [[0, 0.0, 0.0] for _ in range(bins)]
    for x in d:
        q = scale(x["p"], T)
        pred = max(range(len(q)), key=q.__getitem__)
        c = q[pred]; right = pred == x["y"]
        acc += right; nll -= math.log(q[x["y"]])
        brier += sum((q[i] - (1.0 if i == x["y"] else 0.0)) ** 2 for i in range(len(q)))
        b = min(int(c * bins), bins - 1)
        b_stats[b][0] += 1; b_stats[b][1] += c; b_stats[b][2] += right
    ece = sum(abs(bs[1] - bs[2]) for bs in b_stats if bs[0]) / n
    return {"acc": acc / n, "nll": nll / n, "brier": brier / n, "ece": ece}


def best_T(d):
    grid = [0.05 * i for i in range(1, 201)]          # 0.05 .. 10
    return min(grid, key=lambda T: metrics(d, T)["nll"])


random.seed(771)
idx = list(range(len(data))); random.shuffle(idx)
a, b = [data[i] for i in idx[:65]], [data[i] for i in idx[65:]]
Ta, Tb = best_T(a), best_T(b)
tuned = {k: (metrics(b, Ta)[k] + metrics(a, Tb)[k]) / 2 for k in ("acc", "nll", "brier", "ece")}
raw = metrics(data)
out = {"model": tag, "n": len(data), "raw": raw, "tuned_cv2": tuned, "T": [Ta, Tb],
       "per_question": {k: metrics([x for x in data if x["kind"] == k]) for k in ("command", "weapon")}}
json.dump(out, open(f"calibration_{tag}.json", "w"), indent=1)
f = lambda m: " ".join(f"{k}={v:.3f}" for k, v in m.items())
print(f"{tag:12} raw:   {f(raw)}")
print(f"{'':12} tuned: {f(tuned)}   T={Ta:.2f}/{Tb:.2f}")
