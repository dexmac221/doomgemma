"""Calibrazione sulle 130 domande di validazione di Laya v3.

Per ogni modello: accuratezza, NLL, Brier (multi-classe) ed ECE (15 intervalli sulla
confidenza massima), GREZZI e dopo temperature scaling. La temperatura si stima con
validazione incrociata a 2 parti sulla validazione stessa (si stima su una meta',
si misura sull'altra, e viceversa): cosi' nessun numero e' misurato sui dati con cui
e' stato tarato, e nessuno dei modelli usa i dati di addestramento.

Uso: calibra.py <endpoint /predict> <model> <etichetta>   -> stampa e salva calib_<etichetta>.json
"""
import json, math, random, sys, requests

url, model, tag = sys.argv[1:4]
rows = json.load(open("training/v3/validation.json"))
s = requests.Session(); s.trust_env = False
dati = []
for r in rows:
    o = s.post(url, json={"state": r["state"], "questions": {r["kind"]: r["question"]}, "model": model}, timeout=60).json()
    p = o["answers"][r["kind"]]["probabilities"]
    ks = list(r["question"]["criteria"])
    dati.append({"kind": r["kind"], "p": [max(float(p.get(k, 0.0)), 1e-6) for k in ks], "y": ks.index(r["label"])})


def scala(p, T):
    l = [math.log(x) / T for x in p]
    m = max(l); e = [math.exp(x - m) for x in l]; z = sum(e)
    return [x / z for x in e]


def misure(d, T=1.0, bins=15):
    n = len(d); acc = nll = brier = 0.0
    conf_bin = [[0, 0.0, 0.0] for _ in range(bins)]
    for x in d:
        q = scala(x["p"], T)
        pred = max(range(len(q)), key=q.__getitem__)
        c = q[pred]; ok = pred == x["y"]
        acc += ok; nll -= math.log(q[x["y"]])
        brier += sum((q[i] - (1.0 if i == x["y"] else 0.0)) ** 2 for i in range(len(q)))
        b = min(int(c * bins), bins - 1)
        conf_bin[b][0] += 1; conf_bin[b][1] += c; conf_bin[b][2] += ok
    ece = sum(abs(cb[1] - cb[2]) for cb in conf_bin if cb[0]) / n
    return {"acc": acc / n, "nll": nll / n, "brier": brier / n, "ece": ece}


def miglior_T(d):
    griglia = [0.05 * i for i in range(1, 201)]          # 0,05 .. 10
    return min(griglia, key=lambda T: misure(d, T)["nll"])


random.seed(771)
idx = list(range(len(dati))); random.shuffle(idx)
a, b = [dati[i] for i in idx[:65]], [dati[i] for i in idx[65:]]
Ta, Tb = miglior_T(a), miglior_T(b)
cal = {k: (misure(b, Ta)[k] + misure(a, Tb)[k]) / 2 for k in ("acc", "nll", "brier", "ece")}
grezzo = misure(dati)
out = {"modello": tag, "n": len(dati), "grezzo": grezzo, "tarato_cv2": cal, "T": [Ta, Tb],
       "per_domanda": {k: misure([x for x in dati if x["kind"] == k]) for k in ("command", "weapon")}}
json.dump({**out, "probabilita": dati}, open(f"calib_{tag}.json", "w"))
f = lambda m: " ".join(f"{k}={v:.3f}" for k, v in m.items())
print(f"{tag:12} grezzo: {f(grezzo)}")
print(f"{'':12} tarato: {f(cal)}   T={Ta:.2f}/{Tb:.2f}")
