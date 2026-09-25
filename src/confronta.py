import json, glob, collections, sys
tags = sys.argv[1:] or ["laya-v3", "laya-orig", "bonsai", "gemma26"]
for t in tags:
    ds = sorted(glob.glob(f"runs/*_{t}"))
    if not ds:
        continue
    d = ds[-1]
    rows = [json.loads(l) for l in open(d + "/decisions.jsonl")]
    az = collections.Counter(r["directive"]["action"] for r in rows if r.get("directive"))
    ex_off = sum(1 for r in rows if "exit" in r["packet"]["questions"]["command"]["criteria"])
    ex_sc = sum(1 for r in rows if r.get("command") == "exit")
    nem = [r for r in rows if any(k.startswith("shoot_") for k in r["packet"]["questions"]["command"]["criteria"])]
    spara = sum(1 for r in nem if str(r.get("command", "")).startswith("shoot_"))
    conf = sorted(max(r["answers"]["command"]["probabilities"].values()) for r in rows)
    print(f"{t:10} azioni={dict(az.most_common(6))}")
    print(f"{'':10} uscita proposta {ex_off}x, scelta {ex_sc}x | nemici in vista: spara {spara}/{len(nem)} | confidenza mediana {conf[len(conf)//2]:.2f}")
