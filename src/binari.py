"""Quota di tempo in cui l'esecutore NON esegue il comando del modello (bersaglio
irraggiungibile = unavailable, strada bloccata = blocked), per modello e mappa."""
import json, glob, re, collections
agg = collections.defaultdict(lambda: collections.Counter())
for r in glob.glob("runs/*"):
    try:
        cfg = json.load(open(r + "/config.json"))
    except Exception:
        continue
    tag = re.sub(r"-s\d+$", "", cfg["args"].get("tag") or "")
    if not tag:
        continue
    mappa = cfg["args"]["map"]
    for l in open(r + "/telemetry.jsonl"):
        m = re.search(r'"status": "([a-z_]+)"', l)
        if m:
            agg[(tag, mappa)][m.group(1)] += 1
for (tag, mappa), c in sorted(agg.items(), key=lambda x: (x[0][1], x[0][0])):
    tot = sum(c.values())
    print(f"{mappa} {tag:11} campioni {tot:6d}  non eseguito {100*(c['blocked']+c['unavailable'])/tot:5.1f}%  (bloccato {100*c['blocked']/tot:4.1f}%, irraggiungibile {100*c['unavailable']/tot:4.1f}%)")
