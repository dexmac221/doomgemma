"""Share of time in which the executor does NOT execute the model's command
(target unreachable = unavailable, path blocked = blocked), per model and map.
Run it from the doomLaya directory (reads runs/*/telemetry.jsonl)."""
import collections, glob, json, re
agg = collections.defaultdict(collections.Counter)
for r in glob.glob("runs/*"):
    try:
        cfg = json.load(open(r + "/config.json"))
    except Exception:
        continue
    tag = re.sub(r"-s\d+$", "", cfg["args"].get("tag") or "")
    if not tag:
        continue
    game_map = cfg["args"]["map"]
    for l in open(r + "/telemetry.jsonl"):
        m = re.search(r'"status": "([a-z_]+)"', l)
        if m:
            agg[(tag, game_map)][m.group(1)] += 1
for (tag, game_map), c in sorted(agg.items(), key=lambda x: (x[0][1], x[0][0])):
    tot = sum(c.values())
    print(f"{game_map} {tag:18} samples {tot:6d}  not executed {100*(c['blocked']+c['unavailable'])/tot:5.1f}%  "
          f"(blocked {100*c['blocked']/tot:4.1f}%, unreachable {100*c['unavailable']/tot:4.1f}%)")
