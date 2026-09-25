"""Decision patterns of the latest run of each label: action counts, how often the
exit was offered and chosen, how often the model shoots when enemies are visible,
median confidence. Run it from the doomLaya directory.
Usage: decision_patterns.py <label> [label ...]"""
import collections, glob, json, sys
for t in sys.argv[1:]:
    ds = sorted(glob.glob(f"runs/*_{t}"))
    if not ds:
        continue
    rows = [json.loads(l) for l in open(ds[-1] + "/decisions.jsonl")]
    actions = collections.Counter(r["directive"]["action"] for r in rows if r.get("directive"))
    exit_offered = sum(1 for r in rows if "exit" in r["packet"]["questions"]["command"]["criteria"])
    exit_chosen = sum(1 for r in rows if r.get("command") == "exit")
    enemies = [r for r in rows if any(k.startswith("shoot_") for k in r["packet"]["questions"]["command"]["criteria"])]
    shoots = sum(1 for r in enemies if str(r.get("command", "")).startswith("shoot_"))
    conf = sorted(max(r["answers"]["command"]["probabilities"].values()) for r in rows)
    print(f"{t:14} actions={dict(actions.most_common(6))}")
    print(f"{'':14} exit offered {exit_offered}x, chosen {exit_chosen}x | enemies in sight: shoots {shoots}/{len(enemies)} | median confidence {conf[len(conf)//2]:.2f}")
