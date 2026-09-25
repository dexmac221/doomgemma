#!/bin/bash
# play_seeds.sh <model> <endpoint> <label> <seed...>: one game per seed, one JSON summary line each.
# Environment: DOOMLAYA (doomLaya directory with the patch applied, default "."),
# MAP (default MAP01), GAME_SECONDS (default 180). (Not SECONDS: bash reserves it.)
cd "${DOOMLAYA:-.}"
m=$1; ep=$2; tag=$3; shift 3
for s in "$@"; do
  out=$(timeout $(( ${GAME_SECONDS:-180} + 120 )) .venv/bin/python agent.py --model "$m" --endpoint "$ep" --seed $s --skill 3 \
        --map ${MAP:-MAP01} --seconds ${GAME_SECONDS:-180} --record auto --stop-after-level --tag "${tag}-s$s" 2>&1)
  run=$(echo "$out" | grep -o "RUN [^ ]*" | head -1 | cut -d" " -f2)
  python3 - "$run" "$tag" "$s" <<'PY'
import json, os, sys
run, tag, s = sys.argv[1:4]
game_map = os.environ.get("MAP", "MAP01")
x = json.load(open(os.path.join(run, "summary.json")))
ev = [json.loads(l) for l in open(os.path.join(run, "events.jsonl"))]
exit_s = next((e["game_seconds"] for e in ev if e.get("event") == "level_finished" and e.get("map") == game_map), None)
print(json.dumps({"tag": tag, "seed": int(s), "map": game_map, "exit_s": exit_s, "kills": x["kills"],
                  "deaths": x["deaths"], "latency_ms": x["latency_ms_median"], "pickups": x["pickups"],
                  "doors": x["doors_opened"], "run": run}), flush=True)
PY
done
