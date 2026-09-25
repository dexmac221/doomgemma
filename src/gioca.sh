#!/bin/bash
# gioca.sh <model> <endpoint> <tag>  — una partita MAP01 seed 48 skill 3, max 180 s, poi riassunto
cd "${DOOMLAYA:-.}"   # cartella di doomLaya con la patch applicata
out=$(timeout 300 .venv/bin/python agent.py --model "$1" --endpoint "$2" --seed 48 --skill 3 --map MAP01 --seconds 180 --stop-after-level --record auto --tag "$3" 2>&1)
run=$(echo "$out" | grep -o "RUN [^ ]*" | head -1 | cut -d" " -f2)
[ -z "$run" ] && { echo "$out" | tail -20; exit 1; }
python3 - "$run" "$3" <<'PY'
import json,sys,os
run,tag=sys.argv[1],sys.argv[2]
s=json.load(open(os.path.join(run,"summary.json")))
ev=[json.loads(l) for l in open(os.path.join(run,"events.jsonl"))]
fine=next((e["game_seconds"] for e in ev if e.get("event")=="level_finished" and e.get("map")=="MAP01"),None)
dec=[json.loads(l) for l in open(os.path.join(run,"decisions.jsonl"))]
print(f"{tag:14} esito={s['status']:9} uscita={('%.1f s'%fine) if fine else 'NO':8} uccisioni={s['kills']:2} morti={s['deaths']} "
      f"decisioni={s['decisions']:3} errori={s['errors']} latenza p50={s['latency_ms_median']} p90={s['latency_ms_p90']} ms  porte={s['doors_opened']} raccolti={s['pickups']}")
print("   ", run)
PY
