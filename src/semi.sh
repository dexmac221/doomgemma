#!/bin/bash
# semi.sh <model> <endpoint> <tag> <semi...>: una partita per seme, riga di riassunto ciascuna
cd "${DOOMLAYA:-.}"   # cartella di doomLaya con la patch applicata
m=$1; ep=$2; tag=$3; shift 3
for s in "$@"; do
  out=$(timeout 420 .venv/bin/python agent.py --model "$m" --endpoint "$ep" --seed $s --skill 3 --map ${MAPPA:-MAP01} --seconds ${SECONDI:-180} --record auto --stop-after-level --tag "${tag}-s$s" 2>&1)
  run=$(echo "$out" | grep -o "RUN [^ ]*" | head -1 | cut -d" " -f2)
  python3 - "$run" "$tag" "$s" <<'PY'
import json,sys,os
run,tag,s=sys.argv[1:4]
x=json.load(open(os.path.join(run,"summary.json")))
ev=[json.loads(l) for l in open(os.path.join(run,"events.jsonl"))]
f=next((e["game_seconds"] for e in ev if e.get("event")=="level_finished" and e.get("map")==os.environ.get("MAPPA","MAP01")),None)
print(json.dumps({"tag":tag,"seed":int(s),"mappa":os.environ.get("MAPPA","MAP01"),"uscita":f,"kills":x["kills"],"morti":x["deaths"],"lat":x["latency_ms_median"],"raccolti":x["pickups"],"porte":x["doors_opened"],"run":run}),flush=True)
PY
done
