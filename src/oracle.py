"""doomLaya's rule-based teacher used as a PLAYER.

`gold` is copied verbatim from doomLaya's training/build_dataset.py: these are the
rules that assigned the training labels to Laya v3 and to our LoRA adapters. It
receives the same packet and the same raw state as in training, and goes through
the same executor at the same cadence as the models. It answers two questions:
is MAP02 solvable by the teacher itself, and what is the reference on MAP01?

Enabled in doomLaya's agent.py by patches/doomLaya-agent.patch (--model oracle)."""
import time
from items import utility


def gold(packet,s):
 commands=packet['commands'];visible=[k for k,v in commands.items() if v['action']=='attack']
 candidates=[]
 for k,v in commands.items():
  if v['action']!='pickup':continue
  i=v['target'];score=utility(i,s)
  needed=(i['category']=='Weapon' and score>=100) or (i['category']=='Health' and s['hp']<75) or (i['category']=='Armor' and s['armor']<75) or (i['category']=='Ammo' and score>=80) or i['category'] in ('Key','Powerup')
  if needed and i['distance']<12:candidates.append((score-i['distance']*2,k))
 urgent=[(score,key) for score,key in candidates if commands[key]['target']['category']=='Health' and commands[key]['target']['distance']<6 and s['hp']<35]
 if urgent:command=max(urgent)[1]
 elif visible:command=visible[0]
 elif candidates:command=max(candidates)[1]
 elif s['door'] and s['door']['distance']<4:command='open_door'
 else:command='exit'
 usable={int(k) for k,v in s['inventory'].items() if v['owned'] and (int(k)==1 or v['ammo']>0)}
 slot=next(i for i in (6,3,4,2,1) if i in usable)
 weapon=next(k for k,v in packet['weapons'].items() if v==slot)
 return {'command':command,'weapon':weapon}


def decide(client, packet, s):
    """Answer in the /predict format, passed through the SAME validation as the models."""
    t0 = time.perf_counter()
    choices = gold(packet, s)
    # the game executor may have removed 'exit' from the options (no mission):
    # in that case the teacher falls back to 'explore', as a model would
    if choices['command'] not in packet['questions']['command']['criteria']:
        choices['command'] = 'explore'
    answers = {n: {"choice": choices[n],
                   "probabilities": {k: float(k == choices[n]) for k in q['criteria']}}
               for n, q in packet['questions'].items()}
    result = {"answers": answers, "routing": {"model": "oracle"},
              "usage": {"input_tokens": 0, "cost": 0},
              "latency_ms": round((time.perf_counter() - t0) * 1000, 3)}
    return client.validate(result, t0, packet['questions'])
