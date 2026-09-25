"""L'insegnante a regole di doomLaya usato come GIOCATORE.

`gold` e' copiata identica da training/build_dataset.py (le regole che hanno
assegnato le etichette di addestramento a Laya v3 e alle nostre LoRA). Riceve lo
stesso pacchetto e lo stesso stato grezzo che in addestramento, passa dallo stesso
esecutore e dalla stessa cadenza dei modelli. Serve a sapere se MAP02 e' risolvibile
dall'insegnante stesso, e a dare il tetto di riferimento su MAP01."""
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


def decidi(client, packet, s):
    """Risposta nel formato di /predict, passata dalla STESSA validazione dei modelli."""
    t0 = time.perf_counter()
    scelte = gold(packet, s)
    # l'esecutore di gioco puo' aver tolto 'exit' dalle opzioni (nessuna missione):
    # in quel caso l'insegnante ripiega su 'explore', come farebbe un modello
    if scelte['command'] not in packet['questions']['command']['criteria']:
        scelte['command'] = 'explore'
    answers = {n: {"choice": scelte[n],
                   "probabilities": {k: float(k == scelte[n]) for k in q['criteria']}}
               for n, q in packet['questions'].items()}
    risultato = {"answers": answers, "routing": {"model": "oracolo"},
                 "usage": {"input_tokens": 0, "cost": 0},
                 "latency_ms": round((time.perf_counter() - t0) * 1000, 3)}
    return client.validate(risultato, t0, packet['questions'])
