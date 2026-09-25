"""Stessa interfaccia /predict di serve_doom_laya.py, ma le decisioni le prende un
LLM servito da llama.cpp (OpenAI-compatibile). Serve a confrontare Laya con i
nostri modelli ALLA PARI: stesse domande, stesso formato di risposta, stesse
probabilita' per opzione.

Come: ogni opzione di ogni domanda diventa una lettera; una grammatica obbliga il
modello a rispondere solo con le lettere (una per domanda, nell'ordine); la
probabilita' di ogni opzione si legge dai top_logprobs del token di quella
posizione, rinormalizzata sulle sole lettere valide. Un'opzione che non compare fra
i top_logprobs prende una probabilita' minima.

Uso: serve_llm.py --llm http://127.0.0.1:8090 --name gemma26 --port 8002
"""
import argparse
import json
import math
import string
import time
from concurrent.futures import ThreadPoolExecutor

import requests
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

LETTERE = string.ascii_uppercase
MINIMO = 1e-4


def costruisci(state, questions):
    """Prompt e grammatica per le domande a scelta: USATO anche per l'addestramento
    LoRA, cosi' formato di gioco e di addestramento restano identici."""
    nomi = [n for n, q in questions.items() if q.get("type") == "choice"]
    chiavi = {n: list(questions[n]["criteria"]) for n in nomi}
    blocchi = []
    for n in nomi:
        q = questions[n]
        righe = [f"{LETTERE[i]}) {q['criteria'][k]}" for i, k in enumerate(chiavi[n])]
        blocchi.append(f"{n.upper()} — {q.get('instructions', '')}\n" + "\n".join(righe))
    prompt = (state + "\n\n" + "\n\n".join(blocchi) + "\n\nAnswer with "
              + str(len(nomi)) + " letter" + ("s separated by a space" if len(nomi) > 1 else "")
              + ", in this order: " + ", ".join(n.upper() for n in nomi) + ".")
    grammatica = "root ::= " + ' " " '.join("[" + LETTERE[:len(chiavi[n])] + "]" for n in nomi)
    messaggi = [{"role": "system", "content": SISTEMA}, {"role": "user", "content": prompt}]
    return nomi, chiavi, messaggi, grammatica


SISTEMA = ("You control a DOOM player in real time. "
           "Pick the best option for each question. Reply only with the letters.")


def chiedi(sess, args, state, questions):
    nomi, chiavi, messaggi, grammatica = costruisci(state, questions)
    body = {"messages": messaggi, "max_tokens": 2 * len(nomi) + 2, "temperature": 0,
            "grammar": grammatica, "logprobs": True, "top_logprobs": args.top, "stream": False,
            "chat_template_kwargs": {"enable_thinking": False}}
    r = sess.post(args.llm + "/v1/chat/completions", json=body, timeout=30)
    if r.status_code != 200:
        raise HTTPException(502, r.text[:300])
    o = r.json()
    testo = o["choices"][0]["message"].get("content") or ""
    contenuto = ((o["choices"][0].get("logprobs") or {}).get("content")) or []
    pos = [t for t in contenuto if t["token"].strip() and t["token"].strip()[0] in LETTERE]
    answers = {}
    for idx, n in enumerate(nomi):
        ks = chiavi[n]
        valide = {LETTERE[i]: k for i, k in enumerate(ks)}
        p = {k: 0.0 for k in ks}
        if idx < len(pos):
            for alt in pos[idx].get("top_logprobs") or []:
                let = alt["token"].strip()
                if len(let) == 1 and let in valide:
                    p[valide[let]] += math.exp(alt["logprob"])
            scelta_let = pos[idx]["token"].strip()[0]
        else:
            sc = testo.split()
            scelta_let = sc[idx] if idx < len(sc) else "A"
        scelta = valide.get(scelta_let, ks[0])
        if p[scelta] <= 0:
            p[scelta] = 1.0
        p = {k: max(v, MINIMO) for k, v in p.items()}
        tot = sum(p.values())
        answers[n] = {"choice": scelta, "probabilities": {k: v / tot for k, v in p.items()}}
    return answers, (o.get("usage") or {}).get("prompt_tokens", 0), testo


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--llm", default="http://127.0.0.1:8090")
    ap.add_argument("--name", required=True, help="etichetta del modello nei risultati")
    ap.add_argument("--port", type=int, default=8002)
    ap.add_argument("--top", type=int, default=20)
    ap.add_argument("--separate", action="store_true", help="una richiesta per domanda, in parallelo (come in addestramento)")
    args = ap.parse_args()
    sess = requests.Session()
    sess.trust_env = False
    props = sess.get(args.llm + "/props", timeout=10).json()
    modello = (props.get("model_path") or "?").split("/")[-1]
    app = FastAPI()

    class Request(BaseModel):
        state: str
        questions: dict
        model: str = "llm"

    @app.get("/health")
    def health():
        return {"status": "ok", "models": {"llm": modello}, "llm_url": args.llm,
                "name": args.name}

    @app.post("/predict")
    def predict(req: Request):
        t0 = time.perf_counter()
        nomi = [n for n, q in req.questions.items() if q.get("type") == "choice"]
        if args.separate:
            with ThreadPoolExecutor(max_workers=len(nomi) or 1) as ex:
                parti = list(ex.map(lambda n: chiedi(sess, args, req.state, {n: req.questions[n]}), nomi))
        else:
            parti = [chiedi(sess, args, req.state, {n: req.questions[n] for n in nomi})]
        answers, tok, raw = {}, 0, []
        for a, t, r in parti:
            answers.update(a); tok += t; raw.append(r)
        return {"answers": answers,
                "routing": {"model": "llm", "checkpoint": modello, "name": args.name},
                "usage": {"input_tokens": tok, "cost": 0},
                "latency_ms": round((time.perf_counter() - t0) * 1000, 2),
                "raw": " | ".join(raw)}

    uvicorn.run(app, host="127.0.0.1", port=args.port)


if __name__ == "__main__":
    main()
