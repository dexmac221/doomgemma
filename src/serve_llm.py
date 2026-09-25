"""Same /predict interface as doomLaya's serve_doom_laya.py, but the decisions are
taken by an LLM served by llama.cpp (OpenAI-compatible API). It is used to compare
Laya with generative models ON EQUAL TERMS: same questions, same answer format,
same per-option probabilities.

How: every option of every question becomes a letter; a grammar forces the model
to answer only with letters (one per question, in order); the probability of each
option is read from the top_logprobs of the token at that position, renormalised
over the valid letters only. An option that does not appear among the top_logprobs
gets a minimum probability.

Usage: serve_llm.py --llm http://127.0.0.1:8090 --name e2b-doom --port 8002 [--separate]
"""
import argparse
import math
import string
import time
from concurrent.futures import ThreadPoolExecutor

import requests
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

LETTERS = string.ascii_uppercase
MIN_PROB = 1e-4

# NOTE: the prompt text below is exactly the one the published LoRA adapters were
# trained on (lora_train.py imports build_prompt): do not change it.
SYSTEM = ("You control a DOOM player in real time. "
          "Pick the best option for each question. Reply only with the letters.")


def build_prompt(state, questions):
    """Prompt and grammar for the choice questions. Also used for LoRA training,
    so that the in-game and training formats stay identical."""
    names = [n for n, q in questions.items() if q.get("type") == "choice"]
    keys = {n: list(questions[n]["criteria"]) for n in names}
    blocks = []
    for n in names:
        q = questions[n]
        lines = [f"{LETTERS[i]}) {q['criteria'][k]}" for i, k in enumerate(keys[n])]
        blocks.append(f"{n.upper()} — {q.get('instructions', '')}\n" + "\n".join(lines))
    prompt = (state + "\n\n" + "\n\n".join(blocks) + "\n\nAnswer with "
              + str(len(names)) + " letter" + ("s separated by a space" if len(names) > 1 else "")
              + ", in this order: " + ", ".join(n.upper() for n in names) + ".")
    grammar = "root ::= " + ' " " '.join("[" + LETTERS[:len(keys[n])] + "]" for n in names)
    messages = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": prompt}]
    return names, keys, messages, grammar


def ask(sess, args, state, questions):
    """One grammar-constrained request; returns (answers, prompt_tokens, raw_text)."""
    names, keys, messages, grammar = build_prompt(state, questions)
    body = {"messages": messages, "max_tokens": 2 * len(names) + 2, "temperature": 0,
            "grammar": grammar, "logprobs": True, "top_logprobs": args.top, "stream": False,
            "chat_template_kwargs": {"enable_thinking": False}}
    r = sess.post(args.llm + "/v1/chat/completions", json=body, timeout=30)
    if r.status_code != 200:
        raise HTTPException(502, r.text[:300])
    o = r.json()
    text = o["choices"][0]["message"].get("content") or ""
    content = ((o["choices"][0].get("logprobs") or {}).get("content")) or []
    # tokens carrying a letter (the space may be separate or attached: " B")
    pos = [t for t in content if t["token"].strip() and t["token"].strip()[0] in LETTERS]
    answers = {}
    for idx, n in enumerate(names):
        ks = keys[n]
        valid = {LETTERS[i]: k for i, k in enumerate(ks)}
        p = {k: 0.0 for k in ks}
        if idx < len(pos):
            for alt in pos[idx].get("top_logprobs") or []:
                letter = alt["token"].strip()
                if len(letter) == 1 and letter in valid:
                    p[valid[letter]] += math.exp(alt["logprob"])
            chosen_letter = pos[idx]["token"].strip()[0]
        else:
            parts = text.split()
            chosen_letter = parts[idx] if idx < len(parts) else "A"
        choice = valid.get(chosen_letter, ks[0])
        if p[choice] <= 0:
            p[choice] = 1.0          # no usable logprob: the choice takes it all
        p = {k: max(v, MIN_PROB) for k, v in p.items()}
        tot = sum(p.values())
        # the choice stays the grammar's even if it is not the renormalised argmax
        # (rare); the agent logs that case separately
        answers[n] = {"choice": choice, "probabilities": {k: v / tot for k, v in p.items()}}
    return answers, (o.get("usage") or {}).get("prompt_tokens", 0), text


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--llm", default="http://127.0.0.1:8090", help="llama-server base URL")
    ap.add_argument("--name", required=True, help="model label in the results")
    ap.add_argument("--port", type=int, default=8002)
    ap.add_argument("--top", type=int, default=20, help="top_logprobs to read")
    ap.add_argument("--separate", action="store_true",
                    help="one request per question, in parallel (as in training)")
    args = ap.parse_args()
    sess = requests.Session()
    sess.trust_env = False
    props = sess.get(args.llm + "/props", timeout=10).json()
    model_file = (props.get("model_path") or "?").split("/")[-1]
    app = FastAPI()

    class Request(BaseModel):
        state: str
        questions: dict
        model: str = "llm"

    @app.get("/health")
    def health():
        return {"status": "ok", "models": {"llm": model_file}, "llm_url": args.llm,
                "name": args.name}

    @app.post("/predict")
    def predict(req: Request):
        t0 = time.perf_counter()
        names = [n for n, q in req.questions.items() if q.get("type") == "choice"]
        if args.separate:
            with ThreadPoolExecutor(max_workers=len(names) or 1) as ex:
                parts = list(ex.map(lambda n: ask(sess, args, req.state, {n: req.questions[n]}), names))
        else:
            parts = [ask(sess, args, req.state, {n: req.questions[n] for n in names})]
        answers, tokens, raw = {}, 0, []
        for a, t, r in parts:
            answers.update(a); tokens += t; raw.append(r)
        return {"answers": answers,
                "routing": {"model": "llm", "checkpoint": model_file, "name": args.name},
                "usage": {"input_tokens": tokens, "cost": 0},
                "latency_ms": round((time.perf_counter() - t0) * 1000, 2),
                "raw": " | ".join(raw)}

    uvicorn.run(app, host="127.0.0.1", port=args.port)


if __name__ == "__main__":
    main()
