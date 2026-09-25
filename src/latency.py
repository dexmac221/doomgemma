"""Latency on equal hardware: replays the SAME recorded game packets (state + the two
real questions, command and weapon) one after the other and measures the end-to-end
time of the HTTP request. The first 10 are discarded (warm-up).
Usage: latency.py <predict endpoint> <model> <label> <run dir> [run dir ...]"""
import json, statistics, sys, time, requests
url, model, tag, runs = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4:]
packets = []
for r in runs:
    for l in open(r + "/decisions.jsonl"):
        d = json.loads(l)
        if d.get("packet"):
            packets.append(d["packet"])
s = requests.Session(); s.trust_env = False
t = []
t0 = time.perf_counter()
for i, p in enumerate(packets):
    a = time.perf_counter()
    s.post(url, json={"state": p["state"], "questions": p["questions"], "model": model}, timeout=60).raise_for_status()
    if i >= 10:
        t.append((time.perf_counter() - a) * 1000)
total = time.perf_counter() - t0
q = lambda x: sorted(t)[min(len(t) - 1, int(x * len(t)))]
print(f"{tag:12} n={len(t)} p50={q(.5):.1f} p90={q(.9):.1f} p99={q(.99):.1f} ms  mean={statistics.mean(t):.1f}  decisions/s={len(packets)/total:.1f}")
json.dump({"tag": tag, "n": len(t), "p50_ms": q(.5), "p90_ms": q(.9), "p99_ms": q(.99), "mean_ms": statistics.mean(t),
           "decisions_per_s": len(packets) / total}, open(f"latency_{tag}.json", "w"), indent=1)
