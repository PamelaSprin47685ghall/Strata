#!/bin/sh
# 06-phase4 §4.1: the default-path-unchanged check.  Runs the phase-0 fullvram config (TP5
# OFF) with the phase-0 story prompt and compares content_sha256 + token count with the
# phase-0 baseline rows.  Any mismatch fails the check.
#
#   usage: tools/tp5_default_path_check.sh   (from the repo root; needs the GPUs idle)
set -e
RUN=bench/results/20261013T130444Z-tp5-phase0
PORT=18080
OUT=/tmp/tp5-default-check.json
.venv/bin/python -m serve.server --engine strata --config strata-swift-q2_0-fullvram.json \
  --host 127.0.0.1 --port $PORT &
SRV=$!
trap 'kill -TERM $SRV 2>/dev/null || true; wait $SRV 2>/dev/null || true' EXIT INT TERM
for i in $(seq 1 300); do
  .venv/bin/python - <<'EOF' && break || sleep 2
import json, urllib.request
r = json.load(urllib.request.urlopen('http://127.0.0.1:18080/health', timeout=5))
assert r.get('loaded'), r
EOF
done
.venv/bin/python - "$RUN" "$OUT" <<'EOF'
import json, sys, time, hashlib, urllib.request, pathlib
run, out = pathlib.Path(sys.argv[1]), sys.argv[2]
prompt = (run / 'prompt-story.txt').read_text()
url = 'http://127.0.0.1:18080/v1/chat/completions'
rows = []
for k in range(4):   # 1 warmup + 3 measured, exactly the phase-0 protocol
    body = {"messages": [{"role": "user", "content": prompt}], "temperature": 0, "seed": 42,
            "max_tokens": 400, "stream": False, "cache_prompt": False,
            "chat_template_kwargs": {"enable_thinking": False}}
    t = time.monotonic()
    r = json.load(urllib.request.urlopen(urllib.request.Request(
        url, json.dumps(body).encode(), {"Content-Type": "application/json"}), timeout=600))
    text = r["choices"][0]["message"]["content"] or ""
    rows.append({"phase": "warmup" if k == 0 else "measured",
                 "finish": r["choices"][0]["finish_reason"],
                 "n": r.get("timings", {}).get("predicted_n"),
                 "ms": r.get("timings", {}).get("predicted_ms"),
                 "content_sha256": hashlib.sha256(text.encode()).hexdigest()})
pathlib.Path(out).write_text(json.dumps(rows, ensure_ascii=False, indent=1))
base = set()
for r in ('r1', 'r2', 'r3'):
    for e in json.load(open(run / f'fullvram-{r}.json')):
        if e['prompt'] == 'story' and e['phase'] == 'measured':
            base.add((e['content_sha256'], e['timings']['predicted_n']))
got = set((e['content_sha256'], e['n']) for e in rows if e['phase'] == 'measured')
print('baseline (sha, n):', sorted(base))
print('this run  (sha, n):', sorted(got))
ok = got and got <= base
print('DEFAULT PATH ' + ('UNCHANGED' if ok else 'CHANGED - FAIL'))
sys.exit(0 if ok else 1)
EOF
