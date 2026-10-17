#!/usr/bin/env python3
"""Direct token-ID parity: the same prompts decoded alone (solo GEN) and together in the batch
windows (BGEN), through the real engine protocol - the same path tools/batch_test.py drives, with
the same invariants (every slot's tokens equal its solo tokens), plus:

  - mixed prompt lengths (a long prompt read while other slots decode)
  - slot reuse: a conversation's next turn continued from the slot that holds it
  - accepted AND rejected MTP prefixes: the batch-mtp window's second row is only emitted when
    the target picked the proposal (keep[b] = 2), so a run that never accepts is itself evidence;
    the solo reference with --spec 5 gives the drafts-accepted counts to compare against
  - cancel/continuation: BSTOP mid-decode, then the solo path continues from the slot
    (what the server does with a request left alone in a slot)

Greedy, fixed prompts, deterministic.  Exit 0 only when every comparison is IDENTICAL.

  .venv/bin/python tools/batch_test.py --exe engine/strata \
      --config strata-swift-q2_0-tp5-batch2.json --batch 2 --n 2 --max-new 128 \
      --dump bench/results/20261016T-tp5-opt100/batch2-parity.json

  # batch-mtp: the config carries --batch-mtp; the same parity must hold
  .venv/bin/python tools/batch_test.py --exe engine/strata \
      --config strata-swift-q2_0-tp5-batch2-mtp.json --batch 2 --n 2 --max-new 128 \
      --dump bench/results/20261016T-tp5-opt100/batch2-mtp-parity.json

  # mixed prompt lengths: the LAST prompt gets ~--long-tokens tokens of filler, so its chunks
  # run beside the earlier slots' windows
  ... --long-tokens 2000

  # slot reuse + cancel/continuation (the interleave scenarios, same engine):
  .venv/bin/python tools/batch_interleave_test.py --exe engine/strata \
      --config strata-swift-q2_0-tp5-batch2.json --extra "--pcie-frac 0 --adapt-every 1000000"
"""
import argparse, json, os, sys, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from batch_test import Engine, tokenizer  # noqa: E402  (the fixed helper: --tp5 never gains --layer-split)


def run(eng, out, line, slot=None):
    """Send one GEN/BGEN; returns (tokens of the request line, the BADM flag)."""
    eng.send(line)
    got = []
    for l in out:
        if l.startswith("T "):
            got.append(int(l.split()[1]))
        elif l.startswith(("BT ", "BDONE ")):
            eng.pending.append(l)
        elif l.startswith("ERR"):
            raise SystemExit("engine: " + l)
        elif slot is None and l.startswith("DONE"):
            return got, None
        elif slot is not None and l.startswith("BADM "):
            return got, l.split()[2] == "1"
    raise SystemExit(f"the engine ended during: {line[:40]} (see {eng.log_path})")


def drain_slot(eng, out, slot, got, stop_after=None, stop_sent=None):
    """A slot's BT lines until its BDONE.  stop_after: BSTOP once this many tokens are in."""
    while True:
        l = None
        for i, p in enumerate(eng.pending):   # this slot's own lines first; others stay parked
            if p.startswith(f"BT {slot} ") or p.startswith(f"BDONE {slot} "):
                l = eng.pending.pop(i)
                break
        if l is None:
            l = next(out)
        if l.startswith(f"BT {slot} "):
            got.append(int(l.split()[2]))
            if stop_after is not None and len(got) >= stop_after and not (stop_sent or set()):
                eng.send(f"BSTOP {slot}")
                stop_sent.add(True)
        elif l.startswith(f"BDONE {slot} "):
            return l
        elif l.startswith(("BT ", "BDONE ")):     # another slot's line: keep it
            eng.pending.append(l)
        elif l.startswith("ERR"):
            raise SystemExit("engine: " + l)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--exe", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--n", type=int, default=2, help="prompts (<= --batch)")
    ap.add_argument("--max-new", type=int, default=128)
    ap.add_argument("--long-tokens", type=int, default=0,
                    help="the LAST prompt is read from about this many tokens of filler: mixed lengths")
    ap.add_argument("--extra", default="")
    a = ap.parse_args()
    if a.n > a.batch:
        print(f"--n {a.n} exceeds --batch {a.batch}: this test never queues", file=sys.stderr)
        return 2
    cfg = json.loads(Path(a.config).read_text())
    tok = tokenizer(cfg["tokenizer"])
    qs = [
        "请写一个关于灯塔看守人的400字短篇故事。",
        "用 Python 写一个快速排序，并解释时间复杂度。",
    ]
    prompts = []
    for i, q in enumerate(qs[: a.n]):
        text = f"<|im_start|>user\n{q}<|im_end|>\n<|im_start|>assistant\n<|im_end|>\n"
        if a.long_tokens and i == a.n - 1:
            para = ("The committee reviewed item %d of the long agenda, noting the budget, the schedule "
                    "and the open risks, and asked the staff to report back next quarter. ")
            per = max(1, len(tok.encode(para % 100)))
            filler = "".join(para % k for k in range(a.long_tokens // per + 1))
            text = f"<|im_start|>user\n{filler}\n\n{q}<|im_end|>\n<|im_start|>assistant\n<|im_end|>\n"
        prompts.append(tok.encode(text, parse_special=True))
    eng = Engine(a.exe, cfg, a.batch, {"STRATA_IQ_MT_MIN": "1", "STRATA_DECODE_TIMING": "1"},
                 a.extra.split())
    eng.pending = []
    out = eng.lines()
    ids = lambda v: ",".join(map(str, v))
    M = a.max_new
    ok = True

    # 1. solo references (GEN, with drafts)
    solo = []
    for i, P in enumerate(prompts):
        got, _ = run(eng, out, f"GEN {M} {ids(P)}")
        solo.append(got)
        print(f"solo {i}: {len(got)} tokens", flush=True)

    # 2. the same prompts in the slots, admitted together (mixed lengths with --long-tokens:
    # the long one's chunks interleave with the others' windows)
    got = {i: [] for i in range(len(prompts))}
    done = {}
    for s, P in enumerate(prompts):
        first, cont = run(eng, out, f"BGEN {s} {M} {ids(P)}", slot=s)
        got[s] += first
        if not cont:
            done[s] = True
    for s in range(len(prompts)):
        if s in done:
            continue
        drain_slot(eng, out, s, got[s])
    for s in range(len(prompts)):
        same = got[s] == solo[s]
        ok &= same
        d = next((k for k in range(min(len(got[s]), len(solo[s]))) if got[s][k] != solo[s][k]), None)
        print(f"slot {s}: {len(got[s])} tokens, solo {len(solo[s])}: "
              f"{'IDENTICAL' if same else f'DIFFERS at {d}'}", flush=True)

    # 3. slot reuse: prompt 0's next turn, continued from slot 0 (which holds it)
    nxt = tok.encode("<|im_end|>\n<|im_start|>user\nNow do it again, shorter.<|im_end|>\n"
                     "<|im_start|>assistant\n<|im_end|>\n", parse_special=True)
    P2 = prompts[0] + solo[0] + nxt
    s_ref, _ = run(eng, out, f"GEN {M} {ids(P2)}")
    first, cont = run(eng, out, f"BGEN 0 {M} {ids(P2)}", slot=0)
    got2 = list(first)
    if cont:
        drain_slot(eng, out, 0, got2)
    same = got2 == s_ref
    ok &= same
    d = next((k for k in range(min(len(got2), len(s_ref))) if got2[k] != s_ref[k]), None)
    print(f"slot reuse (prompt 0's next turn from slot 0): {len(got2)} tokens, solo {len(s_ref)}: "
          f"{'IDENTICAL' if same else f'DIFFERS at {d}'}", flush=True)

    # 4. cancel/continuation: prompt 1 in slot 1, BSTOP after 40 tokens, then the solo path
    # continues from the slot (GEN of the prompt + what it produced)
    first, cont = run(eng, out, f"BGEN 1 {M} {ids(prompts[1])}", slot=1)
    got4 = list(first)
    stopped = set()
    if cont:
        drain_slot(eng, out, 1, got4, stop_after=40, stop_sent=stopped)
    tail4, _ = run(eng, out, f"GEN {M - len(got4)} {ids(prompts[1] + got4)}") if len(got4) < M else ([], None)
    whole = got4 + tail4
    same = whole == solo[1]
    ok &= same
    d = next((k for k in range(min(len(whole), len(solo[1]))) if whole[k] != solo[1][k]), None)
    print(f"cancel/continuation (BSTOP at 40, then solo): {len(whole)} tokens, solo {len(solo[1])}: "
          f"{'IDENTICAL' if same else f'DIFFERS at {d}'} (stopped after {len(got4)})", flush=True)
    ok &= bool(stopped) or not cont

    eng.send("QUIT")
    eng.p.wait(timeout=180)
    log = Path(eng.log_path).read_text(errors="replace")
    for l in log.splitlines():
        if "windows, avg " in l or "MTP proposals accepted" in l:
            print("  log:", l[:160], flush=True)   # the engine's own window/acceptance counters
    print("PASS" if ok else "FAIL")
    return 0 if ok else 2


if __name__ == "__main__":
    sys.exit(main())
