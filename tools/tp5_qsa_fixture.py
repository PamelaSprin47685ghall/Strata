#!/usr/bin/env python3
"""tools/tp5_qsa_fixture.py - the TP5 QSA tier's numeric-gate fixture (06 §4.2.3).

Writes QSA layer 3's ten tensors (the raw payload bytes, ~22 MB) from the model's
GGUF shards into a directory, one .bin per tensor named blk_3_<name with dots as
underscores>.bin.  The gate (tests/hip/tp5_qsa_test.cpp) and the tier's fixture
mode (STRATA_TP5_QSA_FIXTURE) both read them.

Why a fixture at all: on this stack a test binary's openat of the >32 GiB shard 1
by its long original path returns ENOENT - the same call succeeds from bash, from
a plain gcc binary and from the engine, and flips with the test binary's own
layout (an hour of bisection left the mechanism unexplained; see the note in
src/core/tp5_qsa.cu).  Python's open works, so the gate reads the weights through
this script's output instead.

Usage: python3 tools/tp5_qsa_fixture.py <shard-1-path> <out-dir>
"""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import gguf_reader as G  # noqa: E402

NAMES = ['attn_q.weight', 'attn_k.weight', 'attn_v.weight', 'attn_output.weight',
         'indexer.q_proj.weight', 'indexer.k_proj.weight',
         'attn_q_norm.weight', 'attn_k_norm.weight',
         'indexer.q_norm.weight', 'indexer.k_norm.weight']


def main() -> int:
    if len(sys.argv) != 3:
        print(__doc__)
        return 2
    shard1 = pathlib.Path(sys.argv[1])
    out = pathlib.Path(sys.argv[2])
    out.mkdir(parents=True, exist_ok=True)
    shards = sorted(shard1.parent.glob(shard1.name.replace('00001', '0000*')))
    found = 0
    for p in shards:
        f = G.GGUFFile(p)
        for t in f.tensors:
            if t.name.startswith('blk.3.') and t.name.split('blk.3.', 1)[1] in NAMES:
                be, bb = G.BLOCK_GEOMETRY.get(t.type_name, (None, None))
                if be is None:
                    be, bb = (1, 4) if t.type_name == 'F32' else (1, 2)
                n_elem = 1
                for d in t.shape:
                    n_elem *= d
                n_bytes = n_elem // be * bb
                with open(p, 'rb') as fh:
                    fh.seek(f.data_start + t.offset)
                    data = fh.read(n_bytes)
                (out / (t.name.replace('.', '_') + '.bin')).write_bytes(data)
                print(f"  {t.name:32s} {t.type_name:6s} {n_bytes} bytes")
                found += 1
    print(f"{found}/10 tensors -> {out}")
    return 0 if found == 10 else 1


if __name__ == '__main__':
    raise SystemExit(main())
