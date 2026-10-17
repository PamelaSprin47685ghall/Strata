#!/usr/bin/env python3
"""Write the TP5 GDN tier's fixture: GDN layer 1's ten tensors, raw GGUF payload
bytes, one .bin per tensor (blk_1_<name>.bin).  The numeric gate compares the
5-rank split against the engine's own kernels; every layer's weights are
byte-identical to layer 1's in the test (the fixture IS layer 1's bytes)."""
import pathlib, sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "tools"))
import gguf_reader as G

TENSORS = ["attn_qkv.weight", "attn_gate.weight", "ssm_out.weight", "ssm_alpha.weight",
           "ssm_beta.weight", "ssm_conv1d.weight", "ssm_norm.weight", "ssm_dt.bias", "ssm_a"]

def main(model_dir, out_dir):
    out = pathlib.Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    d = pathlib.Path(model_dir)
    files = sorted(d.glob("Swift-Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-0000*-of-00002.gguf"))
    assert files, f"no shards under {d}"
    for p in files:
        gf = G.GGUFFile(p)
        for t in gf.tensors:
            if t.name.startswith("blk.1."):
                base = t.name.split("blk.1.")[1].replace(".", "_")
                dst = out / f"blk_1_{base}.bin"
                if dst.exists(): continue
                with open(p, "rb") as f:
                    f.seek(gf.data_start + t.offset)   # data_start: the aligned data region's start (the QSA fixture's own lesson)
                    data = f.read(t.expected_bytes())
                dst.write_bytes(data)
                print(f"{dst.name} {len(data)} bytes {t.type_name}")
    print("fixture complete:", out)

if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else "/tmp/tp5_gdn_fixture")
