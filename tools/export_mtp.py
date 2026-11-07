#!/usr/bin/env python3
"""Package an MTP sidecar for release: <out>/mtp/mtp.safetensors (+ draft_vocab) and <out>/mtp/mtp.json.

The engine picks it up automatically when it sits in the model folder (RWKV_MTP_PATH=auto). The meta gains
`n_layer` / `n_embd` / `base_model` of the model it was trained on, so the engine refuses it on another model
(a 7B sidecar would load on a 13B of the same width and only produce useless drafts).

  python tools/export_mtp.py --sidecar mtp-kl-attn-d3-fuse-dd.pt --model /path/to/RWKV7-G1l-7.2B-20261031 [--out <dir>]
"""
import argparse, json, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sidecar", required=True); ap.add_argument("--model", required=True)
    ap.add_argument("--out", default="", help="default: the model folder")
    a = ap.parse_args()
    from rwkv_next.mtp import load_sidecar
    from rwkv_next.model import RWKV7TP
    from rwkv_next.weights import open_weights, model_dir
    from safetensors.torch import save_file
    sd, meta, dv = load_sidecar(a.sidecar)
    cfg = RWKV7TP.config_of(open_weights(a.model))
    C = sd["lh.weight"].shape[0]
    if C != cfg["n_embd"]:
        sys.exit(f"sidecar width {C} != model n_embd {cfg['n_embd']}")
    fl = [int(x) for x in meta.get("fuse_layers") or []]
    if any(not 0 <= l < cfg["n_layer"] for l in fl):
        sys.exit(f"fuse_layers {fl} outside a {cfg['n_layer']}-layer model")
    meta = dict(meta, n_layer=cfg["n_layer"], n_embd=cfg["n_embd"], base_model=os.path.basename(os.path.abspath(a.model).rstrip("/")),
                format="rwkv-mtp", version=1)
    out = os.path.join(a.out or model_dir(a.model), "mtp")
    os.makedirs(out, exist_ok=True)
    t = {k: v.contiguous() for k, v in sd.items()}
    if dv is not None:
        t["draft_vocab"] = dv.to(torch.int64).contiguous()
    save_file(t, os.path.join(out, "mtp.safetensors"))
    json.dump({"meta": meta}, open(os.path.join(out, "mtp.json"), "w"), indent=1)
    n = sum(v.numel() for v in sd.values())
    print(f"wrote {out}/mtp.safetensors ({n / 1e6:.1f}M params) + mtp.json: {json.dumps(meta)}")


if __name__ == "__main__":
    main()
