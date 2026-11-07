#!/usr/bin/env python3
"""Hugging Face Hub resolution without the network: a local folder laid out like a sidecar repository
(index.json + one folder per base model) stands in for the Hub.

  python tools/test_hub.py [--repo-dir /path/to/local/copy/of/OpenMOSE/RWKV7-MTP-Exp]
"""
import argparse, json, os, sys, tempfile
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--repo-dir", default="")
    a = ap.parse_args()
    import huggingface_hub as hh
    from rwkv_next import hub
    tmp = tempfile.mkdtemp()
    repo_dir = a.repo_dir
    if not repo_dir:                                      # a tiny fake repository
        repo_dir = os.path.join(tmp, "repo")
        os.makedirs(os.path.join(repo_dir, "RWKV7-G1l-7.2B-20261031"))
        open(os.path.join(repo_dir, "RWKV7-G1l-7.2B-20261031", "mtp.safetensors"), "wb").write(b"x")
        json.dump({"format": "rwkv-mtp", "version": 1, "sidecars": [
            {"base_model": "RWKV7-G1l-7.2B-20261031", "path": "RWKV7-G1l-7.2B-20261031",
             "hf_base": ["RWKV/RWKV7-G1l-7.2B-20261031"]}]}, open(os.path.join(repo_dir, "index.json"), "w"))
    calls = []
    hh.hf_hub_download = lambda repo, fn, revision=None, **k: (calls.append(("file", repo, fn)), os.path.join(repo_dir, fn))[1]
    hh.snapshot_download = lambda repo, revision=None, allow_patterns=None, **k: (calls.append(("snap", repo, tuple(allow_patterns or ()))), repo_dir)[1]
    ok = True
    def check(name, got, want):
        nonlocal ok
        good = got == want
        ok &= good
        print(f"{'PASS' if good else 'FAIL'}  {name}: {got!r}" + ("" if good else f" (want {want!r})"))
    check("parse repo id", hub.parse_repo("RWKV/RWKV7-G1l-7.2B-20261031@main"), ("RWKV/RWKV7-G1l-7.2B-20261031", "main"))
    check("hf:// prefix", hub.parse_repo("hf://RWKV/RWKV7-G1l-7.2B-20261031"), ("RWKV/RWKV7-G1l-7.2B-20261031", None))
    check("local path is not a repo", hub.parse_repo("/mnt/models/x"), None)
    check("relative existing dir is not a repo", hub.parse_repo(os.path.relpath(tmp)), None)
    check("label of a repo id", hub.model_label("RWKV/RWKV7-G1l-13.3B-20261031"), "RWKV7-G1l-13.3B-20261031")
    check("label of a .pth", hub.model_label("/x/rwkv7-g1-1.5b.pth"), "rwkv7-g1-1.5b")
    q = lambda *a, **k: None
    p = hub.find_hub_sidecar("RWKV/RWKV7-G1l-7.2B-20261031", "OpenMOSE/RWKV7-MTP-Exp", note=q)
    check("sidecar by hf_base", p, os.path.join(repo_dir, "RWKV7-G1l-7.2B-20261031"))
    check("only the matching folder is fetched", calls[-1], ("snap", "OpenMOSE/RWKV7-MTP-Exp", ("RWKV7-G1l-7.2B-20261031/*",)))
    p = hub.find_hub_sidecar("/models/RWKV7-G1l-7.2B-20261031", "OpenMOSE/RWKV7-MTP-Exp", note=q)
    check("sidecar by local folder name", p, os.path.join(repo_dir, "RWKV7-G1l-7.2B-20261031"))
    check("no sidecar for another model", hub.find_hub_sidecar("RWKV/some-other-model", "OpenMOSE/RWKV7-MTP-Exp", note=q), None)
    check("empty MTP_REPO", hub.find_hub_sidecar("RWKV/RWKV7-G1l-7.2B-20261031", "", note=q), None)
    json.dump({"format": "rwkv-mtp", "version": 1, "sidecars": [
        {"base_model": "RWKV7-G1l-7.2B-20261031", "path": "RWKV7-G1l-7.2B-20261031", "n_layer": 32, "n_embd": 4096},
        {"base_model": "RWKV7-G1l-13.3B-20261031", "path": "RWKV7-G1l-13.3B-20261031", "n_layer": 61, "n_embd": 4096}]},
        open(os.path.join(tmp, "index2.json"), "w"))
    os.makedirs(os.path.join(repo_dir, "RWKV7-G1l-13.3B-20261031"), exist_ok=True)
    open(os.path.join(repo_dir, "RWKV7-G1l-13.3B-20261031", "mtp.safetensors"), "ab").close()
    idx2 = os.path.join(tmp, "index2.json")
    hh.hf_hub_download = lambda repo, fn, revision=None, **k: idx2
    new13 = "RWKV/RWKV7-G1x-13.3B-20270101"
    check("new model, no force -> off", hub.find_hub_sidecar(new13, "OpenMOSE/RWKV7-MTP-Exp", note=q), None)
    check("new 13.3B shape, force -> the 13.3B sidecar", hub.find_hub_sidecar(new13, "OpenMOSE/RWKV7-MTP-Exp", note=q, force=True, shape=(61, 4096)),
          os.path.join(repo_dir, "RWKV7-G1l-13.3B-20261031"))
    check("new 7B shape, force -> the 7B sidecar", hub.find_hub_sidecar("RWKV/RWKV7-G1x-7.2B-20270101", "OpenMOSE/RWKV7-MTP-Exp", note=q, force=True, shape=(32, 4096)),
          os.path.join(repo_dir, "RWKV7-G1l-7.2B-20261031"))
    check("force, other layer count -> nearest (40 -> 32)", hub.pick_compatible(json.load(open(idx2))["sidecars"], (40, 4096))["n_layer"], 32)
    check("force, other width -> none", hub.find_hub_sidecar("RWKV/x-2.9B", "OpenMOSE/RWKV7-MTP-Exp", note=q, force=True, shape=(32, 2560)), None)
    hh.hf_hub_download = lambda *a, **k: (_ for _ in ()).throw(OSError("offline"))
    check("offline -> MTP off, no exception", hub.find_hub_sidecar("RWKV/RWKV7-G1l-7.2B-20261031", "OpenMOSE/RWKV7-MTP-Exp", note=q), None)
    print("ALL PASSED" if ok else "FAILED")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
