"""python -m rwkv_next [--model REPO_OR_PATH] [--tp N] [--quant int6|int8|bf16] [--mtp auto|off|PATH] [--host H] [--port P]
                     [--devices 0,1] [--max-batch N] [--stop]

Command-line options are shortcuts for the RWKV_* settings (they win over .env, like environment variables, and
show as locked in the WebUI). Without options the settings come from .env / the environment.

  python -m rwkv_next --model RWKV/RWKV7-G1l-7.2B-20261031 --tp 2      # download (once), load, serve on :8000
  python -m rwkv_next --stop                                          # stop the server started from this directory
"""
import argparse
import os

OPTS = (("model", "MODEL_PATH", str), ("tp", "TP", int), ("quant", "QUANT", str), ("mtp", "MTP_PATH", str),
        ("host", "HOST", str), ("port", "PORT", int), ("devices", "DEVICES", str), ("max_batch", "MAX_BATCH", int))


def cli():
    ap = argparse.ArgumentParser(prog="python -m rwkv_next", description="RWKV-Infer-Next server (OpenAI-compatible API + WebUI)")
    for opt, key, typ in OPTS:
        ap.add_argument("--" + opt.replace("_", "-"), dest=opt, type=typ, help=f"sets RWKV_{key}")
    ap.add_argument("--mtp-force", action="store_true", help="sets RWKV_MTP_FORCE=1: use the closest compatible MTP "
                    "sidecar for a model that has none of its own")
    ap.add_argument("--stop", action="store_true", help="stop the server started from this directory (PID file)")
    a = ap.parse_args()
    if a.mtp_force:
        os.environ["RWKV_MTP_FORCE"] = "1"
    for opt, key, _ in OPTS:                    # before the settings are created (they read the environment once)
        v = getattr(a, opt)
        if v is not None:
            os.environ["RWKV_" + key] = str(v)
    from .server import main
    main()


if __name__ == "__main__":
    cli()
