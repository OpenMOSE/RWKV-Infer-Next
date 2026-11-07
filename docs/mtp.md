# MTP speculative decoding

This document specifies the MTP drafter that RWKV-Infer-Next uses for speculative decoding. It covers:
- what the drafter is (architecture);
- how it is shipped (sidecar format);
- how the engine finds, checks and runs it (state, prefill, the graphed speculative step, acceptance);
- how to configure and measure it.

The engine-wide architecture is in [architecture.md](architecture.md).

Status: the drafter architecture and the sidecar format (v1) are **frozen** (2026-11-08). Changing anything listed as
*fixed* in §1.2 is a new architecture and needs a new format version.

---

## 1. The drafter

### 1.1 Overview

```
                    main model (RWKV-7, L layers, width C), weights frozen
 token ─► emb ─► ln0 ─┬─► layer 0 … layer L-1 ─► ln_out ─► h_j ─────────────► head ─► p_main
                      │      │ outputs of layers f1, f2 ("taps")
                      │      ▼
                      │    m1_j, m2_j
                      └─► e_t = ln0(emb(t))            (shared with the main model)

 drafter: ONE block, applied K times per step, depth k predicts the token k positions ahead
   depth 1 : u = fproj · [ LN_lh(h_j) ; LN_lf0(m1_j) ; LN_lf1(m2_j) ]
   depth k : u = LN_lhd(o_{k-1})                                   (its own previous output)
   x  = LN_ln0( proj · [ u ; LN_le(e_{j+k}) ] + dep[k-1] )
   x  = x + GatedSWA( LN_ln1(x) )         RoPE position j+k-1, window W = 512
   x  = x + ChannelMix( LN_ln2(x) )       token shift = previous step of the drafter's stream
   o_k = LN_lo(x) ─► main head rows ─► draft d_k = argmax
```

- **What it is:** a small model (247.6M parameters for C = 4096), one block of gated sliding-window attention plus an RWKV channel-mix.
- **Inputs:** the main model's final hidden state and two of its intermediate residual streams.
- **Output:** a guess of the next K tokens.
- **What it does not do:** it never decides the output. The main model verifies every draft (§4), so the drafter changes only the speed.

### 1.2 Fixed and per-model choices

| Item | Status | Value |
|---|---|---|
| Blocks | fixed | 1, shared by all depths |
| Attention | fixed | 16 query heads, 4 KV heads (GQA), head_dim 128, q/k RMSNorm (eps 1e-6), sigmoid output gate, for every C |
| RoPE | fixed | rotate-half on the first 32 dims of each head, θ = 1e6, angles in fp32 or better |
| Channel-mix | fixed | RWKV-7 form: token shift with `x_k`, relu², hidden 4C |
| Depth-1 hidden input | fixed | `h` after `ln_out` + the **output** residual streams of 2 layers, fused by `fproj` |
| Depth ≥ 2 hidden input | fixed | the drafter's own post-`lo` output through `lhd` |
| Depth vector | fixed | `dep[8, C]`, row k−1 added at depth k (rows beyond the trained depth are zero) |
| Embedding / head | fixed | the main model's `ln0(emb)` and head; neither is stored in the sidecar |
| LayerNorm | fixed | eps 1e-5, affine; no biases in any Linear |
| Window W | fixed | 512 keys, the query's own key included |
| Training | fixed | main frozen; target = the main model's own distribution (KL); 3 depths unrolled, equal weights |
| `fuse_layers` (f1, f2) | per model | in the sidecar meta. Used: 32-layer models → 9, 19; 61-layer models → 17, 36 |
| Draft vocabulary | per sidecar | token ids by training frequency; the engine uses the first `MTP_DRAFT_VOCAB` (16384) |
| K (drafts per step) | per deployment | `MTP_K` (§5); may exceed the training depth |

**A sidecar belongs to one main model.** Equal width is not enough. Measured: a sidecar trained for a 7B model, loaded on a 13B model of the same width, gives 1.03 tokens per main forward, i.e. no speed-up.

### 1.3 Tensors

C = model width, H = 16, Hk = 4, D = 128. Linear weights are `[out, in]`.

| Name | Shape | Role |
|---|---|---|
| `lh.{weight,bias}` | [C] | LayerNorm of `h` |
| `lf.0.*`, `lf.1.*` | [C] | LayerNorms of the two taps (meta `fuse_layers` order) |
| `fproj.weight` | [C, 3C] | fusion projection |
| `lhd.{weight,bias}`, `dep` | [C], [8, C] | depth adapter |
| `le.{weight,bias}` | [C] | LayerNorm of `e` |
| `proj.weight` | [C, 2C] | input projection |
| `blk.ln0.*`, `blk.ln1.*`, `blk.ln2.*` | [C] | block LayerNorms |
| `attn.q.weight`, `attn.g.weight` | [2048, C] | query, gate |
| `attn.k.weight`, `attn.v.weight` | [512, C] | key, value |
| `attn.o.weight` | [C, 2048] | output |
| `attn.q_norm.weight`, `attn.k_norm.weight` | [128] | per-head RMSNorm (shared over heads) |
| `blk.ffn.x_k` | [1, 1, C] | channel-mix shift mix |
| `blk.ffn.key.weight`, `blk.ffn.value.weight` | [4C, C], [C, 4C] | channel-mix |
| `lo.{weight,bias}` | [C] | output LayerNorm |
| `draft_vocab` | [V] int64 | draft vocabulary order (most frequent first) |

### 1.4 One step (depth k at absolute position p = j + k − 1)

```
e  = ln0(emb(token))                       token = the one being drafted from
u  = (k == 1) ? fproj·[LN_lh(h); LN_lf0(m1); LN_lf1(m2)] : LN_lhd(prev_out)
x  = LN_ln0( proj·[u; LN_le(e)] + dep[k-1] )
xl = LN_ln1(x)
q  = RoPE(RMS(Wq·xl as [H, D]), p);  kk = RoPE(RMS(Wk·xl as [Hk, D]), p);  v = Wv·xl as [Hk, D]
write (kk, v) at position p, then attend: y_h = softmax(q_h·K^T/√D)·V over visible keys (§3.2); head h reads KV head h // 4
x  = x + Wo·( concat(y) ⊙ sigmoid(Wg·xl) )
x2 = LN_ln2(x);  kx = x2 + (shift − x2) ⊙ x_k;  x = x + Wvalue·relu(Wkey·kx)²
out = LN_lo(x);  draft = draft_vocab[ argmax(head[draft_vocab[:N]] · out) ]
next shift = x2
```

Attention scores, softmax and the draft logits are computed in fp32. On an argmax tie, the lowest index wins.

---

## 2. Sidecar files

### 2.1 Release layout (preferred)

```
<model folder>/
  config.json, model.safetensors.index.json, model-*.safetensors, chat_template.jinja, ...
  mtp/
    mtp.safetensors       the tensors of §1.3 (bf16) + draft_vocab (int64)
    mtp.json              {"meta": {...}}
```

`tools/export_mtp.py` writes this layout from a trainer's `.pt` sidecar:
```bash
python tools/export_mtp.py --sidecar <trainer sidecar .pt> --model <model folder> [--out <dir>]
# --out defaults to the model folder; the result is <out>/mtp/
```

It also checks that the sidecar's width and `fuse_layers` fit the model, and records the model in the meta.

### 2.2 Meta keys

| Key | Meaning |
|---|---|
| `format`, `version` | sidecar format identifier and `1`. The loader does not check the identifier string; any v1 sidecar loads |
| `kind` | `"attn"` |
| `n_embd` | C |
| `n_layer`, `base_model` | the main model's layer count and folder name, added by `export_mtp.py`. The engine refuses a sidecar whose `n_layer` differs from the model |
| `fuse_layers` | the two tapped layers (0-based, the layer's output) |
| `dd` | `true` (depth adapter present) |
| `depth` | training depth (3) |
| `window` / `ctx` | W. When both are missing, 512 is used |
| `attn` | optional `{rope_dim, rope_theta}`; defaults 32 / 1e6. Head counts are read from the tensor shapes |
| `ln_eps`, `rms_eps` | optional; defaults 1e-5 / 1e-6 |
| `model`, `tokens`, `target` | provenance (trained-on model, training tokens, `"kl"`) |

### 2.3 Other accepted forms

`rwkv_next/mtp.py: load_sidecar` reads:
- a directory with `mtp.safetensors` + `mtp.json`;
- a single `.safetensors` file whose header metadata holds the meta as JSON under `mtp_meta`;
- a trainer's PyTorch `.pt` file `{"state_dict", "meta", "draft_vocab"}`.

### 2.4 Discovery and checks

`RWKV_MTP_PATH` (`rwkv_next/drafter.py: find_sidecar`):

| Value | Effect |
|---|---|
| `auto` (default) | in the model folder, in this order: `mtp/` (with `mtp.safetensors`), the first `*.mtp.safetensors`, the first `*.mtp.pt`. Nothing found there → the sidecar repository `RWKV_MTP_REPO` (§2.5). Nothing found → MTP off |
| a path | that sidecar (directory or file) |
| `org/name[@rev]` | a sidecar repository on the Hugging Face Hub (§2.5) |

**Force mode** (`RWKV_MTP_FORCE=1`, `--mtp-force`) is for new releases that have no sidecar of their own.
- In the sidecar repository, a model without a match takes the entry of the same `n_embd` whose `n_layer` is closest. On a tie, the later entry wins.
- An `n_layer` mismatch is a warning instead of an error.
- The two tapped layers are remapped to the same relative depth: `l → round((l+1)·L_model/L_trained) − 1`.
- The status shows `[forced: trained for another model]`.
- A forced sidecar never changes the output, only the speed, and the speed-up depends on how close the new model is to the one the sidecar was trained on. A poorly matching sidecar can be slower than no MTP. Measured: the 7.2B sidecar forced onto the 13.3B model gave 66 tok/s against 78 tok/s without MTP.
| empty / `off` / `none` / `0` | MTP off |

At load (`DraftPool.__init__`), the engine refuses the sidecar with an error when any of these fail:
- `n_layer` matches the model (when present);
- the width matches;
- every `fuse_layers` index is inside the model.

The status endpoint reports the sidecar's base model (`info.mtp`).

### 2.5 Sidecar repositories on the Hugging Face Hub

`rwkv_next/hub.py`.

**Layout.** A sidecar repository (default `RWKV_MTP_REPO=OpenMOSE/RWKV7-MTP-Exp`) holds one folder per base model, laid out as in §2.1, and an index:

```json
{"format": "rwkv-mtp", "version": 1,
 "sidecars": [{"base_model": "RWKV7-G1l-7.2B-20261031", "path": "RWKV7-G1l-7.2B-20261031",
               "hf_base": ["RWKV/RWKV7-G1l-7.2B-20261031"], "n_layer": 32, "n_embd": 4096}]}
```

**Matching.** An entry matches when either:
- its `base_model` equals the model's name (the local folder name, or the repository's name part);
- or `hf_base` lists the model's repository id.

**Download.** Only the matching folder is downloaded, into the Hugging Face cache.
- When offline, or when the repository has no index or no entry, the engine starts without MTP and notes why in the log.
- `RWKV_MTP_REPO=` (empty) disables the lookup.

**Resolution happens in the front process, before the ranks start.**
- `RWKV_MODEL_PATH` may itself be a repository id (`RWKV/RWKV7-G1l-7.2B-20261031`, optionally `@revision`). Its safetensors, JSON and template files are downloaded once.
- The ranks receive local paths only.

---

## 3. Runtime state and protocol

### 3.1 Where things run

- **Rank 0:** the drafter runs here only, next to the head and the sampler.
- **Other ranks:** they hold no drafter. They receive the drafts and the accepted counts by NCCL broadcast inside the speculative-step graph (§4).
- **Quantisation:** the drafter's large matrices are quantised weight-only to int6 at load (`MTPDrafter.quantize`; 0.5 GB bf16 → about 0.19 GB). The draft-logit head rows are int8.
  - Drafts only steer speed, so quantisation can change the acceptance rate but never the output.
- **Compilation:** with `RWKV_COMPILE=1` the block is `torch.compile`d with static shapes.

### 3.2 Per-slot drafter state (`DraftPool`, one entry per slot of the main state pool)

| Field | Shape | Content |
|---|---|---|
| `kc`, `vc` | [slots, 4, S+1, 128] | KV ring, S = W + K_max + 1. Position p lives in slot p mod S. Slot S is a dummy that absorbs masked writes, so shapes stay static |
| `slot_pos` | [slots, S+1] | absolute position held by each ring slot (−1 = empty) |
| `ffn_x` | [slots, C] | channel-mix shift = `x2` of the last **committed** position |
| `pos` | [slots] | committed positions = RoPE position of the next depth-1 step |
| `H` | [slots, 3C] | pending drafter input `[h; m1; m2]` of the last position the main model processed |

**Visibility rule.** A query at absolute position qp sees a key at sp only if `0 ≤ sp ≤ qp` and `qp − sp < W`.
- Keys are written before attending, so the query sees itself.
- One call handles several rows at once. Its queries see two kinds of keys under the rule:
  - the ring entries older than the first position written by the call;
  - the call's own new keys.
- This equals one-row-at-a-time processing.
- Chain steps (depth ≥ 2) write their keys at positions ahead of `pos`. Those positions are "future" for every later committed query until a commit overwrites them, so **a rejected draft needs no clean-up**.

The state lives in the same slot pool as the main RWKV state.
- `reset`, `move`, `view` follow the main pool's slot moves (decode rows at `[0, nd)`).
- A decode batch of B rows is the view `[:B]`.

### 3.3 Operations

| Operation | When | What it does |
|---|---|---|
| `commit_rows(slot, rows, toks, start)` | each prefill chunk | depth-1 steps for (H of position i, token i+1) |
| `chain` / `draft(B, t, K)` | start of a speculative step | commit (H, t) at depth 1, then K−1 more depths fed their own outputs. Returns K drafts per row |
| `accept(B, Hs, drafts, j)` / `accept_graph` | after verify | commit the j accepted drafts at depth 1 (one batched call); `H := Hs[j]` |
| `plain_step(B, H_new, t_prev)` | a plain (non-speculative) decode step | commit (H, t_prev); `H := H_new` |

### 3.4 Prefill

`engine.py: Scheduler._prefill_step`.

**Main model.** The prompt runs through the main model in chunks (`PREFILL_CHUNK`, smaller while other rows decode). With MTP on, the main model also returns the drafter input rows `[h; m1; m2]` for the requested positions (`h_positions`).

**Which positions are committed.** Only the last W positions before the final one, `q ∈ [max(0, L−1−W), L−1)`. Each is committed as a (row of q, token q+1) pair in prompt order. Older positions fall outside every future query's window, so they are skipped.

**The final position.** Its row is stored as the pending `H`, and the first speculative step commits it together with the first sampled token.

**Chunks with no drafter rows.** When a prompt is longer than one chunk plus W + 1, its early chunks contain no position of that range. The prefill then returns only the main hidden states, and the scheduler handles that case explicitly.

---

## 4. The speculative step

### 4.1 Steps

`rwkv_next/spec.py: SpecGraphs`. **One CUDA graph per rank** per (batch size, K):

```
rank 0                                              other ranks
  drafts = chain(H, t, K)                               .
  toks = [t | d1 .. dK]     ── NCCL broadcast ──►       toks
  main verify of K+1 tokens per row (compiled layers; 2 all-reduces per layer; identical on every rank)
  picks[i] = sample(main logits at position i), i = 0..K      (in-graph sampler, every position)
  j = length of the prefix with picks[i] == d_{i+1}
  n = j + 1                 ── NCCL broadcast ──►       n
  rollback main state to n tokens (WKV replay of the first n recorded steps, token shifts from step n−1)
  drafter: commit the j accepted drafts, H := H at position j
```

Rank 0 synchronises with the host once per step, to read `picks` and `n`. The workers never synchronise.

### 4.2 The output distribution is unchanged

- **Where every emitted token comes from:** it is the sampler's own pick from the **main model's** distribution at that position, given the tokens already accepted. That is the same sampling procedure plain decoding uses.
- **What a draft decides:** only whether the main model's pick for the next position can be used in the same step.
  - It can be used when the draft equals the pick (the prefix is consistent).
  - Otherwise the step ends with that pick.
- **Consequences:**
  - The token sequence has the same distribution as plain decoding, for greedy and for sampling.
  - In bf16, a K+1-token verify rounds differently from a 1-token step, so individual greedy runs can differ at near-ties. This is not a correctness bug.

### 4.3 When the plain path is used instead

- **K = 0:** the K rule (§5) gives K = 0 for this batch size.
- **No captured graph fits:** an eager speculative step (same math) is used. That step has no penalty state, so a batch where some row uses a presence or frequency penalty goes plain in this case.

The graphed step samples exactly for every setting except `top_k` > 1024:
- top-p / top-k truncation uses the top 1024 logits;
- no truncation uses Gumbel-max over the whole vocabulary;
- presence / frequency penalties: the penalty counts advance position by position along the drafts, and the accepted tokens are committed inside the graph.

Penalties and full-vocabulary sampling therefore keep the MTP speed-up.

---

## 5. Configuration

| Key | Default | Meaning |
|---|---|---|
| `RWKV_MTP_PATH` | `auto` | sidecar discovery (§2.4) |
| `RWKV_MTP_K` | `4:3,16:2,0:0` | K by decode batch size: the first rule `limit:K` with B ≤ limit applies; `0:` = no limit. Default: B ≤ 4 → K = 3, B ≤ 16 → K = 2, larger → no speculation |
| `RWKV_MTP_DRAFT_VOCAB` | 16384 | first N tokens of the draft vocabulary used for drafting (restart) |

Why K shrinks with the batch size:
- At small batch sizes, decoding is memory-bound. Verifying K+1 tokens costs little more than one, so drafts are almost free.
- At large batch sizes, the GPU is already busy. Rejected drafts then cost real compute.

---

## 6. Measured

Setup:
- Models: G1 7.2B / 13.3B, int6, tensor parallel over two PCIe 16 GB GPUs without P2P.
- Load: single stream, mean of 6 chat prompts (ja chat, proofreading, translation, en chat, code, zh).
- Date: 2026-11-07/08.

| Model | Sidecar training | No MTP | With MTP | Speed-up |
|---|---|---|---|---|
| 7.2B | 25M tokens | 140 tok/s | 191–196 tok/s | 1.4× |
| 7.2B | 300M tokens | 140 tok/s | **252–258 tok/s** | **1.8×** |
| 13.3B | 30M tokens | 78 tok/s | 119 tok/s | 1.5× |
| 13.3B | 90M tokens | 78 tok/s | 134 tok/s | 1.7× |
| 13.3B | 270M tokens | 78 tok/s | **154 tok/s** | **2.0×** |

On **one** 16 GB GPU (int6, TP1):

| Model | Slots (`MAX_BATCH`) | GPU memory | No MTP | With MTP | Speed-up |
|---|---|---|---|---|---|
| 7.2B (300M sidecar) | 32 | 9.4 GB | 117 tok/s | 226–228 tok/s | 1.9× |
| 13.3B (270M sidecar) | 16 (32 does not fit) | 14.5 GB | 63–64 tok/s | 136 tok/s | 2.1× |

- **By prompt type:** acceptance is highest for proofreading (7.2B, 300M: 274–339 tok/s), and lowest for English chat and Chinese (≈ 200 tok/s).
- **More training helps:** 25M → 300M tokens took the 7.2B from 1.4× to 1.8×.

---

## 7. Training a sidecar (summary)

Sidecars are trained outside this repository, with the main model frozen. The trainer is RWKV-LM `RWKV-v7/train_temp/tools/mtp_train.py`, documented in its `docs/mtp-training.md` and `docs/mtp-architecture.en.md`.

| Item | Recipe |
|---|---|
| Architecture arm | `kl:attn+d3+fuse+dd` |
| Training windows | ctx 512 |
| Optimiser | lr 1e-3 (warm-up 200 steps), AdamW β = (0.9, 0.99) without weight decay, gradient clip 1.0 |
| Training length | 100–300M tokens of chat-heavy multilingual data |
| Main model during training | bf16, or weight-only int8 when it does not fit |
| Packaging | `tools/export_mtp.py` (§2.1) |

---

## 8. Testing

| Command | Checks |
|---|---|
| `python tools/test_engine.py --ckpt <model> --mtp <sidecar> [--chat]` | batched speculative decoding vs solo plain decoding, token by token (differences only at near-ties) |
| `python tools/test_tp.py --ckpt <model> --check` | verify / rollback consistency under TP |
| `python tools/test_wkv7.py` | the WKV kernel, including the in-place rollback replay |
| `/api/status` | `stats.mtp_tok_per_step` (tokens per main forward while decoding) |
