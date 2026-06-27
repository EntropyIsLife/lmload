# `lmload` — LM Studio model loader

A small CLI wrapper around the LM Studio Python SDK that loads/unloads local models
with good defaults: **Flash Attention + Q8_0 KV cache**, **full GPU offload**, and a
**context length auto-fit to your GPU's VRAM** (computed per-model from its GGUF).

It exists because the stock `lms load` CLI can set only context length and GPU offload —
it **cannot** enable Flash Attention or KV-cache quantization, which is exactly what you
need to run large models at long context. `lmload` fills that gap and adds fuzzy model
matching, auto-unload-on-swap, and a VRAM planner.

> Tuned for a single **Quadro RTX 8000 (48 GB)** that's pinned to LM Studio. Adjust the
> constants at the top of the script for a different card.

---

## Contents
- [Requirements](#requirements)
- [Install](#install)
- [Commands](#commands)
- [Options](#options)
- [Examples](#examples)
- [Adding / removing models](#adding--removing-models)
- [How auto-fit works](#how-auto-fit-works)
- [Tuning](#tuning)
- [Gotchas](#gotchas)
- [Appendix: full script](#appendix-full-script)

---

## Requirements

- LM Studio headless daemon (`llmster`) running with its server up (`lms server status`).
- The LM Studio **Python SDK** in a venv. The script's shebang points at it:
  ```bash
  python3 -m venv ~/.lmstudio-sdk-venv
  ~/.lmstudio-sdk-venv/bin/pip install -U pip lmstudio
  ```
- Models downloaded via `lms get` (see [Adding models](#adding--removing-models)).

---

## Install

```bash
mkdir -p ~/.local/bin
# save the script (see appendix) to ~/.local/bin/lmload, then:
chmod +x ~/.local/bin/lmload
# ensure ~/.local/bin is on PATH:
grep -q '.local/bin' ~/.bashrc || echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc
exec $SHELL
```

The shebang (`#!/home/mumm/.lmstudio-sdk-venv/bin/python`) runs it with the SDK venv, so
you call `lmload` directly — no need to activate anything.

---

## Commands

| Command | What it does |
|---|---|
| `lmload <name>` | Fuzzy-match a downloaded model, **unload whatever's loaded**, then load it with Flash Attention + Q8_0 KV at the largest safe context. |
| `lmload <name> --dry-run` | Print the plan (arch, weights, KV size, chosen context, est VRAM) and load nothing. |
| `lmload --ls` | List downloaded models with size + native context. |
| `lmload --unload <name>` | Unload one loaded model by name. |
| `lmload --unload all` | Unload everything (free all VRAM). |

`<name>` is any unique substring of the model key. If it matches more than one model,
`lmload` prints the candidates so you can narrow it down.

---

## Options

| Flag | Default | Meaning |
|---|---|---|
| `--ctx <n\|auto>` | `auto` | Context length. `auto` = largest that safely fits; or pass a number (e.g. `65536`). |
| `--kv <type>` | `q8_0` | KV-cache quant: `f16`, `q8_0`, `q5_1`, `q5_0`, `q4_1`, `q4_0`, `iq4_nl`. Smaller = more context, lower quality. |
| `--gpu <ratio\|max>` | `max` (1.0) | GPU offload ratio (0–1). Use `<1` for models too big to fully offload. |
| `--ttl <sec\|none>` | `none` | Idle-unload timeout. `none` = stay resident (mapped to the ~24.8-day max). |
| `--no-flash` | off | Disable Flash Attention (forces f16 KV — caps context hard). |
| `--keep` | off | Don't unload other models first (load alongside). |
| `--dry-run` | off | Show the plan, don't load. |

---

## Examples

```bash
# Smart load — auto context, Q8 KV, swaps out the current model
lmload qwen3.6

# Preview the plan for any model without loading
lmload coder --dry-run

# Pin an exact context
lmload qwen3.6 --ctx 32768

# Trade KV quality for much more context
lmload qwen3.6 --kv q4_0            # auto-recomputes a bigger context

# A model too big to fully offload: spill some layers to CPU
lmload big-70b --gpu 0.8

# Auto-unload after 30 min idle instead of staying resident
lmload qwen3.6 --ttl 1800

# Free VRAM
lmload --unload all
```

Typical "trying many models" loop:

```bash
lms get "<hf-url-or-search>@Q6_K" -y   # download
lmload <name>                          # load (auto-config, frees previous)
#   ...test against http://localhost:1234/v1 ...
lmload <next>                          # swaps automatically
```

---

## Adding / removing models

**There is no model list in the script** — it discovers whatever LM Studio has downloaded.
So you never edit `lmload` to add a model; you just download one:

```bash
lms get "<huggingface-url-or-search>@<quant>" -y   # e.g. @Q6_K, @Q8_0, @IQ4_XS
lmload --ls                                        # it now appears
lmload <part-of-name>                              # load it
```

Remove from **VRAM**: `lmload --unload <name>` (or `all`).
Remove from **disk**: delete its folder under `~/.lmstudio/models/<publisher>/<repo>/`.

---

## How auto-fit works

The KV cache is what limits context. `lmload` reads the model's GGUF header for
`block_count`, `attention.head_count_kv`, `attention.key_length/value_length`, and
`context_length`, then:

```
bytes/token = n_layers × n_kv_heads × (key_len + value_len) × bytes_per_element(kv_type)
```

(`bytes_per_element`: f16 = 2.0, q8_0 ≈ 1.06, q4_0 ≈ 0.56.)

It computes the VRAM left after weights + reserves, fills **90 %** of it with KV cache
(rounded down to a multiple of 2048), and caps the result at the model's native context.
`--dry-run` shows the full breakdown. A `⚠ TIGHT` warning only appears when *you* override
`--ctx` past the safe-auto value — `auto` always leaves headroom by design.

---

## Tuning

Constants at the top of the script (edit for a different GPU or risk tolerance):

| Constant | Default | Purpose |
|---|---|---|
| `GPU_TOTAL_MIB` | `49152` | Total VRAM of the target card (RTX 8000 = 48 GiB). |
| `SYS_RESERVE_MIB` | `1024` | VRAM left for the desktop/Xorg on the card. |
| `COMPUTE_RESERVE_MIB` | `3072` | Headroom for llama.cpp compute/graph buffers. |
| `KV_BYTES` | — | Bytes/element per KV quant type. |

The `0.90` fill factor and `2048` rounding for `auto` are in `fit_context()`.

---

## Gotchas

- **TTL cap (~24.8 days).** LM Studio stores TTL as int32 **milliseconds**
  (`2^31 ms ≈ 2147483 s`). A larger value silently fails — the model unloads as soon as the
  client disconnects. `--ttl none` maps to that max (`2147483`). `ttl=None` in the raw SDK is
  *not* infinite either — it inherits the server's 1h JIT default.
- **Loading guardrails.** On a low-RAM box, LM Studio's `modelLoadingGuardrails` can refuse a
  GPU-offloaded model ("insufficient system resources") because it estimates memory ignoring
  GPU offload. Set `"mode": "off"` in `~/.lmstudio/settings.json` (stop the daemon first so the
  edit isn't clobbered). `lmload`'s own VRAM math is the replacement safety net.
- **Fuzzy match is a substring** of the model key — if ambiguous, `lmload` lists candidates and
  exits without loading.
- **Thinking models + small `max_tokens`** return empty content (the budget is spent on the
  hidden reasoning). Use a generous `max_tokens` (512+) when testing.
- **Config is applied at load time.** If a model later idle-unloads and the OpenAI endpoint
  JIT-reloads it, that reload uses LM Studio's *defaults*, not your `lmload` settings. Keep it
  resident (`--ttl none`, the default) or just re-run `lmload`.

---

## Appendix: full script

Save as `~/.local/bin/lmload`:

```python
#!/home/mumm/.lmstudio-sdk-venv/bin/python
"""lmload - load/unload LM Studio models with good defaults (flash-attn + Q8 KV,
auto-fit context to the RTX 8000's VRAM). See `lmload --help`."""
import argparse, os, struct, subprocess, sys

MODELS_DIR = os.path.expanduser("~/.lmstudio/models")
# RTX 8000 is the pinned card; total VRAM in MiB and reserves (MiB)
GPU_TOTAL_MIB = 49152
SYS_RESERVE_MIB = 1024        # Xorg/desktop etc. on the card
COMPUTE_RESERVE_MIB = 3072    # llama.cpp compute/graph buffers
# bytes per KV element by cache quant type
KV_BYTES = {"f16": 2.0, "f32": 4.0, "q8_0": 1.0625, "q5_1": 0.75, "q5_0": 0.6875,
            "q4_1": 0.625, "q4_0": 0.5625, "iq4_nl": 0.5625}

def gguf_arch(path):
    """Read just enough GGUF metadata to size the KV cache."""
    fh = open(path, "rb"); buf = bytearray(fh.read(1 << 16))
    def need(o, n):
        while len(buf) < o + n:
            c = fh.read(1 << 20)
            if not c: raise EOFError
            buf.extend(c)
    def U(o, n, f): need(o, n); return struct.unpack(f, buf[o:o+n])[0]
    if bytes(buf[:4]) != b"GGUF": raise ValueError("not a GGUF file")
    off = 8; U(off, 8, "<Q"); off += 8; kvc = U(off, 8, "<Q"); off += 8
    SZ = {0:1,1:1,2:2,3:2,4:4,5:4,6:4,7:1,10:8,11:8,12:8}
    FM = {0:"<B",1:"<b",2:"<H",3:"<h",4:"<I",5:"<i",6:"<f",7:"<?",10:"<Q",11:"<q",12:"<d"}
    def rstr(o): n = U(o,8,"<Q"); o += 8; need(o,n); return buf[o:o+n].decode("utf-8","replace"), o+n
    def rval(o,t):
        if t in SZ: need(o,SZ[t]); return struct.unpack(FM[t],buf[o:o+SZ[t]])[0], o+SZ[t]
        if t == 8: return rstr(o)
        if t == 9:
            et = U(o,4,"<I"); o += 4; n = U(o,8,"<Q"); o += 8
            for _ in range(n): _, o = rval(o, et)
            return None, o
        raise ValueError(t)
    kv = {}
    for _ in range(kvc):
        k, off = rstr(off); t = U(off,4,"<I"); off += 4; v, off = rval(off,t); kv[k] = v
    a = kv.get("general.architecture", "")
    g = lambda s: kv.get(f"{a}.{s}")
    return {
        "arch": a,
        "n_layer": g("block_count"),
        "n_kv": g("attention.head_count_kv"),
        "k_len": g("attention.key_length"),
        "v_len": g("attention.value_length"),
        "native_ctx": g("context_length"),
    }

def fit_context(meta, weight_mib, kv_type, want=None):
    """Auto-pick a context that fits with headroom (capped to native), or size `want`."""
    per_tok = meta["n_layer"] * meta["n_kv"] * (meta["k_len"] + meta["v_len"]) * KV_BYTES[kv_type]
    free_for_kv = GPU_TOTAL_MIB - SYS_RESERVE_MIB - COMPUTE_RESERVE_MIB - weight_mib
    # 'auto' fills to 90% of the KV budget and rounds down to 2048 -> real headroom
    auto_tok = int(free_for_kv * 0.90 * 1024 * 1024 / per_tok) // 2048 * 2048
    cap = meta["native_ctx"] or auto_tok
    fit = max(0, min(auto_tok, cap))
    chosen = min(want, cap) if want else fit
    kv_mib = chosen * per_tok / 1024 / 1024
    return chosen, fit, kv_mib, per_tok

def list_models():
    import lmstudio as lms
    return list(lms.list_downloaded_models("llm"))

def match(models, q):
    ql = q.lower()
    exact = [m for m in models if m.model_key.lower() == ql]
    if exact: return exact
    return [m for m in models if ql in m.model_key.lower()]

def cmd_ls():
    for m in list_models():
        p = os.path.join(MODELS_DIR, m.path)
        sz = os.path.getsize(p)/1e9 if os.path.exists(p) else 0
        try: nat = gguf_arch(p)["native_ctx"]
        except Exception: nat = "?"
        print(f"  {sz:6.1f}GB  native_ctx={nat:<8}  {m.model_key}")

def cmd_unload(target):
    import lmstudio as lms
    loaded = lms.list_loaded_models("llm")
    n = 0
    for m in loaded:
        ident = getattr(m, "identifier", "") or ""
        if target in ("all", "*") or target.lower() in ident.lower():
            try: m.unload(); print(f"  unloaded {ident}"); n += 1
            except Exception as e: print(f"  ! {ident}: {e}")
    if not n: print("  (nothing to unload)")

def main():
    ap = argparse.ArgumentParser(prog="lmload", description="Load an LM Studio model with flash-attn + quantized KV, context auto-fit to the RTX 8000.")
    ap.add_argument("query", nargs="?", help="fuzzy model name (substring of the model key)")
    ap.add_argument("--ctx", default="auto", help="context length: 'auto' (max that fits) or a number e.g. 65536")
    ap.add_argument("--kv", default="q8_0", choices=list(KV_BYTES), help="KV cache quant (default q8_0)")
    ap.add_argument("--gpu", default="1.0", help="GPU offload ratio 0-1 or 'max' (default max)")
    ap.add_argument("--ttl", default="none", help="idle unload seconds, or 'none' to stay loaded (default none)")
    ap.add_argument("--no-flash", action="store_true", help="disable flash attention (forces f16 KV)")
    ap.add_argument("--keep", action="store_true", help="don't unload other models first")
    ap.add_argument("--dry-run", action="store_true", help="show the plan, don't load")
    ap.add_argument("--ls", action="store_true", help="list downloaded models and exit")
    ap.add_argument("--unload", metavar="NAME", help="unload a loaded model (name or 'all') and exit")
    args = ap.parse_args()

    if args.ls: return cmd_ls()
    if args.unload is not None: return cmd_unload(args.unload)
    if not args.query: ap.error("give a model name, or use --ls / --unload")

    models = list_models()
    hits = match(models, args.query)
    if not hits:
        print(f"no model matches '{args.query}'. Try: lmload --ls"); sys.exit(1)
    if len(hits) > 1:
        print(f"'{args.query}' matches {len(hits)} models — be more specific:")
        for m in hits: print("   ", m.model_key)
        sys.exit(1)
    m = hits[0]; path = os.path.join(MODELS_DIR, m.path)
    weight_mib = os.path.getsize(path) / 1024 / 1024
    meta = gguf_arch(path)
    kv_type = "f16" if args.no_flash else args.kv
    want = None if args.ctx == "auto" else int(args.ctx)
    ctx, fitmax, kv_mib, per_tok = fit_context(meta, weight_mib, kv_type, want)
    est_total = weight_mib + kv_mib + COMPUTE_RESERVE_MIB + SYS_RESERVE_MIB

    print(f"model      {m.model_key}")
    print(f"arch       {meta['arch']}  layers={meta['n_layer']} kv_heads={meta['n_kv']} head_dim={meta['k_len']}+{meta['v_len']}")
    print(f"weights    {weight_mib/1024:.1f} GiB    native_ctx {meta['native_ctx']}")
    print(f"KV         {kv_type}  ({per_tok/1024:.2f} KiB/token)   safe-auto ctx ≈ {fitmax}")
    print(f"-> loading ctx={ctx}  flash={'off' if args.no_flash else 'on'}  gpu={args.gpu}  ttl={args.ttl}")
    warn = bool(want) and est_total > GPU_TOTAL_MIB - 1536
    print(f"   est VRAM ≈ {est_total/1024:.1f} GiB of {GPU_TOTAL_MIB/1024:.0f} GiB"
          + ("   ⚠ TIGHT — may OOM at full context" if warn else ""))
    if want and want > fitmax:
        print(f"   ⚠ requested {want} is above the safe-auto {fitmax}; loading as asked — watch nvidia-smi at full context")
    if args.dry_run: return

    import lmstudio as lms
    if not args.keep:
        cmd_unload("all")
    gpu_ratio = 1.0 if args.gpu == "max" else float(args.gpu)
    cfg = {"context_length": ctx, "gpu": {"ratio": gpu_ratio}}
    if args.no_flash:
        cfg["flash_attention"] = False
    else:
        cfg.update(flash_attention=True,
                   llama_k_cache_quantization_type=kv_type,
                   llama_v_cache_quantization_type=kv_type)
    # LM Studio caps TTL at int32 milliseconds (~24.8 days); larger silently fails to persist.
    # ttl=None would inherit the server's 1h JIT default, so 'none' maps to the max instead.
    MAX_TTL = 2147483
    ttl = MAX_TTL if args.ttl == "none" else min(int(args.ttl), MAX_TTL)
    print("   loading weights into VRAM (~30-90s)...", flush=True)
    model = lms.llm(m.model_key, ttl=ttl, config=cfg)
    eff = "~24.8d (max)" if args.ttl == "none" else f"{ttl}s"
    print(f"✅ loaded {model.identifier} @ ctx {ctx}  (ttl={eff})")

if __name__ == "__main__":
    main()
```
