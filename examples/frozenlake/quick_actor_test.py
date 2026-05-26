"""Standalone trainer-forward sanity check for Gemma4-26B-A4B.

Tests SHORT (16 tok) and LONG (2048 tok) prompts to isolate whether the
training-time uniform-logits bug is sequence-length-specific (sliding window,
RoPE wrap, etc).

Set DEBUG_LOGITS=1 to enable jax.debug.print dumps from model.py + common.py.
"""
import os
import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx
from jax.experimental import mesh_utils
from jax.sharding import Mesh
from transformers import AutoTokenizer

from tunix.models.gemma4 import params_safetensors as params_lib
from tunix.models.gemma4 import model as model_lib
from tunix.rl import common as rl_common

MODEL_DIR = os.environ.get("MODEL_DOWNLOAD_DIR", "/data/models/gemma-4-26B-A4B-it")
MODEL_VERSION = "google/gemma-4-26B-A4B-it"

print(f"=== quick_actor_test ===\nMODEL_DIR={MODEL_DIR}\n", flush=True)
print("jax devices:", jax.devices(), flush=True)

devices = mesh_utils.create_device_mesh((1, len(jax.devices())))
mesh = Mesh(devices, axis_names=("fsdp", "tp"))
print("mesh:", mesh, flush=True)

config = model_lib.ModelConfig.gemma4_26b_a4b()
config.dtype = jnp.bfloat16
print("loading model...", flush=True)
model = params_lib.create_model_from_safe_tensors(
    MODEL_DIR, config, mesh, dtype=jnp.bfloat16
)
print("model loaded.", flush=True)

state = nnx.state(model)
print("\n=== weight-stat probe (post-load) ===", flush=True)

def stat(v, name):
  arr = np.asarray(jax.device_get(v.value.astype(jnp.float32)))
  print(f"  {name}: shape={arr.shape} mean={arr.mean():.6f} std={arr.std():.6f} min={arr.min():.4f} max={arr.max():.4f} first5={arr.flatten()[:5]}", flush=True)

stat(model.embedder.input_embedding, "embedder.input_embedding")
stat(model.final_norm.scale, "final_norm.scale")
stat(model.layers[0].attn._query_norm.scale, "L0.attn._query_norm.scale")
stat(model.layers[0].attn._key_norm.scale, "L0.attn._key_norm.scale")
stat(model.layers[5].moe.gating_einsum, "L5.moe.gating_einsum")
stat(model.layers[5].moe.linear, "L5.moe.linear")
stat(model.layers[5].moe.router_logits, "L5.moe.router_logits")
stat(model.layers[5].moe.router_scale, "L5.moe.router_scale")

# CRITICAL: verify skip_scale (layer_scalar) loaded — HF stores small fractions
# like 0.07, but tunix init is ones(1). If unloaded, forward dynamics are wrong.
print("\n=== skip_scale (layer_scalar) probe ===", flush=True)
for i in range(0, 30):
  arr = np.asarray(jax.device_get(model.layers[i].skip_scale.value.astype(jnp.float32)))
  print(f"  L{i}.skip_scale = {arr.flatten()[:1].tolist()}", flush=True)

tokenizer = AutoTokenizer.from_pretrained(MODEL_VERSION)
pad_id = tokenizer.pad_token_id or 0
eos_id = tokenizer.eos_token_id or 0


def run_forward(prompt_text, L):
  print(f"\n=== forward test: L={L} ===", flush=True)
  ids = tokenizer.encode(prompt_text, add_special_tokens=True)
  ids = ids[:L]  # truncate if too long
  if len(ids) < L:
    ids = ids + [pad_id] * (L - len(ids))
  print(f"len(ids)={len(ids)} first10={ids[:10]} last5={ids[-5:]}", flush=True)
  # 50/50 split — but always provide at least 1 real token in completion
  half = L // 2
  prompt_tokens = jnp.array([ids[:half]], dtype=jnp.int32)
  completion_tokens = jnp.array([ids[half:]], dtype=jnp.int32)
  graphdef, state = nnx.split(model)
  per_token_logps, logits = rl_common.compute_per_token_logps(
      graphdef, state,
      prompt_tokens=prompt_tokens,
      completion_tokens=completion_tokens,
      pad_id=pad_id,
      eos_id=eos_id,
      stop_gradient=True,
      return_logits=True,
  )
  per_token_logps = jax.device_get(per_token_logps)
  logits_np = jax.device_get(logits.astype(jnp.float32))
  print(f"per_token_logps[0,:5]={per_token_logps[0,:5]} mean={per_token_logps.mean():.4f}", flush=True)
  print(f"logits stats: mean={logits_np.mean():.4f} std={logits_np.std():.4f}", flush=True)
  pos_std = logits_np.std(axis=-1)
  print(f"  pos_std mean={pos_std.mean():.4f} max={pos_std.max():.4f} min={pos_std.min():.4f}", flush=True)
  # Top-5 at a few representative positions
  topk_idx = np.argsort(-logits_np, axis=-1)[:, :, :5]
  T = logits_np.shape[1]
  for p in [0, T // 4, T // 2, 3 * T // 4, T - 1]:
    top5 = topk_idx[0, p]
    decoded = tokenizer.decode(top5)
    print(f"  pos[{p}] (target_id={int(completion_tokens[0, min(p, completion_tokens.shape[1]-1)])}): top5_ids={top5.tolist()} decoded='{decoded}'", flush=True)


# Short prompt — known baseline
run_forward("The capital of France is", 16)

# Medium — past local-sliding-window/2
run_forward("The capital of France is Paris. " * 50, 512)

# Long — beyond sliding_window_size=1024
run_forward("The capital of France is Paris. " * 200, 2048)

print("\n=== quick_actor_test done ===", flush=True)
