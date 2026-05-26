"""Agentic FrozenLake GRPO recipe for Gemma4-26B-A4B (MoE) on a single host.

This recipe targets a single TPU host (e.g. v5p-8) where actor, reference, and
rollout share one mesh. Configuration is env-driven so the same image runs
unchanged on any spot VM:

  HF_TOKEN              Hugging Face token for model download.
  WANDB_API_KEY         Wandb API key (auto-picked-up by wandb lib).
  WANDB_PROJECT         Wandb project name (default "tunix-frozenlake").
  WANDB_RUN_NAME        Wandb run name (default uses timestamp).
  MODEL_DOWNLOAD_DIR    Local dir for HF safetensors (default
                        /tmp/models/gemma-4-26B-A4B-it).
  DATA_DIR              Local or gs:// dir holding train.parquet / test.parquet
                        (default /tmp/data/frozenlake).
  CKPT_DIR              Output checkpoint dir. Checkpointing is opt-in; if
                        unset, no checkpoints are written.
  TB_LOG_DIR            TensorBoard log dir (default /tmp/tunix-tb/frozenlake).
  SHARED_MESH_SHAPE     Override the (fsdp, tp) mesh shape. Defaults to
                        (1, jax.device_count()) (pure tensor parallel).
  ROLLOUT_ENGINE        "vanilla" | "vllm"  (default "vllm").
  MODEL_DTYPE           "bf16" (default) | "fp32" — storage/compute dtype for
                        the reference policy and trainer forward path.
  MIX_PRECISION         "1" (default) | "0" — when 0, runs the model in fp32
                        end-to-end (set together with MODEL_DTYPE=fp32).
  FLASH_ATTN            "1" (default) | "0" — splash flash attention kernel.
  REMAT                 "block" (default) | "decoder" | "0" — gradient
                        checkpoint granularity. "block" remats per Attention /
                        FeedForward (finer-grained, lower peak HBM).
  ENABLE_THINKING       "0" (default) | "1" — Gemma4 chat-template thinking
                        channel. Disabled by default so the agent answers
                        directly without producing internal thoughts that
                        consume the response budget.
  TRAIN_MICRO_BS        Trainer forward+backward micro-batch (default 1).
  COMPUTE_LOGPS_MICRO_BS  Logp recomputation micro-batch (default 1).
"""

import contextlib
import datetime
import logging
import math
import os
import sys
from typing import List

from absl import logging as absl_logging
from flax import nnx
import grain
import jax
from jax import numpy as jnp
import numpy as np
import optax
from orbax import checkpoint as ocp
import qwix

absl_logging.use_python_logging()
logging.basicConfig(
    stream=sys.stdout,
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - [%(name)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    force=True,
)
logging.getLogger().setLevel(logging.INFO)
logging.getLogger("absl").setLevel(logging.INFO)
absl_logging.set_verbosity(absl_logging.INFO)
absl_logging.set_stderrthreshold("info")
print("Logging configured at INFO level.")

from tunix.models.gemma4 import params_safetensors as params_lib
from tunix.models.gemma4 import model as model_lib
from tunix.oss import utils as oss_utils
from tunix.sft import metrics_logger
from tunix.rl.agentic.agentic_grpo_learner import GRPOConfig, GRPOLearner
from tunix.rl.agentic.parser.chat_template_parser import parser
from tunix.rl import rl_cluster as rl_cluster_lib
from tunix.rl.rollout import base_rollout
from tunix.sft import utils as sft_utils
from tunix.cli.utils import data as data_lib
from examples.frozenlake.agent import FrozenLakeAgent
from examples.frozenlake.env import FrozenLakeEnv

_DISTRIBUTED_INITIALIZED = False
try:
  import pathwaysutils
  pathwaysutils.initialize()
  _DISTRIBUTED_INITIALIZED = True
except Exception:
  pass

if not _DISTRIBUTED_INITIALIZED:
  try:
    jax.distributed.initialize()
  except Exception as exc:
    print(f"jax.distributed.initialize() skipped: {exc}")

print("jax devices: ", jax.devices())
try:
  stats = jax.devices()[0].client.memory_stats()
  print(f"--- Startup HBM Reserved Memory: {stats['bytes_reserved'] / 1e9:.2f} GB ---")
except Exception as e:
  print(f"Failed to query startup HBM stats: {e}")

# %%
import argparse

arg_parser = argparse.ArgumentParser(
    description="Train FrozenLake on Gemma4-26B-A4B (single-host TPU)."
)
# Gemma4-26B-A4B is much larger than Qwen3-8B; keep batch small by default so
# the debug-run path works on a single v5p-8. Scale up only after the first
# few steps complete cleanly.
arg_parser.add_argument("--batch_size", type=int, default=8)
arg_parser.add_argument("--mini_batch_size", type=int, default=8)
arg_parser.add_argument("--learning_rate", type=float, default=1e-6)
arg_parser.add_argument("--b1", type=float, default=0.9)
arg_parser.add_argument("--b2", type=float, default=0.95)
arg_parser.add_argument("--weight_decay", type=float, default=0.0)
arg_parser.add_argument("--num_batches", type=int, default=150)
arg_parser.add_argument("--num_generations", type=int, default=8)
arg_parser.add_argument("--beta", type=float, default=0.0)
arg_parser.add_argument("--epsilon", type=float, default=0.003)
arg_parser.add_argument("--epsilon_high", type=float, default=0.005)
arg_parser.add_argument(
    "--loss_algo", type=str, default="gspo-token",
    help="'grpo' (per-token PPO) or 'gspo-token' (sequence-mean IS).",
)
arg_parser.add_argument("--max_prompt_length", type=int, default=2048)
arg_parser.add_argument("--max_response_length", type=int, default=2048)
arg_parser.add_argument("--temperature", type=float, default=0.7)
arg_parser.add_argument("--top_p", type=float, default=1.0)
arg_parser.add_argument("--top_k", type=int, default=0)
arg_parser.add_argument("--max_concurrency", type=int, default=64)
arg_parser.add_argument("--shuffle_data", type=bool, default=True)
arg_parser.add_argument("--seed", type=int, default=42)
arg_parser.add_argument(
    "--loss_agg_mode", type=str, default="sequence-mean-token-mean"
)
arg_parser.add_argument("--kl_loss_mode", type=str, default="low_var_kl")
arg_parser.add_argument(
    "--advantage_estimator", type=str, default="rloo",
    help="'grpo' (z-score) or 'rloo' (leave-one-out baseline).",
)
args, _ = arg_parser.parse_known_args()

TRAIN_FRACTION = 1.0
SEED = args.seed

# ====== Sharding ======
# Two layouts supported:
#   1. Shared mesh (default): actor/reference/rollout all live on the same
#      chips. Works only single-host because vLLM-TPU UniProc shares the JAX
#      backend in-process. Set via SHARED_MESH_SHAPE.
#   2. Disjoint meshes (multi-host or when actor+rollout cannot co-locate):
#      set TRAINER_MESH_SHAPE and ROLLOUT_MESH_SHAPE. Devices are sliced from
#      jax.devices() so the two meshes never overlap. Reference shares the
#      trainer mesh unless REFERENCE_MESH_SHAPE is also set.
_trainer_env = os.getenv("TRAINER_MESH_SHAPE")
_rollout_env = os.getenv("ROLLOUT_MESH_SHAPE")
_reference_env = os.getenv("REFERENCE_MESH_SHAPE")
DISJOINT_MESH = bool(_trainer_env and _rollout_env)
if DISJOINT_MESH:
  TRAINER_MESH_SHAPE = tuple(int(x) for x in _trainer_env.split(","))
  ROLLOUT_MESH_SHAPE = tuple(int(x) for x in _rollout_env.split(","))
  REFERENCE_MESH_SHAPE = (
      tuple(int(x) for x in _reference_env.split(",")) if _reference_env else None
  )
  SHARED_MESH_SHAPE = None  # not used in disjoint mode
else:
  _mesh_env = os.getenv("SHARED_MESH_SHAPE")
  if _mesh_env:
    SHARED_MESH_SHAPE = tuple(int(x) for x in _mesh_env.split(","))
  else:
    SHARED_MESH_SHAPE = (1, jax.device_count())
SHARED_MESH_AXIS_NAMES = ("fsdp", "tp")

# ====== GRPO ======
MAX_PROMPT_LENGTH = args.max_prompt_length
MAX_RESPONSE_LENGTH = args.max_response_length
TEMPERATURE = args.temperature
TOP_P = args.top_p
TOP_K = args.top_k
NUM_GENERATIONS = args.num_generations

VLLM_MAX_NUM_SEQS = 64
VLLM_MAX_BATCHED_TOKENS = VLLM_MAX_NUM_SEQS * 4 * 1024 // 8

NUM_ITERATIONS = 1
BETA = args.beta
EPSILON = args.epsilon
EPSILON_HIGH = args.epsilon_high

# ====== Training ======
_REMAT_ENV = os.environ.get("REMAT", "block").lower()
ENABLE_FLASH_ATTENTION = os.environ.get("FLASH_ATTN", "1") not in ("0", "false", "False")
ENABLE_MIX_PRECISION = os.environ.get("MIX_PRECISION", "1") not in ("0", "false", "False")
ENABLE_THINKING = os.environ.get("ENABLE_THINKING", "0") not in ("0", "false", "False")
# LoRA. When set, the actor is wrapped in qwix LoRA adapters and only the
# adapters are trained. Lets the full ~26B base stay bf16 (frozen, no
# precision risk) while keeping a tiny fp32 Adam state on the adapters. The
# only viable option on memory-bound single-host slices (e.g. v6e-8 ~256GB)
# for the 26B parameter MoE.
TRAIN_WITH_LORA = os.environ.get("TRAIN_WITH_LORA", "0") not in ("0", "false", "False")
LORA_RANK = int(os.environ.get("LORA_RANK", "64"))
LORA_ALPHA = float(os.environ.get("LORA_ALPHA", "64.0"))
BATCH_SIZE = args.batch_size
MINI_BATCH_SIZE = args.mini_batch_size
NUM_BATCHES = args.num_batches
NUM_TEST_BATCHES = 2

EVAL_EVERY_N_STEPS = 10
NUM_EPOCHS = 3
MAX_STEPS = int(NUM_BATCHES * NUM_ITERATIONS * TRAIN_FRACTION * NUM_EPOCHS)

MAX_CONCURRENCY = args.max_concurrency
OFF_POLICY_STEPS = 0
MODEL_DTYPE = {"bf16": jnp.bfloat16, "bfloat16": jnp.bfloat16, "fp32": jnp.float32, "float32": jnp.float32}[
    os.environ.get("MODEL_DTYPE", "bf16").lower()
]

LEARNING_RATE = args.learning_rate
B1 = args.b1
B2 = args.b2
WEIGHT_DECAY = args.weight_decay
WARMUP_STEPS = 0
MAX_GRAD_NORM = 100.0

# ====== Checkpoint saving ======
SAVE_INTERVAL_STEPS = 10**9
MAX_TO_KEEP = 1

# ====== Rollout ======
ROLLOUT_ENGINE = os.getenv("ROLLOUT_ENGINE", "vllm")

# ====== Paths ======
# MODEL_VARIANT selects which Gemma4 size to train.
#   "26b-a4b" (default): Gemma4-26B-A4B MoE instruct
#   "31b": Gemma4-31B dense instruct
GEMMA4_VARIANT = os.getenv("GEMMA4_VARIANT", "26b-a4b").lower()
if GEMMA4_VARIANT == "26b-a4b":
  MODEL_VERSION = "google/gemma-4-26B-A4B-it"
  _DEFAULT_MODEL_DIR = "/tmp/models/gemma-4-26B-A4B-it"
elif GEMMA4_VARIANT == "31b":
  MODEL_VERSION = "google/gemma-4-31B-it"
  _DEFAULT_MODEL_DIR = "/tmp/models/gemma-4-31B-it"
else:
  raise ValueError(
      f"Unsupported GEMMA4_VARIANT={GEMMA4_VARIANT}; expected '26b-a4b' or '31b'."
  )
MODEL_DOWNLOAD_DIR = os.getenv("MODEL_DOWNLOAD_DIR", _DEFAULT_MODEL_DIR)
DATA_DIR = os.getenv("DATA_DIR", "/tmp/data/frozenlake")

now_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
CKPT_DIR = os.getenv("CKPT_DIR") or None
TB_LOG_DIR = os.getenv("TB_LOG_DIR", "/tmp/tunix-tb/frozenlake")


# ====== Build the role meshes ======
if DISJOINT_MESH:
  _trainer_chips = math.prod(TRAINER_MESH_SHAPE)
  _rollout_chips = math.prod(ROLLOUT_MESH_SHAPE)
  _ref_chips = math.prod(REFERENCE_MESH_SHAPE) if REFERENCE_MESH_SHAPE else 0
  _need = _trainer_chips + _rollout_chips + _ref_chips
  if jax.device_count() < _need:
    raise ValueError(
        f"Disjoint mesh layout needs {_need} chips "
        f"(trainer={_trainer_chips}+rollout={_rollout_chips}+ref={_ref_chips}), "
        f"have {jax.device_count()}."
    )
  # Layout in jax.devices() order: [rollout | reference | trainer]
  _all_devs = jax.devices()
  _rollout_devs = _all_devs[:_rollout_chips]
  _ref_devs = _all_devs[_rollout_chips:_rollout_chips + _ref_chips]
  _trainer_devs = _all_devs[-_trainer_chips:]
  _rollout_devlist = jax._src.mesh_utils.create_device_mesh(
      ROLLOUT_MESH_SHAPE, _rollout_devs
  )
  rollout_mesh = jax.sharding.Mesh(
      _rollout_devlist,
      axis_names=SHARED_MESH_AXIS_NAMES,
      axis_types=(jax.sharding.AxisType.Auto,) * len(ROLLOUT_MESH_SHAPE),
  )
  _trainer_devlist = jax._src.mesh_utils.create_device_mesh(
      TRAINER_MESH_SHAPE, _trainer_devs
  )
  trainer_mesh = jax.sharding.Mesh(
      _trainer_devlist,
      axis_names=SHARED_MESH_AXIS_NAMES,
      axis_types=(jax.sharding.AxisType.Auto,) * len(TRAINER_MESH_SHAPE),
  )
  if REFERENCE_MESH_SHAPE:
    _ref_devlist = jax._src.mesh_utils.create_device_mesh(
        REFERENCE_MESH_SHAPE, _ref_devs
    )
    reference_mesh = jax.sharding.Mesh(
        _ref_devlist,
        axis_names=SHARED_MESH_AXIS_NAMES,
        axis_types=(jax.sharding.AxisType.Auto,) * len(REFERENCE_MESH_SHAPE),
    )
  else:
    reference_mesh = trainer_mesh
  # shared_mesh kept as an alias for code paths that still reference it
  # (e.g. LoRA reshard); LoRA + disjoint is not a tested combo.
  shared_mesh = trainer_mesh
  print(
      f"disjoint meshes: trainer={trainer_mesh.devices.shape} "
      f"rollout={rollout_mesh.devices.shape} reference={reference_mesh.devices.shape}"
  )
else:
  if jax.device_count() < math.prod(SHARED_MESH_SHAPE):
    raise ValueError(
        f"Expected at least {math.prod(SHARED_MESH_SHAPE)} devices for mesh "
        f"{SHARED_MESH_SHAPE}, got {jax.device_count()}."
    )

  shared_device_list = jax._src.mesh_utils.create_device_mesh(
      SHARED_MESH_SHAPE, jax.devices()[: math.prod(SHARED_MESH_SHAPE)]
  )
  shared_mesh = jax.sharding.Mesh(
      shared_device_list,
      axis_names=SHARED_MESH_AXIS_NAMES,
      axis_types=(jax.sharding.AxisType.Auto,) * len(SHARED_MESH_SHAPE),
  )
  trainer_mesh = shared_mesh
  rollout_mesh = shared_mesh
  reference_mesh = shared_mesh
  print(f"shared_mesh.devices.shape={shared_mesh.devices.shape}")

# ====== Data ======
import pandas as pd
import datasets as datasets_lib
import transformers

try:
  from google.cloud import storage  # noqa: F401
except Exception:
  pass
import fsspec

Dataset = datasets_lib.Dataset
AutoTokenizer = transformers.AutoTokenizer

TRAIN_DATA_PATH = os.path.join(DATA_DIR, "train.parquet")
TEST_DATA_PATH = os.path.join(DATA_DIR, "test.parquet")


def create_datasets(
    train_ds_path: str = TRAIN_DATA_PATH,
    test_ds_path: str = TEST_DATA_PATH,
):
  with fsspec.open(train_ds_path, "rb") as train_f, fsspec.open(
      test_ds_path, "rb"
  ) as test_f:
    train_df = pd.read_parquet(train_f)
    test_df = pd.read_parquet(test_f)

  train_ds = Dataset.from_pandas(train_df)
  test_ds = Dataset.from_pandas(test_df)
  if args.shuffle_data:
    train_ds = train_ds.shuffle(SEED)
    test_ds = test_ds.shuffle(SEED)

  def process_item(item):
    item["prompts"] = ""
    return item

  train_ds = grain.MapDataset.source(train_ds).map(process_item)
  test_ds = grain.MapDataset.source(test_ds).map(process_item)
  return train_ds, test_ds


tokenizer = AutoTokenizer.from_pretrained(
    MODEL_VERSION,
    extra_special_tokens={"video_token": "<|video|>"},
)
chat_parser = parser.Gemma4ChatTemplateParser(
    tokenizer, enable_thinking=ENABLE_THINKING
)

train_dataset, test_dataset = create_datasets()
train_dataset, val_dataset = data_lib.post_init_dataset(
    train_dataset,
    tokenizer,
    batch_size=BATCH_SIZE,
    num_batches=NUM_BATCHES,
    max_prompt_length=MAX_PROMPT_LENGTH,
    fraction=TRAIN_FRACTION,
    num_epochs=NUM_EPOCHS,
)
test_dataset, _ = data_lib.post_init_dataset(
    test_dataset,
    tokenizer,
    batch_size=BATCH_SIZE,
    num_batches=NUM_TEST_BATCHES,
    max_prompt_length=MAX_PROMPT_LENGTH,
)

show_hbm_usage = sft_utils.show_hbm_usage
show_hbm_usage("Done with loading datasets")

# ====== Download + load model ======
if not os.path.isdir(MODEL_DOWNLOAD_DIR) or not any(
    f.endswith(".safetensors") for f in os.listdir(MODEL_DOWNLOAD_DIR)
):
  os.makedirs(MODEL_DOWNLOAD_DIR, exist_ok=True)
  oss_utils.hf_pipeline(MODEL_VERSION, MODEL_DOWNLOAD_DIR)

if GEMMA4_VARIANT == "26b-a4b":
  config = model_lib.ModelConfig.gemma4_26b_a4b()
else:
  config = model_lib.ModelConfig.gemma4_31b()
if _REMAT_ENV == "block":
  config.remat_config = model_lib.RematConfig.BLOCK
elif _REMAT_ENV in ("decoder", "1", "true"):
  config.remat_config = model_lib.RematConfig.DECODER
# REMAT="0" or anything else → leave remat off.
if ENABLE_FLASH_ATTENTION:
  config.use_flash_attention = True
  config.flash_attention_block_size = 256
if ENABLE_MIX_PRECISION:
  config.dtype = jnp.bfloat16

# Reference + actor base share the same backbone in HBM. Loading the model
# twice doubles HBM usage; under LoRA the reference is the un-adapted base,
# so we alias the same params and let qwix wrap the actor with LoRA on top.
gemma4_ref = params_lib.create_model_from_safe_tensors(
    MODEL_DOWNLOAD_DIR, config, reference_mesh, dtype=MODEL_DTYPE
)
show_hbm_usage("after loading gemma4_ref")

if TRAIN_WITH_LORA:
  # bf16 base is safe under LoRA because the base is frozen; only the small
  # fp32 adapter weights are updated by Adam, so the LR≤1e-5 rounding hazard
  # that forces full-finetune storage to fp32 does not apply.
  gemma4_actor_base = gemma4_ref
  show_hbm_usage("after aliasing gemma4_actor base to ref (shared backbone)")

  lora_provider = qwix.LoraProvider(
      module_path=(
          ".*q_einsum|.*kv_einsum|.*gate_proj|.*down_proj|.*up_proj|"
          ".*attn_vec_einsum"
      ),
      rank=LORA_RANK,
      alpha=LORA_ALPHA,
  )
  model_input = gemma4_actor_base.get_model_input()
  gemma4_actor = qwix.apply_lora_to_model(
      gemma4_actor_base, lora_provider, **model_input
  )

  from tunix.rl import reshard as _reshard
  gemma4_actor = _reshard.reshard_model_to_mesh(gemma4_actor, trainer_mesh)
  show_hbm_usage("after wrapping actor with LoRA")
else:
  # Full-finetune actor: storage MUST be fp32 at LR≤1e-5. Adam updates at
  # typical magnitudes round to zero in bf16 storage, so the policy never
  # moves. Forward compute can still be bf16 via config.dtype.
  #
  # Verification-only path (FORWARD_ONLY_VERIFICATION=1): store actor in
  # MODEL_DTYPE (e.g. bf16) so the model fits on a single-host TPU. This is
  # only valid for rollout/trainer numeric-agreement checks (logp_diff,
  # prob_diff); not for actual training since the Adam-rounding hazard above
  # still applies.
  _actor_dtype = (
      MODEL_DTYPE
      if os.environ.get("FORWARD_ONLY_VERIFICATION", "0") not in ("0", "false", "False")
      else jnp.float32
  )
  gemma4_actor = params_lib.create_model_from_safe_tensors(
      MODEL_DOWNLOAD_DIR, config, trainer_mesh, dtype=_actor_dtype
  )
  show_hbm_usage(f"after loading gemma4_actor ({_actor_dtype})")

# ====== Checkpoint + metrics + optimizer ======
if CKPT_DIR:
  checkpointing_options = ocp.CheckpointManagerOptions(
      save_interval_steps=SAVE_INTERVAL_STEPS, max_to_keep=MAX_TO_KEEP
  )
else:
  checkpointing_options = None

wandb_config = vars(args)
wandb_config.update({
    "WARMUP_STEPS": WARMUP_STEPS,
    "num_steps": MAX_STEPS,
    "rollout_engine": ROLLOUT_ENGINE,
    "model_id": MODEL_VERSION,
    "mesh_shape": (
        {
            "trainer": TRAINER_MESH_SHAPE,
            "rollout": ROLLOUT_MESH_SHAPE,
            "reference": REFERENCE_MESH_SHAPE,
        }
        if DISJOINT_MESH
        else SHARED_MESH_SHAPE
    ),
    "remat": _REMAT_ENV,
    "enable_thinking": ENABLE_THINKING,
    "train_with_lora": TRAIN_WITH_LORA,
    "lora_rank": LORA_RANK if TRAIN_WITH_LORA else 0,
})
wandb_kwargs = {"config": wandb_config}
metrics_logging_options = metrics_logger.MetricsLoggerOptions(
    log_dir=TB_LOG_DIR,
    project_name=os.getenv("WANDB_PROJECT", "tunix-frozenlake"),
    run_name=os.getenv("WANDB_RUN_NAME", ""),
    flush_every_n_steps=1,
    backend_kwargs={"wandb": wandb_kwargs},
)

if os.environ.get("FORWARD_ONLY_VERIFICATION", "0") not in ("0", "false", "False"):
  # No-op optimizer (no Adam m/v state) so the trainer fits alongside vllm on a
  # smaller TPU. Only valid when the run's goal is rollout/trainer numeric checks
  # (logp_diff, prob_diff) rather than actually training.
  optimizer = optax.set_to_zero()
else:
  optimizer = optax.adamw(
      learning_rate=LEARNING_RATE,
      b1=B1,
      b2=B2,
      weight_decay=WEIGHT_DECAY,
  )
  if MAX_GRAD_NORM is not None:
    optimizer = optax.chain(
        optax.clip_by_global_norm(max_norm=MAX_GRAD_NORM),
        optimizer,
    )

# ====== Rollout + RL cluster ======
print("Shared mesh:", shared_mesh)

# Vanilla sampler stops on these. Gemma4's `<end_of_turn>` marker is required
# — the instruct model never emits `<eos>` mid-conversation, so without
# `<end_of_turn>` in `eos_tokens` every rollout runs to MAX_RESPONSE_LENGTH.
# Also accept Qwen's `<turn|>` and `<|im_end|>` so this same path works across
# model families.
_eos_token_ids: list[int] = []
for tok_str in ("<eos>", "<end_of_turn>", "<turn|>", "<|im_end|>"):
  ids = tokenizer.encode(tok_str, add_special_tokens=False)
  if len(ids) == 1:
    _eos_token_ids.append(ids[0])
_eos_token_ids = list(dict.fromkeys(_eos_token_ids))
logging.info("Configured rollout eos_token_ids: %s", _eos_token_ids)

base_rollout_dict = {
    "max_prompt_length": MAX_PROMPT_LENGTH,
    "kv_cache_size": MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH + 256,
    "temperature": TEMPERATURE,
    "top_p": TOP_P,
    "top_k": TOP_K,
    "return_logprobs": True,
    "max_tokens_to_generate": MAX_RESPONSE_LENGTH,
    "eos_tokens": _eos_token_ids if _eos_token_ids else None,
}

vllm_rollout_dict = {
    "rollout_vllm_model_version": MODEL_VERSION,
    "rollout_vllm_hbm_utilization": float(os.environ.get("VLLM_HBM_UTIL", "0.30")),
    "rollout_vllm_tpu_backend_type": "jax",
    "rollout_vllm_server_mode": True,
    "rollout_vllm_async_scheduling": False,
    "rollout_vllm_init_with_random_weights": True,
    "tensor_parallel_size": (
        ROLLOUT_MESH_SHAPE[1] if DISJOINT_MESH else SHARED_MESH_SHAPE[1]
    ),
    "data_parallel_size": (
        ROLLOUT_MESH_SHAPE[0] if DISJOINT_MESH else SHARED_MESH_SHAPE[0]
    ),
    "rollout_vllm_delete_dst_buffers": True,
    "rollout_vllm_reshard_chunk_size": 16,
    "rollout_vllm_max_num_seqs": VLLM_MAX_NUM_SEQS,
    "rollout_vllm_max_num_batched_tokens": VLLM_MAX_BATCHED_TOKENS,
    "rollout_vllm_kwargs": {
        "kv_cache_metrics": True,
        "disable_log_stats": False,
        "enable_prefix_caching": False,
        "dtype": "bfloat16",
    },
}

if ROLLOUT_ENGINE == "vllm":
  rollout_engine_config = base_rollout.RolloutConfig(
      **base_rollout_dict, **vllm_rollout_dict
  )
elif ROLLOUT_ENGINE == "vanilla":
  rollout_engine_config = base_rollout.RolloutConfig(**base_rollout_dict)
else:
  raise ValueError(f"Unsupported rollout engine: {ROLLOUT_ENGINE}")

cluster_config = rl_cluster_lib.ClusterConfig(
    role_to_mesh={
        rl_cluster_lib.Role.ACTOR: trainer_mesh,
        rl_cluster_lib.Role.REFERENCE: reference_mesh,
        rl_cluster_lib.Role.ROLLOUT: rollout_mesh,
    },
    rollout_engine=ROLLOUT_ENGINE,
    offload_to_cpu=False,
    training_config=rl_cluster_lib.RLTrainingConfig(
        actor_optimizer=optimizer,
        eval_every_n_steps=EVAL_EVERY_N_STEPS,
        max_steps=MAX_STEPS,
        mini_batch_size=MINI_BATCH_SIZE,
        train_micro_batch_size=int(os.environ.get("TRAIN_MICRO_BS", 1)),
        compute_logps_micro_batch_size=int(os.environ.get("COMPUTE_LOGPS_MICRO_BS", 1)),
        metrics_logging_options=metrics_logging_options,
        checkpoint_root_directory=CKPT_DIR,
        checkpointing_options=checkpointing_options,
    ),
    rollout_config=rollout_engine_config,
)

grpo_config = GRPOConfig(
    num_generations=NUM_GENERATIONS,
    num_iterations=NUM_ITERATIONS,
    max_response_length=MAX_RESPONSE_LENGTH,
    beta=BETA,
    epsilon=EPSILON,
    epsilon_high=EPSILON_HIGH,
    system_prompt="",
    max_concurrency=MAX_CONCURRENCY,
    off_policy_steps=OFF_POLICY_STEPS,
    loss_agg_mode=args.loss_agg_mode,
    kl_loss_mode=args.kl_loss_mode,
    loss_algo=args.loss_algo,
    sampler_is="token",
    sampler_is_threshold=2.0,
    advantage_estimator=args.advantage_estimator,
)

rl_cluster = rl_cluster_lib.RLCluster(
    actor=gemma4_actor,
    reference=gemma4_ref,
    tokenizer=tokenizer,
    cluster_config=cluster_config,
)
show_hbm_usage("after RLCluster creation")


_metric_call_idx = 0


def metric_fn(prompts, completions, rewards, advantages, **kwargs):
  del prompts, completions, advantages, kwargs
  global _metric_call_idx
  _metric_call_idx += 1
  solve_all = (rewards > 0.1).all()
  solve_none = (rewards == 0).all()
  solve_partial = (~solve_all) and (~solve_none)
  solve_ratio = (rewards > 0.1).mean()
  reward_mean = float(rewards.mean())
  reward_max = float(rewards.max())
  absl_logging.info(
      "[rollout-metric] call=%d n=%d solve_ratio=%.3f reward_mean=%.3f"
      " reward_max=%.3f solve_all=%d solve_none=%d",
      _metric_call_idx, len(rewards), float(solve_ratio), reward_mean,
      reward_max, int(solve_all), int(solve_none),
  )
  return {
      "rewards/solve_all": (1 if solve_all else 0, np.mean),
      "rewards/solve_none": (1 if solve_none else 0, np.mean),
      "rewards/solve_partial": (1 if solve_partial else 0, np.mean),
      "rewards/solve_ratio": (solve_ratio, np.mean),
  }


grpo_trainer = GRPOLearner(
    rl_cluster=rl_cluster,
    agent_class=FrozenLakeAgent,
    agent_kwargs={"use_multistep_prompt": True},
    env_class=FrozenLakeEnv,
    env_kwargs={"max_steps": 10},
    algo_config=grpo_config,
    chat_parser=chat_parser,
    metric_fns=[metric_fn],
)
show_hbm_usage("after GRPOLearner creation")

try:
  print("Defragmenting JAX/XLA memory before training...")
  backend = None
  try:
    import jax.extend.backend as jax_backend
    backend = jax_backend.get_backend()
  except Exception:
    try:
      backend = jax.devices()[0].client
    except Exception:
      pass
  if backend is not None and hasattr(backend, "defragment"):
    backend.defragment()
    print("Defragmentation successful!")
  else:
    print("Defragmentation skipped: backend has no defragment attribute.")
except Exception as e:
  print(f"Defragmentation failed: {e}")

import gc
gc.collect()
jax.clear_caches()

grpo_trainer.train(train_dataset, eval_dataset=test_dataset)
