"""Agentic FrozenLake GRPO recipe for Gemma4-e2b on a single TPU host.

Sibling of ``train_frozenlake_qwen3.py``; structure and hyperparameters are
deliberately kept aligned so the two recipes form a clean cross-family
control. Configuration is env-driven so the same image runs unchanged on any
spot VM:

  HF_TOKEN              Hugging Face token for model download.
  WANDB_API_KEY         Wandb API key (auto-picked-up by wandb lib).
  WANDB_PROJECT         Wandb project name (default "tunix-frozenlake").
  WANDB_RUN_NAME        Wandb run name (default uses timestamp).
  MODEL_DOWNLOAD_DIR    Local dir for HF safetensors (default
                        /tmp/models/gemma-4-e2b).
  DATA_DIR              Local or gs:// dir holding train.parquet / test.parquet
                        (default /tmp/data/frozenlake).
  CKPT_DIR              Output checkpoint dir. Checkpointing is opt-in; if
                        unset, no checkpoints are written.
  TB_LOG_DIR            TensorBoard log dir (default /tmp/tunix-tb/frozenlake).
  SHARED_MESH_SHAPE     Override the (fsdp, tp) mesh shape. Defaults to
                        (1, jax.device_count()) (pure tensor parallel).
  ROLLOUT_ENGINE        "vanilla" | "vllm"  (default "vllm" — the disaggregated
                        vLLM server avoids the trace-context issues of running
                        the in-process sampler under REMAT and offers higher
                        throughput at full concurrency).
  MODEL_DTYPE           "bf16" (default) | "fp32" — storage/compute dtype for
                        the reference policy and trainer forward path.
  MIX_PRECISION         "1" (default) | "0" — when 0, runs the model in fp32
                        end-to-end (set together with MODEL_DTYPE=fp32).
  FLASH_ATTN            "1" (default) | "0" — splash flash attention kernel.
  TRAIN_MICRO_BS        Trainer forward+backward micro-batch (default 4).
  COMPUTE_LOGPS_MICRO_BS  Logp recomputation micro-batch (default 4).
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

# ====== Logging Configuration ======
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

# %%
import argparse

arg_parser = argparse.ArgumentParser(
    description="Train FrozenLake on Gemma4-e2b (single-host TPU)."
)
arg_parser.add_argument("--batch_size", type=int, default=64)
arg_parser.add_argument("--mini_batch_size", type=int, default=64)
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
arg_parser.add_argument("--max_prompt_length", type=int, default=8192)
arg_parser.add_argument("--max_response_length", type=int, default=2048)
arg_parser.add_argument("--temperature", type=float, default=0.7)
arg_parser.add_argument("--top_p", type=float, default=1.0)
arg_parser.add_argument("--top_k", type=int, default=0)
arg_parser.add_argument("--max_concurrency", type=int, default=256)
arg_parser.add_argument("--shuffle_data", type=bool, default=True)
arg_parser.add_argument("--seed", type=int, default=42)
arg_parser.add_argument(
    "--loss_agg_mode", type=str, default="sequence-mean-token-mean"
)
arg_parser.add_argument(
    "--kl_loss_mode", type=str, default="low_var_kl"
)
arg_parser.add_argument(
    "--advantage_estimator", type=str, default="rloo",
    help="'grpo' (z-score) or 'rloo' (leave-one-out baseline).",
)
args, _ = arg_parser.parse_known_args()

TRAIN_FRACTION = 1.0
SEED = args.seed

# ====== Sharding ======
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
ENABLE_REMAT = True
ENABLE_FLASH_ATTENTION = os.environ.get("FLASH_ATTN", "1") not in ("0", "false", "False")
ENABLE_MIX_PRECISION = os.environ.get("MIX_PRECISION", "1") not in ("0", "false", "False")
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

# ====== Paths (env-driven so the same image runs anywhere) ======
MODEL_VERSION = os.getenv("MODEL_VERSION", "google/gemma-4-E2B-it")
MODEL_DOWNLOAD_DIR = os.getenv(
    "MODEL_DOWNLOAD_DIR",
    "/tmp/models/" + MODEL_VERSION.replace("/", "_"),
)
DATA_DIR = os.getenv("DATA_DIR", "/tmp/data/frozenlake")

now_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
CKPT_DIR = os.getenv("CKPT_DIR") or None
TB_LOG_DIR = os.getenv("TB_LOG_DIR", "/tmp/tunix-tb/frozenlake")


# ====== Build the single shared mesh ======
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


def _load_gemma4_tokenizer(model_version: str, local_dir: str):
  """Load tokenizer for Gemma 4, patching tokenizer_config.json if needed.

  Gemma 4's tokenizer_config.json ships `extra_special_tokens` as a list
  (e.g. ``['<|video|>']``), but transformers versions older than ~4.46 expect
  a dict and crash with `AttributeError: 'list' object has no attribute
  'keys'`. Pre-download via snapshot, rewrite the field to a dict, and load
  from disk so the recipe works against the bundled image's transformers.
  """
  import json
  from huggingface_hub import snapshot_download

  if not os.path.isdir(local_dir) or not any(
      f.startswith("tokenizer") for f in os.listdir(local_dir)
  ):
    os.makedirs(local_dir, exist_ok=True)
    snapshot_download(
        repo_id=model_version,
        local_dir=local_dir,
        allow_patterns=["tokenizer*", "*special_tokens*", "processor*"],
    )

  cfg_path = os.path.join(local_dir, "tokenizer_config.json")
  if os.path.isfile(cfg_path):
    with open(cfg_path) as f:
      cfg = json.load(f)
    extra = cfg.get("extra_special_tokens")
    if isinstance(extra, list):
      def _key(tok: str) -> str:
        return tok.strip("<>|").strip("_")
      cfg["extra_special_tokens"] = {_key(t): t for t in extra}
      with open(cfg_path, "w") as f:
        json.dump(cfg, f)
  return AutoTokenizer.from_pretrained(local_dir)


tokenizer = _load_gemma4_tokenizer(MODEL_VERSION, MODEL_DOWNLOAD_DIR)
chat_parser = parser.GemmaChatTemplateParser(tokenizer)

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

config = model_lib.ModelConfig.gemma4_e2b()
if ENABLE_REMAT:
  config.remat_config = model_lib.RematConfig.DECODER
if ENABLE_FLASH_ATTENTION:
  config.use_flash_attention = True
  config.flash_attention_block_size = 256
if ENABLE_MIX_PRECISION:
  config.dtype = jnp.bfloat16

# Reference: dtype controlled by MODEL_DTYPE env (bf16 storage is safe for
# the frozen reference; fp32 only needed if testing precision sensitivity).
gemma_ref = params_lib.create_model_from_safe_tensors(
    MODEL_DOWNLOAD_DIR, config, shared_mesh, dtype=MODEL_DTYPE
)
show_hbm_usage("after loading gemma_ref")

# Actor: storage MUST be fp32. At LR=1e-6 with typical weight magnitudes
# ~1e-2, Adam updates are ~1e-6, well below bf16 ULP (~7.8e-5). bf16 storage
# silently rounds every update to zero in optax.apply_updates, so the policy
# never moves. Forward compute can still be bf16 via config.dtype.
gemma_actor = params_lib.create_model_from_safe_tensors(
    MODEL_DOWNLOAD_DIR, config, shared_mesh, dtype=jnp.float32
)
show_hbm_usage("after loading gemma_actor")

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
    "mesh_shape": SHARED_MESH_SHAPE,
})
wandb_kwargs = {"config": wandb_config}
metrics_logging_options = metrics_logger.MetricsLoggerOptions(
    log_dir=TB_LOG_DIR,
    project_name=os.getenv("WANDB_PROJECT", "tunix-frozenlake"),
    run_name=os.getenv("WANDB_RUN_NAME", ""),
    flush_every_n_steps=1,
    backend_kwargs={"wandb": wandb_kwargs},
)

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

# Gemma's chat template terminates every turn with `<end_of_turn>`, but the
# default sampler stop set is only the tokenizer's `eos_token_id` (the `<eos>`
# token, a distinct id). Without `<end_of_turn>` as a stop, the instruct model
# generates its turn-end marker, fails to halt, and keeps producing tokens
# until `max_response_length` is exhausted — yielding `MAX_CONTEXT_LIMIT_REACHED`
# with no parseable action. Include both ids so the sampler stops cleanly.
_EOT_TOKEN_ID = tokenizer.convert_tokens_to_ids("<end_of_turn>")
_EOS_TOKEN_ID = (
    tokenizer.eos_token_id
    if tokenizer.eos_token_id is not None
    else tokenizer.convert_tokens_to_ids("<eos>")
)
ROLLOUT_EOS_TOKENS = sorted({_EOS_TOKEN_ID, _EOT_TOKEN_ID})

# Vanilla sampler rounds the actual tokenized prompt up to the next power of
# two and demands ``cache_size >= rounded_prompt + max_generation_steps``.
# With ``max_prompt_length`` early-termination in the trajectory engine, the
# worst-case rendered prompt is slightly above MAX_PROMPT_LENGTH → rounds to
# 2 * MAX_PROMPT_LENGTH, plus a full ``max_response_length`` generation
# budget, plus margin for the small tokenizer mismatch between the engine's
# prompt check and the sampler.
base_rollout_dict = {
    "max_prompt_length": MAX_PROMPT_LENGTH,
    "kv_cache_size": 2 * MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH + 2048,
    "temperature": TEMPERATURE,
    "top_p": TOP_P,
    "top_k": TOP_K,
    "return_logprobs": True,
    "max_tokens_to_generate": MAX_RESPONSE_LENGTH,
    "eos_tokens": ROLLOUT_EOS_TOKENS,
}

vllm_rollout_dict = {
    "rollout_vllm_model_version": MODEL_VERSION,
    "rollout_vllm_hbm_utilization": float(os.environ.get("VLLM_HBM_UTIL", "0.20")),
    "rollout_vllm_tpu_backend_type": "jax",
    "rollout_vllm_server_mode": True,
    "rollout_vllm_async_scheduling": False,
    "rollout_vllm_init_with_random_weights": True,
    "tensor_parallel_size": SHARED_MESH_SHAPE[1],
    "data_parallel_size": SHARED_MESH_SHAPE[0],
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
        rl_cluster_lib.Role.ACTOR: shared_mesh,
        rl_cluster_lib.Role.REFERENCE: shared_mesh,
        rl_cluster_lib.Role.ROLLOUT: shared_mesh,
    },
    rollout_engine=ROLLOUT_ENGINE,
    offload_to_cpu=False,
    training_config=rl_cluster_lib.RLTrainingConfig(
        actor_optimizer=optimizer,
        eval_every_n_steps=EVAL_EVERY_N_STEPS,
        max_steps=MAX_STEPS,
        mini_batch_size=MINI_BATCH_SIZE,
        train_micro_batch_size=int(os.environ.get("TRAIN_MICRO_BS", 4)),
        compute_logps_micro_batch_size=int(os.environ.get("COMPUTE_LOGPS_MICRO_BS", 4)),
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
    actor=gemma_actor,
    reference=gemma_ref,
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
    env_kwargs={"max_steps": 8},
    algo_config=grpo_config,
    chat_parser=chat_parser,
    metric_fns=[metric_fn],
)
show_hbm_usage("after GRPOLearner creation")

grpo_trainer.train(train_dataset, eval_dataset=test_dataset)
