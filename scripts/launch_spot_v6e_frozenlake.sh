#!/bin/bash
# Provision a v6e-4 spot TPU VM, then SSH-run the FrozenLake training docker
# image on it. Retries spot creation until capacity is granted, then performs
# docker auth + pull + run on the freshly provisioned VM.
#
# Required env on the caller (your dev box):
#   HF_TOKEN          Hugging Face token (model download).
#   WANDB_API_KEY     Wandb key (auto-logged-into when wandb is imported).
#   AR_IMAGE          Artifact-registry path of your training image.
#                     (e.g. us-docker.pkg.dev/<project>/<repo>/tunix)
#   PROJECT           gcloud project to create the TPU VM in.
# Optional:
#   TPU_NAME          Default: tunix-frozenlake-<random>
#   ZONE              Default: try us-east5-a, us-central2-b, us-east1-d.
#   AR_TAG            Default: latest
#   CKPT_DIR          GCS path for checkpoints (default /tmp inside VM).
#   WANDB_PROJECT     Wandb project name (default tunix-frozenlake).
#   ROLLOUT_ENGINE    vanilla | vllm (default vllm).
#   BACKOFF_SECONDS   Sleep between create retries (default 40).
#   MAX_ATTEMPTS      0 = infinite. Default 0.
#
# Usage:
#   AR_IMAGE=... PROJECT=... HF_TOKEN=... WANDB_API_KEY=... \
#     bash scripts/launch_spot_v6e_frozenlake.sh

set -euo pipefail

: "${HF_TOKEN:?HF_TOKEN must be set in the calling env}"
: "${WANDB_API_KEY:?WANDB_API_KEY must be set in the calling env}"
: "${AR_IMAGE:?AR_IMAGE must be set (artifact-registry path to your training image)}"
: "${PROJECT:?PROJECT must be set (gcloud project for the TPU VM)}"

ACCELERATOR_TYPE="v6e-4"
RUNTIME_VERSION="v2-alpha-tpuv6e"

TPU_NAME=${TPU_NAME:-tunix-frozenlake-$(printf '%04d' $((RANDOM % 10000)))}
AR_TAG=${AR_TAG:-latest}
WANDB_PROJECT=${WANDB_PROJECT:-tunix-frozenlake}
ROLLOUT_ENGINE=${ROLLOUT_ENGINE:-vllm}
CKPT_DIR=${CKPT_DIR:-}
BACKOFF_SECONDS=${BACKOFF_SECONDS:-40}
MAX_ATTEMPTS=${MAX_ATTEMPTS:-0}

# Zones likely to have v6e capacity; can override with single zone via ZONE env.
ZONES_DEFAULT=("us-east5-a" "us-central2-b" "us-east1-d")
if [[ -n "${ZONE:-}" ]]; then
  ZONES=("${ZONE}")
else
  ZONES=("${ZONES_DEFAULT[@]}")
fi

echo "TPU_NAME=$TPU_NAME"
echo "ACCELERATOR_TYPE=$ACCELERATOR_TYPE"
echo "RUNTIME_VERSION=$RUNTIME_VERSION"
echo "PROJECT=$PROJECT"
echo "Trying zones: ${ZONES[*]}"
echo "Image: ${AR_IMAGE}:${AR_TAG}"

# ---- Phase 1: create spot VM with retry / zone-rotation ----
attempt=0
created_zone=""
while true; do
  for z in "${ZONES[@]}"; do
    attempt=$((attempt + 1))
    echo "[$(date -u +%H:%M:%SZ)] Attempt $attempt: zone=$z"
    if gcloud compute tpus tpu-vm create "$TPU_NAME" \
        --project="$PROJECT" \
        --zone="$z" \
        --accelerator-type="$ACCELERATOR_TYPE" \
        --version="$RUNTIME_VERSION" \
        --scopes="https://www.googleapis.com/auth/cloud-platform" \
        --spot \
        --quiet; then
      created_zone="$z"
      echo "Created $TPU_NAME in $z"
      break 2
    fi
    if [[ "$MAX_ATTEMPTS" -gt 0 && "$attempt" -ge "$MAX_ATTEMPTS" ]]; then
      echo "Exhausted MAX_ATTEMPTS=$MAX_ATTEMPTS without acquiring capacity." >&2
      exit 1
    fi
    sleep "$BACKOFF_SECONDS"
  done
done

# ---- Phase 2: docker pull + run on the VM ----
ZONE_FLAG=("--zone=$created_zone" "--project=$PROJECT")

echo "Configuring docker auth on $TPU_NAME..."
gcloud compute tpus tpu-vm ssh "$TPU_NAME" "${ZONE_FLAG[@]}" --worker=all --command="
  set -e
  gcloud auth configure-docker $(echo "${AR_IMAGE}" | cut -d/ -f1) --quiet
  sudo usermod -aG docker \$USER || true
"

echo "Pulling image ${AR_IMAGE}:${AR_TAG}..."
gcloud compute tpus tpu-vm ssh "$TPU_NAME" "${ZONE_FLAG[@]}" --worker=all --command="
  set -e
  sudo docker pull ${AR_IMAGE}:${AR_TAG}
"

# Compose docker run command.
DATA_DIR=${DATA_DIR:-/tmp/data/frozenlake}
ENV_FLAGS="-e HF_TOKEN=${HF_TOKEN} -e WANDB_API_KEY=${WANDB_API_KEY} -e WANDB_PROJECT=${WANDB_PROJECT} -e ROLLOUT_ENGINE=${ROLLOUT_ENGINE} -e DATA_DIR=${DATA_DIR}"
if [[ -n "${CKPT_DIR}" ]]; then
  ENV_FLAGS="${ENV_FLAGS} -e CKPT_DIR=${CKPT_DIR}"
fi

CONTAINER_NAME="tunix-frozenlake"
REMOTE_CMD="
  set -e
  sudo docker rm -f ${CONTAINER_NAME} >/dev/null 2>&1 || true
  sudo docker run -d --name ${CONTAINER_NAME} --privileged \
    --net=host \
    -v /tmp:/tmp \
    ${ENV_FLAGS} \
    ${AR_IMAGE}:${AR_TAG} \
    bash -c 'cd /app && \
      if [ ! -f /tmp/data/frozenlake/train.parquet ]; then \
        python examples/frozenlake/data.py --local_dir /tmp/data/frozenlake; \
      fi && \
      python -m examples.frozenlake.train_frozenlake_qwen3_v6e4'
  echo 'Container ${CONTAINER_NAME} started. Tail with: sudo docker logs -f ${CONTAINER_NAME}'
"

echo "Launching training on $TPU_NAME ($created_zone)..."
gcloud compute tpus tpu-vm ssh "$TPU_NAME" "${ZONE_FLAG[@]}" --worker=all --command="$REMOTE_CMD"

cat <<EOF

Detached training started.

Tail logs:
  gcloud compute tpus tpu-vm ssh $TPU_NAME --zone=$created_zone --project=$PROJECT --worker=0 \\
    --command='sudo docker logs -f ${CONTAINER_NAME}'

Delete VM when done:
  gcloud compute tpus tpu-vm delete $TPU_NAME --zone=$created_zone --project=$PROJECT --quiet
EOF
