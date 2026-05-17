# This scripts takes a docker image that already contains the GRL dependencies, copies the local source code in and
# uploads that image into GCR. Once in GCR the docker image can be used for development.

# Each time you update the base image via a "bash docker_build_dependency_image.sh", there will be a slow upload process
# (minutes). However, if you are simply changing local code and not updating dependencies, uploading just takes a few seconds.

# Script to buid a GRL base image locally, example cmd is:
# bash build_docker.sh

set -e

DOCKERFILE=./Dockerfile

if [ ! -f "$DOCKERFILE" ]; then
    echo "Error: Dockerfile not found at $DOCKERFILE"
    exit 1
fi

export LOCAL_IMAGE_NAME=${LOCAL_IMAGE_NAME:-tunix_base_image}
# Artifact Registry destination. Override AR_IMAGE to publish to your own repo,
# e.g. AR_IMAGE=us-docker.pkg.dev/<project>/<repo>/tunix. PUSH_AR=0 skips the push.
export AR_IMAGE=${AR_IMAGE:-}
export AR_TAG=${AR_TAG:-$(git rev-parse --short HEAD 2>/dev/null || date +%Y%m%d_%H%M%S)}
export PUSH_AR=${PUSH_AR:-0}
echo "Building base image: $LOCAL_IMAGE_NAME"
echo "Push target (if enabled): ${AR_IMAGE}:${AR_TAG} (set PUSH_AR=0 to skip)"

echo "Using Dockerfile: $DOCKERFILE"

# Use Docker BuildKit so we can cache pip packages.
export DOCKER_BUILDKIT=1

echo "Starting to build your docker image. This will take a few minutes but the image can be reused as you iterate."

DOCKER_COMMAND="docker"
if docker info >/dev/null 2>&1; then
    DOCKER_COMMAND="docker"
else
    # Avoid invoking sudo interactively which can prompt for a password.
    if sudo -n docker info >/dev/null 2>&1; then
        DOCKER_COMMAND="sudo docker"
    else
        cat <<'MSG'
Docker does not appear usable from this account and the build would prompt for a password.

Run the build with sufficient privileges (will prompt): sudo bash build_docker.sh
On Linux, add your user to the docker group so sudo isn't required (you must re-login):
  sudo usermod -aG docker "$USER" && newgrp docker

MSG
        exit 1
    fi
fi
export DOCKER_COMMAND

build_ai_image() {
    COMMIT_HASH=$(git rev-parse --short HEAD)
    echo "Building Tunix Image at commit hash ${COMMIT_HASH}..."

    $DOCKER_COMMAND build \
        --network=host \
        -t ${LOCAL_IMAGE_NAME} \
        -f ${DOCKERFILE} .
}

build_ai_image

echo ""
echo "*************************
"

echo "Built your docker image and named it ${LOCAL_IMAGE_NAME}.
It now installs Tunix and the pinned vLLM and tpu-inference dependencies from requirements/requirements.txt. "

if [[ "${PUSH_AR}" == "1" ]]; then
  if [[ -z "${AR_IMAGE}" ]]; then
    echo "PUSH_AR=1 but AR_IMAGE is empty. Set AR_IMAGE to your registry path."
    exit 1
  fi
  echo "Tagging ${LOCAL_IMAGE_NAME} as ${AR_IMAGE}:${AR_TAG} and pushing to Artifact Registry..."
  ${DOCKER_COMMAND:-docker} tag "${LOCAL_IMAGE_NAME}" "${AR_IMAGE}:${AR_TAG}"
  ${DOCKER_COMMAND:-docker} tag "${LOCAL_IMAGE_NAME}" "${AR_IMAGE}:latest"
  # Configure docker auth for AR host if not already done. Idempotent.
  gcloud auth configure-docker "$(echo "${AR_IMAGE}" | cut -d/ -f1)" --quiet || true
  ${DOCKER_COMMAND:-docker} push "${AR_IMAGE}:${AR_TAG}"
  ${DOCKER_COMMAND:-docker} push "${AR_IMAGE}:latest"
  echo "Pushed ${AR_IMAGE}:${AR_TAG} (and :latest)."
else
  echo "Skipping push (PUSH_AR=0). To enable, set PUSH_AR=1 AR_IMAGE=<registry-path>."
fi
