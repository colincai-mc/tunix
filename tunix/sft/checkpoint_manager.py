# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Checkpoint manager for PEFT."""

from collections.abc import Mapping
import functools
import os
import time
from typing import Any

from absl import logging
from flax import nnx
import jax
from orbax.checkpoint import v1 as ocp
from tunix.sft import checkpoint_options


class CheckpointManager:
  """Checkpoint manager for PEFT."""

  def __init__(
      self,
      root_directory: str | None = None,
      options: checkpoint_options.CheckpointingOptions | None = None,
  ):
    """Initializes the checkpoint manager.

    Args:
      root_directory: The root directory for the checkpoint manager. If None,
        the checkpoint manager will be disabled.
      options: The options for the checkpoint manager.
    """
    self._checkpointer: ocp.training.Checkpointer | None = None
    self._options = checkpoint_options.resolve_checkpointing_defaults(
        options
    )
    if root_directory is not None:
      with self._context:
        self._checkpointer = ocp.training.Checkpointer(
            root_directory,
            save_decision_policy=self._options.save_decision_policy,
            preservation_policy=self._options.preservation_policy,
            step_name_format=self._options.step_name_format,
        )

  @functools.cached_property
  def _context(self) -> ocp.Context:
    """The orbax context."""
    # When using Pathways, the checkpoint manager only supports persistence
    # APIs now.
    if 'proxy' in os.getenv('JAX_PLATFORMS', ''):
      if os.getenv('ENABLE_PATHWAYS_PERSISTENCE', ''):
        logging.info(
            'Using persistence API for checkpointing with Pathways.'
        )
      else:
        logging.warning(
            'Checkpointing without the persistence API, be aware of potential'
            ' OOM.'
        )
      return ocp.Context(
          array_options=ocp.options.ArrayOptions(
              saving=ocp.options.ArrayOptions.Saving(
                  use_ocdbt=False,
                  use_zarr3=False,
              )
          ),
          async_options=self._options.async_options,
      )
    else:
      return ocp.Context(async_options=self._options.async_options)

  def _save_checkpointables(
      self,
      step: int,
      checkpointables: dict[str, Any],
      force: bool,
      custom_metadata: Mapping[str, Any] | None,
  ) -> bool:
    """Internal helper to call Orbax save methods."""
    if self._checkpointer is None:
      return False

    if self._options.enable_async_checkpointing:
      # TODO(b/505050470): Update this logic once Orbax exposes a public API to
      # inspect the initiation status of an async save. For now, this mirrors
      # the exact internal logic used by Orbax to decide if an async_save is
      # initiated.
      should_save = force or self._checkpointer.should_save(step)

      self._checkpointer.save_checkpointables_async(
          step,
          checkpointables,
          force=force,
          custom_metadata=custom_metadata,
      )
      return should_save
    return self._checkpointer.save_checkpointables(
        step,
        checkpointables,
        force=force,
        custom_metadata=custom_metadata,
    )

  def latest_step(self) -> int | None:
    """Returns the latest step."""
    if self._checkpointer is None or self._checkpointer.latest is None:
      return None
    return self._checkpointer.latest.step

  def save(
      self,
      step: int,
      model: nnx.Module,
      optimizer: nnx.Optimizer | None = None,
      save_only_lora_params: bool = False,
      force: bool = False,
      custom_metadata: Mapping[str, Any] | None = None,
  ) -> bool:
    """Saves the params for the given step.

    Args:
      step: The step to save the params for.
      model: The model to save the params for.
      optimizer: The optimizer to save the params for. If None, the optimizer
        will not be saved.
      save_only_lora_params: Whether to save only the LoRA params.
      force: Whether to save the checkpoint regardless of the save decision
        policy.
      custom_metadata: Custom metadata to save with the checkpoint.

    Returns:
      Whether the checkpoint save operation was successful if synchronous,
      otherwise whether the save operation was initiated.
    """
    if self._checkpointer is None:
      return False
    if save_only_lora_params:
      params = nnx.state(model, nnx.LoRAParam)
    else:
      params = nnx.state(model)

    if optimizer is not None:
      checkpointables = {
          'model_params': params,
          'optimizer_state': nnx.state(optimizer, nnx.optimizer.OptState),
      }
    else:
      checkpointables = {
          'model_params': params,
      }
    with self._context:
      return self._save_checkpointables(
          step, checkpointables, force, custom_metadata
      )

  def maybe_restore(
      self,
      model: nnx.Module,
      optimizer: nnx.Optimizer | None = None,
      step: int | None = None,
      restore_only_lora_params: bool = False,
  ) -> tuple[int, Any]:
    """Restores the params from the latest checkpoint if available and updates the model provided.

    Args:
      model: The model to restore the params for.
      optimizer: The optimizer to restore the params for. If None or if
        optimizer state is not found in the checkpoint, the optimizer will not
        be restored.
      step: The step to restore the params from. If None, the latest step will
        be used.
      restore_only_lora_params: Whether to restore only the LoRA params.

    Returns:
      A tuple (step, custom_metadata), where step is the step of the restored
      checkpoint or 0 if no checkpoint is available, and the custom_metadata.

    Raises:
      RuntimeError: If the checkpoint cannot be restored.
    """
    restore_start = time.time()
    if self._checkpointer is None:
      return 0, {}
    if step is None:
      step = self.latest_step()
      # If no checkpoint is available, return 0.
      if step is None:
        return 0, {}

    with self._context:
      metadata = self._checkpointer.checkpointables_metadata(step)

    if restore_only_lora_params:
      model_params_state = nnx.state(model, nnx.LoRAParam)
      # Set partial_load to True to allow restoration of only the LoRA params.
      ctx = ocp.Context(
          async_options=self._context.async_options,
          array_options=self._context.array_options,
          pytree_options=ocp.options.PyTreeOptions(
              loading=ocp.options.PyTreeOptions.Loading(partial_load=True)
          ),
      )
    else:
      model_params_state = nnx.state(model)
      ctx = self._context
    abstract_checkpointables = {'model_params': model_params_state}

    def fix_sharding(state):
      # Scalar values in optimizer states like step and count is initialized as
      # SingleDeviceSharding, which will fail if optimizer is sharded. To fix
      # it, we will replicate the scalar values.
      mesh = next(
          (
              x.sharding.mesh
              for x in jax.tree_util.tree_leaves(state)
              if getattr(x, 'sharding', None)
              and isinstance(x.sharding, jax.sharding.NamedSharding)
          ),
          None,
      )

      if mesh is None:
        logging.info(
            'Optimizer state contains no NamedSharding. Skipping sharding'
            ' replication.'
        )
        return state

      target_shardings = nnx.get_named_sharding(state, mesh)
      return jax.tree_util.tree_map(
          lambda x, shd: jax.ShapeDtypeStruct(
              getattr(x, 'shape', ()),
              getattr(x, 'dtype', jax.numpy.asarray(x).dtype),
              sharding=shd,
          ),
          state,
          target_shardings,
      )

    if (
        optimizer is not None
        and metadata is not None
        and 'optimizer_state' in metadata.metadata
    ):
      optimizer_state = nnx.state(optimizer, nnx.optimizer.OptState)
      abstract_checkpointables['optimizer_state'] = fix_sharding(
          optimizer_state
      )

    try:
      with ctx:
        restored_checkpointables = self._checkpointer.load_checkpointables(
            step,
            abstract_checkpointables,
        )
    except KeyError as e:
      if not restore_only_lora_params:
        raise ValueError(
            f'Failed to restore from step {step}. If this checkpoint only'
            ' contains LoRA parameters, please set'
            ' `restore_only_lora_params=True`.'
        ) from e
      raise e

    if optimizer is not None and 'optimizer_state' in restored_checkpointables:
      nnx.update(optimizer, restored_checkpointables['optimizer_state'])

    # Update the model state with params from the restored checkpoint.
    nnx.update(model, restored_checkpointables['model_params'])
    logging.info(
        'Restored params from step: %d in %.3f seconds',
        step,
        time.time() - restore_start,
    )
    custom_metadata = metadata.custom_metadata if metadata else {}
    return step, custom_metadata

  def close(self) -> None:
    """Closes the checkpoint manager."""
    if self._checkpointer is None:
      return
    self._checkpointer.close()
