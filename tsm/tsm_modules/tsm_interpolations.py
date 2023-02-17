"""Interpolation modules."""

import collections
import functools
import logging
from typing import (
    Any, Callable, Dict, Optional, Sequence, Tuple, TypeVar, Union,
)

import gin
import jax
import jax.numpy as jnp
from jax_cfd.base import grids
from jax_cfd.base import interpolation
from jax_cfd.ml import layers
from jax_cfd.ml import physics_specifications
from jax_cfd.ml import towers
from jax_cfd.ml import layers_util
from jax_cfd.ml import tiling
import numpy as np
import scipy
import haiku as hk
from tsm_modules.hippo import JaxAdaptiveTransition


GridArray = grids.GridArray
GridArrayVector = grids.GridArrayVector
GridVariable = grids.GridVariable
GridVariableVector = grids.GridVariableVector
InterpolationFn = interpolation.InterpolationFn
InterpolationModule = Callable[..., InterpolationFn]
InterpolationTransform = Callable[..., InterpolationFn]
FluxLimiter = interpolation.FluxLimiter


StencilSizeFn = Callable[
    [Tuple[int, ...], Tuple[int, ...], Any], Tuple[int, ...]]


@gin.configurable
class TsmFusedLearnedInterpolation:
  """Learned interpolator that computes interpolation coefficients in 1 pass.
  Interpolation function that has pre-computed interpolation
  coefficients for a given velocity field `v`. It uses a collection of
  `SpatialDerivativeFromLogits` modules and a single neural network that
  produces logits for all expected interpolations. Interpolations are keyed by
  `input_offset`, `target_offset` and an optional `tag`. The `tag` allows us to
  perform multiple interpolations between the same `offset` and `target_offset`
  with different weights.
  """

  def __init__(
      self,
      grid: grids.Grid,
      dt: float,
      physics_specs: physics_specifications.BasePhysicsSpecs,
      v,
      tags=(None,),
      stencil_size: Union[int, StencilSizeFn] = 4,
      tower_factory=towers.forward_tower_factory,
      name='fused_learned_interpolation',
      extract_patch_method='roll',
      fuse_constraints=False,
      fuse_patches=False,
      constrain_with_conv=False,
      tile_layout=None,
      is_training=False,
      cache=None,
  ):
    """Constructs object and performs necessary pre-computate."""
    del dt, physics_specs  # unused.

    derivative_orders = (0,) * grid.ndim
    derivatives = collections.OrderedDict()

    if isinstance(stencil_size, int):
      stencil_size_fn = lambda *_: (stencil_size,) * grid.ndim
    else:
      stencil_size_fn = stencil_size

    for u in v:
      for target_offset in grids.control_volume_offsets(u):
        for tag in tags:
          key = (u.offset, target_offset, tag)
          derivatives[key] = layers.SpatialDerivativeFromLogits(
              stencil_size_fn(*key),
              u.offset,
              target_offset,
              derivative_orders=derivative_orders,
              steps=grid.step,
              extract_patch_method=extract_patch_method,
              tile_layout=tile_layout,
          )
    output_sizes = [deriv.subspace_size for deriv in derivatives.values()]

    # if we use temporal bundling alone: cache = (0, jnp.zeros(cache_shape))
    # if we use temporal bundling and hippo: cache = (0, jnp.zeros(cache_shape[0]), hippo_cache)

    if cache is not None:
      bundle_steps = cache[1].shape[-1] // sum(output_sizes)
      assert cache[1].shape[-1] % sum(output_sizes) == 0
      logging.info(f"Using bundle steps: {bundle_steps}")
    else:
      bundle_steps = 1

    cnn_network = tower_factory(sum(output_sizes) * bundle_steps, grid.ndim, name=name)
    hippo_transition = None
    if cache is not None and len(cache) == 3:
      hippo_transition = JaxAdaptiveTransition(N=cache[2].shape[-1])
    inputs = jnp.stack([u.data for u in v], axis=-1)
    if inputs.ndim == 4:
      inputs = jnp.transpose(inputs, axes=(1, 2, 3, 0))
    elif inputs.ndim == 3:
      inputs = jnp.transpose(inputs, axes=(1, 2, 0))
    elif inputs.ndim == 5:
      inputs = jnp.transpose(inputs, axes=(1, 2, 3, 4, 0))
    else:
      raise ValueError('Invalid inputs shape: {}'.format(inputs.shape))
    if hippo_transition is not None:
      inputs = inputs[..., -1:]
    else:
      inputs = jnp.reshape(inputs, inputs.shape[:grid.ndim] + (-1,))
    logging.info('inputs shape: %s', inputs.shape)

    if cache is not None:
      def true_func(_):
        _inputs = _[0]
        reshaped_inputs = jnp.reshape(_inputs, _inputs.shape[:grid.ndim] + (-1,))
        new_cache1 = cnn_network(reshaped_inputs, is_training=is_training)
        if hippo_transition is not None:
          return 0, new_cache1, _inputs
        else:
          return 0, new_cache1

      def false_func(_):
        _inputs = _[0]
        _cache = _[1]
        if hippo_transition is not None:
          return _cache[0] + 1, _cache[1], _inputs
        else:
          return _cache[0] + 1, _cache[1]

      if hippo_transition is not None:
        pre, post = hippo_transition.get_init_state()
        inputs = hippo_transition.bilinear_fast(cache[2], inputs, pre, post)[0]

      if hk.running_init():
        cache = false_func((inputs, cache))
        cache = true_func((inputs, cache))
      else:
        cache = hk.cond(
            pred=(cache[0] == bundle_steps),
            true_fun=true_func,
            false_fun=false_func,
            operand=(inputs, cache),
        )
      cache_idx = cache[0]
      all_logits = jax.lax.dynamic_slice(cache[1],
                                         (0, 0, 0)[:grid.ndim] + (cache_idx * sum(output_sizes),),
                                         cache[1].shape[:grid.ndim] + (sum(output_sizes),))
      logging.info('cache shape: %s', cache[1].shape)
      logging.info(all_logits.shape)
      self.cache = cache
    else:
      all_logits = cnn_network(inputs, is_training=is_training)
      self.cache = None
      logging.info('Cache not found.')
    logging.info(inputs.shape)
    logging.info(output_sizes)

    if fuse_constraints:
      raise NotImplementedError
    else:
      split_logits = jnp.split(all_logits, np.cumsum(output_sizes), axis=-1)
      self._interpolators = {
          k: functools.partial(derivative, logits=logits)
          for (k, derivative), logits in zip(derivatives.items(), split_logits)
      }

  def __call__(self,
               c: GridVariable,
               offset: Tuple[int, ...],
               v: GridVariableVector,
               dt: float,
               tag=None) -> GridVariable:
    del dt  # not used.
    # TODO(dkochkov) Add decorator to expand/squeeze channel dim.
    c = grids.GridVariable(
        grids.GridArray(jnp.expand_dims(c.data, -1), c.offset, c.grid), c.bc)
    # TODO(jamieas): Try removing the following line.
    if c.offset == offset:
      return c
    key = (c.offset, offset, tag)
    interpolator = self._interpolators.get(key)
    if interpolator is None:
      raise KeyError(f'No interpolator for key {key}. '
                     f'Available keys: {list(self._interpolators.keys())}')
    result = jnp.squeeze(interpolator(c.data), axis=-1)
    return grids.GridVariable(
        grids.GridArray(result, offset, c.grid), c.bc)
