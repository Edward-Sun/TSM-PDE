"""Decoder modules that help interfacing model states with output data.
All decoder modules generate a function that given an specific model state
return the observable data of the same structure as provided to the Encoder.
Decoders can be either fixed functions, decorators, or learned modules.
"""

from typing import Any, Callable
import gin
import jax.numpy as jnp
from jax_cfd.base import grids
from jax_cfd.ml import physics_specifications
from jax_cfd.ml import towers
from jax_cfd.spectral import utils as spectral_utils

DecodeFn = Callable[[Any], Any]  # maps model state to data time slice.
DecoderModule = Callable[..., DecodeFn]  # generate DecodeFn closed over args.
TowerFactory = towers.TowerFactory


# TODO(dkochkov) generalize this to arbitrary pytrees.
@gin.register
def tsm_aligned_array_decoder(
    grid: grids.Grid,
    dt: float,
    physics_specs: physics_specifications.BasePhysicsSpecs,
) -> DecodeFn:
  """Generates decoder that extracts data from AlignedArrays."""
  del grid, dt, physics_specs  # unused.

  def decode_fn(inputs):
    return tuple(x.data[-1] for x in inputs)

  return decode_fn


@gin.register
def tsm_spectral_array_decoder(
    grid: grids.Grid,
    dt: float,
    physics_specs: physics_specifications.BasePhysicsSpecs,
) -> DecodeFn:
  """Generates decoder that extracts data from AlignedArrays."""
  del dt, physics_specs  # unused.

  def decode_fn(inputs):
    assert len(inputs.shape) == 2, f"{inputs.shape}"
    velocity_solve = spectral_utils.vorticity_to_velocity(grid)
    uhat, vhat = velocity_solve(inputs)
    u = jnp.fft.irfftn(uhat)
    v = jnp.fft.irfftn(vhat)
    return u, v

  return decode_fn
