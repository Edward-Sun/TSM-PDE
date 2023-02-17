"""Models for advection and convection components."""
import functools
import jax_cfd.ml.advections

from typing import Callable, Optional, Any
import gin
from jax_cfd.base import advection
from jax_cfd.base import interpolation
from jax_cfd.base import grids
from jax_cfd.ml import interpolations
from jax_cfd.ml import physics_specifications


GridArray = grids.GridArray
GridArrayVector = grids.GridArrayVector
GridVariable = grids.GridVariable
GridVariableVector = grids.GridVariableVector
InterpolationModule = interpolations.InterpolationModule
InterpolationFn = interpolation.InterpolationFn
InterpolationTransform = Callable[..., InterpolationFn]
AdvectFn = Callable[[GridVariable, GridVariableVector, float], GridArray]
AdvectionModule = Callable[..., AdvectFn]
ConvectFn = Callable[[GridVariableVector], GridArrayVector]
ConvectionModule = Callable[..., ConvectFn]


@gin.configurable
def tsm_modular_self_advection(
    grid: grids.Grid,
    dt: float,
    physics_specs: physics_specifications.NavierStokesPhysicsSpecs,
    interpolation_module: InterpolationModule,
    transformation: InterpolationTransform = interpolations.tvd_limiter_transformation,
    **kwargs
) -> (AdvectFn, Any):
  """Modular self advection using a single interpolation module."""
  # TODO(jamieas): Replace this entire function once
  # `single_tower_navier_stokes` is in place.
  interpolate_fn = interpolation_module(grid, dt, physics_specs, **kwargs)
  cache = interpolate_fn.cache
  interpolate_fn.cache = None
  c_interpolate_fn = functools.partial(interpolate_fn, tag='c')
  c_interpolate_fn = transformation(c_interpolate_fn)
  u_interpolate_fn = functools.partial(interpolate_fn, tag='u')

  def advect(
      c: GridVariable,
      v: GridVariableVector,
      dt: Optional[float] = None,
  ) -> GridArray:
    return advection.advect_general(
        c, v, u_interpolate_fn, c_interpolate_fn, dt)

  return advect, cache


@gin.register
def tsm_self_advection(
    grid: grids.Grid,
    dt: float,
    physics_specs: physics_specifications.NavierStokesPhysicsSpecs,
    advection_module: AdvectionModule = jax_cfd.ml.advections.modular_advection,
    **kwargs
) -> (ConvectFn, Any):
  """Convection module based on simultaneous self-advection of velocities."""
  advect_fn, cache = advection_module(grid, dt, physics_specs, **kwargs)

  def convect(v: GridVariableVector) -> GridArrayVector:
    return tuple(advect_fn(u, v, dt) for u in v)

  return convect, cache
