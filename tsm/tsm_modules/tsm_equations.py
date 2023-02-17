"""Implementations of equation modules."""

import logging
from typing import Callable

import gin
import haiku as hk
import jax
import jax.numpy as jnp
from jax_cfd.base import boundaries
from jax_cfd.base import grids
from jax_cfd.ml import advections
from jax_cfd.ml import diffusions
from jax_cfd.ml import equations
from jax_cfd.ml import forcings
from jax_cfd.ml import physics_specifications
from jax_cfd.ml import pressures

ConvectionModule = advections.ConvectionModule
DiffuseModule = diffusions.DiffuseModule
DiffusionSolveModule = diffusions.DiffusionSolveModule
ForcingModule = forcings.ForcingModule
PressureModule = pressures.PressureModule


@gin.configurable(denylist=("grid", "dt", "physics_specs"))
def tsm_modular_navier_stokes_model(
    grid: grids.Grid,
    dt: float,
    physics_specs: physics_specifications.NavierStokesPhysicsSpecs,
    equation_solver=equations.implicit_diffusion_navier_stokes,
    convection_module: ConvectionModule = advections.self_advection,
    pressure_module: PressureModule = pressures.fast_diagonalization,
    acceleration_modules=(),
):
  """Returns an incompressible Navier-Stokes time step model.
  This model is derived from standard components of numerical solvers that could
  be replaced with learned components. Note that diffusion module is specified
  in the equation_solver due to differences in implicit/explicit schemes.
  Args:
    grid: grid on which the Navier-Stokes equation is discretized.
    dt: time step to use for time evolution.
    physics_specs: physical parameters of the simulation module.
    equation_solver: solver to call to create a time-stepping function.
    convection_module: module to use to simulate convection.
    pressure_module: module to use to perform pressure projection.
    acceleration_modules: additional explicit terms to be added to the equation
      before the pressure projection step.
  Returns:
    A function that performs `steps` steps of the Navier-Stokes time dynamics.
  """
  active_forcing_fn = physics_specs.forcing_module(grid)

  def navier_stokes_step_fn(state, is_training=False, cache=None):
    """Advances Navier-Stokes state forward in time."""
    v = state
    for u in v:
      if not isinstance(u, grids.GridVariable):
        raise ValueError(f"Expected GridVariable type, got {type(u)}")
    convection, new_cache = convection_module(grid, dt, physics_specs, v=v, is_training=is_training, cache=cache)
    bc = boundaries.periodic_boundary_conditions(grid.ndim)
    old_v = v
    v = tuple(
        grids.GridVariable(grids.GridArray(u.data[-1], u.offset, grid), bc)
        for u in v)
    accelerations = [
        acceleration_module(grid, dt, physics_specs, v=v)
        for acceleration_module in acceleration_modules
    ]
    forcing = forcings.sum_forcings(active_forcing_fn, *accelerations)
    pressure_solve_fn = pressure_module(grid, dt, physics_specs)
    step_fn = equation_solver(
        grid=grid,
        dt=dt,
        physics_specs=physics_specs,
        density=physics_specs.density,
        viscosity=physics_specs.viscosity,
        pressure_solve=pressure_solve_fn,
        convect=convection,
        forcing=forcing)
    new_v = step_fn(v)
    if old_v[0].shape[0] > 1:
      logging.info(old_v[0].shape)
      new_v = tuple(
          grids.GridVariable(grids.GridArray(
              jnp.concatenate(
                  (old_u.data[1:], jnp.expand_dims(new_u.data, axis=0)),
                  axis=0),
              new_u.offset, grid), bc)
          for old_u, new_u in zip(old_v, new_v))
    else:
      new_v = tuple(
          grids.GridVariable(grids.GridArray(
              jnp.expand_dims(new_u.data, axis=0),
              new_u.offset, grid), bc)
          for new_u in new_v)
    if new_cache is not None:
      return new_v, new_cache
    else:
      return new_v

  return hk.to_module(navier_stokes_step_fn)()


@gin.register
def learned_predictor(
    grid: grids.Grid,
    dt: float,
    physics_specs: physics_specifications.BasePhysicsSpecs,
    predictor_module: Callable,
):
  """Like learned_corrector, but based on the input rather than output state."""
  predictor = predictor_module(grid, dt, physics_specs)

  def step_fn(state, is_training=False, cache=None):
    corrections = predictor(state)

    v = state
    old_v = v
    bc = boundaries.periodic_boundary_conditions(grid.ndim)

    if len(old_v[0].shape) == 3:
      v = tuple(
          grids.GridVariable(grids.GridArray(u.data[-1], u.offset, grid), bc)
          for u in v)

    new_v = jax.tree_map(lambda x, y: x + dt * y, v, corrections)
    logging.info("Using learned predictor")

    if len(old_v[0].shape) == 3:
      if old_v[0].shape[0] > 1:
        logging.info(old_v[0].shape)
        new_v = tuple(
            grids.GridVariable(grids.GridArray(
                jnp.concatenate(
                    (old_u.data[1:], jnp.expand_dims(new_u.data, axis=0)),
                    axis=0),
                new_u.offset, grid), bc)
            for old_u, new_u in zip(old_v, new_v))
      else:
        new_v = tuple(
            grids.GridVariable(grids.GridArray(
                jnp.expand_dims(new_u.data, axis=0),
                new_u.offset, grid), bc)
            for new_u in new_v)
    return new_v
  return hk.to_module(step_fn)()


@gin.register
def tsm_learned_corrector(
    grid: grids.Grid,
    dt: float,
    physics_specs: physics_specifications.BasePhysicsSpecs,
    base_solver_module: Callable,
    corrector_module: Callable,
):
  """Like learned_corrector, but based on input & output states."""
  base_solver = base_solver_module(grid, dt, physics_specs)
  corrector = corrector_module(grid, dt, physics_specs)

  def step_fn(state, is_training=False, cache=None):
    v = state
    old_v = v
    bc = boundaries.periodic_boundary_conditions(grid.ndim)
    v = tuple(
        grids.GridVariable(grids.GridArray(u.data[-1], u.offset, grid), bc)
        for u in v)

    next_state = base_solver(v)
    corrections = corrector(old_v)
    new_v = jax.tree_map(lambda x, y: x + dt * y, next_state, corrections)
    logging.info("Using my learned corrector")

    if len(old_v[0].shape) == 3 and old_v[0].shape[0] > 1:
      logging.info(old_v[0].shape)
      new_v = tuple(
          grids.GridVariable(grids.GridArray(
              jnp.concatenate(
                  (old_u.data[1:], jnp.expand_dims(new_u.data, axis=0)),
                  axis=0),
              new_u.offset, grid), bc)
          for old_u, new_u in zip(old_v, new_v))
    else:
      new_v = tuple(
          grids.GridVariable(grids.GridArray(
              jnp.expand_dims(new_u.data, axis=0),
              new_u.offset, grid), bc)
          for new_u in new_v)

    return new_v

  return hk.to_module(step_fn)()
