# Copyright 2021 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Functions for computing and applying pressure."""

from typing import Callable, Optional

import jax.numpy as jnp
import jax.scipy.sparse.linalg

from jax_cfd.base import boundaries
from jax_cfd.base import finite_differences as fd
from jax_cfd.base import grids

Array = grids.Array
GridArray = grids.GridArray
GridArrayVector = grids.GridArrayVector
GridVariable = grids.GridVariable
GridVariableVector = grids.GridVariableVector
BoundaryConditions = grids.BoundaryConditions

# Specifying the full signatures of Callable would get somewhat onerous
# pylint: disable=g-bare-generic


# TODO(pnorgaard) Implement bicgstab for non-symmetric operators


def solve_cg(
    v: GridVariableVector,
    q0: GridVariable,
    rtol: float = 1e-6,
    atol: float = 1e-6,
    maxiter: Optional[int] = None) -> GridArray:
  """Conjugate gradient solve for the pressure such that continuity is enforced.

  Returns a pressure correction `q` such that `div(v - grad(q)) == 0`.

  The relationship between `q` and our actual pressure estimate is given by
  `p = q * density / dt`.

  Args:
    v: the velocity field.
    q0: an initial value, or "guess" for the pressure correction. A common
      choice is the correction from the previous time step. Also specifies the
      boundary conditions on `q`.
    rtol: relative tolerance for convergence.
    atol: absolute tolerance for convergence.
    maxiter: optional int, the maximum number of iterations to perform.

  Returns:
    A pressure correction `q` such that `div(v - grad(q))` is zero.
  """
  rhs = fd.centered_divergence(v)

  def laplacian_with_bcs(array: GridArray) -> GridArray:
    if not boundaries.has_all_periodic_boundary_conditions(q0):
      raise ValueError(
          'Laplacian operator implementation requires periodic bc.')
    variable = grids.GridVariable(array, q0.bc)
    gradient = fd.central_difference(variable, axis=None)
    gradient = tuple(grids.GridVariable(g, q0.bc) for g in gradient)
    return fd.centered_divergence(gradient)

  q, _ = jax.scipy.sparse.linalg.cg(
      laplacian_with_bcs,
      rhs,
      x0=q0.array,
      tol=rtol,
      atol=atol,
      maxiter=maxiter)
  return q


def projection(
    v: GridVariableVector,
    solve: Callable = solve_cg,
) -> GridVariableVector:
  """Apply pressure projection to make a velocity field divergence free."""
  grid = grids.consistent_grid(*v)
  pressure_bc = boundaries.get_pressure_bc_from_velocity(v)
  q0 = grids.GridVariable(
      grids.GridArray(jnp.zeros(grid.shape), grid.cell_center, grid),
      pressure_bc)

  q = solve(v, q0)
  q = grids.GridVariable(q, pressure_bc)
  q_grad = fd.central_difference(q, axis=None)
  v_projected = tuple(
      grids.GridVariable(u.array - q_g, u.bc) for u, q_g in zip(v, q_grad))
  return v_projected
