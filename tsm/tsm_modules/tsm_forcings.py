from typing import Callable, Optional, Tuple

import gin
import jax.numpy as jnp
from jax_cfd.base import forcings
from jax_cfd.base import grids

Array = grids.Array
GridArrayVector = grids.GridArrayVector
GridVariableVector = grids.GridVariableVector
ForcingFn = Callable[[GridVariableVector], GridArrayVector]
ForcingModule = Callable[..., ForcingFn]


@gin.register
def tsm_kolmogorov_forcing(grid: grids.Grid,  # pylint: disable=missing-function-docstring
                          scale: float = 0,
                          wavenumber: int = 2,
                          linear_coefficient: float = 0,
                          swap_xy: bool = False) -> ForcingFn:
  force_fn = kolmogorov_forcing(grid, scale, wavenumber, swap_xy)
  if linear_coefficient != 0:
    linear_force_fn = forcings.linear_forcing(grid, linear_coefficient)
    force_fn = forcings.sum_forcings(force_fn, linear_force_fn)
  return force_fn


def kolmogorov_forcing(
    grid: grids.Grid,
    scale: float = 1,
    k: int = 2,
    swap_xy: bool = False,
    offsets: Optional[Tuple[Tuple[float]]] = None,
) -> ForcingFn:
  """Returns the Kolmogorov forcing function for turbulence in 2D."""
  if offsets is None:
    offsets = grid.cell_faces

  if swap_xy:
    x = grid.mesh(offsets[1])[0]
    half_grid = x[0, 0]
    v = scale * grids.GridArray(
        (jnp.cos(k * x - half_grid) - jnp.cos(k * x + half_grid)) / (2 * half_grid),
        offsets[1], grid)
    if grid.ndim == 2:
      u = grids.GridArray(jnp.zeros_like(v.data), (1, 1/2), grid)
      f = (u, v)
    elif grid.ndim == 3:
      u = grids.GridArray(jnp.zeros_like(v.data), (1, 1/2, 1/2), grid)
      w = grids.GridArray(jnp.zeros_like(u.data), (1/2, 1/2, 1), grid)
      f = (u, v, w)
    else:
      raise NotImplementedError
  else:
    y = grid.mesh(offsets[0])[1]
    half_grid = y[0, 0]
    u = scale * grids.GridArray(
        (jnp.cos(k * y - half_grid) - jnp.cos(k * y + half_grid)) / (2 * half_grid),
        offsets[0], grid)
    if grid.ndim == 2:
      v = grids.GridArray(jnp.zeros_like(u.data), (1/2, 1), grid)
      f = (u, v)
    elif grid.ndim == 3:
      v = grids.GridArray(jnp.zeros_like(u.data), (1/2, 1, 1/2), grid)
      w = grids.GridArray(jnp.zeros_like(u.data), (1/2, 1/2, 1), grid)
      f = (u, v, w)
    else:
      raise NotImplementedError

  def forcing(v):
    del v
    return f
  return forcing
