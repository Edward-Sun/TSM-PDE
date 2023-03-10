import contextlib
import os
import timeit
from functools import partial
from typing import Mapping, NamedTuple

import gin
import haiku as hk
import jax
import jax.numpy as jnp
import jax_cfd.base as cfd
import jax_cfd.ml as cfd_ml
import jmp
import numpy as np
import optax
import pkg_resources
import tensorflow.compat.v1 as tf
import tqdm
import xarray
from absl import app
from absl import flags
from absl import logging
from flax import jax_utils

import pde_dataset as dataset

try:
  tf.flags.DEFINE_multi_string("gin_file", None, "Path to a Gin file.")
  tf.flags.DEFINE_multi_string("gin_param", None, "Gin parameter binding.")
  tf.flags.DEFINE_list("gin_location_prefix", [], "Gin file search path.")
except tf.flags.DuplicateFlagError:
  pass

# Hyper parameters.
flags.DEFINE_integer('train_init_random_seed', 42, help='')
flags.DEFINE_string('mp_policy', 'p=f32,c=f32,o=f32', help='')
flags.DEFINE_string('mp_bn_policy', 'p=f32,c=f32,o=f32', help='')
flags.DEFINE_enum('mp_scale_type', 'NoOp', ['NoOp', 'Static', 'Dynamic'], help='')
flags.DEFINE_float('mp_scale_value', 2 ** 15, help='')

flags.DEFINE_integer('num_samples', 16, help='')
flags.DEFINE_float('delta_time', 0.001, help='')
flags.DEFINE_float('max_velocity', 7.0, help='')
flags.DEFINE_float('warmup_time', 1.0, help='')
flags.DEFINE_float('simulation_time', 1.0, help='')
flags.DEFINE_integer('init_peak_wavenumber', 4, help='')
flags.DEFINE_integer('model_input_size', 64, help='')
flags.DEFINE_integer('model_encode_steps', 1, help='')
flags.DEFINE_integer('model_predict_steps', 64, help='')
flags.DEFINE_string('output_dir', None, help='')
flags.DEFINE_integer('inner_steps', 1, help='')
flags.DEFINE_float('jnp_pi', jnp.pi, help='')


FLAGS = flags.FLAGS
Scalars = Mapping[str, jnp.ndarray]
script_start_time = timeit.default_timer()


class TrainState(NamedTuple):
  step: int
  params: hk.Params
  state: hk.State
  opt_state: optax.OptState
  loss_scale: jmp.LossScale


class SaveState(NamedTuple):
  step: int
  params: hk.Params
  state: hk.State
  opt_state: optax.OptState


get_policy = lambda: jmp.get_policy(FLAGS.mp_policy)
get_bn_policy = lambda: jmp.get_policy(FLAGS.mp_bn_policy)


def get_initial_loss_scale() -> jmp.LossScale:
  cls = getattr(jmp, f'{FLAGS.mp_scale_type}LossScale')
  return cls(FLAGS.mp_scale_value) if cls is not jmp.NoOpLossScale else cls()


def _forward(
    batch: dataset.Batch,
    is_training: bool,
    inner_steps: int,
    outer_steps: int,
) -> [jnp.ndarray, jnp.ndarray]:
  """Forward application of the resnet."""
  inputs = batch['inputs']
  size = FLAGS.model_input_size
  grid = cfd.grids.Grid((size, size), domain=((0, 2 * FLAGS.jnp_pi), (0, 2 * FLAGS.jnp_pi)))
  dt = FLAGS.delta_time
  physics_specs = cfd_ml.physics_specifications.get_physics_specs()
  stable_time_step = cfd.equations.stable_time_step(FLAGS.max_velocity, 0.5,
                                                    physics_specs.viscosity, grid, implicit_diffusion=True)
  logging.info("Stable time step: %.10f" % stable_time_step)
  inner_steps = inner_steps * round(dt / stable_time_step)
  model = cfd_ml.model_builder.get_model_cls(grid, stable_time_step, physics_specs)()

  trajectory = jax.vmap(
      partial(
          cfd_ml.model_utils.decoded_trajectory_with_inputs(
              model=model,
              num_init_frames=FLAGS.model_encode_steps),
          outer_steps=outer_steps,
          inner_steps=inner_steps,
      ),
      axis_name='i')

  final, predictions = trajectory(inputs)

  return predictions


# Transform our forwards function into a pair of pure functions.
forward = hk.transform_with_state(_forward)


def make_optimizer() -> optax.GradientTransformation:
  """SGD with nesterov momentum and a custom lr schedule."""
  return optax.chain(
      optax.scale_by_adam(eps=1e-06),
      optax.scale(-1))


def initial_state(rng: jnp.ndarray, batch: dataset.Batch) -> TrainState:
  """Computes the initial network state."""
  params, state = forward.init(rng, batch, is_training=True, inner_steps=1, outer_steps=1)
  opt_state = make_optimizer().init(params)
  loss_scale = get_initial_loss_scale()
  train_state = TrainState(0, params, state, opt_state, loss_scale)
  return train_state


# NOTE: We use `jit` not `pmap` here because we want to ensure that we see all
# eval data once and this is not easily satisfiable with pmap (e.g. n=3).
# TODO(tomhennigan) Find a solution to allow pmap of eval.
def predict_batch(
    params: hk.Params,
    state: hk.State,
    batch: dataset.Batch,
    inner_steps: int,
    return_predictions: bool = True,
) -> [jnp.ndarray, jnp.ndarray, jnp.ndarray]:
  """Evaluates a batch."""
  predictions, _ = forward.apply(params, state, None, batch, is_training=False,
                                 inner_steps=inner_steps, outer_steps=FLAGS.model_predict_steps)
  des_predictions = predictions

  des_predictions = (des_predictions[0][:, -FLAGS.model_predict_steps:],
                     des_predictions[1][:, -FLAGS.model_predict_steps:])

  predictions = (predictions[0][:, -FLAGS.model_encode_steps:],
                 predictions[1][:, -FLAGS.model_encode_steps:])

  if return_predictions:
    return predictions, des_predictions
  else:
    return predictions, None


@contextlib.contextmanager
def time_activity(activity_name: str):
  logging.info('[Timing] %s start.', activity_name)
  start = timeit.default_timer()
  yield
  duration = timeit.default_timer() - start
  logging.info('[Timing] %s finished (Took %.2fs).', activity_name, duration)


def parse_gin_defaults_and_flags(skip_unknown=False, finalize_config=True):
  """Parses all default gin files and those provided via flags."""
  # Register .gin file search paths with gin
  for gin_file_path in FLAGS.gin_location_prefix:
    gin.add_config_file_search_path(gin_file_path)
  # Set up the default values for the configurable parameters. These values will
  # be overridden by any user provided gin files/parameters.
  gin.parse_config_files_and_bindings(
      FLAGS.gin_file, FLAGS.gin_param,
      skip_unknown=skip_unknown,
      finalize_config=finalize_config)


def main(argv):
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

  # Add search path for gin files stored in package.
  gin.add_config_file_search_path(
      pkg_resources.resource_filename(__name__, "gin"))

  parse_gin_defaults_and_flags()
  FLAGS.alsologtostderr = True

  rng = jax.random.PRNGKey(FLAGS.train_init_random_seed)
  sample_vec = np.zeros((1, FLAGS.model_encode_steps, FLAGS.model_input_size, FLAGS.model_input_size))
  train_state = initial_state(rng, {'inputs': (sample_vec, sample_vec)})
  train_state = jax_utils.replicate(train_state)

  local_device_count = jax.local_device_count()
  grid = cfd.grids.Grid((FLAGS.model_input_size, FLAGS.model_input_size),
                        domain=((0, 2 * FLAGS.jnp_pi), (0, 2 * FLAGS.jnp_pi)))
  my_init_condition = partial(cfd.initial_conditions.filtered_velocity_field,
                              grid=grid,
                              maximum_velocity=FLAGS.max_velocity,
                              peak_wavenumber=FLAGS.init_peak_wavenumber)

  init_velocity = jax.vmap(my_init_condition)(jax.random.split(rng, FLAGS.num_samples))
  eval_u, eval_v = init_velocity

  eval_u = eval_u.array.data.reshape((local_device_count, -1, 1) + eval_u.shape[1:])
  eval_v = eval_v.array.data.reshape((local_device_count, -1, 1) + eval_v.shape[1:])

  batch = {"inputs": (eval_u, eval_v)}
  u_batch = []
  v_batch = []

  step = FLAGS.model_encode_steps
  warmup_steps = int(FLAGS.warmup_time / (FLAGS.delta_time * FLAGS.inner_steps))
  total_steps = int((FLAGS.warmup_time + FLAGS.simulation_time) / (FLAGS.delta_time * FLAGS.inner_steps))
  my_fast_predict_batch = jax.pmap(partial(predict_batch, inner_steps=FLAGS.inner_steps, return_predictions=False),
                                   axis_name='i', donate_argnums=(0,))
  my_slow_predict_batch = jax.pmap(partial(predict_batch, inner_steps=FLAGS.inner_steps, return_predictions=True),
                                   axis_name='i', donate_argnums=(0,))

  pbar = tqdm.tqdm(total=total_steps, dynamic_ncols=True)
  while step < total_steps:
    if step < warmup_steps:
      my_predict_batch = my_fast_predict_batch
    else:
      my_predict_batch = my_slow_predict_batch
    prediction, des_prediction = my_predict_batch(train_state.params, train_state.state, batch)

    if step >= warmup_steps:
      u_batch.append(np.array(des_prediction[0]))
      v_batch.append(np.array(des_prediction[1]))

    step += FLAGS.model_predict_steps
    pbar.update(FLAGS.model_predict_steps)

    new_u = prediction[0]
    new_v = prediction[1]

    batch = {"inputs": (new_u, new_v)}
  pbar.close()

  u_batch = np.concatenate(u_batch, axis=2)
  u_batch = u_batch.reshape([-1] + list(u_batch.shape[2:]))
  v_batch = np.concatenate(v_batch, axis=2)
  v_batch = v_batch.reshape([-1] + list(v_batch.shape[2:]))

  logging.info(u_batch.shape)

  x_len = grid.axes()[0].shape[0]
  x = 2 * np.double(grid.axes()[0]).mean() / x_len * np.arange(x_len)

  y_len = grid.axes()[1].shape[0]
  y = 2 * np.double(grid.axes()[1]).mean() / y_len * np.arange(y_len)

  ds = xarray.Dataset(
      {
          'u': (('sample', 'time', 'x', 'y'), u_batch),
          'v': (('sample', 'time', 'x', 'y'), v_batch),
      },
      coords={
          'time': (FLAGS.delta_time * FLAGS.inner_steps * np.arange(u_batch.shape[1])),
          'x': x,
          'y': y,
          'sample': np.arange(u_batch.shape[0]),
      }
  )

  if not os.path.exists(FLAGS.output_dir):
    os.makedirs(FLAGS.output_dir)
  ds.to_netcdf(os.path.join(FLAGS.output_dir, "predict.nc"))

  print(ds)


if __name__ == '__main__':
  dataset.check_versions()
  app.run(main)
