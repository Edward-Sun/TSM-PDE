"""Generate data for training and testing.
"""

import contextlib
import os
import timeit
from functools import partial
from typing import Any, Mapping, NamedTuple

import gin
import h5py
import haiku as hk
import jax
import jax.numpy as jnp
import jax_cfd.base as cfd
import jax_cfd.ml as cfd_ml
import jmp
import numpy as np
import optax
import pkg_resources
import tensorflow as tf
import tqdm
import xarray
from absl import app
from absl import flags
from absl import logging

import pde_dataset as dataset

try:
  tf.compat.v1.flags.DEFINE_multi_string("gin_file", None, "Path to a Gin file.")
  tf.compat.v1.flags.DEFINE_multi_string("gin_param", None, "Gin parameter binding.")
  tf.compat.v1.flags.DEFINE_list("gin_location_prefix", [], "Gin file search path.")
except tf.compat.v1.flags.DuplicateFlagError:
  pass

# Hyper parameters.
flags.DEFINE_integer('train_init_random_seed', 42, help='')
flags.DEFINE_string('mp_policy', 'p=f32,c=f32,o=f32', help='')
flags.DEFINE_string('mp_bn_policy', 'p=f32,c=f32,o=f32', help='')
flags.DEFINE_enum('mp_scale_type', 'NoOp', ['NoOp', 'Static', 'Dynamic'], help='')
flags.DEFINE_float('mp_scale_value', 2 ** 15, help='')
flags.DEFINE_integer('mp_scale_period', 1000, help='')

flags.DEFINE_float('max_velocity', 7.0, help='')
flags.DEFINE_float('delta_time', 0.001, help='')
flags.DEFINE_float('simulation_time', 30.0, help='')
flags.DEFINE_integer('model_input_size', 64, help='')
flags.DEFINE_integer('save_grid_size', 64, help='')

flags.DEFINE_integer('model_encode_steps', 64, help='')
flags.DEFINE_integer('model_decode_steps', 1, help='')
flags.DEFINE_integer('model_predict_steps', 64, help='')
flags.DEFINE_integer('inner_steps', 1, help='')
flags.DEFINE_integer('explicit_inner_steps', 1, help='')

flags.DEFINE_string('predict_split', None, required=True, help='')
flags.DEFINE_string('predict_result', "predict.nc", help='')
flags.DEFINE_string('output_dir', None, help='')
flags.DEFINE_string('host_address', None, help='')
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
  if FLAGS.mp_scale_type == 'Dynamic':
    return cls(FLAGS.mp_scale_value, period=FLAGS.mp_scale_period)
  else:
    return cls(FLAGS.mp_scale_value) if cls is not jmp.NoOpLossScale else cls()


def _forward(
    batch: dataset.Batch,
    is_training: bool,
    inner_steps: int,
    outer_steps: int,
) -> jnp.ndarray:
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
  """Creates the (fake) optimizer."""
  tx = optax.chain(
      optax.scale_by_adam(eps=1e-08),
      optax.scale(-1))
  return tx


def initial_state(rng: jnp.ndarray, batch: Any) -> TrainState:
  """Computes the (fake) initial network state."""
  params, state = forward.init(rng, batch, is_training=True, inner_steps=1, outer_steps=13)
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
) -> (jnp.ndarray, jnp.ndarray):
  """Evaluates a batch."""
  predictions, _ = forward.apply(params, state, None, batch, is_training=False,
                                 inner_steps=inner_steps, outer_steps=FLAGS.model_predict_steps)

  source_grid = cfd.grids.Grid((FLAGS.model_input_size, FLAGS.model_input_size),
                               domain=((0, 2 * FLAGS.jnp_pi), (0, 2 * FLAGS.jnp_pi)))
  destination_grid = cfd.grids.Grid((FLAGS.save_grid_size, FLAGS.save_grid_size),
                                    domain=((0, 2 * FLAGS.jnp_pi), (0, 2 * FLAGS.jnp_pi)))

  def my_downsample(x):
    return cfd.resize.downsample_staggered_velocity(source_grid, destination_grid, x)

  my_downsample = jax.vmap(my_downsample, in_axes=((0, 0),), out_axes=(0, 0))
  my_downsample = jax.vmap(my_downsample, in_axes=((0, 0),), out_axes=(0, 0))
  des_predictions_ds = my_downsample(predictions)

  des_predictions = (predictions[0][:, -1:],
                     predictions[1][:, -1:])

  des_predictions_ds = (des_predictions_ds[0][:, -FLAGS.model_predict_steps:],
                        des_predictions_ds[1][:, -FLAGS.model_predict_steps:])

  predictions = (predictions[0][:, -FLAGS.model_encode_steps:],
                 predictions[1][:, -FLAGS.model_encode_steps:])

  return predictions, des_predictions, des_predictions_ds


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


def prediction(train_state, target):
  eval_data = xarray.open_dataset(FLAGS.predict_split)
  eval_u = eval_data.variables['u'].data
  eval_v = eval_data.variables['v'].data
  total_steps = int(FLAGS.simulation_time / (FLAGS.delta_time * FLAGS.inner_steps * FLAGS.explicit_inner_steps))
  original_shape = eval_u.shape
  n_samples = eval_u.shape[0]
  logging.info(original_shape)

  local_device_count = jax.local_device_count()
  eval_u = eval_u.reshape((local_device_count, -1) + eval_u.shape[1:])
  eval_v = eval_v.reshape((local_device_count, -1) + eval_v.shape[1:])

  source_grid = cfd.grids.Grid((eval_u.shape[-1], eval_u.shape[-1]),
                               domain=((0, 2 * FLAGS.jnp_pi), (0, 2 * FLAGS.jnp_pi)))
  destination_grid = cfd.grids.Grid((FLAGS.model_input_size, FLAGS.model_input_size),
                                    domain=((0, 2 * FLAGS.jnp_pi), (0, 2 * FLAGS.jnp_pi)))

  def my_downsample(x):
    return cfd.resize.downsample_staggered_velocity(source_grid, destination_grid, x)

  my_downsample = jax.vmap(my_downsample, in_axes=((0, 0),), out_axes=(0, 0))
  my_downsample = jax.vmap(my_downsample, in_axes=((0, 0),), out_axes=(0, 0))
  my_downsample = jax.pmap(my_downsample, axis_name='i')

  batch = {"inputs": my_downsample((eval_u[:, :, :FLAGS.model_encode_steps],
                                    eval_v[:, :, :FLAGS.model_encode_steps]))}

  step = 0
  my_predict_batch = jax.pmap(partial(predict_batch, inner_steps=FLAGS.inner_steps),
                              axis_name='i', donate_argnums=(0,))

  if not os.path.exists(FLAGS.output_dir):
    os.makedirs(FLAGS.output_dir)
  example_path = os.path.join(FLAGS.output_dir, target + '.hdf5')

  record_writer_num = n_samples
  record_writers = []
  target_res_list = [16, 32, 64]
  if FLAGS.save_grid_size not in target_res_list:
    target_res_list.append(FLAGS.save_grid_size)

  while step < total_steps:
    step += (FLAGS.model_predict_steps // FLAGS.explicit_inner_steps)
  total_seq_len = step
  step = 0

  pbar = tqdm.tqdm(total=total_steps, dynamic_ncols=True)
  for i in range(record_writer_num):
    if step > 0:
      f = h5py.File(example_path + "-%04d" % i, 'r+')
      if i == 0:
        pbar.update(step)
    else:
      f = h5py.File(example_path + "-%04d" % i, 'w')
      for target_res in target_res_list:
        adjusted_seq_len = int(total_seq_len * target_res // FLAGS.save_grid_size)
        f.create_dataset(f"prefix_{target_res}x{target_res}",
                         data=np.zeros((2, adjusted_seq_len, target_res, target_res)))
        f.create_dataset(f"full_prefix_{target_res}x{target_res}",
                         data=np.zeros((2, total_seq_len, target_res, target_res)))
    record_writers.append(f)

  down_samplers = dict()
  for target_res in target_res_list:
    if target_res == FLAGS.save_grid_size:
      down_samplers[target_res] = lambda x: x
    else:
      source_grid = cfd.grids.Grid((FLAGS.save_grid_size, FLAGS.save_grid_size),
                                   domain=((0, 2 * FLAGS.jnp_pi), (0, 2 * FLAGS.jnp_pi)))
      destination_grid = cfd.grids.Grid((target_res, target_res),
                                        domain=((0, 2 * FLAGS.jnp_pi), (0, 2 * FLAGS.jnp_pi)))

      def gen_downsample(source_grid, destination_grid):
        def my_downsample(x):
          return cfd.resize.downsample_staggered_velocity(source_grid, destination_grid, x)

        my_downsample = jax.vmap(my_downsample, in_axes=((0, 0),), out_axes=(0, 0))
        my_downsample = jax.vmap(my_downsample, in_axes=((0, 0),), out_axes=(0, 0))

        def transform(x):
          u = x[:, 0]
          v = x[:, 1]
          new_u, new_v = my_downsample((u, v))
          return np.stack((new_u, new_v), axis=1)

        return transform

      down_samplers[target_res] = gen_downsample(source_grid, destination_grid)

  while step < total_steps:
    prediction, des_prediction, des_prediction_ds = my_predict_batch(train_state.params, train_state.state, batch)
    uv_batch_np = None
    des_prediction_downsample = des_prediction_ds
    uv_batch_downsampled_np = np.stack([
        np.array(des_prediction_downsample[0]),
        np.array(des_prediction_downsample[1])
    ], axis=0)
    uv_batch_downsampled_np = np.reshape(
        uv_batch_downsampled_np,
        (2, n_samples, -1, FLAGS.save_grid_size, FLAGS.save_grid_size)
    ).transpose((1, 0, 2, 3, 4))

    new_u = prediction[0]
    new_v = prediction[1]
    batch = {"inputs": (new_u, new_v)}
    for target_res in target_res_list:
      start_step = step * target_res // FLAGS.save_grid_size
      end_step = (step + FLAGS.model_predict_steps // FLAGS.explicit_inner_steps) * target_res // FLAGS.save_grid_size
      uv_batch_ds_np = down_samplers[target_res](uv_batch_downsampled_np)
      sep = (FLAGS.save_grid_size // target_res)
      start = sep - 1
      uv_batch_ds_np_drop = uv_batch_ds_np[:, :, start::sep]
      for i in range(record_writer_num):
        record_writers[i][f"prefix_{target_res}x{target_res}"][:, start_step: end_step] = uv_batch_ds_np_drop[i]
        record_writers[i][f"full_prefix_{target_res}x{target_res}"][:,
          step: step + FLAGS.model_predict_steps // FLAGS.explicit_inner_steps] = uv_batch_ds_np[i]
    step += FLAGS.model_predict_steps // FLAGS.explicit_inner_steps
    pbar.update(FLAGS.model_predict_steps // FLAGS.explicit_inner_steps)
  pbar.close()
  for i in range(record_writer_num):
    record_writers[i].close()
  return None


def main(argv):
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

  world_size = int(os.environ.get('WORLD_SIZE', '0'))
  node_rank = int(os.environ.get('NODE_RANK', '0'))
  if world_size > 0:
    jax.distributed.initialize(FLAGS.host_address, world_size, node_rank)
    print('global devices=', jax.devices())
    print('local devices=', jax.local_devices())

  # Add search path for gin files stored in package.
  gin.add_config_file_search_path(
      pkg_resources.resource_filename(__name__, "gin"))

  parse_gin_defaults_and_flags()
  FLAGS.alsologtostderr = True
  mp_policy = get_policy()
  bn_policy = get_bn_policy().with_output_dtype(mp_policy.compute_dtype)
  hk.mixed_precision.set_policy(hk.BatchNorm, bn_policy)
  hk.mixed_precision.set_policy(_forward, mp_policy)

  rng = jax.random.PRNGKey(FLAGS.train_init_random_seed)
  sample_vec = np.zeros((1, FLAGS.model_encode_steps, FLAGS.model_input_size, FLAGS.model_input_size))
  train_state = initial_state(rng, {'inputs': (sample_vec, sample_vec)})
  with time_activity('predict'):
    prediction(train_state, FLAGS.predict_result)


if __name__ == '__main__':
  dataset.check_versions()
  app.run(main)
