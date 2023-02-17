"""Handles training and evaluation of the model."""

import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)
import contextlib
import timeit
from typing import Mapping, NamedTuple, Tuple, Any
from functools import partial
import os

from absl import app
from absl import flags
from absl import logging
import haiku as hk
import jax
import jax.numpy as jnp
import jmp
import numpy as np
import optax

import xarray
import tqdm

import gin
import tensorflow.compat.v1 as tf
import pkg_resources
import jax_cfd.base as cfd
import jax_cfd.ml as cfd_ml
import jax_cfd.data as cfd_data
import flax.errors
from flax import jax_utils
from flax import serialization
from flax.training import checkpoints

from tsm_modules.tsm_model_builder import tsm_decoded_trajectory_with_inputs
from utils import get_lr_schedule
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
flags.DEFINE_integer('mp_scale_period', 1000, help='')
flags.DEFINE_bool('mp_skip_nonfinite', False, help='')

flags.DEFINE_bool('no_train', False, help='')
flags.DEFINE_string('train_split', 'TRAIN_AND_VALID', help='')
flags.DEFINE_integer('train_log_every', 100, help='')
flags.DEFINE_float('train_epochs', 10, help='')
flags.DEFINE_float('train_lr_warmup_epochs', 0.1, help='')
flags.DEFINE_float('train_lr_init', 0.1, help='')
flags.DEFINE_float('train_weight_decay', 1e-4, help='')
flags.DEFINE_float('train_clip_norm', 0.25, help='')
flags.DEFINE_integer('train_device_batch_size', 8, help='')
flags.DEFINE_integer('train_predict_every', -1, help='')
flags.DEFINE_bool('use_exponential_decay', False, help='')
flags.DEFINE_bool('use_rsqrt_decay', False, help='')
flags.DEFINE_bool('use_linear_decay', False, help='')

flags.DEFINE_string('predict_split', None, help='')
flags.DEFINE_string('predict_target', None, help='')
flags.DEFINE_string('predict_result', "predict.nc", help='')
flags.DEFINE_string('predict_valid_split', None, help='')
flags.DEFINE_string('predict_valid_target', None, help='')
flags.DEFINE_string('predict_valid_result', "tmp_predict.nc", help='')
flags.DEFINE_string('output_dir', None, help='')

flags.DEFINE_bool('do_predict', False, help='')
flags.DEFINE_float('delta_time', 0.001, help='')
flags.DEFINE_float('predict_simulation_time', 25.0, help='')
flags.DEFINE_integer('model_input_size', 64, help='')
flags.DEFINE_integer('save_grid_size', 64, help='')
flags.DEFINE_integer('model_encode_steps', 64, help='')
flags.DEFINE_integer('model_decode_steps', 1, help='')
flags.DEFINE_integer('model_predict_steps', 64, help='')

flags.DEFINE_bool('resume_checkpoint', False, help='')
flags.DEFINE_bool('explicit_resume', False, help='')
flags.DEFINE_string('resume_checkpoint_dir', None, help='')
flags.DEFINE_bool('warm_start', False, help='')
flags.DEFINE_bool('no_dropout', False, help='')
flags.DEFINE_float('max_velocity', 7.0, help='')
flags.DEFINE_float('adam_beta2', 0.999, help='')
flags.DEFINE_bool('evaluate_latency', False, help='')
flags.DEFINE_integer('inner_steps', 1, help='')
flags.DEFINE_integer('explicit_inner_steps', 1, help='')
flags.DEFINE_string('host_address', None, help='')
flags.DEFINE_integer('grad_accum', 1, help='')
flags.DEFINE_float('jnp_pi', jnp.pi, help='')


FLAGS = flags.FLAGS
Scalars = Mapping[str, jnp.ndarray]


class TrainState(NamedTuple):
  step: int
  params: hk.Params
  state: hk.State
  opt_state: optax.OptState


get_policy = lambda: jmp.get_policy(FLAGS.mp_policy)
get_bn_policy = lambda: jmp.get_policy(FLAGS.mp_bn_policy)


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
  if stable_time_step <= dt:
    inner_steps = inner_steps * round(dt / stable_time_step)
  else:
    stable_time_step = dt
  model = cfd_ml.model_builder.get_model_cls(grid, stable_time_step, physics_specs)()
  if FLAGS.no_dropout:
    trajectory = jax.vmap(
        partial(
            cfd_ml.model_utils.decoded_trajectory_with_inputs(
                model=model,
                num_init_frames=FLAGS.model_encode_steps),
            outer_steps=outer_steps,
            inner_steps=inner_steps,
        ),
        axis_name='i')
  else:
    trajectory = jax.vmap(
        partial(
            tsm_decoded_trajectory_with_inputs(
                model=model,
                num_init_frames=FLAGS.model_encode_steps),
            outer_steps=outer_steps,
            inner_steps=inner_steps,
            is_training=is_training,
        ),
        axis_name='i')

  final, predictions = trajectory(inputs)
  return predictions


# Transform our forwards function into a pair of pure functions.
forward = hk.transform_with_state(_forward)

def map_nested_fn(fn):
  """Recursively apply `fn to the key-value pairs of a nested dict / pytree."""

  def map_fn(nested_dict):
    return {
        k: (map_fn(v) if hasattr(v, "keys") else fn(k, v))
        for k, v in nested_dict.items()
    }

  return map_fn


def make_optimizer(
    lr_schedule=lambda x: 1.0
) -> optax.GradientTransformation:
  """SGD with nesterov momentum and a custom lr schedule."""
  adam_tx = optax.chain(
      optax.clip_by_global_norm(FLAGS.train_clip_norm),
      optax.scale_by_adam(b2=FLAGS.adam_beta2, eps=1e-08),
      optax.add_decayed_weights(weight_decay=FLAGS.train_weight_decay),
      optax.scale_by_schedule(lr_schedule),
      optax.scale(-1))
  if FLAGS.grad_accum > 1:
    adam_tx = optax.MultiSteps(
        adam_tx, FLAGS.grad_accum,
    )
  return adam_tx


def loss_fn(
    params: hk.Params,
    state: hk.State,
    batch: dataset.Batch,
    inner_steps: int,
    decode_steps: int,
    dropout_rng: jax.random.PRNGKey,
) -> Tuple[jnp.ndarray, Tuple[jnp.ndarray, hk.State, Tuple]]:
  """Computes a regularized loss for the given batch."""
  targets = batch['inputs']
  predictions, state = forward.apply(params, state, dropout_rng, batch, is_training=True,
                                     inner_steps=inner_steps, outer_steps=decode_steps)
  pu = predictions[0][:, FLAGS.model_encode_steps: FLAGS.model_encode_steps + decode_steps]
  pv = predictions[1][:, FLAGS.model_encode_steps: FLAGS.model_encode_steps + decode_steps]
  tu = targets[0][:, FLAGS.model_encode_steps: FLAGS.model_encode_steps + decode_steps]
  tv = targets[1][:, FLAGS.model_encode_steps: FLAGS.model_encode_steps + decode_steps]

  loss_u = optax.l2_loss(predictions=pu, targets=tu).mean()
  loss_v = optax.l2_loss(predictions=pv, targets=tv).mean()
  loss = loss_u + loss_v
  return loss, (loss, state, (pu, pv))


def train_step(
    train_state: TrainState,
    batch: dataset.Batch,
    inner_steps: int,
    decode_steps: int,
    lr_schedule=lambda x: 1.0,
    dropout_rng=None,
) -> Tuple[TrainState, Scalars, Any]:
  """Applies an update to parameters and returns new state."""
  step, params, state, opt_state = train_state
  dropout_rng, new_dropout_rng = jax.random.split(dropout_rng)
  grads, (loss, new_state, predictions) = (
      jax.grad(loss_fn, has_aux=True)(params, state, batch,
                                      inner_steps, decode_steps, dropout_rng))

  assert FLAGS.model_input_size >= FLAGS.save_grid_size

  # Grads are in "param_dtype" (likely F32) here. We cast them back to the
  # compute dtype such that we do the all-reduce below in the compute precision
  # (which is typically lower than the param precision).
  policy = get_policy()
  grads = policy.cast_to_compute(grads)

  # Taking the mean across all replicas to keep params in sync.
  grads = jax.lax.pmean(grads, axis_name='i')

  # We compute our optimizer update in the same precision as params, even when
  # doing mixed precision training.
  grads = policy.cast_to_param(grads)

  # Compute and apply updates via our optimizer.
  updates, new_opt_state = make_optimizer(lr_schedule).update(grads, opt_state, params)
  new_params = optax.apply_updates(params, updates)

  grads_finite = None
  if FLAGS.mp_skip_nonfinite:
    grads_finite = jmp.all_finite(grads)
    new_params = jmp.select_tree(grads_finite, new_params, params)
    new_state = jmp.select_tree(grads_finite, new_state, state)
    new_opt_state = jmp.select_tree(grads_finite, new_opt_state, opt_state)

  # Scalars to log (note: we log the mean across all hosts/devices).
  scalars = {'train_loss': loss}
  if FLAGS.mp_skip_nonfinite:
    scalars['grads_finite'] = grads_finite
  state, scalars = jmp.cast_to_full((state, scalars))
  scalars = jax.lax.pmean(scalars, axis_name='i')
  new_step = step + 1
  train_state = TrainState(new_step, new_params, new_state, new_opt_state)
  return train_state, scalars, new_dropout_rng


def initial_state(rng: jnp.ndarray, batch: Any) -> TrainState:
  """Computes the initial network state."""
  params, state = forward.init(rng, batch, is_training=True, inner_steps=1, outer_steps=13)
  opt_state = make_optimizer().init(params)
  train_state = TrainState(0, params, state, opt_state)
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

  if FLAGS.model_input_size > FLAGS.save_grid_size:
    source_grid = cfd.grids.Grid((FLAGS.model_input_size, FLAGS.model_input_size),
                                 domain=((0, 2 * FLAGS.jnp_pi), (0, 2 * FLAGS.jnp_pi)))
    destination_grid = cfd.grids.Grid((FLAGS.save_grid_size, FLAGS.save_grid_size),
                                      domain=((0, 2 * FLAGS.jnp_pi), (0, 2 * FLAGS.jnp_pi)))

    def my_downsample(x):
      return cfd.resize.downsample_staggered_velocity(source_grid, destination_grid, x)

    my_downsample = jax.vmap(my_downsample, in_axes=((0, 0),), out_axes=(0, 0))
    my_downsample = jax.vmap(my_downsample, in_axes=((0, 0),), out_axes=(0, 0))
    des_predictions = my_downsample(predictions)
  else:
    des_predictions = predictions

  des_predictions = (des_predictions[0][:, -FLAGS.model_predict_steps:],
                     des_predictions[1][:, -FLAGS.model_predict_steps:])

  predictions = (
      jnp.concatenate([batch['inputs'][0], predictions[0]], axis=1),
      jnp.concatenate([batch['inputs'][1], predictions[1]], axis=1),
  )

  predictions = (predictions[0][:, -FLAGS.model_encode_steps:],
                 predictions[1][:, -FLAGS.model_encode_steps:])

  return predictions, des_predictions


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


def prediction(train_state, data_split, target, predict_target=None):
  test_data = dataset.get_dataloader(
      data_split,
      resolution=FLAGS.model_input_size,
      full=True,
      mode="test_trajectory" if FLAGS.model_input_size <= FLAGS.save_grid_size else "test_high_resolution",
  )
  eval_u = test_data[:, 0, -FLAGS.model_encode_steps:]
  eval_v = test_data[:, 1, -FLAGS.model_encode_steps:]
  total_steps = int(FLAGS.predict_simulation_time / (FLAGS.delta_time * FLAGS.inner_steps * FLAGS.explicit_inner_steps))
  original_shape = eval_u.shape
  logging.info(original_shape)

  local_device_count = jax.local_device_count()
  eval_u = eval_u.reshape((local_device_count, -1) + eval_u.shape[1:])
  eval_v = eval_v.reshape((local_device_count, -1) + eval_v.shape[1:])

  if eval_u.shape[-1] > FLAGS.model_input_size:
    source_grid = cfd.grids.Grid((eval_u.shape[-1], eval_u.shape[-1]),
                                 domain=((0, 2 * FLAGS.jnp_pi), (0, 2 * FLAGS.jnp_pi)))
    destination_grid = cfd.grids.Grid((FLAGS.model_input_size, FLAGS.model_input_size),
                                      domain=((0, 2 * FLAGS.jnp_pi), (0, 2 * FLAGS.jnp_pi)))

    def my_downsample(x):
      return cfd.resize.downsample_staggered_velocity(source_grid, destination_grid, x)

    my_downsample = jax.vmap(my_downsample, in_axes=((0, 0),), out_axes=(0, 0))
    my_downsample = jax.vmap(my_downsample, in_axes=((0, 0),), out_axes=(0, 0))
    my_downsample = jax.pmap(my_downsample, axis_name='i')
    batch = {"inputs": my_downsample((eval_u[:, :, -FLAGS.model_encode_steps:],
                                      eval_v[:, :, -FLAGS.model_encode_steps:]))}
  else:
    batch = {"inputs": (eval_u[:, :, -FLAGS.model_encode_steps:],
                        eval_v[:, :, -FLAGS.model_encode_steps:])}

  u_batch = []
  v_batch = []

  step = 0
  my_predict_batch = jax.pmap(partial(predict_batch, inner_steps=FLAGS.inner_steps),
                              axis_name='i', donate_argnums=(0,))
  prediction_start_time = timeit.default_timer()
  while step < total_steps:
    if (step % FLAGS.train_log_every * 10) == 0:
      logging.info('[Predict %s/%s]', step, total_steps)
    prediction, des_prediction = my_predict_batch(train_state.params, train_state.state, batch)
    u_batch.append(np.array(des_prediction[0]))
    v_batch.append(np.array(des_prediction[1]))
    new_u = prediction[0]
    new_v = prediction[1]

    batch = {"inputs": (new_u, new_v)}
    step += (FLAGS.model_predict_steps // FLAGS.explicit_inner_steps)
  prediction_end_time = timeit.default_timer()

  u_batch = np.concatenate(u_batch, axis=2)
  v_batch = np.concatenate(v_batch, axis=2)
  u_batch = u_batch.reshape(
      original_shape[:1] + (-1, FLAGS.save_grid_size, FLAGS.save_grid_size)
  )[:, FLAGS.explicit_inner_steps - 1::FLAGS.explicit_inner_steps]
  v_batch = v_batch.reshape(
      original_shape[:1] + (-1, FLAGS.save_grid_size, FLAGS.save_grid_size)
  )[:, FLAGS.explicit_inner_steps - 1::FLAGS.explicit_inner_steps]

  u_batch = xarray.Variable(("sample", "time", "x", "y"), u_batch)
  v_batch = xarray.Variable(("sample", "time", "x", "y"), v_batch)

  logging.info(u_batch.shape)
  grid = cfd.grids.Grid((FLAGS.save_grid_size, FLAGS.save_grid_size),
                        domain=((0, 2 * FLAGS.jnp_pi), (0, 2 * FLAGS.jnp_pi)))
  x, y = grid.axes()
  ds = xarray.Dataset(
      {
          'u': (('sample', 'time', 'x', 'y'), u_batch),
          'v': (('sample', 'time', 'x', 'y'), v_batch),
      },
      coords={
          'time': (FLAGS.delta_time * FLAGS.inner_steps *
                   FLAGS.explicit_inner_steps * np.arange(u_batch.shape[1])),
          'sample': np.arange(u_batch.shape[0]),
          'x': x,
          'y': y,
      }
  )

  if not os.path.exists(FLAGS.output_dir):
    os.makedirs(FLAGS.output_dir)
  ds.to_netcdf(os.path.join(FLAGS.output_dir, target))

  if predict_target is not None:
    time = int(20.0 / (FLAGS.delta_time * FLAGS.inner_steps * FLAGS.explicit_inner_steps))
    target_ds = xarray.open_dataset(predict_target)
    ds = ds[dict(time=slice(400))]
    target_ds.attrs['ndim'] = 2
    summary = xarray.concat([
        cfd_data.evaluation.compute_summary_dataset(_ds, target_ds)
        for _ds in [ds, target_ds]
    ], dim='model')
    summary.coords['model'] = list(["model", "target"])
    correlation = summary.vorticity_correlation.compute()
    corr = (correlation.data[0, :time] > 0.8).sum()
    latency = prediction_end_time - prediction_start_time
    return ds, {"corr": corr, "latency": latency}
  else:
    return ds, {}


def prepare_batch(batch):
  batch_save_grid = None
  if type(batch) is tuple or type(batch) is list:
    batch, batch_save_grid = batch
  prefix, target, _ = np.split(batch,
                               (FLAGS.model_encode_steps, FLAGS.model_encode_steps + FLAGS.model_decode_steps),
                               axis=2)
  target_save_grid = None
  if batch_save_grid is not None:
    relative_length = batch_save_grid.shape[2] // batch.shape[2]
    _, target_save_grid, _ = np.split(batch_save_grid,
                                      (FLAGS.model_encode_steps * relative_length,
                                       (FLAGS.model_encode_steps + FLAGS.model_decode_steps) * relative_length),
                                      axis=2)
    target_save_grid = target_save_grid[:, :, relative_length - 1::relative_length]
  u_batch_prefix = prefix[:, 0]
  v_batch_prefix = prefix[:, 1]
  u_batch_target = target[:, 0]
  v_batch_target = target[:, 1]
  batch = {
      'inputs': (np.concatenate([u_batch_prefix, u_batch_target], axis=1),
                 np.concatenate([v_batch_prefix, v_batch_target], axis=1),),
  }
  if target_save_grid is not None:
    batch[f'target_{FLAGS.save_grid_size}x{FLAGS.save_grid_size}'] = (
        target_save_grid[:, 0, -1:], target_save_grid[:, 1, -1:])
  return batch


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
  local_device_count = jax.local_device_count()
  if FLAGS.no_train:
    train_dataset, train_num_examples = [], 1
  else:
    train_dataset, train_num_examples = dataset.get_dataloader(
        FLAGS.train_split,
        mode="train",
        encode_steps=FLAGS.model_encode_steps,
        decode_steps=FLAGS.model_decode_steps,
        prepare_batch=prepare_batch,
        full=True,
        resolution=min(FLAGS.model_input_size, FLAGS.save_grid_size),
        batch_dims=[local_device_count, FLAGS.train_device_batch_size])
  # The total batch size is the batch size accross all hosts and devices. In a
  # multi-host training setup each host will only see a batch size of
  # `total_train_batch_size / jax.host_count()`.
  logging.info("train_num_examples=%d", train_num_examples)
  total_batch_size = FLAGS.train_device_batch_size * local_device_count
  num_train_steps = (train_num_examples * FLAGS.train_epochs) // total_batch_size
  num_train_steps = int(num_train_steps)
  lr_schedule = partial(
      get_lr_schedule,
      steps_per_epoch=int(train_num_examples // total_batch_size),
      train_lr_warmup_epochs=FLAGS.train_lr_warmup_epochs,
      train_epochs=FLAGS.train_epochs,
      train_lr_init=FLAGS.train_lr_init,
      use_exponential_decay=FLAGS.use_exponential_decay,
      use_rsqrt_decay=FLAGS.use_rsqrt_decay,
      use_linear_decay=FLAGS.use_linear_decay,
  )

  # Assign mixed precision policies to modules. Note that when training in f16
  # we keep BatchNorm in  full precision. When training with bf16 you can often
  # use bf16 for BatchNorm.
  mp_policy = get_policy()
  bn_policy = get_bn_policy().with_output_dtype(mp_policy.compute_dtype)
  # NOTE: The order we call `set_policy` doesn't matter, when a method on a
  # class is called the policy for that class will be applied, or it will
  # inherit the policy from its parent module.
  hk.mixed_precision.set_policy(hk.BatchNorm, bn_policy)
  hk.mixed_precision.set_policy(_forward, mp_policy)
  # For initialization we need the same random key on each device.
  rng = jax.random.PRNGKey(FLAGS.train_init_random_seed)
  dropout_rngs = jax.random.split(rng, jax.local_device_count())
  # Initialization requires an example input.
  sample_vec = np.zeros((1, FLAGS.model_encode_steps + FLAGS.model_decode_steps,
                         FLAGS.model_input_size, FLAGS.model_input_size))
  train_state = initial_state(rng, {'inputs': (sample_vec, sample_vec)})

  start_step = 0
  if FLAGS.resume_checkpoint:
    logging.info("Loading from %s" % FLAGS.output_dir)
    train_state = checkpoints.restore_checkpoint(FLAGS.output_dir, train_state)
    start_step = int(train_state.step)

    if start_step == 0 and FLAGS.resume_checkpoint_dir is not None or FLAGS.explicit_resume:
      logging.info("Loading from %s" % FLAGS.resume_checkpoint_dir)
      try:
        train_state = checkpoints.restore_checkpoint(FLAGS.resume_checkpoint_dir, train_state)
        start_step = 0
      except KeyError:
        # Legacy checkpoint, try to change key name from my_* to tsm_*
        logging.info("KeyError, try to change key name from my_* to tsm_*")
        restore_state = checkpoints.restore_checkpoint(FLAGS.resume_checkpoint_dir, target=None)
        def change_key_name(state_dict):
          new_sub_restore_state = {}
          for key, value in state_dict.items():
            new_key = key.replace("my_", "tsm_")
            if isinstance(value, dict):
              new_sub_restore_state[new_key] = change_key_name(value)
            else:
              new_sub_restore_state[new_key] = value
          return new_sub_restore_state
        new_restore_state = change_key_name(restore_state)

        train_state = serialization.from_state_dict(train_state, new_restore_state)
        start_step = 0
      except ValueError:
        ckpt_train_state = TrainState(0, train_state.params, train_state.state, train_state.opt_state.inner_opt_state)
        ckpt_train_state = checkpoints.restore_checkpoint(FLAGS.resume_checkpoint_dir, ckpt_train_state)
        train_opt_state = optax.MultiStepsState(
            mini_step=jnp.zeros([], dtype=jnp.int32),
            gradient_step=jnp.zeros([], dtype=jnp.int32),
            inner_opt_state=ckpt_train_state.opt_state,
            acc_grads=jax.tree_map(jnp.zeros_like, train_state.params),
        )
        train_state = TrainState(0, ckpt_train_state.params, ckpt_train_state.state, train_opt_state)
        start_step = 0
        logging.info("Loaded from deprecated checkpoint")

  train_state = jax_utils.replicate(train_state)
  my_train_step = partial(train_step, lr_schedule=lr_schedule,
                          inner_steps=FLAGS.inner_steps, decode_steps=FLAGS.model_decode_steps)
  my_train_step = jax.pmap(my_train_step, axis_name='i', donate_argnums=(0,))

  predict_every = FLAGS.train_predict_every
  log_every = FLAGS.train_log_every
  best_metric = 0.0

  with time_activity('train'):
    train_dataloader = iter(train_dataset)
    for step_num in tqdm.tqdm(range(start_step, num_train_steps)):
      if FLAGS.no_train:
        break
      # Take a single training step.
      with jax.profiler.StepTraceAnnotation('train', step_num=step_num):
        batch = next(train_dataloader)
        train_state, train_scalars, dropout_rngs = my_train_step(
            train_state, batch, dropout_rng=dropout_rngs)

      pred_ds, predict_scalars = None, {}
      if predict_every > 0 and step_num and step_num % predict_every == 0:
        if FLAGS.predict_valid_split is not None:
          with time_activity('predict during train'):
            pred_ds, predict_scalars = prediction(
                train_state, FLAGS.predict_valid_split,
                FLAGS.predict_valid_result, FLAGS.predict_valid_target)
          logging.info('[Predict %s/%s] %s', step_num, num_train_steps, predict_scalars)

      # Log progress at fixed intervals.
      if step_num and step_num % log_every == 0:
        train_scalars = jax.tree_map(lambda v: np.mean(v).item(),
                                     jax.device_get(train_scalars))
        logging.info('[Train %s/%s] %s',
                     step_num, num_train_steps, train_scalars)

      if predict_every > 0 and step_num and step_num % predict_every == 0 and jax.host_id() == 0:
        if FLAGS.predict_valid_split is not None and FLAGS.output_dir is not None:
          checkpoints.save_checkpoint(
              FLAGS.output_dir, jax_utils.unreplicate(train_state), step_num, keep=3)

          if len(predict_scalars) > 0 and best_metric < predict_scalars['corr']:
            best_metric = predict_scalars['corr']
            checkpoints.save_checkpoint(
                os.path.join(FLAGS.output_dir, 'best'),
                jax_utils.unreplicate(train_state), step_num, keep=1)
            pred_ds.to_netcdf(os.path.join(os.path.join(FLAGS.output_dir, 'best'), FLAGS.predict_result))
            logging.info('Saved best checkpoint: %d', step_num)

  if FLAGS.output_dir is not None and start_step + 1 < num_train_steps and jax.host_id() == 0:
    try:
      checkpoints.save_checkpoint(
          FLAGS.output_dir, jax_utils.unreplicate(train_state), step_num, keep=3)
    except flax.errors.InvalidCheckpointError:
      logging.info('Checkpoint already exists at %s', FLAGS.output_dir)

  if FLAGS.do_predict and jax.host_id() == 0:
    # Run prediction on the test set.
    if FLAGS.evaluate_latency:
      with time_activity('warmup-predict'):
        predict_scalars = prediction(train_state, FLAGS.predict_split, FLAGS.predict_result)
      logging.info('[Predict FINAL]: %s', predict_scalars)

    with time_activity('predict'):
      predict_scalars = prediction(train_state, FLAGS.predict_split, FLAGS.predict_result, FLAGS.predict_target)
    logging.info('[Predict FINAL]: %s', predict_scalars)


if __name__ == '__main__':
  dataset.check_versions()
  app.run(main)
