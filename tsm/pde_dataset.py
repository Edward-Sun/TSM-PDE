"""PDE dataset with typical pre-processing."""

import copy
import glob
import logging
import time
import types
from typing import Callable, Generator, Iterable, Mapping, Optional, Sequence

import h5py
import jax
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
import torch
import torch.utils.data
from packaging import version

Batch = Mapping[str, np.ndarray]


def _check_min_version(mod: types.ModuleType, min_ver: str):
  actual_ver = getattr(mod, '__version__')
  if version.parse(actual_ver) < version.parse(min_ver):
    raise ValueError(
        f'{mod.__name__} >= {min_ver} is required, you have {actual_ver}')


def check_versions():
  _check_min_version(tf, '2.5.0')
  _check_min_version(tfds, '4.2.0')


class TorchDataset(torch.utils.data.Dataset):
  def __init__(self,
               data_file,
               encode_steps=1,
               decode_steps=1,
               prefix='prefix_64x64',
               target=None,
               ):
    self.data_file = data_file
    self.encode_steps = encode_steps
    self.decode_steps = decode_steps
    self.h5file = None
    self.stride = None
    self.inv_stride = None
    self.prefix = prefix
    self.target = target
    self.prefix_file = None
    self.target_file = None

  def __getitem__(self, index):
    if self.h5file is None:
      self.h5file = h5py.File(self.data_file, 'r')
      source_length = self.h5file[self.prefix].shape[1]
      self.prefix_file = self.h5file[self.prefix]
      if self.prefix_file.shape[0] == 1:
        self.prefix_file = self.prefix_file[:]
      if self.target:
        target_length = self.h5file[self.target].shape[1]
        stride = max(1, source_length // target_length)
        inv_stride = max(1, target_length // source_length)
        self.target_file = self.h5file[self.target]
        if self.target_file.shape[0] == 1:
          self.target_file = self.target_file[:]
      else:
        stride = 1
        inv_stride = 1
      if self.encode_steps == 0:
        encode_strides = 0
      else:
        encode_strides = max(1, self.encode_steps // stride)
      if self.decode_steps == 0:
        decode_strides = 0
      else:
        decode_strides = max(1, self.decode_steps // stride)
      self.stride = stride
      self.inv_stride = inv_stride
      self.sequence_length = encode_strides + decode_strides
    prefix = self.prefix_file[:, index * self.stride: (index + self.sequence_length) * self.stride]
    if self.target:
      target = self.target_file[:, index * self.inv_stride: (index + self.sequence_length) * self.inv_stride]
      return prefix, target
    else:
      return prefix

  def __len__(self):
    h5file = h5py.File(self.data_file, 'r')
    source_length = h5file[self.prefix].shape[1]
    if self.target:
      target_length = h5file[self.target].shape[1]
      stride = max(1, source_length // target_length)
    else:
      stride = 1
    encode_strides = max(1, self.encode_steps // stride)
    decode_strides = max(1, self.decode_steps // stride)
    sequence_length = encode_strides + decode_strides
    return int(h5file[self.prefix].shape[1]) + 1 - sequence_length


def get_dataloader(
    split: str,
    *,
    mode: str,
    prepare_batch: Optional[Callable] = None,
    batch_dims: Sequence[int] = (1,),
    repeat: bool = True,
    encode_steps: int = 1,
    decode_steps: int = 0,
    workers: int = 16,
    world_size: int = 1,
    rank: int = 0,
    use_jax: bool = True,
    resolution: int = 64,
    standard_prefix_length: int = 64,
    standard_prefix_resolution: int = 64,
    full: bool = True,
):
  """Returns a data loader for the given split."""
  if full:
    prefix_str = f'full_prefix_{resolution}x{resolution}'
  else:
    prefix_str = f'prefix_{resolution}x{resolution}'

  if mode == 'eval':
    datasets = sorted(glob.glob(split))
    if world_size > 1:
      datasets = datasets[rank::world_size]
    print(datasets)
    datasets = map(lambda x: TorchDataset(x, encode_steps=encode_steps,
                                          decode_steps=decode_steps, prefix=prefix_str), datasets)
    final_dataset = torch.utils.data.ConcatDataset(datasets)
    eval_dataset = torch.utils.data.DataLoader(
        final_dataset, batch_size=int(np.prod(batch_dims)), num_workers=workers)
    eval_dataset = double_buffer(eval_dataset, prepare_batch=prepare_batch, use_jax=use_jax)
    return eval_dataset, len(final_dataset)
  elif mode == 'test_trajectory':
    prefix = []
    files = glob.glob(split)
    print(sorted(files))
    for file in sorted(files):
      with h5py.File(file, 'r') as f:
        # print(f['prefix'].shape)
        total_length_64 = f[f'prefix_{standard_prefix_resolution}x{standard_prefix_resolution}'].shape[1]
        total_length_target = f[prefix_str].shape[1]
        target_length = standard_prefix_length * total_length_target // total_length_64
        prefix.append(f[prefix_str][:, :target_length])
    prefix = np.stack(prefix, axis=0)
    return prefix
  elif mode == 'test_high_resolution':
    target = []
    files = glob.glob(split)
    print(sorted(files))
    for file in sorted(files):
      with h5py.File(file, 'r') as f:
        # print(f['target'].shape)
        total_length_64 = f[f'prefix_{standard_prefix_resolution}x{standard_prefix_resolution}'].shape[1]
        total_length_target = f['target'].shape[1]
        total_length_target = f.attrs.get("target_length", total_length_target)
        target_length = standard_prefix_length * total_length_target // total_length_64
        target.append(f['target'][:, target_length-1:target_length])
    target = np.stack(target, axis=0)
    return target
  elif mode == 'train':
    datasets = sorted(glob.glob(split))
    if world_size > 1:
      datasets = datasets[rank::world_size]
    print(datasets)
    datasets = map(lambda x: TorchDataset(x,
                                          encode_steps=encode_steps,
                                          decode_steps=decode_steps,
                                          prefix=prefix_str,
                                          ),
                   datasets)
    final_dataset = torch.utils.data.ConcatDataset(datasets)
    train_dataset = torch.utils.data.DataLoader(
        final_dataset, batch_size=int(np.prod(batch_dims)),
        shuffle=True, num_workers=workers)
    if repeat:
      train_dataset = ds_inf_repeat(train_dataset)
    train_dataset = double_buffer(train_dataset, prepare_batch=prepare_batch, use_jax=use_jax)
    return train_dataset, len(final_dataset)
  else:
    raise ValueError(f'Unknown mode: {mode}')


class GeneratorRestartHandler(object):
  def __init__(self, gen_func, argv, kwargv):
    self.gen_func = gen_func
    self.argv = copy.copy(argv)
    self.kwargv = copy.copy(kwargv)
    self.local_copy = iter(self)

  def __iter__(self):
    return self.gen_func(*self.argv, **self.kwargv)

  def __next__(self):
    return next(self.local_copy)


def restartable(g_func: callable) -> callable:
  def tmp(*argv, **kwargv):
    return GeneratorRestartHandler(g_func, argv, kwargv)

  return tmp


@restartable
def double_buffer(ds: Iterable[Batch], prepare_batch: Optional[Callable] = None,
                  use_jax: bool = True) -> Generator[Batch, None, None]:
  """Keeps at least two batches on the accelerator.
  The current GPU allocator design reuses previous allocations. For a training
  loop this means batches will (typically) occupy the same region of memory as
  the previous batch. An issue with this is that it means we cannot overlap a
  host->device copy for the next batch until the previous step has finished and
  the previous batch has been freed.
  By double buffering we ensure that there are always two batches on the device.
  This means that a given batch waits on the N-2'th step to finish and free,
  meaning that it can allocate and copy the next batch to the accelerator in
  parallel with the N-1'th step being executed.
  Args:
    ds: Iterable of batches of numpy arrays.
    prepare_batch: Optional function to prepare a batch.
    use_jax: Whether to use JAX for the double buffering.
  Yields:
    Batches of sharded device arrays.
  """
  t1, t2, t3 = 0.0, 0.0, 0.0
  batch = None
  devices = jax.local_devices()
  logging.info(devices)
  snapshot1 = time.time()
  cnt = 0
  for next_batch in ds:
    snapshot2 = time.time()
    t1 += snapshot2 - snapshot1
    assert next_batch is not None
    if batch is not None:
      cnt += 1
      yield batch
    if (cnt + 1) % 100 == 0:
      logging.info(f't1 {t1} t2 {t2} t3 {t3}')
      t1, t2, t3 = 0.0, 0.0, 0.0
    snapshot3 = time.time()
    t2 += snapshot3 - snapshot2
    if use_jax:
      try:
        batch = _device_put_sharded(next_batch, devices, prepare_batch)
      except ValueError:
        batch = None
    else:
      if prepare_batch is not None:
        batch = prepare_batch(next_batch)
      else:
        batch = next_batch
    snapshot1 = time.time()
    t3 += snapshot1 - snapshot3
  if batch is not None:
    yield batch


def ds_inf_repeat(ds: Iterable[Batch]) -> Iterable[Batch]:
  """Repeats the dataset n times."""
  while True:
    for batch in ds:
      yield batch
    print('finished one epoch')


def _device_put_sharded(sharded_tree, devices, prepare_batch):
  sharded_tree = jax.tree_map(lambda x: x.numpy(), sharded_tree)
  if prepare_batch is not None:
    sharded_tree = prepare_batch(sharded_tree)
  sharded_tree = jax.tree_map(lambda x: x.reshape((len(devices), -1) + x.shape[1:]), sharded_tree)
  leaves, treedef = jax.tree_flatten(sharded_tree)
  n = leaves[0].shape[0]
  return jax.device_put_sharded(
      [jax.tree_unflatten(treedef, [l[i] for l in leaves]) for i in range(n)],
      devices)
