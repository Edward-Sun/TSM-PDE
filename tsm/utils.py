"""Utility functions."""

import logging

import jax.numpy as jnp

def get_lr_schedule(step: jnp.ndarray,
                    steps_per_epoch: int,
                    train_lr_warmup_epochs: int,
                    train_epochs: int,
                    train_lr_init: float,
                    use_exponential_decay: bool,
                    use_rsqrt_decay: bool,
                    use_linear_decay: bool) -> jnp.ndarray:
  """Cosine learning rate schedule."""
  warmup_steps = int(train_lr_warmup_epochs * steps_per_epoch)
  training_steps = int(train_epochs * steps_per_epoch)
  logging.info(f'warmup_steps: {warmup_steps}')
  logging.info(f'training_steps: {training_steps}')

  if use_exponential_decay:
    lr = train_lr_init
    scaled_step = jnp.maximum(step - warmup_steps, 0.0)
    decay_rate = 0.1
    decay_steps = 100000.0
    lr = lr * (decay_rate ** (scaled_step / decay_steps))
    if warmup_steps:
      lr *= jnp.minimum(step / warmup_steps, 1.0)
  elif use_rsqrt_decay:
    lr = train_lr_init
    decay_steps = 50000.0
    rsqrt_warmup_steps = (decay_steps / 100.0)
    scale = jnp.sqrt(rsqrt_warmup_steps)
    lr = lr * scale / jnp.sqrt(jnp.maximum(step, rsqrt_warmup_steps))
    if warmup_steps:
      lr *= jnp.minimum(step / warmup_steps, 1.0)
  elif use_linear_decay:
    lr = train_lr_init
    scaled_step = (jnp.maximum(step - warmup_steps, 0.0) /
                   (training_steps - warmup_steps))
    lr = lr * (1.0 - scaled_step)
    if warmup_steps:
      lr *= jnp.minimum(step / warmup_steps, 1.0)
  else:
    lr = train_lr_init
    scaled_step = (jnp.maximum(step - warmup_steps, 0.0) /
                   (training_steps - warmup_steps))
    lr *= 0.5 * (1.0 + jnp.cos(jnp.pi * scaled_step))
    if warmup_steps:
      lr *= jnp.minimum(step / warmup_steps, 1.0)
  return lr
