"""Definitions of towers (neural networks based on multioke CNN layers)."""

import functools
from typing import Any, Callable, List, Optional, Tuple, Union

import gin
import haiku as hk

import logging
from jax_cfd.ml import layers
from jax_cfd.ml import nonlinearities
from jax_cfd.ml import towers


Array = layers.Array
ConvModule = Callable[..., Any]
ScaleFn = Callable[[Array, List[int]], Array]
TowerFactory = Callable[..., Any]


@gin.configurable
def tsm_forward_tower_factory(
    num_output_channels: int,
    ndim: int,
    num_hidden_channels: int = 16,
    kernel_size: int = 3,
    num_hidden_layers: int = 2,
    rates: Union[int, Tuple[int, ...]] = 1,
    strides: Union[int, Tuple[int, ...]] = 1,
    output_kernel_size: int = 3,
    output_dilation_rate: int = 1,
    output_stride: int = 1,
    conv_module: ConvModule = towers.periodic_convolution,
    nonlinearity: Callable[[Array], Array] = nonlinearities.relu,
    inputs_scale_fn: ScaleFn = lambda x, axes: x,
    output_scale_fn: ScaleFn = lambda x, axes: x,
    dropout_rate: float = 0.0,
    name: Optional[str] = 'forward_cnn_tower',
):
  """Constructs parametrized feed-forward CNN tower.
  Constructs CNN tower parametrized by fixed number of channels in hidden layers
  and fixed square kernels.
  Args:
    num_output_channels: number of channels in the output layer.
    ndim: number of spatial dimensions to expect in inputs to the network.
    num_hidden_channels: number of channels to use in hidden layers.
    kernel_size: size of the kernel to use along every dimension.
    num_hidden_layers: number of hidden layers to construct in the tower.
    rates: dilation rate(s) of the hidden layers.
    strides: strides to use. Must be `int` or same a `num_hidden_layers`.
    output_kernel_size: size of the output kernel to use along every dimension.
    output_dilation_rate: dilation_rate of the output layer.
    output_stride: stride of the final convolution.
    conv_module: convolution module to use. Must accept
      (output channels, kernel shape and ndim).
    nonlinearity: nonlinearity function to apply between hidden layers.
    inputs_scale_fn: scaling function to be applied to the inputs of the tower.
      Must take inputs as argument and return an `Array` of the same shape.
      Can expect an `axes` arguments specifying spatial axes in inputs.
    output_scale_fn: similar to `inputs_scale_fn` but applied to outputs.
    dropout_rate: dropout rate to apply between hidden layers.
    name: a name for this CNN tower. This name will appear in Xprof traces.
  Returns:
    CNN tower with specified configuration.
  """
  channels = (num_hidden_channels,) * num_hidden_layers
  kernel_shapes = ((kernel_size,) * ndim,) * num_hidden_layers
  output_kernel_shape = (output_kernel_size,) * ndim
  return tsm_forward_flex_tower_factory(
      num_output_channels=num_output_channels, ndim=ndim, channels=channels,
      kernel_shapes=kernel_shapes, rates=rates, strides=strides,
      output_kernel_shape=output_kernel_shape, output_rate=output_dilation_rate,
      output_stride=output_stride, conv_module=conv_module,
      nonlinearity=nonlinearity, inputs_scale_fn=inputs_scale_fn,
      output_scale_fn=output_scale_fn, dropout_rate=dropout_rate, name=name)


@gin.configurable
def tsm_forward_flex_tower_factory(
    num_output_channels: int,
    ndim: int,
    channels: Tuple[int, ...] = (16, 16),
    kernel_shapes: Tuple[Tuple[int, ...], ...] = ((3, 3), (3, 3)),
    rates: Tuple[int, ...] = (1, 1),
    strides: Tuple[int, ...] = (1, 1),
    output_kernel_shape: Tuple[int, ...] = (3, 3),
    output_rate: int = 1,
    output_stride: int = 1,
    conv_module: ConvModule = towers.periodic_convolution,
    nonlinearity: Callable[[Array], Array] = nonlinearities.relu,
    inputs_scale_fn: ScaleFn = lambda x, axes: x,
    output_scale_fn: ScaleFn = lambda x, axes: x,
    dropout_rate: float = 0.0,
    name: Optional[str] = 'forward_flex_cnn_tower',
):
  """Constructs CNN tower with specified architecture.
  Args:
    num_output_channels: number of channels in the output layer.
    ndim: number of spatial dimensions to expect in inputs to the network.
    channels: tuple specifying number of channels in hidden layers.
    kernel_shapes: tuple specifying shapes of kernels in hidden layers.
      Each entry must be a tuple that specifies a valid kernel_shape for the
      provided `conv_module`. Must have the same length as `channels`.
    rates: dilation rates of the convolutions.
    strides: strides to use in convolutions.
    output_kernel_shape: shape of the output kernel.
    output_rate: dilation rate of the final convolution.
    output_stride: stride of the final convolution.
    conv_module: convolution module to use. Must accept
      (output channels, kernel shape and ndim).
    nonlinearity: nonlinearity function to apply between hidden layers.
    inputs_scale_fn: scaling function to be applied to the inputs of the tower.
      Must take `inputs`, `axes` arguments specifying input `Array` and
      spatial dimensions and return an `Array` of the same shape as `inputs`.
    output_scale_fn: similar to `inputs_scale_fn` but applied to outputs.
    dropout_rate: dropout rate to use in the network.
    name: a name for this CNN tower. This name will appear in Xprof traces.
  Returns:
    CNN tower with specified architecture.
  """
  if isinstance(strides, int):
    strides = (strides,) * len(channels)
  if isinstance(rates, int):
    rates = (rates,) * len(channels)

  ndim_axes = list(range(ndim))
  n_convs = len(channels)
  if not all(len(arg) == n_convs for arg in [kernel_shapes, rates, strides]):
    raise ValueError('conflicting lengths for channels/kernels/rates/strides: '
                     f'{channels} / {kernel_shapes} / {rates} / {strides}')

  def forward_pass(inputs, is_training=False):
    components = [functools.partial(inputs_scale_fn, axes=ndim_axes)]
    num_cnn_layers = len(channels)
    conv_args = zip(channels[:num_cnn_layers], kernel_shapes[:num_cnn_layers],
                    rates[:num_cnn_layers], strides[:num_cnn_layers])

    layer = 0
    for num_channels, kernel_shape, rate, stride in conv_args:
      components.append(conv_module(num_channels, kernel_shape, ndim, rate=rate,
                                    stride=stride))
      components.append(nonlinearity)
      components.append(functools.partial(
          dropout_layer, dropout_rate=dropout_rate, is_training=is_training))
      layer += 1
      inputs = hk.Sequential(components)(inputs)
      components = []

    last_conv_module = towers.periodic_convolution
    components.append(last_conv_module(num_output_channels, output_kernel_shape,
                                       ndim, rate=output_rate, stride=output_stride))
    components.append(functools.partial(output_scale_fn, axes=ndim_axes))

    return hk.Sequential(components)(inputs)

  module = hk.to_module(forward_pass)(name=name)
  return hk.experimental.named_call(module, name=name)


def dropout_layer(inputs, dropout_rate, is_training):
  if is_training:
    return hk.dropout(hk.next_rng_key(), dropout_rate, inputs)
  else:
    return inputs
