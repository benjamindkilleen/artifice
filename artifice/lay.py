import numpy as np
import tensorflow as tf
from tensorflow import keras

from artifice.log import logger
from artifice import utils
from artifice import sparse
from artifice import conv_utils
from artifice import img, vis

NEG_INF = np.finfo(np.float32).min

def _apply_activation(activation, inputs):
  if activation is None:
    return inputs
  if callable(activation):
    return activation(inputs)
  if activation == 'relu':
    return tf.nn.relu(inputs)
  if activation == 'softmax':
    return tf.nn.softmax(inputs)
  raise NotImplementedError(f"'{activation}' activation")
    

class PeakDetection(keras.layers.Layer):
  """Finds local maxima in each channel of the image.

  Does this pretty crudely by comparing each pixel with all of those <= 2 units
  away. That is, (i,j) is a local max if inputs[i,j] is greater than the pixels
  marked x shown below:

  |---|---|---|---|---|---|---|
  |   |   |   |   |   |   |   |
  |---|---|---|---|---|---|---|
  |   |   |   | x |   |   |   | 0
  |---|---|---|---|---|---|---|
  |   |   | x | x | x |   |   | 1
  |---|---|---|---|---|---|---|
  |   | x | x |i,j| x | x |   | 2
  |---|---|---|---|---|---|---|
  |   |   | x | x | x |   |   | 3
  |---|---|---|---|---|---|---|
  |   |   |   | x |   |   |   | 4
  |---|---|---|---|---|---|---|
  |   |   |   |   |   |   |   | 5
  |---|---|---|---|---|---|---|
        0   1   2   3   4   5

  We consider pixels near the edge to be local maxima if they satisfy the above,
  assuming marked positions outside the image domain are at -inf.

  In cases of ties, both points are returned.

  Notes:
  * Loop indices work out like so:
    0 => 2, 2+1
    1 => 1, 3+1
    2 => 0, 4+1
    3 => 1, 3+1
    4 => 2, 2+1


  """
  def __init__(self, threshold_abs=None, **kwargs):
    self.threshold_abs = threshold_abs
    super().__init__(**kwargs)

  def compute_output_shape(self, input_shape):
    """(batch_size, num_channels, num_peaks, 2)"""
    return (None, len(input_shape))

  def call(self, inputs):
    """FIXME! briefly describe function

    * Use tf.image.image_gradients to get dx, dy for each channel then scan that
      image for low/zero spots in both directions.
    *

    :param inputs:
    :returns:
    :rtype:

    """

    padded = tf.pad(inputs, [[0,0], [2,2], [2,2], [0,0]],
                    constant_values=NEG_INF)
    mask = np.ones_like(inputs, dtype=tf.bool)
    for di in range(5):
      start = abs(di - 2)
      stop = -abs(di - 2) + 4
      for dj in range(start, stop + 1):
        mask = tf.logical_and(mask, inputs >= padded[:, di : di+inputs.shape[1],
                                                     dj : dj+inputs.shape[2], :])

    if self.threshold_abs is not None:
      mask = tf.logical_and(mask, inputs > tf.constant(self.threshold_abs, tf.float32))

    return tf.where(mask)


class SparseConv2D(keras.layers.Layer):
  """2D convolution using the sbnet library.

  The input to this layer should therefore be a list of tensors `[inputs,
  mask]` where `mask` has shape `[N, W, H, 1]`.

  In theory, additional performance gain can be achieved by making inputs a
  tf.Variable. We have not tested this.

  :param filters:
  :param kernel_size:
  :param strides:
  :param padding:
  :param data_format:
  :param dilation_rate:
  :param activation:
  :param use_bias:
  :param kernel_initializer:
  :param bias_initializer:
  :param kernel_regularizer:
  :param bias_regularizer:
  :param activity_regularizer:
  :param kernel_constraint:
  :param bias_constraint:
  :param block_size:
  :param tol:
  :param avgpool:
  :returns:
  :rtype:

  """

  def __init__(self,
               filters,
               kernel_size,
               strides=[1, 1],
               padding='valid',
               activation=None,
               use_bias=True,
               kernel_initializer='glorot_uniform',
               bias_initializer='zeros',
               kernel_regularizer=None,
               bias_regularizer=None,
               kernel_constraint=None,
               bias_constraint=None,
               block_size=[16, 16],
               tol=0.5,
               avgpool=False,
               **kwargs):
    super().__init__(**kwargs)
    
    self.filters = filters
    self.kernel_size = utils.listify(kernel_size, 2)
    self.strides = utils.listify(strides, 2)
    self.padding = padding
    self.activation = activation
    self.use_bias = use_bias
    self.kernel_initializer = kernel_initializer
    self.bias_initializer = bias_initializer
    self.kernel_regularizer = kernel_regularizer
    self.bias_regularizer = bias_regularizer
    self.kernel_constraint = kernel_constraint
    self.bias_constraint = bias_constraint

    self.block_count = None
    self.block_size = utils.listify(block_size, 2)
    self.output_block_size = list(self.compute_output_shape(
      [[1, self.block_size[0], self.block_size[1], 1], None]))[1:3]

    self.block_offset = [0, 0]
    self.output_block_offset = self.block_offset

    self.block_stride = self.output_block_size
    self.output_block_stride = self.output_block_size

    self.tol = tol
    self.avgpool = avgpool

    pad_h = self.kernel_size[0] // 2
    pad_w = (self.kernel_size[1] - 1) // 2
    if self.padding == 'valid':
      self.pad_size = [0,0]
    else:
      self.pad_size = [pad_h, pad_w]
    self.block_pad_size = [pad_h, pad_w]

    # make the nested layers
    self.pad = keras.layers.ZeroPadding2D(self.pad_size)
      
    self.reduce_mask = keras.layers.Lambda(
      lambda mask: sparse.reduce_mask(
        mask,
        block_count=self.block_count,
        bsize=self.block_size,
        boffset=self.block_offset,
        bstride=self.block_stride,
        tol=self.tol,
        avgpool=self.avgpool))
    
    self.sparse_gather = keras.layers.Lambda(
      lambda ip: sparse.sparse_gather(
        ip[0],                  # inputs
        ip[1],                  # bin_counts
        ip[2],                  # active_block_indices
        transpose=False,
        bsize=self.block_size,
        boffset=self.block_offset,
        bstride=self.block_stride))
    
    self.conv = keras.layers.Conv2D(
      filters,
      kernel_size,
      strides=strides,
      padding='same',
      activation=activation,
      use_bias=use_bias,
      kernel_initializer=kernel_initializer,
      bias_initializer=bias_initializer,
      kernel_regularizer=kernel_regularizer,
      bias_regularizer=bias_regularizer,
      kernel_constraint=kernel_constraint,
      bias_constraint=bias_constraint)

    self.sparse_scatter = keras.layers.Lambda(
      lambda ip: sparse.sparse_scatter(
        ip[0],                  # blocks
        ip[1],                  # bin_counts
        ip[2],                  # active_block_indices
        ip[3],                  # outputs
        bsize=self.output_block_size,
        boffset=self.output_block_offset,
        bstride=self.output_block_stride,
        transpose=False))
      
  
  def compute_output_shape(self, input_shape):
    input_shape, mask_shape = input_shape
    space = input_shape[1:-1]
    new_space = []
    for i in range(len(space)):
      new_dim = conv_utils.conv_output_length(
        space[i],
        self.kernel_size[i],
        padding=self.padding,
        stride=self.strides[i])
      new_space.append(new_dim)
    output_shape = tf.TensorShape([input_shape[0]] + new_space + [self.filters])
    return output_shape

  def call(self, inputs):
    inputs, mask = inputs
    logger.debug(f"inputs keras_history: {inputs._keras_history}")
    if self.padding != 'valid':
      inputs = self.pad(inputs)

    logger.debug(f"padded keras_history: {inputs._keras_history}")
    indices = self.reduce_mask(mask)

    logger.debug(f"indices keras_history: {indices._keras_history}")
    blocks = self.sparse_gather([inputs,
                                 indices.bin_counts,
                                 indices.active_block_indices])
    logger.debug(f"blocks keras_history: {blocks._keras_history}")
    
    blocks = self.conv(blocks)
    logger.debug(f"conv keras_history: {blocks._keras_history}")
    
    output_shape = self.compute_output_shape([inputs.shape, mask.shape])
    outputs = tf.zeros(output_shape, tf.float32) # todo: make a variable?
    # inputs[:, :output_shape[1], :output_shape[2], :]

    outputs = self.sparse_gather([blocks,
                                  indices.bin_counts,
                                  indices.active_block_indices,
                                  outputs])

    logger.debug(f"outputs keras_history: {outputs._keras_history}")
    return outputs

class SparseConv2DTranspose(keras.layers.Conv2DTranspose):
  """2D transpose convolution using the sbnet library.

  :param filters:
  :param kernel_size:
  :param strides:
  :param padding:
  :param data_format:
  :param dilation_rate:
  :param activation:
  :param use_bias:
  :param kernel_initializer:
  :param bias_initializer:
  :param kernel_regularizer:
  :param bias_regularizer:
  :param activity_regularizer:
  :param kernel_constraint:
  :param bias_constraint:
  :param block_size:
  :param tol:
  :param avgpool:
  :returns:
  :rtype:

  """
  def __init__(self,
               filters,
               kernel_size,
               strides=[1, 1],
               padding='valid',
               output_padding=None,
               data_format=None,
               activation=None,
               use_bias=True,
               kernel_initializer='glorot_uniform',
               bias_initializer='zeros',
               kernel_regularizer=None,
               bias_regularizer=None,
               activity_regularizer=None,
               kernel_constraint=None,
               bias_constraint=None,
               block_size=[16, 16],
               tol=0.5,
               avgpool=False,
               **kwargs):
    super().__init__(
      filters,
      kernel_size,
      strides=strides,
      padding=padding,
      data_format=data_format,
      dilation_rate=[1, 1],     # not supported
      activation=activation,
      use_bias=use_bias,
      kernel_initializer=kernel_initializer,
      bias_initializer=bias_initializer,
      kernel_regularizer=kernel_regularizer,
      bias_regularizer=bias_regularizer,
      activity_regularizer=activity_regularizer,
      kernel_constraint=kernel_constraint,
      bias_constraint=bias_constraint,
      **kwargs)

    self.input_spec = None      # needed since Conv2D directly subclassed
    self.block_count = None

    self.block_size = utils.listify(block_size, 2)
    self.output_block_size = [
      (self.strides[0] * self.block_size[0] +
       max(self.kernel_size[0] - self.strides[0], 0)),
      (self.strides[1] * self.block_size[1] +
       max(self.kernel_size[1] - self.strides[1], 0))]
    
    self.block_offset = [0, 0]
    self.output_block_offset = self.block_offset
    
    self.block_stride = self.block_size
    self.output_block_stride = [self.output_block_size[0] - self.kernel_size[0] + 1,
                                self.output_block_size[1] - self.kernel_size[1] + 1]

    self.tol = tol
    self.avgpool = avgpool

  def build(self, input_shape):
    input_shape, mask_shape = input_shape
    if len(input_shape) != 4:
      raise ValueError('Inputs should have rank 4. Received input shape: ' +
                       str(input_shape))
    if self.data_format == 'channels_first':
      h, w, c = 2, 3, 1
    else:
      h, w, c = 1, 2, 3
    if input_shape[c] is None:
      raise ValueError('The channel dimension of the inputs '
                       'should be defined. Found `None`.')
    input_dim = int(input_shape[c])
    kernel_shape = self.kernel_size + (self.filters, input_dim)

    self.kernel = self.add_weight(
      name='kernel',
      shape=kernel_shape,
      initializer=self.kernel_initializer,
      regularizer=self.kernel_regularizer,
      constraint=self.kernel_constraint,
      trainable=True,
      dtype=self.dtype)
    if self.use_bias:
      self.bias = self.add_weight(
        name='bias',
        shape=(self.filters,),
        initializer=self.bias_initializer,
        regularizer=self.bias_regularizer,
        constraint=self.bias_constraint,
        trainable=True,
        dtype=self.dtype)
    else:
      self.bias = None
    self.block_count = [utils.divup(input_shape[h], self.block_stride[0]),
                        utils.divup(input_shape[w], self.block_stride[1])]
    self.built = True

  def compute_output_shape(self, input_shape):
    input_shape, mask_shape = input_shape
    return super().compute_output_shape(input_shape)

  def call(self, inputs):
    inputs, mask = inputs

    if self.data_format == 'channels_first':
      raise NotImplementedError('channels_first data format for SparseConv2DTranspose')

    indices = sparse.reduce_mask(
      mask,
      block_count=self.block_count,
      bsize=self.block_size,
      boffset=self.block_offset,
      bstride=self.block_stride,
      tol=self.tol,
      avgpool=self.avgpool)

    blocks = sparse.sparse_gather(
      inputs,
      indices.bin_counts,
      indices.active_block_indices,
      transpose=False,
      bsize=self.block_size,
      boffset=self.block_offset,
      bstride=self.block_stride)

    blocks_shape = tf.shape(blocks)
    batch_size = blocks_shape[0]
    if self.data_format == 'channels_first':
      h_axis, w_axis = 2, 3
    else:
      h_axis, w_axis = 1, 2

    height, width = blocks_shape[h_axis], blocks_shape[w_axis]
    kernel_h, kernel_w = self.kernel_size
    stride_h, stride_w = self.strides

    if self.output_padding is None:
      out_pad_h = out_pad_w = None
    else:
      raise NotImplementedError("output_padding should be None")
      # out_pad_h, out_pad_w = self.output_padding

    # Infer the dynamic output shape:
    out_height = conv_utils.deconv_output_length(
      height,
      kernel_h,
      padding='valid',
      output_padding=out_pad_h,
      stride=stride_h)
    out_width = conv_utils.deconv_output_length(
      width,
      kernel_w,
      padding='valid',
      output_padding=out_pad_w,
      stride=stride_w)
    if self.data_format == 'channels_first':
      raise NotImplementedError
      # output_shape = (batch_size, self.filters, out_height, out_width)
    else:
      output_shape = (batch_size, out_height, out_width, self.filters)

    strides = [1, self.strides[0], self.strides[1], 1]
    blocks = tf.nn.conv2d_transpose(
      blocks,
      self.kernel,
      output_shape,
      strides=strides,
      padding='VALID',
      data_format='NHWC')

    if not tf.executing_eagerly():
      # Infer the static output shape:
      out_shape = self.compute_output_shape([blocks.shape, mask.shape])
      blocks.set_shape(out_shape)

    if self.use_bias:
      if self.data_format == 'channels_first':
        raise NotImplementedError
        # blocks = tf.nn.bias_add(blocks, self.bias, data_format='NCHW')
      else:
        blocks = tf.nn.bias_add(blocks, self.bias, data_format='NHWC')

    if self.activation is not None:
      blocks = self.activation(blocks)

    output_shape = self.compute_output_shape([inputs.shape, mask.shape])
    outputs = tf.zeros(output_shape, tf.float32)

    outputs = sparse.sparse_scatter(
      blocks,
      indices.bin_counts,
      indices.active_block_indices,
      outputs,
      bsize=self.output_block_size,
      boffset=self.output_block_offset,
      bstride=self.output_block_stride,
      transpose=False)

    logger.debug(f"outputs: {outputs}")
    return outputs

def main():
  tf.enable_eager_execution()
  inputs = tf.constant(
    np.array([img.open_as_float('../data/disks_100x100/images/1001.png'),
              img.open_as_float('../data/disks_100x100/images/1002.png'),
              img.open_as_float('../data/disks_100x100/images/1003.png'),
              img.open_as_float('../data/disks_100x100/images/1004.png')]))
  inputs = tf.expand_dims(inputs, -1)
  mask = inputs
  # outputs = SparseConv2D(1, [3,3], kernel_initializer='ones', padding='valid',
  #                        use_bias=False)([inputs, mask])
  # outputs = SparseConv2DTranspose(1, [2,2], strides=[2,2], kernel_initializer='ones',
  #                                 padding='valid', use_bias=False)([inputs, mask])
  if tf.executing_eagerly():
    inputs = inputs.numpy()
    outputs = outputs.numpy()
  else:
    with tf.Session() as sess:
      sess.run(tf.global_variables_initializer())
      inputs, mask, outputs = sess.run([inputs, mask, outputs])
  vis.plot_image(*inputs, *outputs, columns=4)
  vis.show('../figs/sparse_conv2d_example.pdf')

if __name__ == '__main__':
  main()
