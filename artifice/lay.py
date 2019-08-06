import numpy as np
import tensorflow as tf
from tensorflow import keras

from tensorflow.python.ops import array_ops

from artifice.log import logger  # noqa
from artifice import utils
from artifice import sparse
from artifice import conv_utils
from artifice import img, vis

NEG_INF = np.finfo(np.float32).min


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

  We consider pixels near the edge to be local maxima if they satisfy the
  above, assuming marked positions outside the image domain are at -inf.

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

    Use tf.image.image_gradients to get dx, dy for each channel then scan
    that image for low/zero spots in both directions.

    :param inputs:
    :returns:
    :rtype:

    """

    padded = tf.pad(inputs, [[0, 0], [2, 2], [2, 2], [0, 0]],
                    constant_values=NEG_INF)
    mask = np.ones_like(inputs, dtype=tf.bool)
    for di in range(5):
      start = abs(di - 2)
      stop = -abs(di - 2) + 4
      for dj in range(start, stop + 1):
        mask = tf.logical_and(
          mask,
          inputs >= padded[:, di: di + inputs.shape[1],
                           dj: dj + inputs.shape[2], :])

    if self.threshold_abs is not None:
      mask = tf.logical_and(mask, inputs > tf.constant(
        self.threshold_abs, tf.float32))

    return tf.where(mask)


class SparseConv2D(keras.layers.Layer):
  """2D convolution using the sbnet library.

  The input to this layer should therefore be a list of tensors `[inputs,
  mask]` where `mask` has shape `[N, W, H, 1]`.

  In theory, additional performance gain can be achieved by making inputs a
  tf.Variable. We have not tested this.

  :param filters:
  :param kernel_size:
  :param batch_size: if provided, allows SparseConv2D to use sparse_scatter_var
  (assuming eager execution is not enabled)
  :param strides:
  :param padding:
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
               batch_size=None,  # todo: replace with a use_var option
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
    self.batch_size = batch_size
    self.use_var = batch_size is not None and not tf.executing_eagerly()
    self.strides = utils.listify(strides, 2)
    self.padding = padding
    self.activation = keras.layers.Activation(activation)
    self.use_bias = use_bias
    self.kernel_initializer = kernel_initializer
    self.bias_initializer = bias_initializer
    self.kernel_regularizer = kernel_regularizer
    self.bias_regularizer = bias_regularizer
    self.kernel_constraint = kernel_constraint
    self.bias_constraint = bias_constraint

    self.block_size = utils.listify(block_size, 2)
    self.output_block_size = [conv_utils.conv_output_length(
      self.block_size[i],
      self.kernel_size[i],
      'valid',
      self.strides[i]) for i in [0, 1]]

    self.block_offset = [0, 0]
    self.output_block_offset = self.block_offset

    self.block_stride = self.output_block_size
    self.output_block_stride = self.output_block_size

    self.tol = tol
    self.avgpool = avgpool

    if self.padding == 'valid':
      pad_size = [0, 0]
    else:
      pad_h = self.kernel_size[0] // 2
      pad_w = (self.kernel_size[1] - 1) // 2
      pad_size = [pad_h, pad_w]
    self.pad = keras.layers.ZeroPadding2D(pad_size)

  def build(self, input_shape):
    input_shape, mask_shape = input_shape
    self.block_count = [utils.divup(input_shape[1], self.block_stride[0]),
                        utils.divup(input_shape[2], self.block_stride[1])]

    if len(input_shape) != 4:
      raise ValueError(f'Inputs should have rank 4. Received input shape: '
                       f'{input_shape}')
    if input_shape[3] is None:
      raise ValueError('The channel dimension of the inputs '
                       'should be defined. Found `None`.')
    input_dim = int(input_shape[3])
    kernel_shape = self.kernel_size + [input_dim, self.filters]

    self.kernel = self.add_weight(
      name='kernel',
      shape=kernel_shape,
      dtype=tf.float32,
      initializer=self.kernel_initializer,
      regularizer=self.kernel_regularizer,
      constraint=self.kernel_constraint,
      trainable=True)
    if self.use_bias:
      self.bias = self.add_weight(
        name='bias',
        shape=(self.filters,),
        dtype=tf.float32,
        initializer=self.bias_initializer,
        regularizer=self.bias_regularizer,
        constraint=self.bias_constraint,
        trainable=True)
    else:
      self.bias = None

    if self.use_var:
      output_shape = list(self.compute_output_shape([input_shape, mask_shape]))
      self.outputs = self.add_variable(
        name='outputs',
        shape=[self.batch_size] + output_shape[1:],
        dtype=tf.float32,
        initializer='zeros',
        trainable=False,
        use_resource=False)

  def compute_output_shape(self, input_shape):
    input_shape, mask_shape = input_shape
    shape = conv_utils.conv_output_shape(
      input_shape,
      self.filters,
      self.kernel_size,
      self.padding,
      self.strides)
    return tf.TensorShape(shape)

  def call(self, inputs):
    inputs, mask = inputs

    if self.use_var:
      self.outputs.assign(tf.zeros_like(self.outputs))
      outputs = self.outputs
    else:
      output_shape = list(
        self.compute_output_shape([inputs.shape, mask.shape]))
      batch_size = array_ops.shape(inputs)[0]
      outputs = tf.zeros([batch_size] + output_shape[1:], tf.float32)

    if self.padding == 'same':
      inputs = self.pad(inputs)
      mask = self.pad(mask)

    indices = sparse.reduce_mask(
      mask,
      block_count=self.block_count,
      bsize=self.block_size,
      boffset=self.block_offset,
      bstride=self.block_stride,
      tol=self.tol,
      avgpool=self.avgpool)

    blocks = sparse.gather(
      inputs,
      indices.bin_counts,
      indices.active_block_indices,
      bsize=self.block_size,
      boffset=self.block_offset,
      bstride=self.block_stride)

    strides = [1, self.strides[0], self.strides[1], 1]
    blocks = tf.nn.conv2d(
      blocks,
      self.kernel,
      strides=strides,
      padding='VALID')

    if self.use_bias:
      blocks = tf.nn.bias_add(blocks, self.bias, data_format='NHWC')

    if self.activation is not None:
      blocks = self.activation(blocks)

    outputs = sparse.scatter(
      blocks,
      indices.bin_counts,
      indices.active_block_indices,
      outputs,
      bsize=self.output_block_size,
      boffset=self.output_block_offset,
      bstride=self.output_block_stride,
      use_var=self.use_var)

    if self.use_var:
      outputs.set_shape([None] + outputs.shape.as_list()[1:])

    return outputs


class SparseConv2DTranspose(keras.layers.Layer):
  """2D transpose convolution using the sbnet library.

  :param filters:
  :param kernel_size:
  :param batch_size: needed to allocate space for outputs, if using a variable
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
               batch_size,
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
    self.batch_size = batch_size
    self.use_var = batch_size is not None and not tf.executing_eagerly()
    self.strides = utils.listify(strides, 2)
    self.padding = padding
    self.activation = keras.layers.Activation(activation)
    self.use_bias = use_bias
    self.kernel_initializer = kernel_initializer
    self.bias_initializer = bias_initializer
    self.kernel_regularizer = kernel_regularizer
    self.bias_regularizer = bias_regularizer
    self.kernel_constraint = kernel_constraint
    self.bias_constraint = bias_constraint

    self.block_size = utils.listify(block_size, 2)
    self.output_block_size = [conv_utils.deconv_output_length(
      self.block_size[i],
      self.kernel_size[i],
      'valid',
      stride=self.strides[i]) for i in [0, 1]]

    self.block_offset = [0, 0]
    self.output_block_offset = self.block_offset

    self.block_stride = self.block_size
    self.output_block_stride = self.output_block_size  # might not be correct

    self.tol = tol
    self.avgpool = avgpool

  def build(self, input_shape):
    input_shape, mask_shape = input_shape
    self.block_count = [utils.divup(input_shape[1], self.block_stride[0]),
                        utils.divup(input_shape[2], self.block_stride[1])]

    if len(input_shape) != 4:
      raise ValueError(f'Inputs should have rank 4. Received input shape: '
                       f'{input_shape}')
    if input_shape[3] is None:
      raise ValueError('The channel dimension of the inputs '
                       'should be defined. Found `None`.')
    input_dim = int(input_shape[3])
    kernel_shape = self.kernel_size + [self.filters, input_dim]

    self.kernel = self.add_weight(
      name='kernel',
      shape=kernel_shape,
      initializer=self.kernel_initializer,
      regularizer=self.kernel_regularizer,
      constraint=self.kernel_constraint,
      trainable=True,
      dtype=tf.float32)
    if self.use_bias:
      self.bias = self.add_weight(
        name='bias',
        shape=(self.filters,),
        initializer=self.bias_initializer,
        regularizer=self.bias_regularizer,
        constraint=self.bias_constraint,
        trainable=True,
        dtype=tf.float32)
    else:
      self.bias = None

    if self.use_var:
      output_shape = self.compute_output_shape([input_shape, mask_shape])
      self.outputs = self.add_variable(
        name='outputs',
        shape=[self.batch_size] + list(output_shape)[1:],
        dtype=tf.float32,
        initializer='zeros',
        trainable=False,
        use_resource=False)

  def compute_output_shape(self, input_shape):
    input_shape, mask_shape = input_shape
    shape = conv_utils.deconv_output_shape(
      input_shape,
      self.filters,
      self.kernel_size,
      self.padding,
      self.strides)
    return tf.TensorShape(shape)

  def call(self, inputs):
    inputs, mask = inputs

    if self.padding == 'valid':
      raise NotImplementedError('valid padding for transpose convolution')

    indices = sparse.reduce_mask(
      mask,
      block_count=self.block_count,
      bsize=self.block_size,
      boffset=self.block_offset,
      bstride=self.block_stride,
      tol=self.tol,
      avgpool=self.avgpool)

    blocks = sparse.gather(
      inputs,
      indices.bin_counts,
      indices.active_block_indices,
      bsize=self.block_size,
      boffset=self.block_offset,
      bstride=self.block_stride)

    blocks_shape = array_ops.shape(blocks)
    num_blocks = blocks_shape[0]
    height, width = blocks_shape[1], blocks_shape[2]
    kernel_h, kernel_w = self.kernel_size
    stride_h, stride_w = self.strides
    out_pad_h = out_pad_w = None  # output padding not implemented

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
    blocks_output_shape = (num_blocks, out_height, out_width, self.filters)

    strides = [1, self.strides[0], self.strides[1], 1]
    blocks = tf.nn.conv2d_transpose(
      blocks,
      self.kernel,
      blocks_output_shape,
      strides=strides,
      padding='VALID',
      data_format='NHWC')

    if not tf.executing_eagerly():
      # Infer the static output shape:
      out_shape = self.compute_output_shape([blocks.shape, mask.shape])
      blocks.set_shape(out_shape)

    if self.use_bias:
      blocks = tf.nn.bias_add(blocks, self.bias, data_format='NHWC')

    if self.activation is not None:
      blocks = self.activation(blocks)

    if self.use_var:
      self.outputs.assign(tf.zeros_like(self.outputs))
      outputs = self.outputs
    else:
      output_shape = list(
        self.compute_output_shape([inputs.shape, mask.shape]))
      batch_size = array_ops.shape(inputs)[0]
      outputs = tf.zeros([batch_size] + output_shape[1:],
                         tf.float32)  # todo: might not work

    outputs = sparse.scatter(
      blocks,
      indices.bin_counts,
      indices.active_block_indices,
      outputs,
      bsize=self.output_block_size,
      boffset=self.output_block_offset,
      bstride=self.output_block_stride,
      use_var=self.use_var)

    if self.use_var:
      outputs.set_shape([None] + outputs.shape.as_list()[1:])

    return outputs


class ReduceMask(keras.layers.Layer):
  """Perform the sparse gather operation.

  Outputs is a list containing [bin_counts, active_block_indices] rather than
  the usual namedtuple.

  :param block_size:
  :param block_offset:
  :param block_stride:
  :param tol:
  :param avgpool:
  :returns:
  :rtype:

  """

  def __init__(self,
               block_size=[16, 16],
               block_offset=[0, 0],
               block_stride=[16, 16],
               tol=0.5,
               avgpool=False,
               **kwargs):
    super().__init__(**kwargs)

    self.block_size = utils.listify(block_size, 2)
    self.block_offset = utils.listify(block_offset, 2)
    self.block_stride = utils.listify(block_stride, 2)
    self.tol = tol
    self.avgpool = avgpool

  def build(self, mask_shape):
    self.block_count = [utils.divup(mask_shape[1], self.block_stride[0]),
                        utils.divup(mask_shape[2], self.block_stride[1])]

  def compute_output_shape(self, _):
    return [tf.TensorShape([]), tf.TensorShape([None, 3])]

  def call(self, mask_):
    indices = sparse.reduce_mask(
      mask_,
      block_count=self.block_count,
      bsize=self.block_size,
      boffset=self.block_offset,
      bstride=self.block_stride,
      tol=self.tol,
      avgpool=self.avgpool)

    return [indices.bin_counts, indices.active_block_indices]


class SparseGather(keras.layers.Layer):
  """Perform the sparse gather operation.

  :param block_size:
  :param block_offset:
  :param block_stride:
  :returns:
  :rtype:

  """

  def __init__(self,
               block_size=[16, 16],
               block_offset=[0, 0],
               block_stride=[16, 16],
               **kwargs):
    super().__init__(**kwargs)

    self.block_size = utils.listify(block_size, 2)
    self.block_offset = utils.listify(block_offset, 2)
    self.block_stride = utils.listify(block_stride, 2)

  def compute_output_shape(self, input_shape):
    input_shape, _, _ = input_shape
    return tf.TensorShape(
      [None, self.block_size[0], self.block_size[1], input_shape[3]])

  def call(self, inputs):
    inputs, bin_counts, active_block_indices = inputs
    return sparse.gather(
      inputs,
      bin_counts,
      active_block_indices,
      bsize=self.block_size,
      boffset=self.block_offset,
      bstride=self.block_stride)


class SparseScatter(keras.layers.Layer):
  """Perform the sparse scatter operation.

  :param block_size:
  :param block_offset:
  :param block_stride:
  :returns:
  :rtype:

  """

  def __init__(self,
               output_shape,
               block_size=[16, 16],
               block_offset=[0, 0],
               block_stride=[16, 16],
               use_var=False,
               **kwargs):
    super().__init__(**kwargs)

    self.output_shape_ = list(output_shape)
    self.block_size = utils.listify(block_size, 2)
    self.block_offset = utils.listify(block_offset, 2)
    self.block_stride = utils.listify(block_stride, 2)
    self.use_var = use_var

  def compute_output_shape(self, _):
    return tf.TensorShape(self.output_shape_)

  def call(self, inputs):
    inputs, bin_counts, active_block_indices = inputs
    outputs = tf.zeros(self.output_shape, tf.float32)
    return sparse.scatter(
      inputs,
      bin_counts,
      active_block_indices,
      outputs,
      bsize=self.block_size,
      boffset=self.block_offset,
      bstride=self.block_stride,
      use_var=self.use_var)


def main():
  # tf.enable_eager_execution()
  inputs = keras.layers.Input(shape=(100, 100, 1))
  x = SparseConv2D(1, [3, 3], 4, padding='same')([inputs, inputs])
  x = SparseConv2D(1, [1, 1], 4, padding='same')([x, x])
  # x = SparseConv2DTranspose(1, [2, 2], strides=[2, 2], padding='same')([x, x]) # noqa
  # x = keras.layers.MaxPool2D()(x)
  model = keras.Model(inputs, x)
  model.compile(optimizer=tf.train.AdadeltaOptimizer(0.1), loss='mse',
                metrics=['mae'])

  images = np.array([
    img.open_as_float('../data/disks_100x100/images/1001.png'),
    img.open_as_float('../data/disks_100x100/images/1002.png'),
    img.open_as_float('../data/disks_100x100/images/1003.png'),
    img.open_as_float('../data/disks_100x100/images/1004.png')])
  images = images[:, :, :, np.newaxis]

  dataset = tf.data.Dataset.from_tensor_slices((images, images))
  dataset = dataset.batch(4).repeat(-1)
  model.fit(dataset, epochs=5, steps_per_epoch=1000)

  x = images
  y = model.predict(images)
  vis.plot_image(*x, *y, columns=4, vmin=0., vmax=1.)
  vis.show('../figs/sparse_conv2d_example.pdf')


if __name__ == '__main__':
  main()
