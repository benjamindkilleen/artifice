import numpy as np
import tensorflow as tf
from tensorflow import keras

from artifice.log import logger
from artifice import utils
from artifice import sparse
from artifice import img, vis

NEG_INF = np.finfo(np.float32).min

def _apply_activation(inputs, activation):
  if activation is None:
    outputs = inputs
  elif callable(activation):
    outputs = activation(inputs)
  elif activation == 'relu':
    outputs = tf.nn.relu(inputs)
  elif activation == 'crelu':
    outputs = tf.nn.crelu(inputs)
  elif activation == 'elu':
    outputs = tf.nn.elu(inputs)
  elif activation == 'leaky_relu':
    outputs = tf.nn.leaky_relu(inputs)
  elif activation == 'softmax':
    outputs = tf.nn.softmax(inputs)
  elif activation == 'tanh':
    outputs = tf.math.tanh(inputs)
  else:
    raise ValueError(f"unknown activation {activation}")
  return outputs

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


class SparseConv2D(keras.layers.Conv2D):
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
  :returns: 
  :rtype:

  """

  def __init__(self,
               filters,
               kernel_size,
               strides=[1, 1],
               padding='valid',
               data_format=None,
               dilation_rate=[1, 1],
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
               threshold=0.5,
               avgpool=False,
               **kwargs):
    super().__init__(
      filters,
      kernel_size,
      strides=strides,
      padding=padding,
      data_format=data_format,
      dilation_rate=dilation_rate,
      activation=activation,
      use_bias=use_bias,
      kernel_initializer='glorot_uniform',
      bias_initializer='zeros',
      kernel_regularizer=None,
      bias_regularizer=None,
      activity_regularizer=None,
      kernel_constraint=None,
      bias_constraint=None,
      **kwargs)

    self.block_count = None

    self.block_size = utils.listify(block_size, 2)
    self.block_offset = [0, 0]
    self.block_stride = [self.block_size[0] - self.kernel_size[0] + 1,
                         self.block_size[1] - self.kernel_size[1] + 1]
    
    self.output_block_size = self.block_stride
    self.output_block_offset = self.block_offset
    self.output_block_stride = self.input_block_stride

    self.threshold = threshold
    self.avgpool = avgpool
    
    pad_h = self.kernel_size[0] // 2
    pad_w = self.kernel_size[1] // 2
    if self.padding == 'valid':
      self.pad_size = [0,0]
    else:
      self.pad_size = [pad_h, pad_w]
    self.block_pad_size = [pad_h, pad_w]
    
  def build(self, input_shape):
    input_shape, mask_shape = input_shape
    input_shape = tensor_shape.TensorShape(input_shape)
    if self.data_format == 'channels_first':
      h, w, c = 2, 3, 1
    else:
      h, w, c = 1, 2, 3
    if input_shape.dims[c].value is None:
      raise ValueError('The channel dimension of the inputs '
                       'should be defined. Found `None`.')
    input_dim = int(input_shape[c])
    kernel_shape = self.kernel_size + (input_dim, self.filters)

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

    if self.padding != 'valid':
      paddings = [
        [0, 0],
        [self.pad_size[0], self.pad_size[0]],
        [self.pad_size[1], self.pad_size[1]],
        [0, 0]]
      inputs = tf.pad(inputs, paddings)

    indices = sparse.reduce_mask(
      mask,
      block_count=self.block_count,
      bsize=self.bsize,
      boffset=self.boffset,
      bstride=self.bstride,
      tol=self.tol,
      avgpool=self.avgpool)

    blocks = sparse.sparse_gather(
      inputs,
      indices.bin_counts,
      indices.active_block_indices,
      transpose=False,
      bsize=self.bsize,
      boffset=self.boffset,
      bstride=self.bstride)

    conv_strides = [1, self.strides[0], self.strides[1], 1]
    blocks = tf.nn.conv2d(
      blocks,
      self.kernel,
      strides=conv_strides,
      padding='VALID')

    if self.use_bias:
      if self.data_format == 'channels_first':
        if self.rank == 1:
          # nn.bias_add does not accept a 1D input tensor.
          bias = array_ops.reshape(self.bias, (1, self.filters, 1))
          blocks += bias
        else:
          blocks = nn.bias_add(blocks, self.bias, data_format='NCHW')
      else:
        blocks = nn.bias_add(blocks, self.bias, data_format='NHWC')

    if self.activation is not None:
      blocks = self.activation(blocks)

    inputs = inputs[:,
                    self.pad_size[0] : inputs.shape[1] - self.pad_size[0],
                    self.pad_size[1] : inputs.shape[2] - self.pad_size[1],
                    :]

    return sparse.sparse_scatter(
      blocks,
      indices.bin_counts,
      indices.active_block_indices,
      inputs,
      bsize=self.bsize,
      boffset=self.boffset,
      bstride=self.bstride)

class SparseConv2DTranspose(SparseConv2D):
  """2D transpose convolution using the sbnet library.

  :param filters: 
  :param kernel_size: 
  :param strides: 
  :param padding: 
  :param activation: 
  :param use_bias: 
  :param kernel_initializer: 
  :param bias_initializer: 
  :param norm: apply batch normalization
  :param bsize: 
  :param boffset: 
  :param tol: 
  :param avgpool: 
  :returns: 
  :rtype: 

  """

  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)
    
  # todo: implement compute_output_shape
  def compute_output_shape(self, input_shape):
    input_shape, _ = input_shape
    

  def call(self, inputs):
    inputs, mask = inputs

    indices = sparse.reduce_mask(mask, block_count=self.block_count,
                                 bsize=self.bsize, boffset=self.boffset,
                                 bstride=self.bstride, tol=self.tol,
                                 avgpool=self.avgpool)

    block_stack = sparse.sparse_gather(inputs, indices.bin_counts,
                                       indices.active_block_indices,
                                       transpose=False, bsize=self.bsize,
                                       boffset=self.boffset,
                                       bstride=self.bstride)
    
    block_stack = tf.nn.conv2d_transpose(
      value=block_stack, filter=self.w,
      output_shape=[None, ],
      strides=[1, self.strides[0], self.strides[1], 1])
  
    
def main():
  tf.enable_eager_execution()
  inputs = tf.constant(
    np.array([img.open_as_float('../data/disks_100x100/images/1001.png'),
              img.open_as_float('../data/disks_100x100/images/1002.png'),
              img.open_as_float('../data/disks_100x100/images/1003.png'),
              img.open_as_float('../data/disks_100x100/images/1004.png')]))
  inputs = tf.expand_dims(inputs, -1)
  mask = inputs
  outputs = SparseConv2D(64, [3,3], kernel_initializer='ones')([inputs, mask])
  if tf.executing_eagerly():
    inputs = inputs.numpy()
    outputs = outputs.numpy()
  else:
    with tf.Session() as sess:
      sess.run(tf.global_variables_initializer())
      inputs, mask, outputs = sess.run([inputs, mask, outputs])
  vis.plot_image(*inputs, *outputs, columns=4)
  vis.show('../figs/sparse_conv2d_example.pdf', save=True)

if __name__ == '__main__':
  main()
