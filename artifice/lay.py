import numpy as np
import tensorflow as tf
from tensorflow import keras

from artifice import utils
from artifice import sparse
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

  Note that this does not implement convolution over SparseTensors, as the class
  name might suggest. Instead, it uses the SBNet API to mask the areas where
  convolution is performed. The inputs to this layer should therefore be a list
  of tensors `[inputs, mask]` where `mask` is broadcastable to
  `inputs`.

  In theory, additional performance gain can be achieved by making inputs a
  tf.Variable. We have not tested this.

  """
  def __init__(self, filters, kernel_size, strides=(1, 1), padding='valid',
               activation=None, use_bias=True,
               kernel_initializer='glorot_uniform', bias_initializer='zeros',
               kernel_regularizer=None, bias_regularizer=None,
               activity_regularizer=None, kernel_constraint=None,
               bias_constraint=None, bsize=[16,16], boffset=[0,0], tol=0.5,
               avgpool=False, **kwargs):

    self._conv_layer = keras.layers.Conv2D(filters, kernel_size,
                                            strides=strides, padding='valid',
                                            activation=activation,
                                            use_bias=use_bias,
                                            kernel_initializer=kernel_initializer,
                                            bias_initializer=bias_initializer,
                                            kernel_regularizer=kernel_regularizer,
                                            bias_regularizer=bias_regularizer,
                                            activity_regularizer=activity_regularizer,
                                            kernel_constraint=kernel_constraint,
                                            bias_constraint=bias_constraint)
    self.filters = self._conv_layer.filters
    self.kernel_size = self._conv_layer.kernel_size
    self.strides = self._conv_layer.strides
    self.padding = padding      # block-layer must have valid padding
    self.activation = self._conv_layer.activation
    self.use_bias = self._conv_layer.use_bias
    self.kernel_initializer = self._conv_layer.kernel_initializer
    self.bias_initializer = self._conv_layer.bias_initializer
    self.kernel_regularizer = self._conv_layer.kernel_regularizer
    self.bias_regularizer = self._conv_layer.bias_regularizer
    self.activity_regularizer = self._conv_layer.activity_regularizer
    self.kernel_constraint = self._conv_layer.kernel_constraint
    self.bias_constraint = self._conv_layer.bias_constraint

    self.block_count = None
    self.bsize = bsize
    self.boffset = boffset
    self.output_bsize = self.bstride = [bsize[0] - self.kernel_size[0] + 1,
                                        bsize[1] - self.kernel_size[1] + 1]
    self.tol = tol
    self.avgpool = avgpool
    self.pad_size = [self.kernel_size[0] // 2, self.kernel_size[1] // 2]
    super().__init__(**kwargs)

  def build(self, input_shape):
    input_shape, _ = input_shape
    self.block_count = [utils.divup(input_shape[1], self.bstride[0]),
                        utils.divup(input_shape[2], self.bstride[1])]

  def compute_output_shape(self, input_shape):
    try:
      input_shape, _ = input_shape
    except ValueError:
      raise ValueError("SparseConv2D takes [inputs, mask] as a list")
    return self._conv_layer.compute_output_shape(input_shape)
    
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
    block_stack = self._conv_layer(block_stack)
    if self.padding == 'valid':
      valid = inputs[:, self.pad_size[0] : inputs.shape[1] - self.pad_size[0],
                     self.pad_size[1] : inputs.shape[2] - self.pad_size[1], :]
      outputs = sparse.sparse_scatter(block_stack, indices.bin_counts,
                                      indices.active_block_indices,
                                      bsize=self.bsize, boffset=self.boffset,
                                      bstride=self.bstride)
    else:
      raise NotImplementedError("'same' padding not supported yet")
    return outputs

def main():
  tf.enable_eager_execution()
  inputs = tf.constant(
    np.array([img.open_as_float('../data/disks_100x100/images/1001.png'),
              img.open_as_float('../data/disks_100x100/images/1002.png'),
              img.open_as_float('../data/disks_100x100/images/1003.png'),
              img.open_as_float('../data/disks_100x100/images/1004.png')]))
  mask = inputs > 0.1
  outputs = SparseConv2D(64, [3,3])([inputs, mask])
  vis.plot_image(*inputs.numpy, *mask.numpy, *outputs.numpy, columns=4)

if __name__ == '__main__':
  main()
