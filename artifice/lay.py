import numpy as np
import tensorflow as tf
from tensorflow import keras

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

  def build(self, input_shape):
    pass
  
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
