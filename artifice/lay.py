import tensorflow as tf
from tensorflow import keras

class Crop(keras.layers.Layer):
  def __init__(self, shape, **kwargs):
    """Crop layer to fit `inputs`'s shape with `shape`

    :param shape: 
    :returns: 
    :rtype: 

    """
    self.shape = shape
    super().__init__(**kwargs)
    
  def call(self, inputs):
    offset_height = (inputs.shape[1] - self.shape[1]) // 2
    offset_width  = (inputs.shape[2] - self.shape[2]) // 2

    return tf.image.crop_to_bounding_box(
      inputs, offset_height, offset_width,
      self.shape[1], self.shape[2])
  
  def compute_output_shape(self, input_shape):
    return self.shape

class Untile(keras.layers.Layer):
  def __init__(self, image_shape, **kwargs):
    self.image_shape = image_shape
    self._num_tiles = None
    super().__init__(**kwargs)

  @property
  def num_tiles(self):
    return self._num_tiles
    
  def call(self, inputs):
    """

    :param inputs: batched set of tiles, as produced by dat.Data.tiled
    :returns: batch of images
    :rtype: 

    """
    # inputs will have shape
    # [batch_size * num_tiles] + tile_shape
    batch_size = tf.shape(inputs)[0]
    tf.reshape(inputs, [-1, self.num_tiles] + inputs.shape[1:])
    raise NotImplementedError
    
    for i in range(0, self.image_shape[0], self.tile_shape[0]):
      if i + self.tile_shape[0] < self.image_shape[0]:
        si = self.tile_shape[0]
      else:
        si = self.image_shape[0] % self.tile_shape[0]
      for j in range(0, self.image_shape[1], self.tile_shape[1]):
        if j + self.tile_shape[1] < self.image_shape[1]:
          sj = self.tile_shape[1]
        else:
          sj = self.image_shape[1] % self.tile_shape[1]
        try:
          tile = next(next_tile)
        except StopIteration:
          break
        image[i:i + si, j:j + sj] = tile[:si,:sj]

  def compute_num_tiles(self, input_shape):
    tile_shape = input_shape[1:]
    return int(np.ceil(self.image_shape[0] / tile_shape[0]) *
               np.ceil(self.image_shape[1] / tile_shape[1]))

  def compute_output_shape(self, input_shape):
    self._num_tiles = self.compute_num_tiles(input_shape)
    batch_size = (None if input_shape[0] is None
                  else input_shape[0] // self.num_tiles)
    return [batch_size] + self.image_shape

  
class LocalMaxima(keras.layers.Layer):
  def __init__(self, num_objects, **kwargs):
    """Detect at most `num_objects` local maxima.

    Useful for prediction, not for analysis.

    :param num_objects: 
    :returns:
    :rtype: 

    """
    self.num_objects = num_objects
    super().__init__(**kwargs)

  def call(self, inputs):
    """Expects a `field` output from the HourglassModel network.

    :param inputs: batched tensor
    :returns: a (num_objects, 3) tensor with [obj_id,x,y]. obj_id is 0 is object
    not present
    :rtype: 

    """
    raise NotImplemented
    
  def compute_output_shape(self, input_shape):
    return (self.num_objects, 3)
