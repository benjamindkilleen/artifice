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
    raise NotImplementedError
    super().__init__(**kwargs)
  
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
