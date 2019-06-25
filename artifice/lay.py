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

