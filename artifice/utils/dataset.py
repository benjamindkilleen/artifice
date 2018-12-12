"""Functions for reading and writing datasets (mainly in tfrecords), as needed
by artifice and test_utils.

.tfrecord writer largely based on:
http://warmspringwinds.github.io/tensorflow/tf-slim/2016/12/21/tfrecords-guide/

"""

import numpy as np
import tensorflow as tf
# import matplotlib.pyplot as plt
import os

def _bytes_feature(value):
  # Helper function for writing a string to a tfrecord
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _int64_feature(value):
  # Helper function for writing an array to a tfrecord
  return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def example_string_from_scene(image, annotation):
  """Creates a tf example from the scene, which contains an image and an
  annotation.
  
  output: 
    example_string: a tf.train.Example, serialized to a string with four
      elements: the original images, as strings, and their shapes.

  """
  assert(type(scene) == tuple and len(scene) == 2)
  
  image = np.atleast_3d(image)
  annotation = np.atleast_3d(annotation)

  # TODO: rather than assert this, change image and annotation if neeeded
  assert(image.dtype == np.uint8 and annotation.dtype == np.uint8)
  image_string = image.tostring()
  annotation_string = annotation.tostring()
  image_shape = np.array(image.shape, dtype=np.int64)
  annotation_shape = np.array(annotation.shape, dtype=np.int64)
  
  feature = {"image" : _bytes_feature(image_string),
             "annotation" : _bytes_feature(annotation_string),
             "image_shape" : _int64_feature(image_shape),
             "annotation_shape" : _int64_feature(annotation_shape)}
        
  features = tf.train.Features(feature=feature)
  example = tf.train.Example(features=features)
  return example.SerializeToString()


def scene_from_feature(feature):
  """Using an example's feature dict, as created above, return the reconstructed
  numpy arrays `image` and `annotation`.
  """

  image_string = feature['image'].bytes_list.value[0]
  annotation_string = feature['annotation'].bytes_list.value[0]
  image_shape = np.array(feature['image_shape'].int64_list.value,
                         dtype=np.int64)
  annotation_shape = np.array(feature['annotation_shape'].int64_list.value,
                              dtype=np.int64)

  image = np.fromstring(image_string, dtype=np.uint8).reshape(image_shape)
  annotation = np.fromstring(annotation_string, dtype=np.uint8) \
                 .reshape(annotation_shape)

  return image, annotation

def scene_from_example_string(example_string):

  """Take a serialized tf.train.Example, as created by example_string_from_scene(),
  and convert it back to a scene, containing an image and a annotation."""

  example = tf.train.Example()
  example.ParseFromString(example_string)

  return scene_from_feature(example.features.feature)


def entry_from_example_proto(example_proto):
  # TODO: finish this function, according to features generated by 
  # features = {"image": tf.FixedLenFeature((), tf.string, default_value=""),
  #             "annotation": tf.FixedLenFeature((), tf.string, default_value="")}
  

  # image, annotation = scene_from_feature()
  return tf.constant(32)        # TODO: dummy value

def write_tfrecord(fname, gen):
  """Write a tfrecord from the generator, gen, which yields a serialized string
  to write for every example. Save it to fname."""
  writer = tf.python_io.TFRecordWriter(fname)
  
  for e in gen():
    writer.write(e)

  writer.close()


def read_tfrecord(fname, parse_example_string=scene_from_example_string):
  """Reads a tfrecord into a generator over each parsed example, using
  parse_example to turn each serialized tf string into a value returned from the
  generator. parse_example=None, then just return the unparsed string on each
  call to the generator.
  """

  if parse_example_string == None:
    parse_example_string = lambda x : x
  record_iter = tf.python_io.tf_record_iterator(path=fname)

  for string_record in record_iter:
    yield parse_example_string(string_record)


def save_first_scene(fname):
  """Saves the first scene from fname, a tfrecord, in the same directory. Meant
  for testing.
  """
  root = os.path.join(*fname.split(os.sep)[:-1])

  gen = read_tfrecord(fname)
  image, annotation = next(gen)

  plt.imshow(image[:,:,0], cmap='gray')
  plt.savefig(os.path.join(root, "example_image.png"))

  plt.imshow(annotation[:,:,0], cmap='tab20')
  plt.savefig(os.path.join(root, "example_annotation.png"))
  

class Loader:
  """Loads a dataset and, using the __call__() method, yields through batches over
  that dataset.

  args:
  * fname: tfrecord file to open.
  
  """
  def __init__(self, fname):
    self.fname = fname
    self._data = tf.data.TFRecordDataset(self.fname)
    self.iterator = self._data.make_one_shot_iterator()
    self._data.map(entry_from_example_proto)
    
  def __call__(self, batch_size=16):
    """Yield a batch of examples, in the form of a [batch_size, None, None,
    channels] tensor.

    """
    with tf.Session() as sess:
      value = sess.run(self.iterator.get_next())
      value = scene_from_example_string(value)
    return value
  
