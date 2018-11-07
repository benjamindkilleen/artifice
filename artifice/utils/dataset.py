"""Functions for reading and writing datasets (mainly in tfrecords), as needed
by artifice and test_utils.

.tfrecord writer largely based on:
http://warmspringwinds.github.io/tensorflow/tf-slim/2016/12/21/tfrecords-guide/

"""

import numpy as np
import tensorflow as tf

def _bytes_feature(value):
  # Helper function for writing a string to a tfrecord
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _int64_feature(value):
  # Helper function for writing an array to a tfrecord
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def tf_string_from_scene(scene):
  """Creates a tf example from the scene, which contains an image and a annotation.

  args:
    scene: a tuple containing two elements, image and annotation, as
      numpy arrays. These should have ndim==3.
  
  output: 
    example_string: a tf.train.Example, serialized to a string with four
      elements: the original images, as strings, and their shapes.

  """
  assert(type(scene) == tuple and len(scene) == 2)
  
  image = scene[0]
  annotation = scene[1]

  # TODO: rather than assert this, change image and annotation if neeeded
  assert(image.ndim == 3 and annotation.ndim == 3)
  assert(image.dtype == np.uint8 and annotation.dtype == np.uint8)
  image_string = image.tostring()
  annotation_string = annotation.tostring()
  image_shape = np.array(image.shape, dtype=np.int64)
  annotation_shape = annotation.shape(annotation.shape, dtype=np.int64)
  
  feature = {"image" : _bytes_feature(image_string),
             "annotation" : _bytes_feature(annotation_string),
             "image_shape" : _int64_feature(image_shape),
             "annotation_shape" : _int64_feature()}
        
  features = tf.train.Features(feature=feature)
  example = tf.train.Example(features=features)
  return example.SerializeToString()

def scene_from_tf_string(example_string):
  """Take a serialized tf.train.Example, as created by tf_string_from_scene(),
  and convert it back to a scene, containing an image and a annotation."""

  example = tf.train.Example()
  example.ParseFromString(example_string)

  image_string = example.features.feature['image'].bytes_list.value[0]
  annotation_string = example.features.feature['annotation'].bytes_list.value[0]
  image_shape = np.array(example.features.feature['image_shape']
                         .int64_list.value, dtype=np.int64)
  annotation_shape = np.array(example.features.feature['annotation_shape']
                         .int64_list.value, dtype=np.int64)

  image = np.fromstring(image_string, dtype=np.uint8).reshape(image_shape)
  annotation = np.fromstring(annotation_string, dtype=np.uint8) \
                 .reshape(annotation_shape)

  return image, annotation

def write_tfrecord(fname, gen):
  """Write a tfrecord from the generator, gen, which yields a serialized string
  to write for every example. Save it to fname."""
  writer = tf.python_io.TFRecordWriter(fname)
  
  for e in gen():
    writer.write(e)

  writer.close()

def read_tfrecord(fname, parse_example_string=scene_from_tf_string):
  """Reads a tfrecord into a generator over each parsed example, using
  parse_example to turn each serialized tf string into a value returned from the
  generator. parse_example=None, then just return the unparsed string on each
  call to the generator.
  """

  if parse_example_string == None:
    parse_example_string = lambda x : x
  record_iter = tf.python_io.tf_record_iterator(path=fname)

  def gen():
    for string_record in record_iter:
      yield parse_example_string(string_example)

  return gen
