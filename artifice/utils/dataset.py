"""Functions for reading and writing datasets (mainly in tfrecords), as needed
by artifice and test_utils.

.tfrecord writer largely based on:
http://warmspringwinds.github.io/tensorflow/tf-slim/2016/12/21/tfrecords-guide/

"""

import numpy as np
import tensorflow as tf
from artifice.utils import img
import os

def _bytes_feature(value):
  # Helper function for writing a string to a tfrecord
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _int64_feature(value):
  # Helper function for writing an array to a tfrecord
  return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def proto_from_scene(scene):
  """Creates a tf example from the scene, which contains an image and an
  annotation.
  
  output: 
    example_string: a tf.train.Example, serialized to a string with four
      elements: the original images, as strings, and their shapes.

  """
  image, annotation = scene
  image = np.atleast_3d(image)
  annotation = np.atleast_3d(annotation)

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


def tensors_from_proto(proto):
  """Parse the serialized string PROTO into tensors (IMAGE, ANNOTATION).
  """
  features = tf.parse_single_example(
    proto,
    # Defaults are not specified since both keys are required.
    features={
      'image': tf.FixedLenFeature([], tf.string),
      'annotation': tf.FixedLenFeature([], tf.string),
      'image_shape': tf.FixedLenFeature([3], tf.int64),
      'annotation_shape': tf.FixedLenFeature([3], tf.int64)
    })

  # decode strings
  image = tf.decode_raw(features['image'], tf.uint8)
  image = tf.reshape(image, features['image_shape'])
  image = tf.cast(image, tf.float32) / 255.

  annotation = tf.decode_raw(features['annotation'], tf.uint8)
  annotation = tf.reshape(annotation, features['annotation_shape'])

  return image, annotation


def scene_from_proto(proto):
  """Take a serialized tf.train.Example, as created by proto_from_scene(),
  and convert it back to a scene, containing an image and a annotation."""

  example = tf.train.Example()
  example.ParseFromString(proto)

  feature = example.features.feature
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


def read_tfrecord(fname, parse_entry=scene_from_proto):
  """Reads a tfrecord into a generator over each parsed example, using
  parse_entry to turn each serialized tf string into a value returned from the
  generator. parse_entry=None, then just return the unparsed string on each
  call to the generator.
  
  Note that this is not intended for any training/testing pipeline but rather
  for accessing the actual data.

  """

  if parse_entry == None:
    parse_entry = lambda x : x
  record_iter = tf.python_io.tf_record_iterator(path=fname)

  for string_record in record_iter:
    yield parse_entry(string_record)


def save_first_scene(fname):
  """Saves the first scene from fname, a tfrecord, in the same directory, as npy
  files. Meant for testing.

  """
  root = os.path.join(*fname.split(os.sep)[:-1])

  gen = read_tfrecord(fname)
  image, annotation = next(gen)

  np.save(os.path.join(root, "example_image.npy"), image)
  np.save(os.path.join(root, "example_annotation.npy"), annotation)




"""A DataInput object encompasses all of them. The
__call__ method for a Data returns an iterator function over the associated
tf.Dataset.
* Data objects can be used for evaluation, testing, or prediction. They iterate
once over the dataset.
* TrainData is meant for training. It's __call__ method returns an input
  function that includes shuffling, repeating, etc.
* 

"""
class DataInput(object):
  def __init__(self, dataset, **kwargs):
    """
    :dataset: a tf.data.Dataset
    :batch_size:
    :num_parallel_calls:
    """
    self._dataset = dataset
    self.batch_size = kwargs.get('batch_size', 1)
    self.num_parallel_calls = kwargs.get('num_parallel_calls')
    self.prefetch_buffer_size = kwargs.get('prefetch_buffer_size', 1)
    self._kwargs = kwargs

  def make_input(self):
    """Make an input function for an estimator. Can be overwritten by subclasses
    withouting overwriting __call__."""
    return lambda : (
      self._dataset
      .batch(self.batch_size)
      .prefetch(self.prefetch_buffer_size)
      .make_one_shot_iterator()
      .get_next())

  def __call__(self, *args):
    return self.make_input(*args)


class TrainDataInput(DataInput):
  def __init__(self, *args, **kwargs):
    self.num_shuffle = kwargs.get('num_shuffle', 10000)
    super().__init__(*args, **kwargs)

  def make_input(self, 
                 num_epochs=1):
    return lambda : (
      self._dataset
      .repeat(num_epochs)
      .shuffle(self.num_shuffle)
      .map(lambda *args : args)          # TODO: augmentations
      .batch(self.batch_size)
      .prefetch(self.prefetch_buffer_size)
      .make_one_shot_iterator()
      .get_next())



def load(record_names,
         parse_entry=tensors_from_proto,
         input_classes=[DataInput],
         input_sizes=[-1],
         num_parallel_calls=None,
         **kwargs):
  """Load the tfrecord. By default, parses scenes (image, annotation).
  
  args:
  :record_names: one or more tfrecord files.
  :parse_entry: a function parsing each raw entry in the loaded dataset
  :num_parallel_calls: see tf.data.Dataset.map
  :input_classes: list of DataInput or a subclass thereof, determining the type
    of each data that will be loaded from this tfrecord. A type of None adds
    None to the list of data_inputs (useful for standardizing calls to this
    function)
  :input_sizes: list of sizes corresponding to input_classes. A value of -1
    takes the remainder of the dataset. Note that a NoneType input class still
    skips data, so be sure to enter 0 for these entries.
  :kwargs: additional keyword args for DataInput objects.
  """
  
  dataset = tf.data.TFRecordDataset(record_names)
  dataset = dataset.map(parse_entry, num_parallel_calls=num_parallel_calls)

  data_inputs = []
  for input_class, input_size in zip(input_classes, input_sizes):
    assert issubclass(input_class, DataInput)
    if input_class is None or input_size == 0:
      data_inputs.append(None)
    else:
      data_inputs.append(
        input_class(dataset.take(input_size)))
    dataset = dataset.skip(input_size)
    
  return data_inputs

def load_single(record_names, train=False, **kwargs):
  """Wrapper around load that returns a single dataset object."""
  if train:
    return load(record_names,
                input_classes=[dataset.TrainDataInput],
                **kwargs)
  else:
    return load(record_names, **kwargs)

