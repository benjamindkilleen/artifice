"""Functions for reading and writing datasets in tfrecords, as needed by
artifice and test_utils. A "scene" is the info needed for a single, labeled
example. It should always be a 2-tuple, usually (image, annotation),
although it could also be (image, (annotation, label)).

"""

import numpy as np
import tensorflow as tf
from artifice.utils import augment
import os
import logging


logger = logging.getLogger('artifice')


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

  TODO: allow for creating an unlabeled example. Include flag in the data?

  """
  image, (annotation, label) = scene
  image = np.atleast_3d(image)
  annotation = np.atleast_3d(annotation)

  assert(image.dtype == np.uint8 and annotation.dtype == np.float32)
  image_string = image.tostring()
  image_shape = np.array(image.shape, dtype=np.int64)

  annotation_string = annotation.tostring()
  annotation_shape = np.array(annotation.shape, dtype=np.int64)

  label_string = label.tostring()
  label_shape = np.array(label.shape, dtype=np.int64)
  
  feature = {"image" : _bytes_feature(image_string),
             "image_shape" : _int64_feature(image_shape),
             "annotation" : _bytes_feature(annotation_string),
             "annotation_shape" : _int64_feature(annotation_shape),
             "label" : _bytes_feature(label_string),
             "label_shape" : _int64_feature(label_shape)}
        
  features = tf.train.Features(feature=feature)
  example = tf.train.Example(features=features)
  return example.SerializeToString()

def proto_from_tensor_scene(scene):
  """Create a proto string holding the given scene properly. Useful for writing
  tfrecord from tf.data.Dataset objects.
  """


def tensor_scene_from_proto(proto):
  """Parse the serialized string PROTO into tensors (IMAGE, ANNOTATION).

  TODO: allow for loading an unlabelled example. Can do this by including a
  boolean in the data? That's one possibility.

  """
  features = tf.parse_single_example(
    proto,
    # Defaults are not specified since both keys are required.
    features={
      'image': tf.FixedLenFeature([], tf.string),
      'image_shape': tf.FixedLenFeature([3], tf.int64),
      'annotation': tf.FixedLenFeature([], tf.string),
      'annotation_shape': tf.FixedLenFeature([3], tf.int64),
      'label' : tf.FixedLenFeature([], tf.string),
      'label_shape' : tf.FixedLenFeature([2], tf.int64)
    })

  # decode strings
  image = tf.decode_raw(features['image'], tf.uint8)
  image = tf.reshape(image, features['image_shape'])
  image = tf.cast(image, tf.float32) / 255.

  annotation = tf.decode_raw(features['annotation'], tf.float32)
  annotation = tf.reshape(annotation, features['annotation_shape'],
                          name='reshape_annotation_proto')

  label = tf.decode_raw(features['label'], tf.float32)
  label = tf.reshape(label, features['label_shape'],
                     name='reshape_label_proto')

  return image, (annotation, label)


def scene_from_proto(proto):
  """Take a serialized tf.train.Example, as created by proto_from_scene(),
  and convert it back to a scene.
  
  TODO: allow for loading an unlabelled example.
  
  """

  example = tf.train.Example()
  example.ParseFromString(proto)

  feature = example.features.feature

  image_string = feature['image'].bytes_list.value[0]
  image_shape = np.array(feature['image_shape'].int64_list.value,
                         dtype=np.int64)
  image = np.fromstring(image_string, dtype=np.uint8).reshape(image_shape)

  annotation_string = feature['annotation'].bytes_list.value[0]
  annotation_shape = np.array(feature['annotation_shape'].int64_list.value,
                              dtype=np.int64)
  annotation = np.fromstring(annotation_string, dtype=np.float32) \
                 .reshape(annotation_shape)

  label_string = feature['label'].bytes_list.value[0]
  label_shape = np.array(feature['label_shape'].int64_list.value,
                         dtype=np.int64)
  label = np.fromstring(label_string, dtype=np.float32).reshape(label_shape)
  
  return image, (annotation, label)


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
  image, (annotation, _) = next(gen)

  np.save(os.path.join(root, "example_image.npy"), image)
  np.save(os.path.join(root, "example_annotation.npy"), annotation)


def load_dataset(record_name,
                 parse_entry=None,
                 num_parallel_calls=None):
  """Load the record_name as a tf.data.Dataset"""
  if parse_entry is None:
    parse_entry = tensor_scene_from_proto
  dataset = tf.data.TFRecordDataset(record_name)
  return dataset.map(parse_entry, num_parallel_calls=num_parallel_calls)


  
class Data(object):
  """A Data object contains scenes, as defined above. It's mainly useful as a
  way to save and then distribute data."""
  def __init__(self, data, **kwargs):
    """args:
    :data: a tf.data.Dataset OR tfrecord file name(s) OR another Data
      subclass. In this case, the other object's _dataset is adopted, allows
      kwargs to be overwritten.
    """
    if issubclass(type(data), Data):
      self._kwargs = data._kwargs.update(kwargs)
    else:
      self._kwargs = kwargs

    self.batch_size = self._kwargs.get('batch_size', 1)
    self.num_parallel_calls = self._kwargs.get('num_parallel_calls', None)
    self.parse_entry = self._kwargs.get('parse_entry', None)

    if issubclass(type(data), Data):
      self._dataset = data._dataset
    elif issubclass(type(data), tf.data.Dataset):
      self._dataset = dataset
    elif type(data) == str or hasattr(data, '__iter__'):
      self._dataset = load_dataset(data,
                                   parse_entry=self.parse_entry,
                                   num_parallel_calls=self.num_parallel_calls)
    else:
      raise ValueError(f"unrecognized data type '{type(data)}'")

    
  def save(self, record_name, encode_entry=proto_from_tensor_scene):
    """Save the dataset to record_name."""
    logger.info(f"Writing dataset to {record_name}")
    dataset = self._dataset.map(encode_entry,
                                num_parallel_calls=self.num_parallel_calls)
    writer = tf.data.experimental.TFRecordWriter(record_name)
    write_op = writer.write(dataset)
    with tf.Session() as sess:
      sess.run(write_op)
    
  def accumulate(self, *accumulators):
    """Given a tf.data.dataset with scene entries (image, (annotation, label)),
    accumulate the labels using given functions.

    An accumulator function should take a scene and an aggregate object. On the
    first call, aggregate will be None. Afterward, each accumulator will be
    passed the output from its previous call. Finally, the final call, scene
    will be None.

    """

    iterator = self._dataset.make_initializable_iterator()
    next_scene = iterator.get_next()
    logger.debug("made label iterator (?)")
    
    aggregates = [None] * len(accumulators)
  
    with tf.Session() as sess:
      logger.debug("started session")
      sess.run(iterator.initializer)
      logger.debug("initialized iterator, accumulating labels")
      while True:
        try:
          image, (annotation, label) = sess.run(next_scene)
          for i in range(len(accumulators)):
            aggregates[i] = accumulators[i](scene, aggregates[i])
        except tf.errors.OutOfRangeError:
          break

    return [acc(None, agg) for acc, agg in zip(accumulators, aggregates)]


"""A DataInput object encompasses all of them. The
__call__ method for a Data returns an iterator function over the associated
tf.Dataset.
* Data objects can be used for evaluation, testing, or prediction. They iterate
once over the dataset.
* TrainData is meant for training. It's __call__ method returns an input
  function that includes shuffling, repeating, etc. It also allows for
  augmentation.

"""
class DataInput(Data):
  def __init__(self, *args, **kwargs):
    """
    :batch_size:
    :num_parallel_calls:
    """
    super().__init__(*args, **kwargs)
    self.prefetch_buffer_size = kwargs.get('prefetch_buffer_size', 1)

  def make_input(self):
    """Make an input function for an estimator. Can be overwritten by subclasses
    withouting overwriting __call__."""
    return lambda : (
      self._dataset
      .batch(self.batch_size)
      .prefetch(self.prefetch_buffer_size)
      .make_one_shot_iterator()
      .get_next())

  def __call__(self, *args, **kwargs):
    return self.make_input(*args, **kwargs)

"""TrainDataInput is where augmentation takes place. New augmentations can be
added to it using the add_augmentation() method, or the "+" operator.
"""
class TrainDataInput(DataInput):
  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)
    self.num_shuffle = kwargs.get('num_shuffle', 10000)
    self.augmentation = kwargs.get('augmentation', augment.identity)

  def make_input(self, num_epochs=1):
    return lambda : (
      self.augmentation(self._dataset)
      .shuffle(self.num_shuffle) # TODO: shuffle before or after?
      .repeat(num_epochs)
      .batch(self.batch_size)
      .prefetch(self.prefetch_buffer_size)
      .make_one_shot_iterator()
      .get_next())

  def add_augmentation(self, aug):
    self.augmentation += aug
    return self

  def __add__(self, other):
    if issubclass(type(other), augment.Augmentation):
      return self.add_augmentation(aug)
    elif issubclass(type(other), TrainDataInput):
      return self.add_augmentation(other.augmentation)
    else:
      raise ValueError(f"unrecognized type '{type(other)}'")

  def __radd__(self, other):
    if other == 0:
      return self
    else:
      return self.__add__(other)

  def mul_augmentation(self, aug):
    self.augmentation *= aug
    return self

  def __mul__(self, other):
    if issubclass(type(other), augment.Augmentation):
      return self.mul_augmentation(aug)
    elif issubclass(type(other), TrainDataInput):
      return self.mul_augmentation(other.augmentation)
    else:
      raise ValueError(f"unrecognized type '{type(aug)}'")


def load_data_input(record_name,
                    input_classes=[DataInput],
                    input_sizes=[-1],
                    parse_entry=tensor_scene_from_proto,
                    num_parallel_calls=None,
                    **kwargs):
  """Load the tfrecord as a DataInput object. By default, parses
  scenes. Generalizes load().
  
  args:
  :record_name: one or more tfrecord files.
  :parse_entry: a function parsing each raw entry in the loaded dataset
  :num_parallel_calls: see tf.data.Dataset.map
  :input_classes: list of DataInput or a subclass thereof, determining the type
    of each data that will be loaded from this tfrecord. A type of None adds
    None to the list of data_inputs (useful for standardizing calls to this
    function)
  :input_sizes: list of sizes corresponding to input_classes. A value of -1
    takes the remainder of the dataset. Note that a NoneType input class still
    skips data, so be sure to enter 0 for these entries.
  :kwargs: additional keyword args for each DataInput objects. 
    TODO: allow passing separate kwargs to each object.
  """
  dataset = load_dataset(record_name, parse_entry=parse_entry,
                         num_parallel_calls=num_parallel_calls)

  data_inputs = []
  for input_class, input_size in zip(input_classes, input_sizes):
    assert issubclass(input_class, DataInput)
    if input_class is None or input_size == 0:
      data_inputs.append(None)
    else:
      data_inputs.append(
        input_class(dataset.take(input_size), **kwargs))
    dataset = dataset.skip(input_size)
    
  return data_inputs

def load(record_name, train=False, **kwargs):
  """Wrapper around load that returns a single dataset object. Most often used."""
  if train:
    return load_data_input(record_name,
                           input_classes=[dataset.TrainDataInput],
                           **kwargs)[0]
  else:
    return load_data_input(record_name, **kwargs)[0]


"""A DataAugmenter object should work with augmentations to create new tfrecord
files. It belongs in the dataset module because it should handle all the
reading/writing of data to files. Perhaps this should be the object that
processes data initially (i.e. processes label-space, inpainting stuff), then
the augmentation object can use that data??? Maybe, but probably
not. We can use transformation objects, maybe, except that those take
tensors. We'd just have to use eager execution, I guess. So we can still use
individual transformations, composed or otherwise, and apply them selectively to
examples in the dataset.

It doesn't matter too much if this step takes a while to run. But it does
prevent training from happening while it's running? Yeah, probably. Whatever.

Don't actually need to do eager execution. Could just run inside a session,
which would be fine, after building up the graph. This allows
augment.Transformation to continue to just use tensor operations, and
augmentation will be more efficient with a computation graph, etc. 

Ideas:
- can use tf.data.experimental.enumerate_dataset() to give each original example
  a unique label. (Returns a transformation function, give to dataset.apply)
- tf.experimental.

"""




class DataAugmenter(Data):
  """The DataAugmenter object interfaces with tfrecord files for a labeled
  dataset to create a new augmentation set (not including the originals).

  It needs:
  - an initial processing ability, either internal or using a module-level
    function (more likely).
  - the ability to apply transformations to select examples of the dataset. Not
    sure how that's going to happen.
  - a method to create a DataInput object (or TrainDataInput) using all the
    tfrecords it has incorporated or created.
  """

  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)

  @staticmethod
  def label_accumulator(scene, labels):
    if labels is None:
      labels = []
    if scene is None:
      return np.array(labels)

    _, (_, label) = scene
    labels.append(label)
    return labels
    
  @staticmethod
  def background_accumulator(scene, agg):
    """Iterate over the dataset, taking a running average of the pixels where no
    objects exist.

    COMMENTS: Need to make a suitable background image or images to draw from, when
    inpainting. This should be based on the whole dataset, under the assumption
    that the camera never moves, although objects in the image do. Maybe there
    should be a coupled background images produced? We could go through the
    dataset in order, until we have accumulated enough images to get a good
    background for all pixels, save that, etc.

    Better yet, can take a running average for all the pixels? This can be a
    good place to start.

    """
    if agg is None:
      assert scene is not None
      background = np.zeros_like(scene[0])
      n = np.zeros_like(background, dtype=np.float64)
    else:
      background, n = agg

    if scene is None:
      return background

    image, (annotation, _) = scene

    # Update the elements with a running average
    bg_indices = np.equal(annotation, 0)
    indices = np.logical_and(np.not_equal(n, 0), bg_indices)
    n[indices] += 1
    background[indices] = (background[indices] + 
                           (image[indices] - background[indices]) / n[indices])

    # initialize the new background elements
    indices = np.logical_and(np.equal(n, 0), bg_indices)
    background[indices] = image[indices]
    n[indices] += 1
    
    return background, n
