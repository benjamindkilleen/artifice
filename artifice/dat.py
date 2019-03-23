"""Functions for reading and writing datasets in tfrecords, as needed by
artifice and test_utils. A "scene" is the info needed for a single, labeled
example. It should always be a 2-tuple, usually (image, annotation),
although it could also be (image, (annotation, label)).

"""

import numpy as np
import tensorflow as tf
from artifice import tform
import os
import logging


logger = logging.getLogger('artifice')


def as_float(image):
  """Return image as a grayscale float32 array at least 3d."""
  if image.dtype in [np.float32, np.float64]:
    image = image.astype(np.float32)
  elif image.dtype in [np.uint8, np.int32, np.int64]:
    image = image.astype(np.float32) / 255.
  else:
    raise ValueError(f"image dtype '{image.dtype}' not allowed")
  return np.atleast_3d(image)
  

def _bytes_feature(value):
  # Helper function for writing a string to a tfrecord
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _int64_feature(value):
  # Helper function for writing an array to a tfrecord
  return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def proto_from_example(example):
  """Creates a tf example proto from an (image, label) pair.

  :param example: (image, label) pair
  :returns: proto string
  :rtype: str

  """
  image, label = example
  image = as_float(image)
  image_string = image.tostring()
  image_shape = np.array(image.shape, dtype=np.int64)

  label_string = label.tostring()
  label_shape = np.array(label.shape, dtype=np.int64)

  feature = {"image" : _bytes_feature(image_string),
             "image_shape" : _int64_feature(image_shape),
             "label" : _bytes_feature(label_string),
             "label_shape" : _int64_feature(label_shape)}

  features = tf.train.Features(feature=feature)
  example = tf.train.Example(features=features)
  return example.SerializeToString()

def example_from_proto(proto):
  """Parse `proto` into tensors `(image, label)`.

  """
  features = tf.parse_single_example(
    proto,
    features={
      'image': tf.FixedLenFeature([], tf.string),
      'image_shape': tf.FixedLenFeature([3], tf.int64),
      'label' : tf.FixedLenFeature([], tf.string),
      'label_shape' : tf.FixedLenFeature([2], tf.int64)
    })

  # decode strings
  image = tf.decode_raw(features['image'], tf.uint8)
  image = tf.reshape(image, features['image_shape'])
  image = tf.to_float(image) / 255.

  label = tf.decode_raw(features['label'], tf.float32)
  label = tf.reshape(label, features['label_shape'],
                     name='reshape_label_proto')

  return (image, label)


def proto_from_scene(scene):
  """Creates a tf example from the scene, which contains an (image,label) example
  and an annotation.

  example_string: a tf.train.Example, serialized to a string with four
  elements, the original images, as strings, and their shapes.

  """
  example, annotation = scene
  image, label = example
  image = np.as_float(image)
  annotation = np.as_float(annotation)

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


def scene_from_proto(proto):
  """Parse `proto` into tensors `(image, label), annotation`.

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
  image = tf.to_float(image) / 255.

  annotation = tf.decode_raw(features['annotation'], tf.float32)
  annotation = tf.reshape(annotation, features['annotation_shape'],
                          name='reshape_annotation_proto')

  label = tf.decode_raw(features['label'], tf.float32)
  label = tf.reshape(label, features['label_shape'],
                     name='reshape_label_proto')

  return (image, label), annotation


def load_dataset(record_name,
                 parse_entry=example_from_proto,
                 num_parallel_calls=None):
  """Load the record_name as a tf.data.Dataset"""
  dataset = tf.data.TFRecordDataset(record_name)
  return dataset.map(parse_entry, num_parallel_calls=num_parallel_calls)


def save_dataset(record_name, dataset,
                 proto_from_example=proto_from_example,
                 num_parallel_calls=None):
  """Save the dataset in tfrecord format

  :param record_name: filename to save to
  :param dataset: tf.data.Dataset to save
  :param proto_from_example: 
  :param num_parallel_calls: 
  :returns: 
  :rtype: 

  """
  next_example = dataset.make_one_shot_iterator().get_next()

  logger.info(f"writing dataset to {record_name}...")
  writer = tf.python_io.TFRecordWriter(record_name)
  with tf.Session() as sess:
    i = 0
    while True:
      try:
        example = sess.run(next_example)
        writer.write(proto_from_example(example))
      except tf.errors.OutOfRangeError:
        break
      i += 1


  writer.close()
  logger.info(f"wrote {i} examples")

  
class Data(object):
  """Wrapper around tf.data.Dataset of examples.

  Chiefly useful for feeding into models in various forms."""
  def __init__(self, data, **kwargs):
    """args:
    :param data: a tf.data.Dataset OR tfrecord file name(s) OR another Data
      subclass. In this case, the other object's _dataset is adopted, allows
      kwargs to be overwritten.
    """

    if issubclass(type(data), Data):
      self._dataset = data._dataset
    elif issubclass(type(data), tf.data.Dataset):
      self._dataset = data
    elif type(data) in [str, list, tuple]:
      # Loading tfrecord files
      self._dataset = load_dataset(data, **kwargs)
    else:
      raise ValueError(f"unrecognized data '{data}'")

    self.image_shape = self._kwargs.get('image_shape', None)
    self.batch_size = self._kwargs.get('batch_size', 1)
    self.num_parallel_calls = kwargs.get('num_parallel_calls')
    self._kwargs = kwargs
    
  def save(self, record_name, **kwargs):
    """Save the dataset to record_name."""
    save_dataset(record_name, self._dataset, **kwargs)

  def __iter__(self):
    return self._dataset.__iter__()
  
  def split(self, *splits, types=None):
    """Split the dataset into different sets.

    Pass in the number of examples for each split. Returns a new Data object
    with datasets of the corresponding number of examples.

    :param types: optional list of Data subclasses to instantiate the splits
    :returns: list of Data objects with datasets of the corresponding sizes.
    :rtype: [Data]

    """
    
    datas = []
    for i, n in enumerate(splits):
      dataset = self._dataset.skip(sum(splits[:i])).take(n)
      if types is None or i >= len(types):
        datas.append(type(self)(dataset, **self._kwargs))
      elif types[i] is None:
        datas.append(None)
      else:
        datas.append(types[i](dataset, **self._kwargs))
      
    return datas

  def sample(self, sampling):
    """Draw a sampling from the dataset, returning a new dataset of the same type.

    :param sampling: 1-D array indexing the dataset, e.g. a boolean array, with
    possible repetitions.
    :returns: new data object with examples selected by sampling.
    :rtype: a Data subclass, same as self

    """
    sampling = tf.constant(sampling, dtype=tf.int64)
    dataset = self._dataset.apply(tf.data.experimental.enumerate_dataset())
    def map_func(idx, example):
      return tf.data.Dataset.from_tensors(example).repeat(sampling[idx])
    dataset = dataset.flat_map(map_func)
    return type(self)(dataset, **self._kwargs)
  
  def accumulate(self, accumulator):
    """Eagerly runs the accumulators across the dataset.

    An accumulator function should take a `scene` and an `aggregate` object. On
    the first call, `aggregate` will be None. Afterward, each accumulator will
    be passed the output from its previous call as `aggregate`, as well as the
    next scene in the data as 'scene'. On the final call, `scene` will be None,
    allowing for post-processing.

    :param accumulator: an accumulator function OR a dictionary mapping names to
      accumulator functions
    :returns: aggregate from `accumulator` OR a dictionary of aggregates with
      the same keys as `accumulators`.
    :rtype: dict

    """

    iterator = self._dataset.make_initializable_iterator()
    next_item = iterator.get_next()

    if type(accumulator) == dict:
      accumulators = accumulator
    else:
      accumulators = {0 : accumulator}
    aggregates = dict.fromkeys(accumulators.keys())

    with tf.Session() as sess:
      sess.run(iterator.initializer)
      logger.info("initialized iterator, starting accumulation...")
      while True:
        try:
          scene = sess.run(next_item)
          for k, acc in accumulators.items():
            aggregates[k] = acc(scene, aggregates[k])
        except tf.errors.OutOfRangeError:
          break

    logger.info("finished accumulation")
    for k, acc in accumulators.items():
      aggregates[k] = acc(None, aggregates[k])
    
    if type(accumulator) == dict:
      return aggregates
    else:
      return aggregates[0]

  @property
  def (self):
    raise NotImplementedError


class AnnotatedData(Data):
  pass
    
class DataInput(Data):
  """A DataInput object encompasses all of them. The
  __call__ method for a Data returns an iterator function over the associated
  tf.Dataset.
  * Data objects can be used for evaluation, testing, or prediction. They iterate
    once over the dataset.
  * TrainData is meant for training. It's __call__ method returns an input
    function that includes shuffling, repeating, etc. It also allows for
    augmentation.

  """
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


class TrainDataInput(DataInput):
  """TrainDataInput is where in-place augmentation takes place. New augmentations
  can be added to it using the add_augmentation() method, or the "+" operator.

  """

  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)
    self.num_shuffle = kwargs.get('num_shuffle', 10000)

  def make_input(self, num_epochs=1):
    return lambda : (
      self.augmentation(self._dataset)
      .shuffle(self.num_shuffle) # TODO: shuffle before or after?
      .repeat(num_epochs)
      .batch(self.batch_size)
      .prefetch(self.prefetch_buffer_size)
      .make_one_shot_iterator()
      .get_next())


def load_data_input(record_name,
                    input_classes=[DataInput],
                    input_sizes=[-1],
                    parse_entry=tensor_scene_from_proto,
                    num_parallel_calls=None,
                    **kwargs):
  """Load the tfrecord as a DataInput object. By default, parses
  scenes. Generalizes load().

  :param record_name: one or more tfrecord files.
  :param parse_entry: a function parsing each raw entry in the loaded dataset
  :param num_parallel_calls: see tf.data.Dataset.map
  :param input_classes: list of DataInput or a subclass thereof, determining the type
    of each data that will be loaded from this tfrecord. A type of None adds
    None to the list of data_inputs (useful for standardizing calls to this
    function)
  :param input_sizes: list of sizes corresponding to input_classes. A value of -1
    takes the remainder of the dataset. Note that a NoneType input class still
    skips data, so be sure to enter 0 for these entries.
  :param **kwargs: additional keyword args for each DataInput objects.
    TODO: allow passing separate kwargs to each object.
  :returns: list of DataInput objects.
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


"""TODO: Note:

So, doing this augmentation and update step with tensors just isn't
working. Better to do:
1. Accumulate over the dataset initial stuff, like prime examples, etc.
2. Decide which ObjectTransformation is going to go with which example, except
   now make it so that ObjectTransformation works on numpy arrays.
3. Accumulate over the dataset again, but this time write out the new examples
   to the tfrecord in the accumulation.

"""
class DataAugmenter(Data):
  """The DataAugmenter object interfaces with tfrecord files for a labeled
  dataset to create a new augmentation set (not including the originals).

  Aggregate attributes may be set by keyword arguments (below). These also
  accept custom accumulators (see Data.accumulate).

  :param N: number of examples to have in the augmented set, default: 10,000.
  :param labels: label set. Should be used only when the tfrecord is unlabeled.
  :param background_image: used for inpainting transformed examples.
  :param image_shape: shape of the image. 

  """

  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)
    self._inserted_labels = None
    self.N = kwargs.get('N', 10000)

    self._accs = {'labels' : DataAugmenter.label_accumulator,
                  'background_image' : DataAugmenter.mean_background_accumulator,
                  'image_shape' : DataAugmenter.image_shape_accumulator}

    accumulators = {}
    for k, acc in self._accs.items():
      v = kwargs.get(k)
      if v is None:
        accumulators[k] = acc
      elif callable(v):
        accumulators[k] = v
      else:
        setattr(self, k, v)

    # May take a while.
    aggregates = self.accumulate(accumulators)

    for k,v in aggregates.items():
      setattr(self, k, v)

  def run(self, record_name='augmented.tfrecord',
          encode_entry=proto_from_scene):
    """Run the transformations on self._original_dataset. 

    Save novel examples in a new tfrecord file. Update self._dataset to
    incorporate both datasets.

    :param record_name: saves the augmented set. Defaults to
      'augmented.tfrecord' in the current directory.
    :param encode_entry: function to encode numpy scenes as example protocols
    :returns: self
    :rtype: DataAugmenter

    TODO:
    1. Determine the new labels that would more evenly sample the label-space
    2. For each of these, create an augmentation transformation, drawing an
       example from prime_examples at random.
    3. Run the augmentation on self._original_dataset
    4. save the newly-created dataset in record_name, reload it (forces
       execution).
    5. update: self._dataset = self._original_dataset `concat` augmentations

    """
    if True:
      raise NotImplementedError()
  
    transform = tform.ObjectTransformation(
      background_image=self.background_image)

    # Problem: would like to iterate over the labels.
    # Store prime examples in memory, some random selection of them, after the
    # prime_examples accumulator. This way we can actually multithread the
    # creation and writing of examples.

    def augment_accumulator(scene, agg):
      if agg is None:
        # initialization
        agg = {}
        agg['idx'] = 0
        agg['writer'] = tf.python_io.TFRecordWriter(record_name)
      if scene is None:
        agg['writer'].close()
        return

      if agg['idx'] in idx_to_labels:
        for new_label in idx_to_labels[agg['idx']]:
          agg['writer'].write(encode_entry(transform(scene, new_label)))

      agg['idx'] += 1

      return agg
    
    # May take a while
    self.accumulate(augment_accumulator)

    augmented = load_dataset(record_name)
    self._dataset = self._original_dataset.concatenate(augmented)

    return self

  def __call__(self, *args, **kwargs):
    return self.run(*args, **kwargs)

  def _compute_inserted_labels(self):
    """Create labels required for a label space according to some multivariate
    distribution.

    FOR NOW: draw from a uniform distribution in R^4, the location space for two
    spheres, with a hypercube location space. Always map sphere two in front of
    sphere one in case of overlap. (shouldn't matter)

    TODO: extend space to include rotation, scaling, etc.

    :returns: new numpy labels
    :rtype:

    """

    # TODO: generalize
    points = np.random.uniform(
      [0,0,0,0],
      [self.image_shape[0], self.image_shape[1],
       self.image_shape[0], self.image_shape[1]],
      size=(self.N,4)
    )

    original_points = self.labels[:,:,1:3].reshape()

    for point in points:
      np.argmin()
    
    # TODO: figure out the better place to instantiate and place this code
    bounds = [None,                     # object 1
              (0, self.image_shape[0]), # X position
              None, # (0, self.image_shape[1]), # Y position
              None,                     # rotation
              None,                     # X scaling
              None,                     # Y scaling
              None,                     # object 2
              None, # (0, self.image_shape[0]), # X position
              None, # (0, self.image_shape[1]), # Y position
              None,                     # rotation
              None,                     # X scaling
              None]                     # Y scaling
    
    smoother = smoothing.MultiSmoother(self.labels, bounds)
    smoother.smooth(max_iter=1)
    return smoother.inserted

  @property
  def inserted_labels(self):
    """Iterator over the labels required to smooth the dataset."""
    if self._inserted_labels is None:
      self._inserted_labels = self._compute_inserted_labels()
    return self._inserted_labels
  
  @staticmethod
  def label_accumulator(scene, labels):
    """Accumulate labels in the original dataset.

    :param scene: 
    :param labels: 
    :returns: 
    :rtype: 

    """
    if labels is None:
      labels = []
    if scene is None:
      return np.array(labels)

    _, (_, label) = scene
    labels.append(label)
    return labels


  @staticmethod
  def fill_background(background, mode='gaussian'):
    """Fill the negative values in background, depending on mode. Returns the
    new background.

    mode: ['gaussian']
    - gaussian: draw from a normal distribution with the same mean and stddev as
      the rest of the background.

    """

    if mode != 'gaussian':
      raise NotImplementedError()

    background = background.copy()
    indices = background >= 0

    mean = background[indices].mean()
    std = background[indices].std()

    indices = background < 0
    background[indices] = np.random.normal(mean, std, size=background[indices].shape)
    return background


  @staticmethod
  def mean_background_accumulator(scene, agg):
    """Iterate over the dataset, taking a running average of the pixels where no
    objects exist.

    Fills pixels with no values at the end of the accumulation using fill_background.

    """
    if agg is None:
      assert scene is not None
      background = -np.ones_like(scene[0], dtype=np.float64)
      n = np.zeros_like(background, dtype=np.int64)
    else:
      background, n = agg

    if scene is None:
      return DataAugmenter.fill_background(background)

    image, (annotation, _) = scene

    # Update the elements with a running average
    bg_indices = np.atleast_3d(np.equal(annotation[:,:,0], 0))
    indices = np.logical_and(background >= 0, bg_indices)
    n[indices] += 1
    background[indices] = (background[indices] +
                           (image[indices] - background[indices]) / n[indices])

    # initialize the new background elements
    indices = np.logical_and(background < 0, bg_indices)
    background[indices] = image[indices]
    n[indices] += 1

    return background, n


  @staticmethod
  def median_background_accumulator(scene, agg):
    raise NotImplementedError("TODO: median_background_accumulator")

  @staticmethod
  def image_shape_accumulator(scene, agg):
    """Get the shape of the images in the dataset.

    :param scene:
    :param agg:
    :returns:
    :rtype:

    """
    
    if agg is None:
      assert scene is not None
      image, (annotation, label) = scene
      agg = image.shape
      
    return agg
