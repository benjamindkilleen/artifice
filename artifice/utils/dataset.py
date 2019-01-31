"""Functions for reading and writing datasets in tfrecords, as needed by
artifice and test_utils. A "scene" is the info needed for a single, labeled
example. It should always be a 2-tuple, usually (image, annotation),
although it could also be (image, (annotation, label)).

"""

import numpy as np
import tensorflow as tf
from artifice.utils import augment, tform, inpaint
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
  image, (annotation, label) = scene
  raise NotImplementedError("implement proto_from_tensor_scene")
  

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
  image = tf.to_float(image) / 255.

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

def save_dataset(record_name, dataset,
                 encode_entry=proto_from_tensor_scene):
  """FIXME! briefly describe function

  :param record_name: 
  :param dataset: 
  :param encode_entry: 
  :returns: 
  :rtype: 

  """
  logger.info(f"Writing dataset to {record_name}")
  encoded_dataset = dataset.map(encode_entry,
                                num_parallel_calls=self.num_parallel_calls)
  writer = tf.data.experimental.TFRecordWriter(record_name)
  write_op = writer.write(encoded_dataset)
  with tf.Session() as sess:
    sess.run(write_op)


class Data(object):
  """A Data object contains scenes, as defined above. It's mainly useful as a
  way to save and then distribute data.

  Attributes:
  :_dataset: the dataset associated with the object, including any
    augmentations with ObjectTransformations.
  :_original_dataset: the dataset without any augmentations. This is used mainly
    for DataAugmenter, which will discard the _dataset object.

  Data subclasses should be instantiated from one another in chains, rarely if
  ever invoking the Data class directly. Instead, create a DataAugmenter by
  passing it a DataInput or TrainDataInput object. After running the
  DataAugmenter instance (either by run() or __call__()).

  Subclasses which do not want to accept keyword arguments from a parent should
  overwrite them with the **kwargs passed to them directly rather than using
  `self._kwargs`, as this is left alone.

  Data objects that support augmentation (mainly TrainDataInput and
  DataAugmenter) use the self.augmentation object. Data objects pass '+' and
  '*' operations onto their augmentations. The augmentation is not preserved
  when transforming between Data subclasses (unless explicitly passed as a
  keyword arg).

  """

  def __init__(self, data, **kwargs):
    """args:
    :param data: a tf.data.Dataset OR tfrecord file name(s) OR another Data
      subclass. In this case, the other object's _dataset is adopted, allows
      kwargs to be overwritten.
    """
    if issubclass(type(data), Data):
      self._kwargs = data._kwargs.copy()
      self._kwargs.update(kwargs)
    else:
      self._kwargs = kwargs.copy()

    self.batch_size = self._kwargs.get('batch_size', 1)
    self.num_parallel_calls = self._kwargs.get('num_parallel_calls', None)
    self.parse_entry = self._kwargs.get('parse_entry', None)
    self.image_shape = self._kwargs.get('image_shape', None)
    self.augmentation = kwargs.get('augmentation', augment.identity)

    if issubclass(type(data), Data):
      self._dataset = data._dataset
      self._original_dataset = getattr(data, '_original_dataset', data._dataset)
    elif issubclass(type(data), tf.data.Dataset):
      self._dataset = self._original_dataset = dataset
    elif type(data) == str or hasattr(data, '__iter__'):
      self._dataset = self._original_dataset = load_dataset(
        data, parse_entry=self.parse_entry, 
        num_parallel_calls=self.num_parallel_calls)
    else:
      raise ValueError(f"unrecognized type '{type(data)}'")


  def save(self, record_name, **kwargs):
    """Save the dataset to record_name."""
    save_dataset(record_name, self._dataset, **kwargs)


  def save_original(self, record_name, **kwargs):
    """Save the original dataset to record_name."""
    save_dataset(record_name, self._original_dataset, **kwargs)
    

  def accumulate(self, accumulators):
    """Eagerly runs the accumulators across the original_dataset to gather metadata.
    
    An accumulator function should take a `scene` and an `aggregate` object. On
    the first call, `aggregate` will be None. Afterward, each accumulator will
    be passed the output from its previous call as `scene`. On the final call,
    `scene` will be None, allowing for post-processing.
    
    :param accumulators: a dictionary mapping names to accumulator functions
    :returns: aggregates from the accumulations, with the same keys as
      `accumulators`.
    :rtype: dict

    """

    iterator = self._original_dataset.make_initializable_iterator()
    next_scene = iterator.get_next()
    logger.debug("made label iterator (?)")
    
    aggregates = dict.fromkeys(accumulators.keys())
    
    with tf.Session() as sess:
      sess.run(iterator.initializer)
      logger.info("initialized iterator, starting accumulation...")
      while True:
        try:
          scene = sess.run(next_scene)
          for k, accumulator in accumulators.items():
            aggregates[k] = accumulator(scene, aggregates[k])
        except tf.errors.OutOfRangeError:
          break
    
    logger.info("finished accumulation")
    for k, accumulator in accumulators.items():
      aggregates[k] = accumulator(None, aggregates[k])
    return aggregates

  def add_augmentation(self, aug):
    self.augmentation += aug
    return self

  def __add__(self, other):
    if issubclass(type(other), augment.Augmentation):
      return self.add_augmentation(aug)
    elif issubclass(type(other), Data):
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
    elif issubclass(type(other), Data):
      return self.mul_augmentation(other.augmentation)
    else:
      raise ValueError(f"unrecognized type '{type(aug)}'")


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

  :param labels: label set. Should be used only when the tfrecord is unlabeled.
  :param background_image: used for inpainting transformed examples.
  :param prime_examples: indices of examples which are prime for transformation. In
    principle, a prime example can be any example. When a new, transformed example
    is introduced, it will be created from a random example in `prime_examples`.

  """

  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)
    
    self._accs = {'labels' : DataAugmenter.label_accumulator,
                  'background_image' : DataAugmenter.mean_background_accumulator,
                  'prime_examples' : DataAugmenter.prime_examples_accumulator}

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


  def compute_new_labels(self):
    """Create new labels to generate examples from.

    TODO: implement label smoothing, incorporating label-space boundaries.

    :returns: 
    :rtype: 

    """
    return tf.expand_dims(self.labels[0], 0)

  def run(self, record_name='augmented.tfrecord'):
    """Run the transformations on self._original_dataset and save novel examples
    in a new tfrecord file. Update self._dataset to incorporate both datasets.
    
    :param record_name: saves the augmented set. Defaults to
      'augmented.tfrecord' in the current directory.
    :returns: self
    :rtype: DataAugmenter

    """

    """TODO:
    1. Determine the new labels that would more evenly sample the label-space
    2. For each of these, create an augmentation transformation, drawing an
       example from prime_examples at random.
    3. Run the augmentation on self._original_dataset 
    4. save the newly-created dataset in record_name, reload it (forces
       execution).
    5. update: self._dataset = self._original_dataset `concat` augmentations

    """
    new_labels = self.compute_new_labels()
    
    transformations = []
    for new_label in tf.unstack(new_labels, axis=0):
      transformations.append(tform.ObjectTransformation(
        new_label,
        which_examples=0,
        background_image=self.background_image))
    aug = augment.Augmentation(transformations)
    augmented = aug(self._original_dataset)
    save_dataset(record_name, augmented)
    # augmented = load_datset(record_name) # TODO: necessary?
    self._dataset = self._original_dataset.concatenate(augmented)
    
    return self

  def __call__(self, *args, **kwargs):
    return self.run(*args, **kwargs)

  @staticmethod
  def prime_examples_accumulator(scene, prime_examples):
    """Selects only the first examples as "prime."

    TODO: be more exact with this definition.

    :param scene: 
    :param prime_examples: 
    :returns: 
    :rtype: 

    """
    return [0]

    
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
      return tf.constant(DataAugmenter.fill_background(background), tf.float32)

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

    TODO: fix this function

    """
    if agg is None:
      image, (annotation, label) = scene
      return image.shape
    if scene is None:
      return agg
  
