"""Module for artifice's annotator, probably labelimg.

"""

import os
from time import sleep, strftime
from sortedcontainers import SortedList
from skimage.draw import circle

from artifice.shared import SharedDict
from artifice import dat, utils


class AnnotationInfo(SharedDict):
  """Maintain a sorted list or heap of (index, priority) pairs, as well as a
  list of annotated indices.

  """
  
  def __init__(self, path, clear=False):
    """Create a new annotation info dict.

    :param path: path to save this dict.
    :param clear: start with a fresh heap. 

    """
    super().__init__(path)
    self.acquire()
    if self.get('annotated') is None:
      self['annotated'] = set()
    if clear:
      self['selections'] = dict()
      self['sorted_selections'] = SortedList(key=lambda t : t[1])
    self.release()

  def push(self, item):
    """Update sortec selections with (idx, priority) item (or items).

    If item already present, updates it. No-op if idx already annotated.

    """
    items = utils.listwrap(item)
    self.acquire()
    for idx, priority in items:
      if idx in self['annotated']:
        continue
      old_priority = self['selections'].get(idx)
      if old_priority is not None:
        self['ordered_selections'].remove((idx, old_priority))
      self['selections'].update((idx, priority))
      self['sorted_selections'].update((idx, priority))
    self.release()
    
  def pop(self):
    """Pop an example idx off the stack and add it to the annotated list.

    :returns: the popped idx, or None if stack is empty

    """
    self.acquire()
    if self['annotated'] is None:
      self['annotated'] = set()
    if self['selections']:
      idx, _ = self['sorted_selections'].popitem()
      del self['selections'][idx]
      self['annotated'].add(idx)
    else:
      idx = None
    self.release()
    return idx

  
class Annotator:
  """The annotator takes the examples off the annotation stack and annotates
  them, if they have not already been annotated.

  Subclasses should implement the annotate method, which takes an index or list
  of indices and returns the (image, label, annotation) annotated example in
  numpy form.

  """
  
  def __init__(self, data_set, *, info_path, annotated_dir,
               record_size=10, sleep_duration=15):
    """Annotator abstract class

    :param data_set: ArtificeData object to annotate. Typically an UnlabeledData
    object, but could be LabeledData, if simulating annotation.
    :param info_path: 
    :param annotated_dir: directory to store annotation tfrecords in
    :param record_size: number of examples to save in each
    tfrecord. Must be small enough for `record_size` annotated examples to fit
    in memory.
    :param sleep_duration: seconds to sleep if no examples on the annotation
    stack, before checking again.
    :returns: 
    :rtype: 

    """
    self.data_set = data_set
    self.info = AnnotationInfo(info_path, clear=False)
    self.annotated_dir = annotated_dir
    self.record_size = record_size

  def _generate_record_name(self):
    return os.path.join(self.annotated_dir, strftime(
      f"%Y%m%d%H%m%S_size-{self.record_size}.tfrecord"))

  def run(self):
    while True:
      examples = []
      for _ in range(self.record_size):
        idx = self.info.pop()
        while idx is None:
          sleep(self.sleep_duration)
          idx = self.info.pop()
        entry = self.data_set.get_entry(idx)
        examples.append(self.annotate(entry))
      dataset = tf.data.Dataset.from_tensor_slices(examples)
      dat.save_dataset(self._generate_record_name(), dataset,
                       serialize=dat.proto_from_annotated_example)
      
  def annotate(self, entry):
    """Abstract method for annotating an example.

    :param entry: numpy-form entry in the dataset
    :returns: `(image, label, annotation)` tuple of numpy arrays
    :rtype: 

    """
    
    raise NotImplementedError("subclasses should implement")

  
def SimulatedAnnotator(Annotator):
  """Simulate a human annotator.

  Expects self.data_set to be a subclass of artifice LabeledData. Abstract
  method `annotation_from_label()` used to convert a label to an annotation.

  Loads all the labels in the dataset, which must fit in memory.

  """
  def __init__(self, *args, annotation_delay=60, **kwargs):
    """Simulate a human annotator.

    :param annotation_delay: time that the simulated annotator spends for each
    example. 

    """
    self.annotation_delay = annotation_delay
    super().__init__(*args, **kwargs)
    assert issubclass(type(self.data_set), LabeledData)

    
class DiskAnnotator(SimulatedAnnotator)
  def annotate(self, entry):
    sleep(self.annotation_delay)
    image, label = entry[:2]
    annotation = -1 * np.ones((image.shape[0], image.shape[1], 1), np.float32)
    for i in range(label.shape[0]):
      rr, cc = circle(label[i,0], label[i,1], 5, shape=image.shape[:2])
      xs = []
      ys = []
      for x,y in zip(rr,cc):
        if image[x,y] >= 0.4:   # arbitrary threshold
          xs.append(x)
          ys.append(y)
      annotation[xs,ys] = i
    return image, label, annotation
