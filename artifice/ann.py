"""Module for artifice's annotator, probably labelimg.

"""

import os
import logging
from time import sleep, strftime
import itertools
from operator import itemgetter
import numpy as np
from sortedcontainers import SortedList
from skimage.draw import circle
import tensorflow as tf

from artifice.shared import SharedDict
from artifice import dat, utils

logger = logging.getLogger('artifice')

class AnnotationInfo(SharedDict):
  """Maintain a sorted list or heap of (index, priority) pairs, as well as a
  list of annotated indices.

  'priorities' and 'sorted_priorities' are (idx, priority) pairs guaranteed to
  not have an annotation yet.

  'limbo' is a set of indices which have been selected for annotation but don't
  yet have an annotation. These cannot be added to priorities, but they cannot
  belong to 'annotated' yet, in case the annotator is killed before it saves
  their annotation. A new prioritizer should not clear limbo, but a new
  annotator should.

  'annotated' is a set of annotated indices. These are guaranteed to have
  annotations.

  Under this scheme, there may exist annotations for indices not yet added to
  'annotated' (in case the annotator is killed just after it saves the
  annotation but before it can move the indices out of limbo), but there will
  never be an index in 'annotated' that does not have an annotation.

  """
  
  def __init__(self, path, *, clear_priorities, clear_limbo):
    """Create a new annotation info dict.

    :param path: path to save this dict.
    :param clear_priorities: start with a fresh set of priorities. Typically,
    will be True for the prioritizer, must be False for the Annotator.
    :param clear_limbo: clear the limbo index set. Annotator should call with
    clear_limbo=True. The Prioritizer should always set clear_limbo=False.

    """
    super().__init__(path)
    self.acquire()
    if self.get('annotated') is None:
      self['annotated'] = set()
    if (clear_limbo or self.get('limbo') is None):
      self['limbo'] = set()
    if (clear_priorities or self.get('priorities') is None or
        self.get('sorted_priorities') is None):
      self['priorities'] = dict()
      self['sorted_priorities'] = SortedList(key=itemgetter(1))
    self.release()

  def push(self, item):
    """Update sortec priorities with (idx, priority) item (or items).

    If item already present, updates it. No-op if idx already in 'annotated' or
    'limbo'.

    """
    items = utils.listwrap(item)
    self.acquire()
    for idx, priority in items:
      if idx in self['annotated'] or idx in self['limbo']:
        continue
      old_priority = self['priorities'].get(idx)
      if old_priority is not None:
        self['sorted_priorities'].remove((idx, old_priority))
      self['priorities'][idx] = priority
      self['sorted_priorities'].add((idx, priority))
    self.release()
    
  def pop(self):
    """Pop an example idx off the stack and add it to the 'limbo' set.

    :returns: the popped idx, or None if stack is empty

    """
    self.acquire()
    if self['priorities']:
      idx, _ = self['sorted_priorities'].pop()
      del self['priorities'][idx]
      self['limbo'].add(idx)
    else:
      idx = None
    self.release()
    return idx

  def finalize(self, idx):
    """Discard idx or idxs from 'limbo' and add to 'annotated'. 

    Note that these need not be in limbo (could have multiple annotators, or
    multiple prioritizers). Caller is responsible for making sure that all of
    these indices have actually been annotated.

    """
    idxs = utils.listwrap(idx)
    self.acquire()
    for idx in idxs:
      self['limbo'].discard(idx)
      self['annotated'].add(idx)
    self.release()
  
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
    self.info = AnnotationInfo(info_path, clear_priorities=False,
                               clear_limbo=True)
    self.annotated_dir = annotated_dir
    self.record_size = record_size
    self.sleep_duration = sleep_duration

  def _generate_record_name(self):
    return os.path.join(self.annotated_dir, strftime(
      f"%Y%m%d%H%m%S_size-{self.record_size}.tfrecord"))

  def run(self):
    for i in itertools.count():
      examples = []
      idxs = []
      for _ in range(self.record_size):
        idx = self.info.pop()
        while idx is None:
          logger.info("waiting {self.sleep_duration}s for more selections...")
          sleep(self.sleep_duration)
          idx = self.info.pop()
        entry = self.data_set.get_entry(idx)
        logger.info(f"annotating example {idx}...")
        examples.append(self.annotate(entry))
        idxs.append(idx)
      record_name = self._generate_record_name()
      dat.write_set(map(dat.proto_from_annotated_example, examples),
                    record_name)
      self.info.finalize(idxs)
      logger.info(f"saved {i}'th annotated set to {record_name}.")
      
  def annotate(self, entry):
    """Abstract method for annotating an example.

    :param entry: numpy-form entry in the dataset
    :returns: `(image, label, annotation)` tuple of numpy arrays
    :rtype: 

    """
    
    raise NotImplementedError("subclasses should implement")

  
class SimulatedAnnotator(Annotator):
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
    assert issubclass(type(self.data_set), dat.LabeledData)

    
class DiskAnnotator(SimulatedAnnotator):
  def annotate(self, entry):
    sleep(self.annotation_delay)
    image, label = entry[:2]
    annotation = -1 * np.ones((image.shape[0], image.shape[1], 1), np.float32)
    for i in range(label.shape[0]):
      rr, cc = circle(label[i,0], label[i,1], 8, shape=image.shape[:2])
      xs = []
      ys = []
      for x,y in zip(rr,cc):
        if image[x,y] >= 0.2:   # arbitrary threshold
          xs.append(x)
          ys.append(y)
      annotation[xs,ys] = i
    return image, label, annotation
