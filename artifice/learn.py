import tensorflow as tf
import numpy as np
from artifice import mod, dat
from os.path import join
import logging

logger = logging.getLogger('artifice')


class Detector():
  def __init__(self, model):
    self.model = model

  def predict(self, data, steps=None, verbose=1):
    """Yield reassembled fields from the data.

    Requires batch_size to be a multiple of num_tiles

    :param data: dat.Data set
    :param steps: number of image batches to do at once
    :param verbose: 
    :returns: 
    :rtype: 

    """
    if steps is None:
      steps = data.size // data.batch_size
    round_size = steps*data.batch_size
    rounds = int(np.ceil(data.size / round_size))
    for r in range(rounds):
      logger.info(f"predicting round {r}...")
      tiles = self.model.predict(data.eval_input.skip(r*round_size),
                                 steps=steps, verbose=verbose)
      logger.debug(f"tiles: {tiles.shape}")
      for field in data.untile(tiles):
        yield field

  def detect(self, data):
    """Detect objects in the reassembled fields.
    
    :param data: dat.Data set
    :returns: matched_detections
    
    """
    detections = np.zeros((data.size, data.num_objects, 3), np.float32)
    for i, field in enumerate(self.predict(data)):
      detections[i] = data.from_field(field)
    return dat.match_detections(detections, data.labels)

  def __call__(self, *args, **kwargs):
    return self.detect(*args, **kwargs)

class ActiveLearner(Detector):
  def __init__(self, model, annotated_set_dir,
               num_candidates=1000,
               query_size=1,
               **kwargs):
    """Wrapper around model's that performs active learning on dat.Data objects.

    :param model: a `mod.Model`
    :param annotated_set_dir: directory to save annotated sets
    :param num_candidates: max number of images to consider for query
    :param query_size: number of images in each query
    :returns: 
    :rtype: 

    """
    super().__init__(model, **kwargs)
    self.annotated_set_dir = annotated_set_dir
    self.num_candidates = num_candidates
    self.query_size = query_size
    self.candidate_idx = 0

  def compute_uncertainty(self, field, data):
    """Quantify the uncertainty of the predicted field.

    One method: compute the squared error between `field` and a constructed
    field with max peaks at the same locations, using `data.from_field` and
    `data.to_field`.

    :param field: predicted field
    :param data: `dat.Data` from which `field` was predicted
    :returns: uncertainty measure
    :rtype: float

    """
    detection = data.from_field(field)
    other_field = data.to_numpy_field(detection)
    uncertainty = np.linalg.norm(field - other_field)
    return uncertainty

  def choose_query(self, unlabeled_set):
    """Run inference on the unlabeled set, choose `query_size` images for
    annotation.

    Runs inference for at most num_candidates at once, starting where the
    previous call left off.

    :param unlabeled_set: the unlabeled set
    :returns: list of indices into the dataset, at most `self.query_size`

    """
    query = []
    candidate_set = unlabeled_set.skip(self.candidate_idx).take(self.num_candidates)
    
    for i, field in enumerate(self.predict(candidate_set)):
      idx = i + self.candidate_idx
      uncertainty = self.compute_uncertainty(candidate_set, field)
      query.append((idx, uncertainty))
      query.sort(key=lambda t: t[1], reverse=True)
      query = query[:self.query_size]

    self.candidate_idx += self.num_candidates
    return query
  
  def fit(self, unlabeled_set, epochs=1, **kwargs):
    """Fit using an active learning approach to the unlabeled data.

    TODO: use already annotated examples, if they exist.
    TODO: determine better method for training size in between queries

    :param unlabeled_set: a dat.Data object with an `annotate()` method.
    :param epochs: number of epochs to run
    :returns: annotated_set created during active_learning
    :rtype: 

    """
    sampling = np.zeros(unlabeled_set.size, np.int64)
    
    for epoch in range(epochs):
      logger.info(f"Epoch {epoch} / {epochs}")
      if epoch == 0:
        sampling[0] = 1
      else:
        query = self.choose_query(unlabeled_set)
        logger.debug(f"querying {query}...")
        sampling[query] += 1
      annotated_set_path = join(
        self.annotated_set_dir, f'annotated_set_{epoch}.tfrecord')
      annotated_set = unlabeled_set.sample_and_annotate(
        sampling, annotated_set_path)
      self.model.fit(annotated_set.training_input,
                     epochs=epoch+1, initial_epoch=epoch, **kwargs)
      
    return annotated_set
      
