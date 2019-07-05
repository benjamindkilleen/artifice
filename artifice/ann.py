"""Module for artifice's annotator, probably labelimg.

"""

import os

class Annotator:
  """The annotator takes the examples off the annotation stack and annotates them.

  """
  
  def __init__(self, info):
    self.info = info

  def __call__(self):
    while True:
      info = self.shared_info.acquire()
      self.shared_info.save(info)
      
    
