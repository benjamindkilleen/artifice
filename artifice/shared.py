
from os.path import exists
import json
from filelock import Timeout, FileLock
  
class SharedDict(dict):
  """Maintain a dict. Doesn't allow reads or writes unless the dict is
  acquired.

  """
  def __init__(self, path):
    self.path = path
    if not exists(path):
      self._save()
    self.lock = FileLock(self.path + ".lock")
    super().__init__()

  def __getitem__(self, key):
    if not self.lock.is_locked:
      raise RuntimeError("SharedDict: call acquire() before accessing dict.")
    return super().__getitem__(key)

  def __setitem__(self, key, val):
    if not self.lock.is_locked:
      raise RuntimeError("SharedDict: call acquire() before accessing dict.")
    return super().__setitem__(key, val)

  def _save(self):
    with open(self.path, 'w') as f:
      f.write(json.dumps(self))

  def _load(self):
    with open(self.path, 'r') as f:
      self.update(json.loads(f.read()))
  
  def acquire(self):
    """Acquire this dictionary in order to modify it.

    Loads the dictionary from the file and updates `self` with it, so if some
    keys were deleted by another process, they will be reinserted when this
    process closes. This should not be a problem, as most use-cases won't be
    deleting keys, but something to be aware of.

    """
    self.lock.acquire()
    self._load()
    return self

  def release(self):
    """Release this dictionary, allowing other processes to acquire it.

    """
    self._save()
    self.lock.release()
