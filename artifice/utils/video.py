"""Make a nice little video writer."""

import numpy as np
import subprocess as sp
import logging
import os
import matplotlib.pyplot as plt

logger = logging.getLogger('artifice')


class MP4Writer:
  def __init__(self, fname, shape, fps=30):
    """Opens a process to start writing frames.

    :param fname: file to write the mp4 to
    :param shape: shape of the video. For rgb, include number of channels.
    :param fps: frames per second. Default is 30

    """

    self.fname = fname
    self.shape = tuple(shape)
    self.fps = fps
    if len(self.shape) == 2:
      self.fmt = 'gray'
    elif len(self.shape) == 3 and self.shape[2] == 3:
      self.fmt = 'rgba'
    elif len(self.shape) == 3 and self.shape[2] == 4:
      self.fmt = 'rgba'
    else:
      raise ValueError(f"Unrecognized shape: {self.shape}.")

    cmd = [
      'ffmpeg',
      '-y',                     # overwrite existing file
      '-f', 'rawvideo',
      '-vcodec', 'rawvideo',      
      '-s', f'{self.shape[1]}x{self.shape[0]}', # WxH
      '-pix_fmt', 'rgba',                       # byte format
      '-r', str(self.fps),                      # frames per second
      '-i', '-',                                # input from pipe
      '-an',                                    # no audio
      '-vcodec', 'mpeg4',
      self.fname]

    logger.info(' '.join(cmd))
    self.log = open(self.fname.split('.')[0] + '.log', 'w')
    self.proc = sp.Popen(cmd, stdin=sp.PIPE, stderr=self.log)
    
  def write(self, frame):
    frame = frame.astype(np.uint8)
    if frame.shape[2] == 3:
      frame = np.insert(frame, 3, np.zeros((frame.shape[:2])), axis=2)
    self.proc.stdin.write(frame.tobytes())

  def close(self):
    self.proc.stdin.close()
    self.proc.wait()
    self.log.close()
