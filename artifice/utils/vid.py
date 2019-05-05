"""Make a nice little video writer."""

import numpy as np
import subprocess as sp
import logging
import os
import matplotlib.pyplot as plt
from artifice.utils import img

logger = logging.getLogger('artifice')

class MP4Writer:
  def __init__(self, fname, shape=None, fps=30):
    """Write frames to a video.

    If shape is provided, opens the process here. Otherwise, process is opened
    on the first call to `write()`.

    :param fname: file to write the mp4 to
    :param shape: (optional) shape of the video. For rgb, include number of channels.
    :param fps: frames per second. Default is 30

    """
    self.fname = fname
    self.fps = fps
    self.shape = shape
    if shape is not None:
      self.open(shape)
      
  def open(self, shape):
    """FIXME! briefly describe function

    :param shape: 
    :returns: 
    :rtype: 

    """
    self.shape = tuple(shape)
    if len(self.shape) == 2:
      fmt = 'gray'
    elif len(self.shape) == 3 and self.shape[2] == 1:
      fmt = 'gray'
    elif len(self.shape) == 3 and self.shape[2] == 3:
      fmt = 'rgba'
    elif len(self.shape) == 3 and self.shape[2] == 4:
      fmt = 'rgba'
    else:
      raise ValueError(f"Unrecognized shape: {self.shape}.")

    cmd = [
      'ffmpeg',
      '-y',                     # overwrite existing file
      '-f', 'rawvideo',
      '-vcodec', 'rawvideo',
      '-s', f'{self.shape[1]}x{self.shape[0]}', # WxH
      '-pix_fmt', fmt,                          # byte format
      '-r', str(self.fps),                      # frames per second
      '-i', '-',                                # input from pipe
      '-an',                                    # no audio
      '-b', '40000k',                           # bitrate, controls compression, TODO: customize
      '-vcodec', 'mpeg4',
      self.fname]

    logger.info(' '.join(cmd))
    self.log = open(self.fname.split('.')[0] + '.log', 'w')
    self.proc = sp.Popen(cmd, stdin=sp.PIPE, stderr=self.log)
    
  def write(self, frame):
    if self.shape is None:
      self.open(frame.shape)
    frame = img.as_uint(frame)
    if frame.shape[2] == 3:
      frame = np.insert(frame, 3, np.zeros((frame.shape[:2])), axis=2)
    self.proc.stdin.write(frame.tobytes())

  def write_fig(self, fig, close=True):
    """Write the matplotlib figure to the feed.

    :param fig: matplotlib figure
    :param close: close the figure when done with it.

    """
    fig.canvas.draw()
    self.write(np.array(fig.canvas.renderer._renderer))
    if close:
      plt.close()
    
  def close(self):
    self.proc.stdin.close()
    self.proc.wait()
    self.log.close()
    del self
