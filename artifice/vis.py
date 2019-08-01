"""Utils for visualizing artifice output. (Mostly for testing).

TODO: make `show` functions wrappers around `plot` functions, which can be
called without clearing the matplotlib buffer.

"""

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

from artifice.log import logger
from artifice import utils


_show = True
def set_show(val):
  global _show
  _show = val
  if not val:
    mpl.use('Agg')
    plt.ioff()

    
def show(fname=None, save=False):
  """Show the figure currently in matplotlib or save it, if not self.show.

  If no fname provided, and self.show is False, then closes the figure. If save
  is True, figure is saved regardless of show.

  """
  if _show and not save:
    logger.info("showing figure...")
    plt.show()
  elif fname is None:
    logger.warning("Cannot save figure. Did you forget to set --show?")
    plt.close()
  else:
    plt.savefig(fname)
    logger.info(f"saved figure to {fname}.")


def plot_image(*images, columns=10, ticks=True, scale=20, colorbar=False,
               cmap='gray', cram=False, **kwargs):
  cmaps = utils.listify(cmap, len(images))
  columns = min(columns, len(images))
  rows = max(1, len(images) // columns)
  fig, axes = plt.subplots(rows, columns, squeeze=False,
                           figsize=(scale, scale*rows/columns))
  for i, image in enumerate(images):
    ax = axes[i // columns, i % columns]
    if image is None:
      ax.axis('off')
      continue
    im = ax.imshow(np.squeeze(image), cmap=cmaps[i], **kwargs)
    if colorbar:
      fig.colorbar(im, ax=ax, orientation='horizontal', fraction=0.046, pad=0.04)
  for ax in axes.ravel():
    if not ticks:
      ax.axis('off')
    ax.set_aspect('equal')
  if cram:
    fig.subplots_adjust(wspace=0, hspace=0)
  return fig, axes


def plot_hist(hist):
  fig, axes = plt.subplots(2, 1)
  for name, values in hist.items():
    if type(values) is not list:
      continue
    if 'loss' in name:
      axes[0].plot(values, label=name)

  axes[0].set_title("Loss (Weigted MSE)")
  axes[0].set_xlabel("Epoch")
  axes[0].set_ylabel("Loss")

  axes[1].set_title("Mean Absolute Error")
  axes[1].set_xlabel("Epoch")
  axes[1].set_ylabel("MAE")
  fig.suptitle("Training")
  return fig, axes
