"""Utils for visualizing artifice output. (Mostly for testing).

TODO: make `show` functions wrappers around `plot` functions, which can be
called without clearing the matplotlib buffer.

"""

from glob import glob
from os.path import join, basename
from stringcase import pascalcase, titlecase
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

  :param fname: name of the file to save to.
  :param save: whether to save the file.

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
                           figsize=(scale, scale * rows / columns))
  for i, image in enumerate(images):
    ax = axes[i // columns, i % columns]
    if image is None:
      ax.axis('off')
      continue
    im = ax.imshow(np.squeeze(image), cmap=cmaps[i], **kwargs)
    if colorbar:
      fig.colorbar(im, ax=ax, orientation='horizontal',
                   fraction=0.046, pad=0.04)
  for ax in axes.ravel():
    if not ticks:
      ax.axis('off')
    ax.set_aspect('equal')
  if cram:
    fig.subplots_adjust(wspace=0, hspace=0)
  return fig, axes


def plot_hists_from_dir(model_root, columns=10, scale=20):
  """Plot all the histories in `model_dir`.

  For each named property, creates a plot with all the model histories that had
  that named property (loss or metric)

  :returns: fig, axes

  """

  history_fnames = glob(join(model_root, '*history.json'))
  logger.debug(f"history_fnames: {history_fnames}")
  if not history_fnames:
    logger.warning(f"no history saved at {model_root}")
    return None, None

  hist_data = {}                # {property_name -> {model_name -> [values]}}
  for fname in history_fnames:
    hist = utils.json_load(fname)
    model_name = pascalcase(basename(fname).replace('_history.json', ''))
    for prop_name, values in hist.items():
      if not isinstance(values, list):
        continue
      if hist_data.get(prop_name) is None:
        hist_data[prop_name] = {}
      hist_data[prop_name][model_name] = values

  columns = min(columns, len(hist_data))
  rows = max(1, len(hist_data) // columns)
  fig, axes = plt.subplots(rows, columns, squeeze=False,
                           figsize=(scale, scale * rows / columns))

  for i, (prop_name, prop_data) in enumerate(hist_data.items()):
    ax = axes[i // columns, i % columns]
    for model_name, values in prop_data.items():
      ax.plot(values, '-', label=model_name)
    ax.set_title(titlecase(prop_name))
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
  fig.suptitle("Training")
  plt.legend()
  return fig, axes


if __name__ == '__main__':
  print('Hello world!')
