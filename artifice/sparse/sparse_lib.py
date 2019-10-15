"""Sparse ops implementation using tf primitives.
"""

import numpy as np
from collections import namedtuple
import tensorflow as tf

from artifice.log import logger  # noqa: unused
from artifice import utils


def _compute_bcount(size, bstride):
  return [utils.divup(size[0], bstride[0]),
          utils.divup(size[1], bstride[1])]


def _compute_input_padding(size, bcount, bsize, boffset, bstride):
  """Computes the padding for the operation.

  :param size: `[SZH, SZW]` list-like of ints, size of image
  :param bcount: `[BCH, BCW]` list of ints
  :param bsize:
  :param boffset:
  :param bstride:
  :returns: `pad_h, pad_w` for _pad_inputs function, possibly negative.
  :rtype:

  """
  pad_h = [boffset[0],
           boffset[0] + bstride[0] * bcount[0] + bsize[0] - size[0]]
  pad_w = [boffset[1],
           boffset[1] + bstride[1] * bcount[1] + bsize[1] - size[1]]
  return pad_h, pad_w


def _pad_inputs(mask, bcount, bsize, boffset, bstride):
  """Pad the inputs for then

  :param mask: 4D tensor containing the inputs.
  :param bcount:
  :param bsize:
  :param boffset:
  :param bstride:
  :returns: Padded (or cropped) mask
  :rtype: tf.Tensor

  """
  pad_h, pad_w = _compute_input_padding(
    mask.shape[1:3], bcount, bsize, boffset, bstride)

  if pad_h[0] < 0:
    mask = mask[:, -pad_h[0]:, :, :]
    pad_h[0] = 0
  if pad_h[1] < 0:
    mask = mask[:, :-pad_h[1], :, :]
    pad_h[1] = 0
  if pad_w[0] < 0:
    mask = mask[:, :, -pad_w[0]:, :]
    pad_w[0] = 0
  if pad_w[1] < 0:
    mask = mask[:, :, :-pad_w[1], :]
    pad_w[1] = 0

  pad_n = pad_c = [0, 0]
  return tf.pad(mask, [pad_n, pad_h, pad_w, pad_c])


def _compute_upsample_offsets(bsize):
  """Compute the offsets for blocks with bsize.

  Assumes that the given coordinate is at the top left of the block.

  So for example, if the block size were [3, 4], the returned offsets would be:
```
  [[[0], [1], [2], [3]],
   [[1], [2], [3], [4]],
   [[2], [3], [4], [5]]]
```
  which has shape [1, 3, 4, 1]

  :param bsize: `[BSZH, BSZW]` size of the blocks.
  :returns: [1, bsize[0], bsize[1], 1] array of offsets to upsample a set of
  block_indices.
  :rtype: tf.Tensor

  """
  offsets = np.array([[[0, i, j] for j in range(bsize[1])]
                      for i in range(bsize[0])], np.int32)
  # todo: fix this so it isn't crazy
  offsets = tf.constant(offsets, tf.int32)
  offsets = tf.expand_dims(offsets, 0)
  return offsets


def _upsample_block_indices(active_block_indices, bsize, boffset, bstride):
  """Upsamples the indices to have all indices in a rectangle.

  :param active_block_indices: [M,3] Tensor. Corresponds to top left coordinate
  after offset and scaling.
  :param bsize: block size
  :param boffset:
  :param bstride:
  :returns: [M, bsize[0], bsize[1], 3] locations of all pixels in the blocks.
  :rtype:

  """
  ops = []
  logger.debug(f"bsize: {bsize}")
  logger.debug(f"bstride: {bstride}")
  # ops.append(tf.print(active_block_indices, summarize=-1))
  offset = tf.constant([0, boffset[0], boffset[1]], dtype=tf.int32)
  scale = tf.constant([1, bstride[0], bstride[1]], dtype=tf.int32)
  indices = tf.cast(active_block_indices, tf.int32) + offset
  indices *= scale                                       # [M, 3]
  indices = tf.expand_dims(indices, 1)
  indices = tf.expand_dims(indices, 2)  # [M, 1, 1, 3]
  upsample_offsets = _compute_upsample_offsets(
    bsize)  # [1, bsize[0], bsize[1], 3]
  logger.debug(f"indices: {indices.shape}")
  logger.debug(f"upsample_offsets: {upsample_offsets.shape}")
  # ops.append(tf.print(indices, summarize=-1))
  # ops.append(tf.print(upsample_offsets, summarize=-1))
  with tf.control_dependencies(ops):
    indices += upsample_offsets  # [M, bsize[0], bsize[1], 3]

  return indices


"""
TensorFlow primitive implementations.
"""


def reduce_mask(mask,
                bcount, *,
                bsize,
                boffset,
                bstride,
                tol=0.5,
                avgpool=False):
  """Reduce the mask to namedtuple `(bin_counts, active_block_indices)`, indices.

  :param mask:
  :param block_count:
  :param bsize:
  :param boffset:
  :param bstride:
  :param tol:
  :param avgpool:
  :returns:
  :rtype:

  """
  logger.debug(f"mask: {mask.shape}")
  mask = _pad_inputs(mask, bcount, bsize, boffset, bstride)
  logger.debug(f"padded mask: {mask.shape}")
  mask = tf.nn.pool(
    mask,
    window_shape=bsize,
    pooling_type='AVG' if avgpool else 'MAX',
    padding='SAME',
    strides=bstride)
  mask = tf.squeeze(mask, axis=3)
  active_block_indices = tf.where(mask > tf.constant(tol, mask.dtype))
  active_block_indices = tf.cast(active_block_indices, tf.int32)
  bin_counts = tf.shape(active_block_indices)[0]
  Indices = namedtuple('Indices', ['active_block_indices', 'bin_counts'])
  return Indices(active_block_indices, bin_counts)


def gather(
        inputs,
        bin_counts,
        active_block_indices, *,
        bsize,
        boffset,
        bstride):
  """FIXME! briefly describe function

  :param inputs:
  :param bin_counts: number of blocks?
  :param active_block_indices:
  :param bsize:
  :param boffset:
  :param bstride:
  :returns:
  :rtype:

  """
  logger.debug(f"inputs: {inputs.shape}")
  size = inputs.shape[1:3]
  bcount = _compute_bcount(size, bstride)
  inputs = _pad_inputs(inputs, bcount, bsize, boffset, bstride)

  logger.debug(f"padded inputs: {inputs.shape}")
  indices = _upsample_block_indices(
    active_block_indices,
    bsize,
    boffset,
    bstride)
  ops = []
  # ops.append(tf.print(indices, summarize=-1))
  logger.debug(f"gather indices: {indices.shape}")
  with tf.control_dependencies(ops):
    blocks = tf.gather_nd(inputs, indices)  # todo: fix index error
  blocks = tf.reshape(
    blocks, [bin_counts, bsize[0], bsize[1], tf.shape(inputs)[3]])
  return blocks


def scatter(
        blocks,
        bin_counts,                 # pylint: disable=unused-argument
        active_block_indices,
        outputs, *,
        bsize,
        boffset,
        bstride,
        add=False):
  """Scatter the blocks back onto outputs.

  Note that currently this only uses `outputs.shape` to scatter onto a tensor
  of zeros.

  In tf >= 1.14, the functions tf.tensor_scatter_nd_update and
  tf.tensor_scatter_nd_add would overcome this barrier.

  :param blocks: [M, bsize[0], bsize[1], C]
  :param bin_counts:
  :param active_block_indices:
  :param outputs: [N, H, W, C]
  :param bsize:
  :param boffset:
  :param bstride:
  :returns:
  :rtype:

  """
  size = outputs.shape[1:3]
  bcount = _compute_bcount(size, bstride)
  outputs = _pad_inputs(outputs, bcount, bsize, boffset, bstride)

  indices = _upsample_block_indices(
    active_block_indices,
    bsize,
    boffset,
    bstride)                    # [M, bsize[0], bsize[1], 3]

  if add:
    raise NotImplementedError
  else:
    outputs = tf.case(
      [(tf.equal(tf.shape(blocks)[0], tf.constant(0, tf.int32)),
        (lambda: outputs))],
      default=lambda: tf.scatter_nd(indices, blocks, tf.shape(outputs)))

  return outputs[:, :size[0], :size[1], :]


def scatter_var(
        blocks,
        bin_counts,                 # pylint: disable=unused-argument
        active_block_indices,
        outputs, *,
        bsize,
        boffset,
        bstride,
        add=False):

  raise NotImplementedError("no gradient for sparse_lib.scatter_var")

  indices = _upsample_block_indices(
    active_block_indices,
    bsize,
    boffset,
    bstride)                    # [M, bsize[0], bsize[1], 3]

  if add:
    outputs = tf.scatter_nd_add(outputs, indices, blocks)
  else:
    outputs = tf.scatter_nd_update(outputs, indices, blocks)

  return outputs
