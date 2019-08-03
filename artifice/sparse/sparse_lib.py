"""

   Sparse Blocks Network
   Copyright (c) 2017, Uber Technologies, Inc.

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.

"""

import numpy as np
from collections import namedtuple
import tensorflow as tf

from artifice.log import logger

def _compute_mask_padding(size, bcount, bsize, boffset, bstride):
  """Computes the padding for the reduce_mask operation.

  :param size: `[SZH, SZW]` list-like of ints, size of image
  :param bcount: `[BCH, BCW]` list of ints
  :param bsize:
  :param boffset:
  :param bstride:
  :returns: `pad_h, pad_w` for _pad_mask function, possibly negative.
  :rtype:

  """
  pad_h = [boffset[0], boffset[0] + bstride[0]*(bcount[0] - 1) + bsize[0] - size[0]]
  pad_w = [boffset[1], boffset[1] + bstride[1]*(bcount[1] - 1) + bsize[1] - size[1]]
  return pad_h, pad_w

def _pad_mask(mask, bcount, bsize, boffset, bstride):
  """Pad the mask for then

  :param mask: 4D tensor containing the mask.
  :param bcount:
  :param bsize:
  :param boffset:
  :param bstride:
  :returns: Padded (or cropped) mask
  :rtype: tf.Tensor

  """
  pad_h, pad_w = _compute_mask_padding(
    mask.shape[1:3], bcount, bsize, boffset, bstride)

  if pad_h[0] < 0:
    mask = mask[:, -pad_h[0]:, :, :]
    pad_h[0] = 0
  if pad_h[1] < 0:
    mask = mask[:, :-pad_h[1], :, :]
    mad_h[1] = 0
  if pad_w[0] < 0:
    mask = mask[:, :, -pad_w[0]:, :]
    pad_w[0] = 0
  if pad_w[1] < 0:
    mask = mask[:, :, :-pad_w[1], :]
    mad_w[1] = 0

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
  offsets = np.array([np.arange(i, i + bsize[1]) for i in range(bsize[0])], np.int32)
  offsets = tf.constant(offsets, tf.int32)
  offsets = tf.expand_dims(offsets, 0)
  offsets = tf.expand_dims(offsets, 3)
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
  offset = tf.constant([1, boffset[0], boffset[1]], dtype=tf.int32)
  scale = tf.constant([1, bstride[0], bstride[1]], dtype=tf.int32)
  indices = active_block_indices + offset
  indices *= scale                                       # [M, 3]
  indices = tf.expand_dims(indices, 1)
  indices = tf.expand_dims(indices, 2) # [M, 1, 1, 3]
  upsample_offsets = _compute_upsample_offsets(bsize) # [1, bsize[0], bsize[1], 1]
  indices += upsample_offsets # [M, bsize[0], bsize[1], 3]

  return indices


#################### tf primitive implementations ####################


def reduce_mask(mask,
                block_count, *,
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
  mask = _pad_mask(mask, block_count, bsize, boffset, bstride)
  mask = tf.nn.pool(
    mask,
    window_shape=bsize,
    pooling_type='AVG' if avgpool else 'MAX',
    padding='SAME',
    strides=bstride)
  logger.debug(f"mask values: {mask.numpy().min()} to {mask.numpy().max()}")
  mask = tf.squeeze(mask, axis=3)
  active_block_indices = tf.where(mask > tf.constant(tol, mask.dtype))
  active_block_indices = tf.cast(active_block_indices, tf.int32)
  bin_counts = tf.shape(active_block_indices)[0]
  logger.debug(f"bin_counts: {bin_counts.numpy()}")
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
  indices = _upsample_block_indices(
    active_block_indices,
    bsize,
    boffset,
    bstride)
  blocks = tf.gather_nd(inputs, indices)
  blocks = tf.reshape(blocks, [bin_counts, bsize[0], bsize[1], tf.shape(inputs)[3]])
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

  Note that currently this only uses `outputs.shape` to scatter onto a tensor of zeros.

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
  indices = _upsample_block_indices(
    active_block_indices,
    bsize,
    boffset,
    bstride)                    # [M, bsize[0], bsize[1], 3]

  if add:
    raise NotImplementedError
  else:
    # if indices.shape[1] != blocks.shape[1] and indices.shape[2] != indices.shape[2]:
    #   raise ValueError(f'indices and blocks have incompatible shapes: '
    #                    f'{indices.shape} vs {blocks.shape}')
    outputs = tf.cond(
      tf.cast(tf.shape(blocks)[0], tf.bool),
      lambda: tf.scatter_nd(indices, blocks, tf.shape(outputs)),
      lambda: outputs)

  return outputs

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
