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

from collections import namedtuple
import tensorflow as tf

from tf_conv_dims import calc_padding_4d

def _calc_block_strides(bsize, ksize, strides):
  """Calculates strides for blocks.

  :param bsize:     [list]        List of 4 int. Size of blocks, or downsample ratio.
  :param ksize:     [list]        List of 4 int. Sparse convolution kernel size.
  :param strides:   [list]        List of 4 int. Sparse convolution strides.

  :return           [list]        List of 4 int. Block strides.
  """
  return [1, bsize[1] - ksize[0] + strides[1], bsize[2] - ksize[1] + strides[2], 1]


def _pad_input(x, ksize, strides, padding, bsize=None, bstrides=None):
  """Pads the input tensor.

  Optional to pass in block strides. The right hand side padding will be increased
  if the last block does not fit in (no effect on the convolution results.

  :param x:        [Tensor]   [N, H, W, C]. input tensor, dtype float32.
  :param ksize:    [list]     List of 4 int. Sparse convolution kernel size.
  :param strides:  [list]     List of 4 int. Sparse convolution stride size.
  :param padding:  [string]   `VALID` or `SAME`, padding method for sparse convolution.
  :param bsize     [list]     List of 4 int. Block size. Optional.
  :param bstrides: [list]     List of 4 int. Block strides. Optional.

  :return          [Tensor]   [N, H+Ph, W+Pw, C]. Padded input tensor.

  """
  x_shape = tf.shape(x)
  if padding == 'SAME':
    pad_h0, pad_h1, pad_w0, pad_w1 = calc_padding_4d(x_shape, ksize, strides, padding)

    if bstrides is not None:
      # Here we do not use the standard padding on the right hand side.  If the
      # convolution results is larger than expected, the scatter function will
      # not use out-of-boundary points.
      assert bsize is not None, 'Must pass in bsize and bstrides together.'
      h = x_shape[1] + pad_h0 + pad_h1
      w = x_shape[2] + pad_w0 + pad_w1
      pad_h1 += tf.mod(-h + bsize[1], bstrides[1])
      pad_w1 += tf.mod(-w + bsize[2], bstrides[2])
    return tf.pad(x, [[0, 0], [pad_h0, pad_h1], [pad_w0, pad_w1], [0, 0]])
  else:
    if bstrides is not None:
      assert bsize is not None, 'Must pass in bsize and bstrides together.'
      h = x_shape[1]
      w = x_shape[2]
      pad_h1 = tf.mod(-h + bsize[1], bstrides[1])
      pad_w1 = tf.mod(-w + bsize[2], bstrides[2])
      return tf.cond(
        tf.logical_or(tf.greater(pad_h1, 0), tf.greater(pad_w1, 0)),
        lambda: tf.pad(x, [[0, 0], [0, pad_h1], [0, pad_w1], [0, 0]]), lambda: x)
    else:
      return x


def convert_mask_to_indices(mask, *, bsize, ksize, strides, padding, tol):
  """
  Converts a binary mask to sparse indices.

  :param mask:     [Tensor]   [N, H, W]. 1 indicates non-sparse locations. Dtype float32.
  :param bsize:    [list]     List of 4 int. Size of blocks, or downsample ratio.
  :param ksize:    [list]     List of 4 int. Sparse convolution kernel size.
  :param strides:  [list]     List of 4 int. Sparse convolution stride size.
                              Currently only supports when,
                              1) (bsize[1] - ksize[0]) % strides[1] == 0 and,
                              2) (bsize[2] - ksize[1]) % strides[2] == 0
  :param padding:  [string]   `VALID` or `SAME`, padding method for sparse convolution.
  :param tol:      [float]    Lower bound of occupancy for creating a rectangle.

  :return          [Tensor]   [M, 3]. Center locations (N, H, W) of M rectangles. Dtype int32.
  """
  ERR_MSG_RANK = 'Expect mask rank = 3'
  ERR_MSG_DIV = 'Expect `stride` divides `bsize` - `ksize`. stride {}, bsize {}, ksize {}.'
  ERR_MSG_DIM = 'Expect first and last dimensions of strides = 1. Dim {}.'

  assert len(mask.get_shape()) == 3, ERR_MSG_RANK
  assert type(bsize) in [list, tuple], '`bsize` needs to be a list or tuple.'
  assert type(ksize) in [list, tuple], '`ksize` needs to be a list or tuple.'
  assert type(strides) in [list, tuple], '`strides` needs to be a list or tuple.'
  assert (bsize[1] - ksize[0]) % strides[1] == 0, ERR_MSG_DIV.format(
    strides[1], bsize[1], ksize[0])
  assert (bsize[2] - ksize[1]) % strides[2] == 0, ERR_MSG_DIV.format(
    strides[2], bsize[2], ksize[1])
  assert strides[0] == strides[3] == 1, ERR_MSG_DIM.format(strides)

  bstrides = _calc_block_strides(bsize, ksize, strides)

  # Pad mask.
  mask_ = tf.expand_dims(mask, 3)
  mask_ = _pad_input(mask_, ksize, strides, padding, bsize=bsize, bstrides=bstrides)
  mask_ = tf.nn.max_pool(mask_, bsize, bstrides, 'VALID')    # Blocks are always valid conv.
  mask_ = tf.squeeze(mask_, [3])
  indices = tf.where(tf.greater(mask_, tol))
  indices = tf.cast(indices, tf.int32)
  return indices

#################### cpu or primitive implementations ####################

# todo: need to implement these, based on tensorflow primitives

def reduce_mask(mask, *,
                block_count,
                bsize,
                boffset,
                bstride,
                tol=0.5,
                avgpool=False):
  pass


def gather(
    inputs,
    bin_counts,
    active_block_indices, *,
    bsize,
    boffset,
    bstride):
  pass


def scatter(
    block_stack,
    bin_counts,
    active_block_indices,
    outputs, *,
    bsize,
    boffset,
    bstride):
  pass
