"""The implementations in this file rely on SBNet, a project from
uber-research. The original repository containing this code can be found
[here](https://github.com/uber-research/sbnet), and the paper describing it
[here](https://arxiv.org/abs/1801.02108). Documentation for my wrapper functions
comes mostly from the git repo README, albeit in a more pythonic format.

"""


import logging
import numpy as np
import tensorflow as tf
from tensorflow import keras

from artifice.log import logger
from artifice import utils
from artifice import sbnet

def reduce_mask(mask, *, block_count, bsize, boffset, bstride, tol=0.5,
                avgpool=False):
  """Reduce `mask` to indices for sparse_gather or sparse_scatter.

  Thin wrapper around sbnet.reduce_mask, which offers the GPU implementation. If
  no GPU is available, implements the operation in tensorflow primitives.

  Blocks, in the sbnet framework, refer to patches of the image which are either
  collected for convolution or ignored (and thus implicitly zeroed). In numpy
  terms each block is defined as a slice from the input mask of dimensions
  `[N,H,W,1]`, with following dimensions: `[ni, BOFFSH+BSTRH*hi :
  BOFFSH+BSTRH*hi+BSZH, BOFFSW+BSTRW*wi : BOFFSW+BSTRW*wi+BSZW, :]`

  See https://arxiv.org/abs/1801.02108 for more info.

  :param mask: `[N, H, W, 1]` shape tensor containing mask values.
  :param block_count: `[BCH, BCW]` block counts in
  height and width dimensions (axes 0 and 1 respectively).
  :param bsize: `[BSZH, BSZW]` block size
  :param boffset: `[BOFFSH, BOFFSW]` block offset
  :param bstride: `[BSTRH, BSTRW]` block stride
  :param tol: pooling threshold to consider a block as active
  :param avgpool: if True, use average pooling. If False (default), use max
  pooling.
  :returns: `namedtuple` which contains tensors `bin_counts` and
  `active_block_indices`, for passing to `sparse_gather` and `sparse_scatter`.

  """
  if tf.test.is_gpu_available() and tf.test.is_built_with_cuda():
    return sbnet.reduce_mask(mask, block_count, dynamic_bsize=bsize,
                             dynamic_boffset=boffset, dynamic_bstride=bstride,
                             tol=tol, avgpool=avgpool)
  raise NotImplementedError("reduce_mask for CPU")


def sparse_gather(inputs, bin_counts, active_block_indices, *, bsize, boffset,
                  bstride, transpose=False):
  """Gather the active blocks from `inputs` into a `block_stack`.

  Gathers the blocks from `inputs` as in the following pseudocode:
```
for (ni, hi, wi) in indices.active_block_indices:
  channel_slice = x[ni, BOFFSH+BSTRH*hi : BOFFSH+BSTRH*hi+BSZH,
                    BOFFSW+BSTRW*wi : BOFFSW+BSTRW*wi+BSZW, :]
  block_stack[ni, :, :, :] = channel_slice
```

  :param inputs: `[N, H, W, C]` shaped input tensor
  :param bin_counts:
  :param active_block_indices: `[nBlocks, 3]` set of active block indices.
  :param bsize: `[BSZH, BSZW]` block size
  :param boffset: `[BOFFSH, BOFFSW]` block offset
  :param bstride: `[BSTRH, BSTRW]` block stride
  :param transpose: If `transpose` is true, a fused transpose operation will
  also be performed and the resulting tensor will have dimensions
  `[nBlocks, C, BSZH, BSZW]`.
  :returns: `[nBlocks, BSZH, BSZW, C]` tensor stack of blocks

  """
  if tf.test.is_gpu_available() and tf.test.is_built_with_cuda():
    return sbnet.sparse_gather(inputs, bin_counts, active_block_indices,
                               dynamic_bsize=bsize, dynamic_boffset=boffset,
                               dynamic_bstride=bstride, transpose=transpose)
  raise NotImplementedError("sparse_gather for CPU")

def sparse_scatter(block_stack, bin_counts, active_block_indices, outputs, *,
                   bsize, boffset, bstride, add=False, atomic=False,
                   transpose=False):
  """Scatter blocks in `block_stack` back onto `outputs`.

  Note that due to a limitation of TensorFlow API an intermediate tensor cannot
  be modified in place unless it's specified to be a tf.Variable. This
  necessitates creating an intermediate tensor inside the op and performing a
  copy which has negative implications for performance. So the creators of SBNet
  made a second version of the op sbnet_module.sparse_scatter_var that expects
  `outputs` to be a tf.Variable and modifies it in place. We automatically
  detect whether `outputs` is a Tensor or a Variable and using the proper
  fucntion. Using a Variable is strongly recommended for maximum performance.

  The effect of this operation is opposite to sparse_gather - the input blocks
  will be written on top of base tensor x, or added to it's contents if do_add
  is True. The following pseudo-code snippet illustrates the semantics of
  sparse_scatter:
```
for (ni, hi, wi) in indices.active_block_indices:
  if add:
    x[ni, BOFFSH+BSTRH*hi : BOFFSH+BSTRH*hi+BSZH,
      BOFFSW+BSTRW*wi : BOFFSW+BSTRW*wi+BSZW, :] += blockStack[ni, :, :, :]
  else:
    x[ni, BOFFSH+BSTRH*hi : BOFFSH+BSTRH*hi+BSZH,
      BOFFSW+BSTRW*wi : BOFFSW+BSTRW*wi+BSZW, :] = blockStack[ni, :, :, :]
```

  :param block_stack: `[nBlocks, BSZH, BSZW, C]` tensor stack of blocks
  :param bin_counts:
  :param active_block_indices: `[nBlocks, 3]` set of active block indices.
  :param outputs: base tensor to copy to output and overwrite on top of
  :param bsize: `[BSZH, BSZW]` block size
  :param boffset: `[BOFFSH, BOFFSW]` block offset
  :param bstride: `[BSTRH, BSTRW]` block stride
  :param add: perform an add operation rather than replacement
  :param atomic: use atomic or regular adds
  :param transpose:  if `transpose` is true, a fused transpose operation will
  also be performed by sparse_scatter, permuting the input `[N,C,H,W]` dimensions
  to `[N,H,W,C]` in the output.
  :returns:
  :rtype:

  """

  if tf.test.is_gpu_available() and tf.test.is_built_with_cuda():
    return sbnet.sparase_scatter(
      block_stack,
      bin_counts,
      active_block_indices,
      outputs,
      dynamic_bsize=bsize,
      dynamic_boffset=boffset,
      dynamic_bstride=bstride,
      add=add,
      atomic=atomic,
      transpose=transpose)
  raise NotImplementedError("sparse_scatter for CPU")

def main():
  """For testing/understanding sbnet."""
  # tf.enable_eager_execution()
  # Specify input tensor dimensions and block-sparsity parameters
  batch = 4
  hw = 256
  channels = 64
  blockSize = [16, 16]
  blockStride = [14, 14]
  blockOffset = [0, 0]
  blockCount = [utils.divup(hw, blockStride[0]), utils.divup(hw, blockStride[1])]

  # build kwargs to simplify op calls
  inBlockParams = {"dynamic_bsize": blockSize, "dynamic_boffset": blockOffset, "dynamic_bstride": blockStride }
  outBlockParams = {"dynamic_bsize": [blockSize[0]-2, blockSize[1]-2], "dynamic_boffset": blockOffset, "dynamic_bstride": blockStride}

  # create a random mask representing attention/a priori sparsity
  # threshold the mask to a specified percentile sparsity
  mask = np.random.randn(batch, blockCount[0], blockCount[1],
                         channels).astype(np.float32)
  threshold = np.percentile(mask, 90)
  sparseMask = np.greater(mask, threshold).astype(np.float32)

  # upsample the mask to full resolution
  upsampledMask = sparseMask.repeat(blockStride[0],
                                    axis=1).repeat(blockStride[1], axis=2)

  # create a random input tensor
  x = tf.constant( np.random.randn(batch, hw, hw, channels).astype(np.float32) )

  # create a random weight tensor
  w = tf.constant( np.random.randn(3, 3, channels, channels).astype(np.float32) )

  # reduce the mask to indices by using a fused pooling+indexing operation
  indices = sbnet.reduce_mask(mask, blockCount, tol=0.5, **inBlockParams)
  print("using gpu:", tf.test.is_gpu_available() and tf.test.is_built_with_cuda())
  print("bin_counts:", indices.bin_counts)
  print("bin_counts:", indices.bin_counts.shape)
  print("active_block_indices:", indices.active_block_indices)
  print("active_block_indices:", indices.active_block_indices.shape)

  # stack active overlapping tiles to batch dimension
  blockStack = sbnet.sparse_gather(x, indices.bin_counts,
                                   indices.active_block_indices,
                                   transpose=True, **inBlockParams)
  print("block_stack:", blockStack.shape)

  # perform dense convolution on a sparse stack of tiles
  convBlocks = tf.nn.conv2d(blockStack, w, strides=[1, 1, 1, 1],
                            padding='VALID', data_format='NCHW')
  # convBlocks = keras.layers.Conv2D(channels, (3,3), padding='valid', data_format='channels_first')(blockStack)

  # write/scatter the tiles back on top of original tensor
  # note that the output tensor is reduced by 1 on each side due to 'VALID' convolution
  validX = x[:, 1:hw-1, 1:hw-1, :]
  y = sbnet.sparse_scatter(
    convBlocks, indices.bin_counts, indices.active_block_indices,
    validX, transpose=True, add=False, atomic=False, **outBlockParams)

  if not tf.executing_eagerly():
    sess = tf.Session()
    y_output, = sess.run([y])

if __name__ == '__main__':
  main()
