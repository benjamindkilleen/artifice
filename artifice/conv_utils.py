"""Borrowed largely from Tensorflow source.
"""



def divup(a, b):
  return (a+b-1) // b

def conv_output_length(input_length, filter_size, padding, stride, dilation=1):
  """Determines output length of a convolution given input length.
  Arguments:
      input_length: integer.
      filter_size: integer.
      padding: one of "same", "valid", "full", "causal"
      stride: integer.
      dilation: dilation rate, integer.
  Returns:
      The output length (integer).
  """
  if input_length is None:
    return None
  assert padding in {'same', 'valid', 'full', 'causal'}
  dilated_filter_size = filter_size + (filter_size - 1) * (dilation - 1)
  if padding in ['same', 'causal']:
    output_length = input_length
  elif padding == 'valid':
    output_length = input_length - dilated_filter_size + 1
  elif padding == 'full':
    output_length = input_length + dilated_filter_size - 1
  return (output_length + stride - 1) // stride

def conv_output_shape(input_shape, filters, kernel_size, padding, strides):
  """Compute the output shape of the given convolutional layer.

  :param input_shape: 
  :param filters: 
  :param kernel_size: 
  :param padding: 
  :param strides: 
  :returns: 
  :rtype: 

  """
  output_h = conv_output_length(
    input_shape[1],
    kernel_size[0],
    padding,
    strides[0])
  output_w = conv_output_length(
    input_shape[2],
    kernel_size[1],
    padding,
    strides[1])
  return [input_shape[0].value, output_h, output_w, filters]
    
def deconv_output_length(input_length, filter_size, padding,
                         output_padding=None, stride=0, dilation=1):
  """Determines output length of a transposed convolution given input length.
    
  Original source can be found [here](https://github.com/tensorflow/tensorflow/blob/5912f51d580551e5cee2cfde4cb882594b4d3e60/tensorflow/python/keras/utils/conv_utils.py#L140).

  :param input_length: Integer.
  :param filter_size: Integer.
  :param padding: one of `"same"`, `"valid"`, `"full"`.
  :param output_padding: Integer, amount of padding along the output dimension.
        Can be set to `None` in which case the output length is inferred.
  :param stride: Integer.
  :param dilation: Integer.
  :returns: The output length (integer).

  """
  assert padding in {'same', 'valid', 'full'}
  if input_length is None:
    return None

  # Get the dilated kernel size
  filter_size = filter_size + (filter_size - 1) * (dilation - 1)

  # Infer length if output padding is None, else compute the exact length
  if output_padding is None:
    if padding == 'valid':
      length = input_length * stride + max(filter_size - stride, 0)
    elif padding == 'full':
      length = input_length * stride - (stride + filter_size - 2)
    elif padding == 'same':
      length = input_length * stride

  else:
    if padding == 'same':
      pad = filter_size // 2
    elif padding == 'valid':
      pad = 0
    elif padding == 'full':
      pad = filter_size - 1

    length = ((input_length - 1) * stride + filter_size - 2 * pad +
              output_padding)
  return length


def deconv_output_shape(input_shape, filters, kernel_size, padding, strides):
  """Compute the output shape of the given transpose convolutional layer.

  :param input_shape: 
  :param filters: 
  :param kernel_size: 
  :param padding: 
  :param strides: 
  :returns: 
  :rtype: 

  """
  
  output_h = deconv_output_length(
    input_shape[1],
    kernel_size[0],
    padding,
    stride=strides[0])
  output_w = deconv_output_length(
    input_shape[2],
    kernel_size[1],
    padding,
    stride=strides[1])
  return [input_shape[0].value, output_h, output_w, filters]
    
