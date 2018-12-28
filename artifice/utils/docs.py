"""Documentation strings for artifice.
"""

description = """\
Artifice. An object detection system for scientific images, using boundary-aware
data augmentation.
- experiment: train a model using a simulated experiment.
- predict: run artifice on input examples, showing predictions.
"""

command_help = "Artifice command to run."

input_help = """\
Input file:
- experiment: tfrecord containing examples.
"""

output_help = """\
Output file:
- experiment: directory to store the trained model in.
"""

overwrite_help = """\
- experiment: overwrite existing model; restart training from scratch.
"""

image_shape_help = """\
First two dimensions of the image.
"""
