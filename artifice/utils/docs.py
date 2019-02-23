"""Documentation strings for artifice.
"""

description = """\
Artifice. An object detection system for scientific images, using boundary-aware
data augmentation.
- experiment: train a model using a simulated experiment.
- predict: run artifice on input examples, showing predictions.
"""

command_choices = ['experiment', 'predict', 'augment']
command_help = "Artifice command to run."

input_help = """Input:
- experiment: one or more tfrecord files containing examples.
- predict: one or more tfrecord files containing examples to predict
"""

output_help = """Output:
- experiment: not used. Set model-dir.
- predict: TODO: directory to save outputs (created if nonexistent)
  OR 'show' to display one prediction at a time using matplotlib (DEFAULT)
"""

model_dir_help = """Model directory.
- experiment: continues training from model_dir. Created if nonexistent.
- predict: REQUIRED for prediction.
"""

overwrite_help = """
overwrite existing model; restart training from scratch.
"""

image_shape_help = """Shape of the image. Must be 3D. Grayscale uses 1 for last
dimension. Default is '388 388 1'"""

epochs_help = """Number of training EPOCHS. Default is -1, repeats indefinitely."""

num_examples_help = """
- experiment: not used
- predict: limit prediction to NUM_EXAMPLES from INPUT. -1 (DEFAULT) takes all
  examples."""

num_classes_help = """\
Number of classes, including background. Can be larger than necessary. Default
is 3."""

eval_secs_help = """\
- experiment: evaluate model every EVAL_SECS during training. Default = 1200. Set
  to 0 for no evaluation. Default = 100.
"""

eval_mins_help = """\
- experiment: see EVAL_SECS. Default is 20 minutes (1200 seconds).
"""

num_eval_help = """\
- experiment: number of examples to take from the training set for evaluation.
  Set to 0 for no evaluation.
"""

eval_data_help = """\
- experiment: specifies an evaluation set."""

l2_reg_help = """\
L2 regularization factor. Default = 0.0001. l2_reg = 0 disables regularization.
"""

cores_help = """Number of CPU cores to parallelize over for data
processing/augmentation. Set to -1 to use all available cores. Default is -1."""

augment_help = """experiment: Apply augmentations to the training set."""
