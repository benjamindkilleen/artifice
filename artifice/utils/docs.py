"""Documentation strings for artifice.
"""

description = """Artifice."""

command_choices = ['experiment', 'predict', 'augment']
command_help = "Artifice command to run."

data_dir_help = """Data input directory. Initially, this can take the shape of
an "images" directory and a "labels.npy" file with associated labels. This gets
converted "data.tfrecord" before anything else, which has the associated
(image, label) pairs."""

output_help = """Output."""

model_dir_help = """Model directory. Contains "model/" with the actual model
checkpoints and associated files/plots."""

overwrite_help = """overwrite existing model; restart training from scratch."""

image_shape_help = """Shape of the image. Must be 3D. Grayscale uses 1 for last
dimension."""

epochs_help = """Number of training epochs. Default is 1."""

num_examples_help = """Not used."""

num_classes_help = """Not used."""

splits_help = """Splits to use for training, validation, and testing."""

l2_reg_help = """Not used."""

cores_help = """Number of CPU cores to parallelize over. Default (-1) uses
available cores."""

eager_help = """Enable eager execution."""

show_help = """Show plots rather than save them."""
