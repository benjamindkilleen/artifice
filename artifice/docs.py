description = "Artifice."
commands = "Artifice command to run. Can be multiple."

# file settings
data_root = "todo"
model_root = "todo"
overwrite = "Overwrite existing models."
deep = "todo"
figs_dir = "todo"

# data settings
convert_mode = "todo"
transformation = "todo"
identity_prob = "todo"
priority_mode = "todo"
labeled = """use the labeled and not the annotated set for training. This is
subtly different from the AUGMENT option. If LABELED is true, then AUGMENT
should not be used, but if LABELED is false, augment may or may not be used,
since an annotated set can be used with or without augmentation."""

# annotation settings
annotation_mode = "todo"
record_size = "Number of examples to save in each annotated tfrecord."
annotation_delay = "todo"

# data sizes
image_shape = "Shape of the image as: HEIGHT WIDTH CHANNELS"
base_shape = "todo"
data_size = "Number of examples per training epoch."
test_size = "Number of examples withheld for testing."
batch_size = "Batch size."
subset_size = "Number of examples to annotate."
num_objects = "Maximum number of objects."
pose_dim = "todo"
num_shuffle = "todo"

# model architecture
base_size = "Height/width of the output of the first layer of the lower level."
level_filters = "Number of filters for each level in the unet."
level_depth = "todo"

# sparse eval and other optimization settings
model = "Which model to use."
model_choices = ['unet', 'sparse', 'dynamic']
multiscale = "todo"
use_var = "todo"

# model hyperparameters
dropout = "todo"
initial_epoch = """Initial epoch, starting at 0."""
epochs = """Number of training epochs. Default is 1."""
learning_rate = """Learning rate."""

# runtime settings
num_parallel_calls = "Threadpool size. Default (-1) uses number of available cores."
verbose = "Artifice verbosity. Default is 2 (debug level)."
keras_verbose = "Keras verbosity. Default is 1 (progress bars)."
patient = "Disable eager execution."
show = "Show plots rather than save them."
cache = "cache the pipelined dataset"
seconds = """Limits runtime for "prioritize" and "annotate" commands. For "train," sets
the time after which the dataset is no longer reloaded every epoch, and caching
can occur."""
