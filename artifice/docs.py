description = "Artifice."
commands = "Artifice command to run. Can be multiple."
mode = "Training mode for augmentation."

# file settings
data_root = "todo"
model_root = "todo"
overwrite = "Overwrite existing models."

# data conversion
convert_mode = "todo"

# data sizes
image_shape = "Shape of the image as: HEIGHT WIDTH CHANNELS"
base_shape = "todo"
data_size = "Number of total examples."
test_size = "Number of examples to withhold for testing."
batch_size = "Batch size."
subset_size = "Number of examples to annotate."
epoch_size = "Number of artificial examples per training epoch."
num_objects = "Maximum number of objects."
pose_dim = "todo"

# model hyperparameters
base_size = "Height/width of the output of the first layer of the lower level."
level_filters = """Number of filters for each level in the unet."""
level_depth = "todo"
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
