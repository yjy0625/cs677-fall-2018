import tensorflow as tf

# limit GPU's visible by the program
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

flags = tf.app.flags

# experiment params
flags.DEFINE_string("prefix", "test_run", "Nickname for the experiment [default]")

# training params
flags.DEFINE_integer("max_steps", 50000, "Number of steps to train. [50000]")
flags.DEFINE_float("learning_rate", 1e-3, "Learning rate for the model. [1e-3]")
flags.DEFINE_integer("batch_size", 64, "Number of images in batch [64]")

# system variations
flags.DEFINE_boolean("apply_batch_norm", False, "Whether to apply batch normalization [False]")
flags.DEFINE_boolean("add_more_layers", False, "Whether to add more conv2d layers [False]")
flags.DEFINE_boolean("larger_filter_size", False, "Whether to use larger filter size [False]")
flags.DEFINE_boolean("add_dropout", False, "Whether to apply dropout to fc layers [False]")

# dataset params
flags.DEFINE_integer("num_classes", 100, "Number of classes in the dataset [100]")
flags.DEFINE_integer("num_superclasses", 20, "Number of superclasses in the dataset [20]")

# logging params
flags.DEFINE_integer("log_step", 500, "Interval for console logging [500]")
flags.DEFINE_integer("val_step", 500, "Interval for validation [5000]")
flags.DEFINE_integer("save_checkpoint_step", 5000, "Interval for checkpoint saving [5000]")

config = flags.FLAGS
