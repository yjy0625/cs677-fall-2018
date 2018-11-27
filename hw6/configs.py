import tensorflow as tf

flags = tf.app.flags

# experiment params
flags.DEFINE_string("prefix", "default", "Nickname for the experiment [default]")

# model params
flags.DEFINE_boolean("use_bilinear_deconv2d", False, "Whether to use bilinear deconv2d [False]")

# training params
flags.DEFINE_integer("max_steps", 28000, "Number of steps to train. [50000]")
flags.DEFINE_float("learning_rate", 1e-3, "Learning rate for the model. [1e-3]")
flags.DEFINE_float("momentum", 0.99, "Momentum for the optimizer. [0.99]")
flags.DEFINE_integer("batch_size", 1, "Number of images in batch [1]")

# logging params
flags.DEFINE_integer("log_step", 100, "Interval for console logging [100]")
flags.DEFINE_integer("val_step", 100, "Interval for validation [100]")
flags.DEFINE_integer("save_checkpoint_step", 500, "Interval for checkpoint saving [500]")

# checkpoint for testing
flags.DEFINE_string("checkpoint", None, "Checkpoint path for testing [None]")

# gpu
flags.DEFINE_integer("gpu", 0, "GPU to use [0]")

config = flags.FLAGS
