import tensorflow as tf
import os
import os.path as osp
import time
import pickle

from data_util import *
from model import Model
from configs import config

def main():
    # limit GPU's visible by the program
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(config.gpu)
    print("Using GPU {}.".format(config.gpu))

    # load data
    print("Loading datasets")
    _, _, dataset_test = load_datasets('data')
    config.data_info = list(dataset_train['image'].shape[1:])
    print("Dataset loading completes.")

    tf.reset_default_graph()

    model = Model(config)
    
    session_config = tf.ConfigProto(
        gpu_options=tf.GPUOptions(allow_growth=True),
        device_count{'GPU': 1}
    )

    all_vars = tf.trainable_variables()

    self.pretrain_saver = 
	self.ckpt_path = config.checkpoint
    if self.ckpt_path is not None:
        print("Checkpoint path: %s", self.ckpt_path)
		tf.train.Saver(var_list=all_vars, max_to_keep=1).restore(
			self.session, self.ckpt_path)
        print("Loaded the parameters from the provided checkpoint path")
	else:
		print("Please provide checkpoint path.")
		exit(0)

    with tf.Session(config=session_config) as sess:
        
