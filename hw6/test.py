import tensorflow as tf
import os
import os.path as osp
import time
import pickle

from data_util import *
from model import Model
from configs import config
from main import run_single_step

def main():
	# limit GPU's visible by the program
	os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
	os.environ["CUDA_VISIBLE_DEVICES"] = str(config.gpu)
	print("Using GPU {}.".format(config.gpu))

	# load data
	print("Loading datasets")
	_, _, dataset_test = load_datasets('data')
	config.data_info = list(dataset_test['image'].shape[1:])
	print("Dataset loading completes.")

	tf.reset_default_graph()

	model = Model(config)
	
	session_config = tf.ConfigProto(
		gpu_options=tf.GPUOptions(allow_growth=True),
		device_count={'GPU': 1}
	)

	all_vars = tf.trainable_variables()

	ckpt_path = config.checkpoint
	ckpt_dir = osp.dirname(ckpt_path)
	summary_writer = tf.summary.FileWriter(ckpt_dir)

	with tf.Session(config=session_config) as sess:
		sess.run(tf.global_variables_initializer())
		sess.run(tf.local_variables_initializer())

		# restore checkpoint
		if ckpt_dir is not None:
			print("Checkpoint path: %s", ckpt_path)
			tf.train.Saver(var_list=all_vars).restore(
				sess, ckpt_path)
			print("Loaded the parameters from the provided checkpoint path")
		else:
			print("Please provide checkpoint path.")
			exit(0)

		test_logfile = os.path.join(ckpt_dir, 'test_result.txt')
		n_test = dataset_test['image'].shape[0]
		ind_exp_results = {
			"loss": [],
			"tp": [],
			"fp": [],
			"fn": [],
			"pixel_level_iou": [],
			"pred": [],
			"probs": []
		}
		for i in range(n_test):
			test_batch = {key: dataset_test[key][[i]] for key in dataset_test}
			exp_results = run_single_step(sess, model, test_batch, 
										  summary_writer, 
										  mode='test',
										  test_logfile=test_logfile)
			for key in exp_results:
				ind_exp_results[key].append(exp_results[key])
		exp_results = {}
		for key in ind_exp_results:
			exp_results[key] = np.stack(ind_exp_results[key])
		exp_results["loss"] = np.mean(exp_results["loss"])
		exp_results["tp"] = np.sum(exp_results["tp"])
		exp_results["fp"] = np.sum(exp_results["fp"])
		exp_results["fn"] = np.sum(exp_results["fn"])
		exp_results["pixel_level_iou"] = exp_results["tp"] / (
				exp_results["tp"] + exp_results["fp"] + exp_results["fn"])
		print("Test Average Loss: {:3f}; Test IOU: {:3f}".format(
				exp_results["loss"], exp_results["pixel_level_iou"]))

		# log test results
		with open(os.path.join(ckpt_dir, 'test_result.p'), 'wb') as f:
			pickle.dump(exp_results, f)
			print("Logged experiment results to {}".format(f.name))

		# flush Tensorboard summaries
		summary_writer.flush()

		
if __name__ == '__main__':
	main()
