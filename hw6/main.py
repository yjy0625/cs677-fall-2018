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
    print("Loading datasets...")
    dataset_train, dataset_val, dataset_test = load_datasets('data')
    config.data_info = list(dataset_train['image'].shape[1:])
    print("Dataset loading completes.")

    # create log directory
    hyper_parameter_str = 'bs_{}_lr_{}'.format(
        config.batch_size,
        config.learning_rate,
    )

    train_dir = './train_dir/{}_{}_{}_{}'.format(
        'kitti',
        config.prefix,
        hyper_parameter_str,
        time.strftime("%Y%m%d-%H%M%S")
    )

    if not osp.exists(train_dir): os.makedirs(train_dir)
    print("Train Dir: {}".format(train_dir))

    # reset default graph
    tf.reset_default_graph()

    # create model
    model = Model(config)

    # training setups
    saver = tf.train.Saver(max_to_keep=100)
    summary_writer = tf.summary.FileWriter(train_dir)
    session_config = tf.ConfigProto(
        gpu_options=tf.GPUOptions(allow_growth=True),
        device_count={'GPU': 1}
    )

    with tf.Session(config=session_config) as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())

        # buffers for train and val losses
        train_losses = []
        val_losses = []

        # train model
        for step in range(config.max_steps):
            # run validation step
            if step % config.val_step == 0:
                val_batch = sample_batch(dataset_val, config.batch_size)
                val_stats = run_single_step(sess, model, val_batch, 
                                            summary_writer, 
                                            mode='val')
                val_losses.append([val_stats['step'], val_stats['loss']])

            # run train step
            train_batch = sample_batch(dataset_train, config.batch_size)
            train_stats = run_single_step(sess, model, train_batch, 
                                          summary_writer, 
                                          mode='train', 
                                          log=step % config.log_step == 0)
            train_losses.append([train_stats['step'], train_stats['loss']])

            # save checkpoint
            if step % config.save_checkpoint_step == 0:
                print("Saved checkpoint at step {}".format(step))
                saver.save(sess, os.path.join(train_dir, 'model'), global_step=step)

        # test model
        test_logfile = os.path.join(train_dir, 'test_result.txt')
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

        # add loss curves to experiment results
        exp_results['train_losses'] = train_losses
        exp_results['val_losses'] = val_losses

        # log test results
        with open(os.path.join(train_dir, 'test_result.p'), 'wb') as f:
            pickle.dump(exp_results, f)
            print("Logged experiment results to {}".format(f.name))

        # flush Tensorboard summaries
        summary_writer.flush()

def run_single_step(session, model, batch, summary_writer, 
                    mode='train', log=True, test_logfile=None):
    # construct feed dict
    feed_dict = {
        model.images: batch['image'],
        model.labels: batch['gt'],
        model.is_training: mode == 'train'
    }
    
    # select proper summary op
    if mode == 'train':
        summary_op = model.train_summary_op
    elif mode == 'val':
        summary_op = model.val_summary_op
    else:
        summary_op = model.test_summary_op
    
    # construct fetch list
    fetch_list = [model.global_step, summary_op, 
                  model.loss, model.pixel_level_iou,
                  model.tp, model.fp, model.fn]
    if mode == 'train':
        fetch_list.append(model.optimizer_op)

    # run single step
    _start_time = time.time()
    _step, _summary, _loss, _iou, _tp, _fp, _fn = session.run(fetch_list, feed_dict=feed_dict)[:7]
    _end_time = time.time()
    
    # collect step statistics
    step_time = _end_time - _start_time
    batch_size = batch['image'].shape[0]
    
    # log in console
    if log:
        print(('[{:5s} step {:4d}] loss: {:.5f}; iou: {:.5f}; tp: {:5d}; fp: {:5d}; fn: {:5d} ' +
              '({:.3f} sec/batch; {:.3f} instances/sec)'
              ).format(mode, _step, _loss, _iou, _tp, _fp, _fn,
                       step_time, batch_size / step_time))
    
        # log in Tensorboard
        summary_writer.add_summary(_summary, global_step=_step)
    
    # log results to file and return statistics
    if mode == 'test':
        test_fetch_list = [model.loss, model.pixel_level_iou, model.pred, model.probs]
        _loss, _iou, _pred, _probs = \
                session.run(test_fetch_list, feed_dict=feed_dict)
        
        # Log detailed test results to file
        with open(os.path.join(test_logfile), 'w+') as f:
            f.write("Loss: {:.3f} \n".format(_loss))
            f.write("Pixel-level IOU: {:.3f} \n".format(_iou))
            print("Logged test results to {}".format(f.name))
        
        # Log detailed test results in pickle format
        stats = {
            "loss": _loss,
            "tp": _tp,
            "fp": _fp,
            "fn": _fn,
            "pixel_level_iou": _iou,
            "pred": _pred,
            "probs": _probs
        }
    else:
        stats = {
            "step": _step,
            "loss": _loss,
            "pixel_level_iou": _iou
        }
        
    return stats

def sample_batch(dataset, batch_size):
    N = dataset['image'].shape[0]
    # indices = list(range(batch_size))
    indices = np.random.randint(N, size=batch_size)
    return {key: dataset[key][indices] for key in dataset}
    
if __name__ == '__main__':
    main()
