import tensorflow as tf
import numpy as np
import time
import os
import pickle

from data_util import load_and_preprocess_data, get_metadata, get_label_mapping
from model import Model
from configs import config

def main():
    # load data
    print("Loading datasets...")
    dataset, dataset_val, dataset_test = load_and_preprocess_data('cifar-100-python/')
    print("Dataset loading completes.")
    
    # setup metadata
    metadata = get_metadata('cifar-100-python', 'meta')
    config.data_info = list(dataset['data'].shape[1:])
    config.fine_label_names = metadata['fine_label_names']
    config.coarse_label_names = metadata['coarse_label_names']
    config.label_mapping = get_label_mapping(metadata)
    
    # create log directory
    hyper_parameter_str = 'bs_{}_lr_{}'.format(
        config.batch_size,
        config.learning_rate,
    )
    
    if config.apply_batch_norm: hyper_parameter_str += '_batchnorm'
    if config.add_more_layers: hyper_parameter_str += '_morelayers'
    if config.larger_filter_size: hyper_parameter_str += '_largefilter'
    if config.add_dropout: hyper_parameter_str += '_dropout'

    train_dir = './train_dir/{}_{}_{}_{}'.format(
        'cifar10',
        config.prefix,
        hyper_parameter_str,
        time.strftime("%Y%m%d-%H%M%S")
    )
    
    if not os.path.exists(train_dir): os.makedirs(train_dir)
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

    with tf.Session(config=session_config) as session:
        session.run(tf.global_variables_initializer())
        
        # buffers for train and val losses
        train_losses = []
        val_losses = []
        
        # train model
        for step in range(config.max_steps):
            # run validation step
            if step % config.val_step == 0:
                val_batch = sample_batch(dataset_val, config.batch_size)
                val_stats = run_single_step(session, model, val_batch, 
                                            summary_writer, config.label_mapping, 
                                            mode='val')
                val_losses.append([val_stats['step'], val_stats['loss']])

            # run train step
            train_batch = sample_batch(dataset, config.batch_size)
            train_stats = run_single_step(session, model, train_batch, 
                                          summary_writer, config.label_mapping,
                                          mode='train', 
                                          log=step % config.log_step == 0)
            train_losses.append([train_stats['step'], train_stats['loss']])

            # save checkpoint
            if step % config.save_checkpoint_step == 0:
                print("Saved checkpoint at step {}".format(step))
                saver.save(session, os.path.join(train_dir, 'model'), global_step=step)

        # test model
        test_logfile = os.path.join(train_dir, 'test_result.txt')
        exp_results = run_single_step(session, model, dataset_test, 
                                      summary_writer, config.label_mapping,
                                      mode='test',
                                      test_logfile=test_logfile)
        
        # add loss curves to experiment results
        exp_results['train_losses'] = train_losses
        exp_results['val_losses'] = val_losses
        
        # log test results
        with open(os.path.join(train_dir, 'test_result.p'), 'wb') as f:
            pickle.dump(exp_results, f)
            print("Logged experiment results to {}".format(f.name))
        
        # flush Tensorboard summaries
        summary_writer.flush()

def run_single_step(session, model, batch, summary_writer, label_mapping, 
                    mode='train', log=True, test_logfile=None):
    # construct feed dict
    feed_dict = {
        model.images: batch['data'],
        model.coarse_labels: batch['coarse_labels'],
        model.fine_labels: batch['fine_labels'],
        model.label_mapping: label_mapping,
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
    fetch_list = [model.global_step, summary_op, model.loss, model.accuracy, model.top_5_accuracy]
    if mode == 'train':
        fetch_list.append(model.optimizer_op)

    # run single step
    _start_time = time.time()
    _step, _summary, _loss, _top_1, _top_5 = session.run(fetch_list, feed_dict=feed_dict)[:5]
    _end_time = time.time()
    
    # collect step statistics
    step_time = _end_time - _start_time
    batch_size = batch['data'].shape[0]
    
    # log in console
    if log:
        print(('[{:5s} step {:4d}] loss: {:.5f}; top_1_accuracy: {:.5f}; top_5_accuracy: {:5f} ' +
              '({:.3f} sec/batch; {:.3f} instances/sec)'
              ).format(mode, _step, _loss, _top_1, _top_5, 
                       step_time, batch_size / step_time))
    
    # log in Tensorboard
    summary_writer.add_summary(_summary, global_step=_step)
    
    # log results to file and return statistics
    if mode == 'test':
        test_fetch_list = [model.per_superclass_accuracy, model.per_class_accuracy,
                model.top_5_per_superclass_accuracy, model.top_5_per_class_accuracy,
                model.confusion_matrix, model.superclass_confusion_matrix,
                model.pred, model.probs]
        _top_1_s, _top_1_c, _top_5_s, _top_5_c, _cm, _scm, _pred, _probs = \
                session.run(test_fetch_list, feed_dict=feed_dict)
        
        # Log detailed test results to file
        with open(os.path.join(test_logfile), 'w') as f:
            f.write("Loss: {:.3f} \n".format(_loss))
            f.write("Top 1 Accuracy: {:.3f} \n".format(_top_1))
            f.write("Top 5 Accuracy: {:.3f} \n\n".format(_top_5))
            f.write("Top 1 Per-superclass Accuracy: \n{} \n\n".format(_top_1_s))
            f.write("Top 5 Per-superclass Accuracy: \n{} \n\n".format(_top_5_s))
            f.write("Top 1 Per-class Accuracy: \n{} \n\n".format(_top_1_c))
            f.write("Top 5 Per-class Accuracy: \n{} \n\n".format(_top_5_c))
            f.write("Confusion Matrix: \n{} \n\n".format(_cm.tolist()))
            f.write("Superclass Confusion Matrix: \n{} \n".format(_scm.tolist()))
            
            print("Logged test results to {}".format(f.name))
        
        # Log detailed test results in pickle format
        stats = {
            "loss": _loss,
            "top_1_accuracy": _top_1,
            "top_5_accuracy": _top_5,
            "top_1_perclass_accuracy": _top_1_c,
            "top_5_perclass_accuracy": _top_5_c,
            "top_1_persuperclass_accuracy": _top_1_s,
            "top_5_persuperclass_accuracy": _top_5_s,
            "confusion_matrix": _cm,
            "superclass_confusion_matrix": _scm,
            "pred": _pred,
            "probs": _probs
        }
    else:
        stats = {
            "step": _step,
            "loss": _loss,
            "top_1_accuracy": _top_1,
            "top_5_accuracy": _top_5
        }
        
    return stats

def sample_batch(dataset, batch_size):
    N = dataset['data'].shape[0]
    indices = np.random.randint(N, size=batch_size)
    return {key: dataset[key][indices] for key in dataset}

if __name__ == '__main__':
    main()
