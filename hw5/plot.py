import pickle
import os
import sys
import numpy as np
from matplotlib import pyplot as plt
from data_util import get_metadata, load_data_from_file, get_label_mapping

plt.rcParams["font.family"] = "Times New Roman"

def main():
    # get dir from args
    if len(sys.argv) <= 1:
        print('plot.py <data_source_directory>')
        exit(0)
    DIR = sys.argv[1]
    
    # get logs for the directory
    logs = pickle.load(open(os.path.join(DIR, 'test_result.p'), 'rb'))

    # get images used for testing
    test_data = load_data_from_file('cifar-100-python/test')

    # get metadata
    metadata = get_metadata('cifar-100-python/', 'meta')

    # create plot saving directory
    save_dir = os.path.join(DIR, 'plots')
    if not os.path.exists(save_dir): 
        os.makedirs(save_dir)
        
    scm_filename = os.path.join(save_dir, 'superclass_confusion_matrix.pdf')
    vis_superclass_confusion_matrix(scm_filename, logs, metadata)
    
    cm_filename = os.path.join(save_dir, 'confusion_matrix.pdf')
    vis_confusion_matrix(cm_filename, logs)
    
    loss_curve_filename = os.path.join(save_dir, 'loss_curve.pdf')
    vis_loss_curve(loss_curve_filename, logs)
    
    success_examples_filename = os.path.join(save_dir, 'success_examples.pdf')
    vis_retrieval(success_examples_filename, logs, metadata, test_data, 'flowers')
    
    failure_examples_filename = os.path.join(save_dir, 'failure_examples.pdf')
    vis_retrieval(failure_examples_filename, logs, metadata, test_data, 'flowers', success=False)

# visualize superclass confusion matrix
def vis_superclass_confusion_matrix(filename, logs, metadata):
    superclass_confusion_matrix = logs['superclass_confusion_matrix']
    coarse_label_names = metadata['coarse_label_names']

    # setup figure
    fig, ax = plt.subplots(figsize=(20, 10))
    im = ax.imshow(superclass_confusion_matrix)

    # create axis ticks
    ax.set_xticks(np.arange(len(coarse_label_names)))
    ax.set_yticks(np.arange(len(coarse_label_names)))

    # set axis tick labels
    ax.set_xticklabels(coarse_label_names)
    ax.set_yticklabels(coarse_label_names)

    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # create text annotations
    for i in range(len(coarse_label_names)):
        for j in range(len(coarse_label_names)):
            text = ax.text(j, i, superclass_confusion_matrix[i, j],
                           ha="center", va="center", color="w")

    # ax.set_title("Superclass Confusion Matrix for Testing Data (Total 20 Superclasses)")
    fig.tight_layout()
    plt.savefig(filename, bbox_inches='tight')

# visualize confusion matrix
def vis_confusion_matrix(filename, logs):
    confusion_matrix = logs['confusion_matrix']

    # setup figure
    fig, ax = plt.subplots(figsize=(8, 8))
    im = ax.imshow(confusion_matrix)

    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # ax.set_title("Confusion Matrix for Testing Data (Total 100 Classes)")
    fig.tight_layout()
    plt.savefig(filename, bbox_inches='tight')

# visualize loss curves
def vis_loss_curve(filename, logs):
    plt.figure(figsize=(8, 6))

    def smooth_curve(scalars, weight):
        last = scalars[0]
        smoothed = list()
        for point in scalars:
            smoothed_val = last * weight + (1 - weight) * point
            smoothed.append(smoothed_val)
            last = smoothed_val
        return smoothed

    train_losses = np.transpose(np.array(logs['train_losses']))
    plt.plot(train_losses[0], smooth_curve(train_losses[1], 0.95))

    val_losses = np.transpose(np.array(logs['val_losses']))
    plt.plot(val_losses[0], smooth_curve(val_losses[1], 0.95))

    plt.legend(['Training', 'Validation'], loc='upper left')

    plt.savefig(filename, bbox_inches='tight')

# success and failure cases
def vis_retrieval(filename, logs, metadata, test_data, target_coarse_label, success=True):
    pred = np.array(logs['pred'])
    label_mapping = get_label_mapping(metadata)
    coarse_label_names = metadata['coarse_label_names']
    target_coarse_label_ix = coarse_label_names.index(target_coarse_label)
    target_fine_labels = np.argwhere(np.array(label_mapping) == target_coarse_label_ix).flatten().tolist()

    target_label_names = np.array(metadata['fine_label_names'])[target_fine_labels].tolist()
    images = []
    for label in target_fine_labels:
        if success:
            image_ix = np.argwhere((pred == label) & (pred == test_data['fine_labels']))[:10].flatten()
        else:
            image_ix = np.argwhere((pred == label) & (pred != test_data['fine_labels']))[:10].flatten()
        images.append(test_data['data'][image_ix])
    images = np.stack(images)
    nrow, ncol, h, w, c = images.shape
    vis_images = images.transpose((0,2,1,3,4)) \
        .reshape(nrow * h, ncol * w, c)

    # setup figure
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(vis_images)

    # configure axis ticks
    ax.set_yticks(np.arange(len(target_label_names)) * 32 + 16)
    ax.set_yticklabels(target_label_names)
    ax.get_xaxis().set_visible(False)

    plt.savefig(filename, bbox_inches='tight')

if __name__ == '__main__':
    main()