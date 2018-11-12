import os
import numpy as np
import pickle
import cv2

''' Loads and preprocesses cifar 10 data from file.

Args:
    dir: directory that includes the cifar 10 data.
Returns:
    3 dictionaries (train_data, val_data, test_data),
    each with two keys "data", and "label".
'''
def load_and_preprocess_data(dir):
    # collect train_val data from files
    train_val_data = load_data_from_file(os.path.join(dir, 'train'))
    
    # split train and val data
    train_data, val_data = split_train_val_data(train_val_data)
    
    # get test data
    test_data = load_data_from_file(os.path.join(dir, 'test'))
    
    # get mean image from training set
    mean_image = np.mean(train_data['data'], 0)

    # collect train, val, and test datasets
    datasets = [train_data, val_data, test_data]
    
    # augment datasets
    datasets = [preprocess_data(data, mean_image, augment=(ix==0)) for ix, data in enumerate(datasets)]
    
    return tuple(datasets)

def load_data_from_file(filename):
    with open(filename, 'rb') as f:
        data = pickle.load(f, encoding='latin1')
    data = {
        'data': np.transpose(data['data'].reshape((-1, 3, 32, 32)), (0, 2, 3, 1)).astype(np.float32) / 255.0,
        'fine_labels': np.array(data['fine_labels']),
        'coarse_labels': np.array(data['coarse_labels'])
    }
    return data

def split_train_val_data(data):
    N = data['data'].shape[0]
    perm = np.random.permutation(N)
    train_ix, val_ix = perm[:int(N * 0.8)], perm[int(N * 0.8):]
    train_data = {key: data[key][train_ix] for key in data}
    val_data = {key: data[key][val_ix] for key in data}
    return train_data, val_data

def preprocess_data(data, mean_image, augment=False):
    # mean subtraction
    images = data['data']
    images = images - mean_image
    
    # augmentation
    if augment:
        orignal_size = images.shape[0]
        
        images = augment_images_by_cropping(images)
        print("Aug #1: {}".format(images.shape))
        images = augment_images_by_flipping(images)
        print("Aug #2: {}".format(images.shape))
        images = augment_images_by_cropping(images)
        print("Aug #3: {}".format(images.shape))
        
        multiplier = int(images.shape[0] / orignal_size)
        
        data['fine_labels'] = np.tile(data['fine_labels'], multiplier)
        data['coarse_labels'] = np.tile(data['coarse_labels'], multiplier)
    
    data['data'] = images
    
    return data
    
def augment_images_by_cropping(images):
    resize = lambda img, factor: cv2.resize(img, tuple((np.array(img.shape[:2]) * factor).astype(np.int)))
    enlarged_images = np.stack([resize(img, 1.1) for img in images])
    h, w = images.shape[1:3]
    H, W = enlarged_images.shape[1:3]
    crop_starts = [
        (0, 0), (0, W - w), (H - h, 0), (H - h, W - w), 
        (int((H - h) / 2), int((W - w) / 2))
    ]
    augmented_images = []
    # augmented_images.append(images)
    for crop_start in crop_starts:
        start_h, start_w = crop_start
        augmented_images.append(enlarged_images[:, start_h:start_h + h, start_w:start_w + w,:])
    return np.concatenate(augmented_images)

def augment_images_by_flipping(images):
    augmented_images = []
    # augmented_images.append(images)
    augmented_images.append(np.stack([np.fliplr(img) for img in images]))
    return np.concatenate(augmented_images)

def get_metadata(dir, filename):
    with open(os.path.join(dir, filename), 'rb') as f:
        label_info = pickle.load(f)
    return label_info