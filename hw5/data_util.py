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
        print("Augmentation Step #1: {}".format(images.shape))
        images = augment_images_by_flipping(images)
        print("Augmentation Step #2: {}".format(images.shape))
        images = augment_images_by_cropping(images)
        print("Augmentation Step #3: {}".format(images.shape))
        
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
    for crop_start in crop_starts:
        start_h, start_w = crop_start
        augmented_images.append(enlarged_images[:, start_h:start_h + h, start_w:start_w + w,:])
    return np.concatenate(augmented_images)

def augment_images_by_flipping(images):
    augmented_images = []
    augmented_images.append(images)
    augmented_images.append(np.stack([np.fliplr(img) for img in images]))
    return np.concatenate(augmented_images)

def get_metadata(dir, filename):
    with open(os.path.join(dir, filename), 'rb') as f:
        label_info = pickle.load(f)
    return label_info

''' Computes mapping from fine labels to corresponding superclass.

Args:
    metadata: dictionary containing metadata for the dataset.

Returns:
    Numpy array where each element is the superclass label for corresponding
    fine label.
'''
def get_label_mapping(metadata):
    mapping_str = '''aquatic mammals	beaver, dolphin, otter, seal, whale
fish	aquarium fish, flatfish, ray, shark, trout
flowers	orchid, poppy, rose, sunflower, tulip
food containers	bottle, bowl, can, cup, plate
fruit and vegetables	apple, mushroom, orange, pear, sweet pepper
household electrical devices	clock, keyboard, lamp, telephone, television
household furniture	bed, chair, couch, table, wardrobe
insects	bee, beetle, butterfly, caterpillar, cockroach
large carnivores	bear, leopard, lion, tiger, wolf
large man-made outdoor things	bridge, castle, house, road, skyscraper
large natural outdoor scenes	cloud, forest, mountain, plain, sea
large omnivores and herbivores	camel, cattle, chimpanzee, elephant, kangaroo
medium mammals	fox, porcupine, possum, raccoon, skunk
non-insect invertebrates	crab, lobster, snail, spider, worm
people	baby, boy, girl, man, woman
reptiles	crocodile, dinosaur, lizard, snake, turtle
small mammals	hamster, mouse, rabbit, shrew, squirrel
trees	maple tree, oak tree, palm tree, pine tree, willow tree
vehicles 1	bicycle, bus, motorcycle, pickup truck, train
vehicles 2	lawn mower, rocket, streetcar, tank, tractor'''
    
    lines = [s.split('\t') for s in mapping_str.splitlines()]
    mapping = {item[0] : item[1].split(', ') for item in lines}
    
    mapping_coarse_to_fine = {}
    for key, value in mapping.items():
        mapping_coarse_to_fine[metadata['coarse_label_names'].index(key.replace(' ', '_'))] \
            = [metadata['fine_label_names'].index(item.replace(' ', '_')) for item in value]

    mapping_fine_to_coarse = np.zeros((100,), dtype=np.int32)
    for key, values in mapping_coarse_to_fine.items():
        for value in values:
            mapping_fine_to_coarse[value] = key

    return mapping_fine_to_coarse