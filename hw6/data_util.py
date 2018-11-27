import os.path as osp
import numpy as np
import glob
import cv2
import pickle

def load_datasets(dir):
    train_val_data = load_dataset('data', 'train')
    train_data, val_data = split_train_val(train_val_data, factor=199/244)
    test_data = load_dataset('data', 'test')
    
    return train_data, val_data, test_data

def load_dataset(dir, mode):
    image_files = glob.glob(osp.join(dir, "image/{}/*.png".format(mode)))
    data = {
        "id": [],
        "image": [],
        "gt": []
    }

    for image_file in image_files:
        sample_img = cv2.cvtColor(cv2.imread(image_file), cv2.COLOR_BGR2RGB)
        sample_id = image_file.split('/')[-1].split('.')[0]

        gt_filename = sample_id.split('_')[0] + '_road_' + sample_id.split('_')[1]
        gt_file = osp.join(dir, "label/{}/{}.label".format(mode, gt_filename))
        with open(gt_file, 'rb') as f:
            sample_gt = pickle.load(f, encoding='latin1')

        data["id"].append(sample_id)
        data["image"].append(sample_img)
        data["gt"].append(sample_gt)

    data = {key: np.array(data[key]) for key in data.keys()}
    data["image"] = data["image"].astype(np.float32) / 255.0
    
    return data

def split_train_val(data, factor=0.8):
    N = len(data['id'])
    perm = np.random.permutation(N)
    train_ix, val_ix = perm[:int(N * factor)], perm[int(N * factor):]
    train_data = {key: data[key][train_ix] for key in data}
    val_data = {key: data[key][val_ix] for key in data}
    return train_data, val_data
