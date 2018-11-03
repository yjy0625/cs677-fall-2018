''' utils.py '''
import os
import cv2
import numpy as np

''' Safe mkdir that checks directory before creation. '''
def mkdir(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)

''' Display an image in a window with given name. '''
def show_img(window_name, img):
    cv2.imshow(window_name, img)
    cv2.waitKey(0)
    cv2.destroyWindow(window_name)

''' Compute a 4x2 numpy matrix of corner coords of an image. '''
def get_corner_coords(shape):
    h, w, _ = shape
    return np.array([
        [0, 0],
        [w, 0],
        [w, h],
        [0, h]
    ])

''' Compute the distance between two key points. '''
def dist_kp(kp1, kp2):
    return np.linalg.norm(np.array(kp1.pt) - np.array(kp2.pt))
