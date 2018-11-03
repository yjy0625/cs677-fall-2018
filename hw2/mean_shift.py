''' mean_shift.py '''
import argparse
import numpy as np
import cv2
import os
from utils import mkdir, show_img

''' Apply meanshift on a given image.

Args:
    img_path: path to the image.
    sp: spatial window radius for meanshift.
    sr: color window radius for meanshift.

Returns:
    image after meanshift, in RGB color space
'''
def apply_meanshift(img_path, sp, sr):
    # read image
    img = cv2.imread(img_path)

    # convert image to LAB color space
    img_lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    img_filtered = img_lab

    img_filtered = cv2.pyrMeanShiftFiltering(img_lab, sp, sr, img_filtered, maxLevel=1)

    filtered_bgr = cv2.cvtColor(img_filtered, cv2.COLOR_LAB2BGR)

    return filtered_bgr

''' Performs a single mean shift experiment and outputs images to specified
    output directory.
'''
def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--imgdir', type=str, default=None, help='directory to source images.')
    parser.add_argument('--outdir', type=str, default=None, help='directory to save output data.')
    parser.add_argument('--sp', type=int, default=50, help='spatial window radius for meanshift.')
    parser.add_argument('--sr', type=int, default=50, help='color window radius for meanshift.')
    parser.add_argument("--show_img", help='show image results', action="store_true")
    args = parser.parse_args()

    exp_name = 'sp{}_sr{}'.format(args.sp, args.sr)
    mkdir(args.outdir)

    files = os.listdir(args.imgdir)
    for filename in files:
        if filename.endswith('jpg'):
            full_path = os.path.join(args.imgdir, filename)

            # apply meanshift on image
            res = apply_meanshift(full_path, args.sp, args.sr)

            cv2.imwrite(os.path.join(args.outdir, exp_name + '-' + filename[:-4] + '_ms.jpg'), res)

            if args.show_img:
                window_name = 'Meanshift Result for {}'.format(filename)
                show_img(window_name, res)

if __name__ == '__main__':
    main()
