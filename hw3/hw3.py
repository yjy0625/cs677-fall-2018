''' hw3.py '''
import numpy as np
import cv2
import os
import glob
import argparse
from utils import mkdir, show_img, get_corner_coords, dist_kp

''' Compute sift features and then compute homography transformation using RANSAC.

Args:
    query_img: the image to query in a scene (src_img)
    train_img: the scene that contains the query img (dst_img)
    f: log file to write program results

Returns:
    img_query_features: query image with sift features
    img_train_features: train image with sift features
    img_matches_before: image showing matches before homography matrix is computed
    img_matches_after: image showing matches after homography matrix is computed
    h: computed homography matrix
'''
def compute_homo_transformation(query_img, train_img, f=None):
    # initiate SIFT detector
    sift = cv2.xfeatures2d.SIFT_create()

    # find the keypoints and descriptors with SIFT
    kp_query, des_query = sift.detectAndCompute(query_img, None)
    kp_train, des_train = sift.detectAndCompute(train_img, None)

    n_features_query = len(kp_query)
    n_features_train = len(kp_train)

    if f:
        f.write('Number of features for query image: {}\n'.format(n_features_query))
        f.write('Number of features for query image: {}\n'.format(n_features_train))

    # convert image colors to grey so that sift features can be drawn on top
    gray_query_img = cv2.cvtColor(query_img, cv2.COLOR_BGR2GRAY)
    gray_train_img = cv2.cvtColor(train_img, cv2.COLOR_BGR2GRAY)

    # draw sift features on gray image
    draw_kp_params = dict(outImage=np.array([]), flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    img_query_features = cv2.drawKeypoints(gray_query_img, kp_query, **draw_kp_params)
    img_train_features = cv2.drawKeypoints(gray_train_img, kp_train, **draw_kp_params)
    
    # BFMatcher with variable k parameter
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des_query, des_train, k=2)
    
    # get number of matches
    n_matches = len(matches)

    # filter only good matches using ratio test
    matches = filter(lambda f: f[0].distance < 0.7 * f[1].distance, matches)
    matches = map(lambda f: [f[0]], matches)

    # sort matches by distance
    matches = sorted(matches, key=lambda f: f[0].distance)

    # get number of matches and number of good matches
    n_good_matches = len(matches)

    if f:
        f.write("Number of matches: {}\n".format(n_matches))
        f.write("Number of good matches: {}\n".format(n_good_matches))

    # draw first 20 matches
    draw_params = dict(matchColor=(0, 255, 0), singlePointColor=None, outImg=np.array([]), flags=2)
    img_matches_before = cv2.drawMatchesKnn(query_img, kp_query, train_img, kp_train, 
                    matches[:20], **draw_params)

    # extract points in the matches
    matched_points_query = np.vstack([kp_query[match[0].queryIdx].pt for match in matches])
    matched_points_train = np.vstack([kp_train[match[0].trainIdx].pt for match in matches])
    
    # find homography between query and train image using RANSAC algorithm
    h, status = cv2.findHomography(matched_points_query, matched_points_train, cv2.RANSAC)

    # write 
    if f:
        f.write('h Matrix:\n')
        f.write(str(h))
        f.write('\n')

    # calculate boundaries of query image after homography transformation
    corners = get_corner_coords(query_img.shape)
    transformed_corners = cv2.perspectiveTransform(np.float32([corners]), h).astype(np.int32)
    train_img_with_transformed_corners = cv2.polylines(train_img, transformed_corners, True, (246, 148, 59), 4)

    # caculate key points after homography transformation
    kp_warp = cv2.perspectiveTransform(np.array([[[kp.pt[0], kp.pt[1]] for kp in kp_query]]), h)[0]
    kp_warp = np.array([cv2.KeyPoint(pt[0], pt[1], 1) for pt in kp_warp.tolist()])
    
    # filter matched key points
    matches_after_homo = list(filter(
        lambda match: dist_kp(kp_warp[match[0].queryIdx], kp_train[match[0].trainIdx]) < 1, 
        matches
    ))

    # get total number of matches consitent with the computed homography
    n_consistent_matches = len(matches_after_homo)

    if f:
        f.write("Number of matches consistent with homography: {}\n".format(n_consistent_matches))

    # show top 10 matches after homography transformation
    img_matches_after = cv2.drawMatchesKnn(query_img, kp_query,  
                    train_img_with_transformed_corners, kp_train, 
                    matches_after_homo[:10], **draw_params)

    return img_query_features, img_train_features, img_matches_before, img_matches_after, h

def main():
    # setup argparser
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--imgdir', type=str, default=None, help='directory to source images.')
    parser.add_argument('--outdir', type=str, default=None, help='directory to save output data.')
    parser.add_argument('--show_img', help='show image results', action='store_true')
    args = parser.parse_args()

    # make out directory if not already done
    mkdir(args.outdir)

    # open log file
    f = open(os.path.join(args.outdir, 'logs.txt'), 'w')

    # get all src and dst files
    src_files = glob.glob(os.path.join(args.imgdir, 'src_*.jpg'))
    dst_files = glob.glob(os.path.join(args.imgdir, 'dst_*.jpg'))

    # loop through all combinations of source and destination files
    for i, src_file in enumerate(src_files):
        for j, dst_file in enumerate(dst_files):
            # read image files
            src_img = cv2.imread(src_file)
            dst_img = cv2.imread(dst_file)

            # get image file names without extension
            src_filename = src_file.split('/')[-1][:-4]
            dst_filename = src_file.split('/')[-1][:-4]

            f.write('Experiment: src image {} and dst image {}.'.format(i + 1, j + 1))

            # run SIFT and RANSAC on given images
            q_feature, t_feature, match_b, match_a, h = compute_homo_transformation(src_img, dst_img, f=f)

            # show annotated images if needed
            if args.show_img:
                show_img('SIFT Features for Query Image', q_feature)
                show_img('SIFT Features for Train Image', t_feature)
                show_img('Top 20 Matches before Homography', match_b)
                show_img('Top 10 Matches after Homography', match_a)

            # save images to file
            cv2.imwrite(os.path.join(args.outdir, 'img{}{}_qf.jpg'.format(i, j)), q_feature)
            cv2.imwrite(os.path.join(args.outdir, 'img{}{}_tf.jpg'.format(i, j)), t_feature)
            cv2.imwrite(os.path.join(args.outdir, 'img{}{}_mb.jpg'.format(i, j)), match_b)
            cv2.imwrite(os.path.join(args.outdir, 'img{}{}_ma.jpg'.format(i, j)), match_a)

            f.write('\n')
                
    # close log file
    f.close()
        
if __name__ == '__main__':
    main()