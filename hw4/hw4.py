''' hw4.py '''

############################################
#                                          #
#  Edits Made to the File by Jingyun Yang  #
#   1. Refactoring                         #
#   2. Edit intrinsic matrix               #
#   3. Added code to plot point cloud      #
#                                          #
############################################

import numpy as np
import cv2
import os
import argparse

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

MIN_MATCH_COUNT = 10

''' Print a numpy matrix (or vector) with name.

Args:
    name: name of the matrix to print
    matrix: matrix object to be printed
    latex: whether to print the matrix in latex format
'''
def print_matrix(name, matrix, latex=False):
    print("{}:".format(name))
    if latex:
        numbers = matrix.tolist()
        print('\\\\'.join(['&'.join(["{:.4f}".format(s) for s in row]) for row in numbers]))
    else:
        print(matrix)

''' Safe mkdir that checks directory before creation. 

Args:
    dir: directory to make
'''
def mkdir(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)

''' Perform SFM on a pair of images.

Args:
    img1: first image
    img2: second image
    outdir: output directory for results
    latex: whether to print matrices in latex format
    show_plot: whether to show plot or not
'''
def process_image_pair(img1, img2, outdir, latex=False, show_plot=False):
    #############################
    #   0. Intrinsic Matrix K   #
    #############################

    K = np.array([[518.86,     0., 285.58],
                  [    0., 519.47, 213.74],
                  [    0.,     0.,     1.]])

    ################################
    #   1. SIFT Feature Matching   #
    ################################

    # detect sift features for both images
    sift = cv2.xfeatures2d.SIFT_create()
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)

    # use flann to perform feature matching
    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    search_params = dict(checks = 50)

    flann = cv2.FlannBasedMatcher(index_params, search_params)

    matches = flann.knnMatch(des1, des2, k=2)

    # store all the good matches as per Lowe's ratio test.
    good = []
    for m,n in matches:
        if m.distance < 0.7 * n.distance:
            good.append(m)

    if len(good) > MIN_MATCH_COUNT:
        p1 = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
        p2 = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)

    draw_params = dict(matchColor = (0, 255, 0), # draw matches in green color
                         singlePointColor = None,
                         flags = 2)

    img_siftmatch = cv2.drawMatches(img1, kp1, img2, kp2, good, None, **draw_params)
    cv2.imwrite(os.path.join(outdir, 'sift_match.png'), img_siftmatch)

    ###########################
    #   2. Essential Matrix   #
    ###########################

    E, mask = cv2.findEssentialMat(p1, p2, K, cv2.RANSAC, 0.999, 1.0);

    matchesMask = mask.ravel().tolist() 

    draw_params = dict(matchColor = (0,255,0), # draw matches in green color
                         singlePointColor = None,
                         matchesMask = matchesMask, # draw only inliers
                         flags = 2)

    img_inliermatch = cv2.drawMatches(img1, kp1, img2, kp2, good, None, **draw_params)
    cv2.imwrite(os.path.join(outdir, 'inlier_match.png'), img_inliermatch)
    print_matrix("Essential matrix", E, latex=latex)

    #######################
    #   3. Recover Pose   #
    #######################

    points, R, t, mask = cv2.recoverPose(E, p1, p2)
    print_matrix("Rotation", R, latex=latex)
    print_matrix("Translation", t, latex=latex)

    # (R*p2+t)-p1
    p1_tmp = np.ones([3, p1.shape[0]])
    p1_tmp[:2,:] = np.squeeze(p1).T
    p2_tmp = np.ones([3, p2.shape[0]])
    p2_tmp[:2,:] = np.squeeze(p2).T
    print_matrix("(R*p2+t)-p1", (np.dot(R, p2_tmp) + t) - p1_tmp, latex=latex)

    ########################
    #   4. Triangulation   #
    ########################

    # calculate projection matrix for both camera
    M_r = np.hstack((R, t))
    M_l = np.hstack((np.eye(3, 3), np.zeros((3, 1))))

    P_l = np.dot(K, M_l)
    P_r = np.dot(K, M_r)

    # undistort points
    p1 = p1[np.asarray(matchesMask)==1,:,:]
    p2 = p2[np.asarray(matchesMask)==1,:,:]
    p1_un = cv2.undistortPoints(p1,K,None)
    p2_un = cv2.undistortPoints(p2,K,None)
    p1_un = np.squeeze(p1_un)
    p2_un = np.squeeze(p2_un)

    # triangulate points this requires points in normalized coordinate
    point_4d_hom = cv2.triangulatePoints(M_l, M_r, p1_un.T, p2_un.T)
    point_3d = point_4d_hom / np.tile(point_4d_hom[-1, :], (4, 1))
    point_3d = point_3d[:3, :].T

    # print point locations
    print_matrix("Point Locations", point_3d, latex=latex)

    ################################
    #   5. Output 3D Point Cloud   #
    ################################

    # plot 3D points
    fig = plt.figure()
    ax = Axes3D(fig)

    X = point_3d[:, 0].flatten()
    Y = point_3d[:, 1].flatten()
    Z = point_3d[:, 2].flatten()

    ax.scatter(X, Y, Z)

    # save figure
    plt.savefig(os.path.join(outdir, 'point_cloud.png'), format='png')
    
    # show plot if requested
    if show_plot:
        plt.show()

def main():
    # setup argparser
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--imgdir', type=str, default=None, help='directory to source images.')
    parser.add_argument('--img1', type=str, default=None, help='filename of the 1st image.')
    parser.add_argument('--img2', type=str, default=None, help='filename of the 2nd image.')
    parser.add_argument('--outdir', type=str, default=None, help='directory to save output data.')
    parser.add_argument('--latex', help='show matrices in latex format', action='store_true')
    parser.add_argument('--show_plot', help='show image results', action='store_true')
    args = parser.parse_args()

    img1 = cv2.imread(os.path.join(args.imgdir, args.img1))
    img2 = cv2.imread(os.path.join(args.imgdir, args.img2))

    mkdir(args.outdir)
    process_image_pair(img1, img2, args.outdir, args.latex, args.show_plot)

if __name__ == '__main__':
    main()
