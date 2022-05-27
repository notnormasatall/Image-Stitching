# Example of stitching two images
import numpy as np
import imageio as iio
import cv2
import matplotlib.pyplot as plt
from ransac import ransac
from sift import SIFT, draw_matches, match_feautures

if __name__ == "__main__":
    img3 = iio.imread('images/building1.JPG')
    img2 = iio.imread('images/building2.JPG')
    img1 = iio.imread('images/building3.JPG')
    kp3, des3 = SIFT('images/building1.JPG')
    kp2, des2 = SIFT('images/building2.JPG')
    kp1, des1 = SIFT('images/building3.JPG')
    print("Finished extracting features")

    matches_from1_to2 = match_feautures(des1, des2)
    matches_from2_to3 = match_feautures(des2, des3)
    print("finished matching features")
    src_pts_23 = np.float32([ kp2[m[0].queryIdx].pt for m in matches_from2_to3]).reshape(-1, 1, 2)
    dst_pts_23 = np.float32([ kp3[m[0].trainIdx].pt for m in matches_from2_to3]).reshape(-1, 1, 2)
    src_pts_12 = np.float32([ kp1[m[0].queryIdx].pt for m in matches_from1_to2]).reshape(-1, 1, 2)
    dst_pts_12 = np.float32([ kp2[m[0].trainIdx].pt for m in matches_from1_to2]).reshape(-1, 1, 2)
    H_23 = ransac(src_pts_23, dst_pts_23)
    H_12 = ransac(src_pts_12, dst_pts_12)
    dst = cv2.warpPerspective(img1,H_12,((img1.shape[1] + img2.shape[1]), img2.shape[0])) #wraped image
    dst[0:img2.shape[0], 0:img2.shape[1]] = img2 #stitched image
    plt.imshow(dst)
    plt.show()
    H = H_23
    next_dst = cv2.warpPerspective(dst,H,((dst.shape[1] + img3.shape[1]), img3.shape[0])) #wraped image
    next_dst[0:img3.shape[0], 0:img3.shape[1]] = img3 #stitched image
    plt.imshow(next_dst)
    plt.show()