# Example of stitching two images
import numpy as np
import imageio as iio
import cv2
import matplotlib.pyplot as plt
from ransac import ransac
from sift import SIFT, draw_matches, match_feautures

if __name__ == "__main__":
    img2 = iio.imread('images/rivne2.JPG')
    img1 = iio.imread('images/rivne1.JPG')
    plt.imshow(img2)
    plt.show()
    plt.imshow(img1)
    plt.show()
    kp2, des2 = SIFT('images/rivne2.JPG')
    kp1, des1 = SIFT('images/rivne1.JPG')

    matches = match_feautures(des1, des2)

    draw_matches(img1, img2, kp1, kp2, matches)
    src_pts = np.float32([kp1[m[0].queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m[0].trainIdx].pt for m in matches]).reshape(-1, 1, 2)
    H = ransac(src_pts, dst_pts)

    dst = cv2.warpPerspective(img1,H,((img1.shape[1] + img2.shape[1]), img2.shape[0])) #wraped image
    dst[0:img2.shape[0], 0:img2.shape[1]] = img2 #stitched image
    plt.imshow(dst)
    plt.show()
