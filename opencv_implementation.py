import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import cv2
import imageio as iio

def match_feautures(des1: list, des2: list, number_of_neighbours = 2, ratio = 0.75) -> list:
  # creating FLANN object(FLANN - Fast Library for Approximate Nearest Neighbours)
  FLANN_INDEX_KDTREE = 1
  index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
  search_params = dict(checks=50)
  flann = cv2.FlannBasedMatcher(index_params,search_params)

  matches = flann.knnMatch(des1,des2,k=2)

  # ratio test(to exclude bad matches)
  good = []
  for m, n in matches:
    if m.distance < ratio * n.distance:
      good.append([m])
  return good

def draw_matches(img1, img2, kp1, kp2, good):
  img3 = cv2.drawMatchesKnn(img1,kp1,img2,kp2,good,None,flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
  plt.imshow(img3),plt.show()

if __name__ == "__main__":
    img2_to_show = iio.imread('images/rivne2.JPG')
    img1_to_show = iio.imread('images/rivne1.JPG')
    img2 = cv2.imread('images/rivne2.JPG',cv2.IMREAD_GRAYSCALE)
    img1 = cv2.imread('images/rivne1.JPG',cv2.IMREAD_GRAYSCALE)

    # extracting SIFT keypoints and detectors(here should be our code)
    sift = cv2.SIFT_create()
    kp1, des1 = sift.detectAndCompute(img1,None)
    kp2, des2 = sift.detectAndCompute(img2,None)

    matches = match_feautures(des1, des2)

    draw_matches(img1, img2, kp1, kp2, matches)
    src_pts = np.float32([ kp1[m[0].queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([ kp2[m[0].trainIdx].pt for m in matches]).reshape(-1, 1, 2)
    H, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    dst = cv2.warpPerspective(img1_to_show, H, ((img1_to_show.shape[1] + img2_to_show.shape[1]), img2_to_show.shape[0])) #wraped image
    dst[0:img2_to_show.shape[0], 0:img2_to_show.shape[1]] = img2_to_show #stitched image
    plt.imshow(dst)
    plt.show()