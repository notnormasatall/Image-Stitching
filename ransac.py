import numpy as np
from random import sample

def find_homography(src_pts, dst_pts):
  n = len(src_pts)
  A = np.zeros((2*n, 9))
  for i in range(n):
    x1 = src_pts[i][0][0]
    y1 = src_pts[i][0][1]
    x2 = dst_pts[i][0][0]
    y2 = dst_pts[i][0][1]
    A[2*i, :] = np.array([x1, y1, 1, 0, 0, 0, -x2*x1, -x2*y1, -x2])
    A[2*i+1,:] = np.array([0, 0, 0, x1, y1, 1, -y2*x1, -y2*y1, -y2])
  quadratic_form_matrix = A.T @ A
  # our homography matrix correspond to the eigenvector of this matrix with the smallest eigenvalue
  lambdas, v = np.linalg.eigh(quadratic_form_matrix)
  h = v[:, 0]
  h = [[h[0], h[1], h[2]],
       [h[3], h[4], h[5]],
       [h[6], h[7], h[8]]]
  h = np.array(h, dtype=np.float64)
  return h

def apply_homography(pt, h):
  x1, y1 = pt
  c = h[2, 0] * x1 + h[2, 1] * y1 + h[2, 2] 
  x2 = (h[0, 0] * x1 + h[0, 1] * y1 + h[0, 2]) / c
  y2 = (h[1, 0] * x1 + h[1, 1] * y1 + h[1, 2]) / c
  return [x2, y2]

def calculate_inliers(src_pts, dst_pts, h, error=3):
  inliers = 0
  n = len(src_pts)
  for i in range(n):
    x_pred, y_pred = apply_homography(src_pts[i][0], h)
    x_true, y_true = dst_pts[i][0]
    if np.sqrt((x_pred - x_true) ** 2 + (y_pred - y_true) ** 2) < error:
      inliers += 1
  return inliers

def ransac(src_pts, dst_pts, num_of_samples=4, num_of_iter=5000):
  max_inliers = 0
  n = len(src_pts)
  possible_indices = list(range(n))
  for _ in range(num_of_iter):
    indices = sample(possible_indices, 4)
    new_src_pts = [src_pts[i] for i in indices]
    new_dst_pts = [dst_pts[i] for i in indices]
    current_homography = find_homography(new_src_pts, new_dst_pts)
    current_inliers = calculate_inliers(src_pts, dst_pts, current_homography)
    if current_inliers > max_inliers:
      max_inliers = current_inliers
      homography = current_homography
  return homography