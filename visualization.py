import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import imageio as iio


def plot_DoG(DoG: list):
    """
    Plots octave space.
    """
    fig = plt.figure(figsize=(25, 15))

    m = len(DoG)
    n = len(DoG[0])
    i = 1

    for oct_idx, octave in enumerate(DoG):
        for lay_idx, layer in enumerate(octave):
            ax = fig.add_subplot(n, m, i)
            imgplot = plt.imshow(DoG[oct_idx][lay_idx])
            i += 1

    plt.show()


def get_plot_data(points: list, magnitude: list, vector=False, blob=False):
    '''
    Extracts location and angle from the keypoint to construct point's orientation.
    '''
    x = []
    y = []
    u = []
    v = []
    r = []

    for idx, keypoint in enumerate(points):
        x.append(keypoint.pt[0])
        y.append(keypoint.pt[1])

        if vector:
            theta = keypoint.angle
            r = magnitude[idx]
            u.append(r * np.cos(theta))
            v.append(r * np.sin(theta))

        if blob:
            scale = keypoint.size ** 2
            r.append(int(round(1.5 * scale)))

    return x, y, u, v, r


def visualize_points(points: list, path: str):
    '''
    Plots keypoints for a given image.
    '''
    if points:
        x, y, u, v, r = get_plot_data(points=points)

    image = iio.imread(path)
    imgplot = plt.imshow(image, cmap='gray')
    plt.scatter(x=x, y=y, c='r', s=20)
    plt.show()


def make_two_plots(points1: list, points2: list, subtitiles: list, path: str):
    '''
    Plots comparison between two points' sets.
    '''
    x1, y1, u1, v1, r1 = get_plot_data(points1)
    x2, y2, u2, v2, r2 = get_plot_data(points2)

    fig = plt.figure(figsize=(25, 15))

    image = iio.imread(path)
    ax1 = fig.add_subplot(1, 2, 1)
    imgplot = plt.imshow(image, cmap='gray')
    plt.scatter(x=x1, y=y1, c='r', s=10)
    ax2 = fig.add_subplot(1, 2, 2)
    imgplot = plt.imshow(image, cmap='gray')
    plt.scatter(x=x2, y=y2, c='r', s=10)

    ax1.title.set_text(subtitiles[0])
    ax2.title.set_text(subtitiles[1])
    plt.show()


def plot_orientations(path, oriented_keypoints: list, magnitudes: list):
    '''
    Plots points' orientations
    '''
    x, y, u, v, r = get_plot_data(oriented_keypoints, magnitudes, vector=True)

    fig = plt.figure(figsize=(25, 15))

    image = iio.imread(path)
    imgplot = plt.imshow(image, cmap='gray')
    plt.scatter(x=x, y=y, c='r', s=20)
    plt.quiver(x, y, u, v, color='r')
    plt.title("Points Orientations")
    plt.show()


def plot_blobs(path, oriented_keypoints: list):
    '''
    Plots points' orientations.
    '''
    x, y, u, v, r = get_plot_data(oriented_keypoints, blob=True)

    fig = plt.figure(figsize=(25, 15))

    image = iio.imread(path)
    imgplot = plt.imshow(image, cmap='gray')
    plt.scatter(x=x, y=y, c='white', s=10)
    plt.scatter(x, y, s=r,  facecolors='none', edgecolors='white')
    plt.title("Detected Blobs")
    plt.show()
