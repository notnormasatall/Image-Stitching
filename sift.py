from functools import cmp_to_key
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import cv2
from cv2 import KeyPoint
from visualization import *

IMG_PATH = "./images/naruto.png"


def read_image(path: str, rgb=True) -> np.array:
    '''
    '''
    if not rgb:
        img = cv2.imread(path, 0)
    else:
        img = cv2.imread(path)
    img = img.astype('float32')
    return img


def gen_base_image(path: str, sigma=1.6, initial_blur=0.5) -> np.array:
    """
    Reads image from a given path and converts it to a base image
    by doubling its resolution and altering the amount of blur for
    octaves' construction.

    Params:
        path: path to the image
    """
    img = read_image(path, rgb=False)
    x, y = img.shape
    img = cv2.resize(img, (0, 0), fx=2, fy=2, interpolation=cv2.INTER_LINEAR)
    blur = np.sqrt(sigma**2 - (2*initial_blur)**2)

    return cv2.GaussianBlur(img, (0, 0), sigmaX=blur, sigmaY=blur)


def generate_octaves_sigmas(img: np.array, intervals_num=3) -> list:
    '''
    Generates the ovtave pyramid from the given base image.
    '''
    x, y = img.shape

    layers_num = intervals_num + 3
    octaves_num = int(round(np.log(min(x, y)) / np.log(2) - 1))

    k = 2 ** (1 / intervals_num)
    sigmas = [1.6]

    for i in range(1, layers_num):
        sigma_prev = (k ** (i - 1)) * 1.6
        sigma_total = k * sigma_prev
        sigmas.append(np.sqrt(sigma_total ** 2 - sigma_prev ** 2))
    return octaves_num, np.array(sigmas)


def construct_octave_pyramid(img: np.array) -> list:
    '''
    Constructs an octave pyramid from the given base image.
    While iterating over octaves, takes the third last image as
    a base for the next octave.

    Params:
        img: base image
    '''
    octaves_num, sigmas = generate_octaves_sigmas(img=img)

    octaves = []

    for octave_index in range(octaves_num):
        layers = [img]

        for gaussian_kernel in sigmas[1:]:
            img = cv2.GaussianBlur(
                img, (0, 0), sigmaX=gaussian_kernel, sigmaY=gaussian_kernel)
            layers.append(img)

        octaves.append(layers)
        new_octave = layers[-3]
        img = cv2.resize(new_octave, (int(
            new_octave.shape[1] / 2), int(new_octave.shape[0] / 2)), interpolation=cv2.INTER_NEAREST)

    return octaves


def generate_dog(pyramid: list) -> list:
    '''
    Takes the difference-of-Gaussians for each octave in the octave pyramid.
    '''
    DoG = []

    for octave in pyramid:
        differences = []

        for i in range(1, len(octave)):
            differences.append(cv2.subtract(octave[i], octave[i-1]))

        DoG.append(differences)

    return DoG


def detect_extrema(dog: list, intervals_num=3, debug=False):
    '''
    Iterates over each octave, takes a layer and its lower and upper neighbours,
    iterates over 3x3x3 pixel subarea to detect if cube's center point is an
    extrema.
    '''
    threshold = np.floor(0.5 * 0.04 / intervals_num * 255)

    octaves = []
    images = []
    unfiltered_extremes = []

    for oct_idx, octave in enumerate(dog):
        for lay_idx, layer in enumerate(octave):

            if lay_idx != 0 and lay_idx + 1 != len(octave):
                if debug:
                    print(
                        f"For layer {oct_idx+1}.{lay_idx+1} of size {layer.shape}: ", end="")
                count = 0
                for i in range(5, layer.shape[0]-5):
                    for j in range(5, layer.shape[1]-5):

                        if if_extrema(
                                octave[lay_idx-1][i-1:i+2, j-1:j+2],
                                octave[lay_idx][i-1:i+2, j-1:j+2],
                                octave[lay_idx+1][i-1:i+2, j-1:j+2], threshold):
                            keypoint = KeyPoint()
                            keypoint.pt = (
                                (j) * (2 ** (oct_idx)), (i) * (2 ** (oct_idx)))
                            keypoint.octave = oct_idx + lay_idx * (2 ** 8)

                            keypoint.size = 1.6 * \
                                (2 ** ((lay_idx) / np.float32(intervals_num))
                                 ) * (2 ** (oct_idx + 1))
                            unfiltered_extremes.append(keypoint)
                            octaves.append(oct_idx)
                            images.append(lay_idx)
                        count += 1
                if debug:
                    print(f"{count} areas observed!")

    return octaves, images, unfiltered_extremes


def if_extrema(layer_one, layer_two, layer_three, threshold):
    '''
    Checks if the center point in 3x3x3 cube is extrema (min\max)
    '''
    center = layer_two[1][1]
    extrema = False

    if np.abs(center) < threshold:
        return False

    if center >= layer_one[0][0] and center >= layer_one[0][1] and \
       center >= layer_one[0][2] and center >= layer_one[1][0] and \
       center >= layer_one[1][1] and center >= layer_one[1][2] and \
       center >= layer_one[2][0] and center >= layer_one[2][1] and \
       center >= layer_one[2][2] and center >= layer_two[0][0] and \
       center >= layer_two[0][1] and center >= layer_two[0][2] and \
       center >= layer_two[1][0] and center >= layer_two[1][2] and \
       center >= layer_two[2][0] and center >= layer_two[2][1] and \
       center >= layer_two[2][2] and center >= layer_three[0][0] and \
       center >= layer_three[0][1] and center >= layer_three[0][2] and \
       center >= layer_three[1][0] and center >= layer_three[1][1] and \
       center >= layer_three[1][2] and center >= layer_three[2][0] and \
       center >= layer_three[2][1] and center >= layer_three[2][2]:
        extrema = True

    elif center <= layer_one[0][0] and center <= layer_one[0][1] and \
            center <= layer_one[0][2] and center <= layer_one[1][0] and \
            center <= layer_one[1][1] and center <= layer_one[1][2] and \
            center <= layer_one[2][0] and center <= layer_one[2][1] and \
            center <= layer_one[2][2] and center <= layer_two[0][0] and \
            center <= layer_two[0][1] and center <= layer_two[0][2] and \
            center <= layer_two[1][0] and center <= layer_two[1][2] and \
            center <= layer_two[2][0] and center <= layer_two[2][1] and \
            center <= layer_two[2][2] and center <= layer_three[0][0] and \
            center <= layer_three[0][1] and center <= layer_three[0][2] and \
            center <= layer_three[1][0] and center <= layer_three[1][1] and \
            center <= layer_three[1][2] and center <= layer_three[2][0] and \
            center <= layer_three[2][1] and center <= layer_three[2][2]:
        extrema = True

    return extrema


def localize_extremas(unfiltered_keypoints: list, dog: list, border_width=5, contrast_threshold=0.04, intervals_num=3, edge_ratio=10, num_localizations=5, sigma=1.6) -> list:
    '''
    Localizes the given kepoints set.
    '''
    correct_keypoints = []  # for storing filtered keypoints
    incorrect_keypoints = []

    for keypoint in unfiltered_keypoints:
        octave_num = keypoint.octave % (2**8)
        layer_num = keypoint.octave // (2**8)
        shape = dog[octave_num][layer_num].shape
        # converting idx to the correct resoltuion
        j, i = keypoint.pt
        j = int(j // (2**(octave_num)))
        i = int(i // (2**(octave_num)))
        is_bad = False
        prev_i, prev_j, prev_layer_num = i, j, layer_num
        for attempt in range(num_localizations):
            # creating "pixel cube" near our keypoint
            prev_image, cur_image, next_image = dog[octave_num][layer_num-1:layer_num+2]
            layer_one = prev_image[i-1:i+2, j-1:j+2].astype('float32') / 255.
            layer_two = cur_image[i-1:i+2, j-1:j+2].astype('float32') / 255.
            layer_three = next_image[i-1:i+2, j-1:j+2].astype('float32') / 255.
            grad = compute_gradient(layer_one, layer_two, layer_three)
            hessian = compute_hessian(layer_one, layer_two, layer_three)
            # best quadratic fit
            update = -np.linalg.lstsq(hessian, grad, rcond=None)[0]
            # in this case, we have already correctly localized
            if abs(update[0]) < 0.5 and abs(update[1]) < 0.5 and abs(update[2]) < 0.5:
                break
            # finding new nearest point
            prev_i, prev_j, prev_layer_num = i, j, layer_num
            j += int(round(update[0]))
            i += int(round(update[1]))
            layer_num += int(round(update[2]))
            # checking whether new point is inside image
            if i < border_width or i >= shape[0] - border_width or j < border_width or j >= shape[1] - border_width or layer_num < 1 or layer_num > intervals_num:
                is_bad = True
                break
        if is_bad:
            bad_keypoint = KeyPoint()
            bad_keypoint.pt = keypoint.pt
            bad_keypoint.octave = keypoint.octave
            bad_keypoint.size = keypoint.size
            incorrect_keypoints.append(bad_keypoint)
            continue
        # too much iterations to localize, this keypoint is considered unstable
        if attempt >= num_localizations - 1:
            bad_keypoint = KeyPoint()
            bad_keypoint.pt = keypoint.pt
            bad_keypoint.octave = keypoint.octave
            bad_keypoint.size = keypoint.size
            incorrect_keypoints.append(bad_keypoint)
            continue
        # calculating value in the point of extremum to check contrast and edge
        value_of_function = dog[octave_num][prev_layer_num][prev_i, prev_j].astype(
            'float32') / 255. + 1/2 * np.dot(grad, update)
        # check of contrast
        if abs(value_of_function) * intervals_num >= contrast_threshold:
            # checking edge
            hessian_for_xy = np.matrix([[hessian[0][0], hessian[0][1]],
                                        [hessian[1][0], hessian[1][1]]])
            tr = np.trace(hessian_for_xy)
            det = np.linalg.det(hessian_for_xy)
            if det > 0 and (tr ** 2) / det < ((edge_ratio+1) ** 2) / edge_ratio:
                # edge check passed - our keypoint is correct
                good_keypoint = KeyPoint()
                good_keypoint.pt = (
                    (j + update[0]) * (2 ** (octave_num)), (i + update[1]) * (2 ** (octave_num)))
                good_keypoint.octave = octave_num + layer_num * \
                    (2 ** 8) + int(round((update[2] + 0.5) * 255)) * (2 ** 16)
                good_keypoint.size = sigma * \
                    (2 ** ((layer_num + update[2]) /
                     np.float32(intervals_num))) * (2 ** (octave_num+1))
                good_keypoint.response = abs(value_of_function)
                correct_keypoints.append(good_keypoint)
                continue

        # at this moment, we can have only bad keypoint
        bad_keypoint = KeyPoint()
        bad_keypoint.pt = keypoint.pt
        bad_keypoint.octave = keypoint.octave
        bad_keypoint.size = keypoint.size
        incorrect_keypoints.append(bad_keypoint)
    return [correct_keypoints, incorrect_keypoints]


def compute_gradient(layer_one, layer_two, layer_three) -> list:
    dx = 1/2 * (layer_two[1][2] - layer_two[1][0])
    dy = 1/2 * (layer_two[2][1] - layer_two[0][1])
    ds = 1/2 * (layer_three[1][1] - layer_one[1][1])
    return [dx, dy, ds]


def compute_hessian(layer_one, layer_two, layer_three) -> list:
    dxx = layer_two[1][2] - 2 * layer_two[1][1] + layer_two[1][0]
    dyy = layer_two[2][1] - 2 * layer_two[1][1] + layer_two[0][1]
    dss = layer_three[1][1] - 2 * layer_two[1][1] + layer_one[1][1]
    dxy = 1/4 * (layer_two[2][2] + layer_two[0][0] -
                 layer_two[2][0] - layer_two[0][2])
    dxs = 1/4 * (layer_three[1][2] + layer_one[1][0] -
                 layer_three[1][0] - layer_one[1][2])
    dys = 1/4 * (layer_three[2][1] + layer_one[0][1] -
                 layer_three[0][1] - layer_one[2][1])
    return [[dxx, dxy, dxs],
            [dxy, dyy, dys],
            [dxs, dys, dss]]


def keypoints_with_orientation(keypoints: list, images: list) -> list:
    '''
    Performs orientation assignment onto the given set of keypoints.
    '''
    oriented_keypoints = []
    magnitudes = []

    for keypoint in keypoints:

        octave_num = keypoint.octave % (2**8)
        layer_num = ((keypoint.octave - octave_num) // (2**8)) % (2**8)
        oriented_keypoint, magnitude = compute_orientation(
            keypoint, octave_num, images[octave_num][layer_num])
        oriented_keypoint = descale_keypoints(oriented_keypoint)
        oriented_keypoints += oriented_keypoint
        magnitudes += magnitude

    return oriented_keypoints, magnitudes


def descale_keypoints(keypoints: list):
    '''
    Alters keypoint's size and location with respect on the initial image shape.
    '''
    for k in keypoints:
        x, y = k.pt
        k.pt = (x / 2, y / 2)
        k.size /= 2
        k.octave %= (2 ** 8)

    return keypoints


def compute_orientation(keypoint: KeyPoint, octave_num, image, hist_areas=36, radius_factor=3, scale_factor=1.5) -> list:
    '''
    Computes the orientation (multiple if there are several spikes within 80%
    of the maximum one) for the given keypoint.
    '''
    keypoint_orrientations = []
    magnitudes = []
    im_shape = image.shape

    # calculate the radius of area around keypoint given its scale
    scale = scale_factor*keypoint.size / np.float32(2 ** (octave_num + 1))
    radius = int(round(radius_factor * scale))
    weight_factor = -0.5 / (scale ** 2)
    raw_histogram = np.zeros(hist_areas)
    smooth_histogram = np.zeros(hist_areas)

    for i in range(-radius, radius + 1):
        for j in range(-radius, radius + 1):

            x = int(round(keypoint.pt[0] / np.float32(2 ** octave_num))) + i
            y = int(round(keypoint.pt[1] / np.float32(2 ** octave_num))) + j

            if (0 < y < im_shape[0] - 1) and (0 < x < im_shape[1] - 1):

                # compute gradient magnitude and orientation
                dx = image[y, x + 1] - image[y, x - 1]
                dy = image[y - 1, x] - image[y + 1, x]

                grad_magnitude = np.sqrt(np.square(dx) + np.square(dy))
                grad_orientation = np.rad2deg(np.arctan2(dy, dx))

                # apply weightning given the distance from the keypoint
                weight = np.exp(weight_factor * (i ** 2 + j ** 2))
                hist_idx = int(round(grad_orientation / 10)) % hist_areas
                raw_histogram[hist_idx] += weight * grad_magnitude

    # histogram smoothing
    for n in range(hist_areas):
        smooth_histogram[n] = (6 * raw_histogram[n] + 4 * (raw_histogram[n - 1] + raw_histogram[(
            n + 1) % hist_areas]) + raw_histogram[n - 2] + raw_histogram[(n + 2) % hist_areas]) / 16.

    max_orientation = max(smooth_histogram)
    orientation_peaks = np.where(np.logical_and(smooth_histogram > np.roll(
        smooth_histogram, 1), smooth_histogram > np.roll(smooth_histogram, -1)))[0]

    # after finding all top peaks save them as unique keypoint representation
    for peak_index in orientation_peaks:
        peak_value = smooth_histogram[peak_index]
        if peak_value / max_orientation >= 0.8:

            left_value = smooth_histogram[(peak_index - 1) % hist_areas]
            right_value = smooth_histogram[(peak_index + 1) % hist_areas]
            interpolated_peak_index = (peak_index + 0.5 * (left_value - right_value) / (
                left_value - 2 * peak_value + right_value)) % hist_areas
            orientation = 360 - interpolated_peak_index * 360 / hist_areas

            if abs(orientation - 360) < 1e-7:
                orientation = 0

            new_keypoint = KeyPoint(
                *keypoint.pt, keypoint.size, orientation, keypoint.response, keypoint.octave)
            keypoint_orrientations.append(new_keypoint)
            magnitudes.append(peak_value)

    return keypoint_orrientations, magnitudes


def generate_descriptors(keypoints: list, images: list, desc_size=4, num_bins=8, scale_multiplier=3, descriptor_max_value=0.2):

    descriptors = []

    for kp in keypoints:
        oct_num = kp.octave % 256
        lay_num = (kp.octave >> 8) & 255
        scale = 1 / np.float32(2 ** (oct_num))
        img = images[oct_num+1][lay_num]
        x_size, y_size = img.shape
        x, y = round(scale * kp.pt[0]), round(scale * kp.pt[1])

        counterrotation_angle = 360. - kp.angle
        cos_angle = np.cos(np.deg2rad(counterrotation_angle))
        sin_angle = np.sin(np.deg2rad(counterrotation_angle))
        weight_multiplier = -0.5 / ((0.5 * desc_size) ** 2)

        row_rel_lst = []
        col_rel_lst = []
        magnitude_lst = []
        orientation_lst = []

        descriptor = {
            "row_pos": [],
            "col_pos": [],
            "magnitude": [],
            "orientation": [],
        }

        desc_3d_arr = np.zeros((desc_size + 2, desc_size + 2, num_bins))

        kp_region = scale_multiplier * kp.size * \
            scale / 2  # size of region around keypoint
        radius = int(round(kp_region / np.sqrt(2) * (desc_size + 1)))
        radius = int(min(radius, np.sqrt(x_size ** 2 + y_size ** 2)))

        for row in range(-radius, radius+1):
            for col in range(-radius, radius+1):
                row_inv = col * sin_angle + row * cos_angle
                col_inv = col * cos_angle - row * sin_angle
                row_descaled = (row_inv / kp_region) + 0.5 * desc_size - 0.5
                col_descaled = (col_inv / kp_region) + 0.5 * desc_size - 0.5

                if row_descaled > -1 and row_descaled < desc_size and col_descaled > -1 and col_descaled < desc_size:
                    abs_row = int(round(y + row))
                    abs_col = int(round(x + col))
                    if abs_row > 0 and abs_row < x_size - 1 and abs_col > 0 and abs_col < y_size - 1:
                        x_contrast = img[abs_row, abs_col + 1] - \
                            img[abs_row, abs_col - 1]
                        y_contrast = img[abs_row - 1, abs_col] - \
                            img[abs_row + 1, abs_col]

                        gradient_magnitude = np.sqrt(
                            x_contrast ** 2 + y_contrast ** 2)
                        gradient_orientation = np.rad2deg(
                            np.arctan2(y_contrast, x_contrast)) % 360
                        weight = np.exp(
                            weight_multiplier * ((row_inv / kp_region) ** 2 + (col_inv / kp_region) ** 2))

                        descriptor["row_pos"].append(row_descaled)
                        descriptor["col_pos"].append(col_descaled)
                        descriptor["magnitude"].append(
                            weight * gradient_magnitude)
                        descriptor["orientation"].append(
                            (gradient_orientation - counterrotation_angle) * num_bins / 360)

        for idx in range(len(descriptor["row_pos"])):
            row, col, mag, ornt = descriptor["row_pos"][idx], descriptor["col_pos"][idx], \
                descriptor["magnitude"][idx], descriptor["orientation"][idx]

            row_int, col_int, or_int = int(np.floor(row)), int(
                np.floor(col)), int(np.floor(ornt))
            row_dec, col_dec, or_dec = row - row_int, col - col_int, ornt - or_int
            or_int %= 8

            c1 = mag * row_dec
            c0 = mag * (1 - row_dec)
            c11 = c1 * col_dec
            c10 = c1 * (1 - col_dec)
            c01 = c0 * col_dec
            c00 = c0 * (1 - col_dec)
            c111 = c11 * or_dec
            c110 = c11 * (1 - or_dec)
            c101 = c10 * or_dec
            c100 = c10 * (1 - or_dec)
            c011 = c01 * or_dec
            c010 = c01 * (1 - or_dec)
            c001 = c00 * or_dec
            c000 = c00 * (1 - or_dec)

            desc_3d_arr[row_int + 1, col_int + 1, or_int] += c000
            desc_3d_arr[row_int + 1, col_int + 1,
                        (or_int + 1) % num_bins] += c001
            desc_3d_arr[row_int + 1, col_int + 2, or_int] += c010
            desc_3d_arr[row_int + 1, col_int + 2,
                        (or_int + 1) % num_bins] += c011
            desc_3d_arr[row_int + 2, col_int + 1, or_int] += c100
            desc_3d_arr[row_int + 2, col_int + 1,
                        (or_int + 1) % num_bins] += c101
            desc_3d_arr[row_int + 2, col_int + 2, or_int] += c110
            desc_3d_arr[row_int + 2, col_int + 2,
                        (or_int + 1) % num_bins] += c111

        descriptor_vector = desc_3d_arr[1:-1, 1:-1, :].flatten()

        threshold = np.linalg.norm(descriptor_vector) * descriptor_max_value
        descriptor_vector[descriptor_vector > threshold] = threshold
        descriptor_vector /= max(np.linalg.norm(descriptor_vector), 1e-7)

        descriptor_vector = np.ndarray.round(
            np.multiply(descriptor_vector, 2 ** 9))
        descriptor_vector[descriptor_vector < 0] = 0
        descriptor_vector[descriptor_vector > 255] = 255
        descriptors.append(descriptor_vector)

    return np.array(descriptors, dtype='float32')


def cmp_keypoints(keypoint1, keypoint2):
    if keypoint1.pt[0] != keypoint2.pt[0]:
        return keypoint1.pt[0] - keypoint2.pt[0]
    if keypoint1.pt[1] != keypoint2.pt[1]:
        return keypoint1.pt[1] - keypoint2.pt[1]
    if keypoint1.size != keypoint2.size:
        return keypoint2.size - keypoint1.size
    if keypoint1.angle != keypoint2.angle:
        return keypoint1.angle - keypoint2.angle
    if keypoint1.response != keypoint2.response:
        return keypoint2.response - keypoint1.response
    if keypoint1.octave != keypoint2.octave:
        return keypoint2.octave - keypoint1.octave
    return keypoint2.class_id - keypoint1.class_id


def handle_duplicates(keypoints: list) -> list:
    if len(keypoints) < 2:
        return keypoints
    keypoints.sort(key=cmp_to_key(cmp_keypoints))
    without_duplicates = [keypoints[0]]
    for keypoint in keypoints[1:]:
        last = without_duplicates[-1]
        if last.pt[0] != keypoint.pt[0] or last.pt[1] != keypoint.pt[1] or last.size != keypoint.size or last.angle != keypoint.angle:
            without_duplicates.append(keypoint)
    return without_duplicates


def SIFT(path: str) -> list:
    img = gen_base_image(path)
    pyramid = construct_octave_pyramid(img)
    diff = generate_dog(pyramid=pyramid)
    octaves, images, keypoints = detect_extrema(diff, debug=False)
    correct_keypoints, incorrect_keypoints = localize_extremas(keypoints, diff)
    oriented_keypoints, magnitudes = keypoints_with_orientation(keypoints=correct_keypoints,
                                                                images=pyramid)
    keypoints = handle_duplicates(oriented_keypoints)
    descriptors = generate_descriptors(keypoints, pyramid)
    return [keypoints, descriptors]


def match_feautures(des1: list, des2: list, number_of_neighbours=2, ratio=0.75) -> list:
    # creating FLANN object(FLANN - Fast Library for Approximate Nearest Neighbours)
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)

    matches = flann.knnMatch(des1, des2, k=2)

    # ratio test(to exclude bad matches)
    good = []
    for m, n in matches:
        if m.distance < ratio * n.distance:
            good.append([m])
    return good


def draw_matches(img1, img2, kp1, kp2, good):
    img3 = cv2.drawMatchesKnn(img1, kp1, img2, kp2, good,
                              None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    fig = plt.figure(figsize=(25, 15))
    plt.imshow(img3, cmap='gray')
    plt.show()


if __name__ == "__main__":
    BASE_IMG = gen_base_image(IMG_PATH)

    # octave pyramid
    pyramid = construct_octave_pyramid(BASE_IMG)
    diff = generate_dog(pyramid=pyramid)

    # visualize octaves
    plot_DoG(diff, save=True)

    # find extremes
    octaves, images, keypoints = detect_extrema(diff, debug=False)

    # visualize all extremes
    visualize_points(points=keypoints, path=IMG_PATH, save=True)

    # localization
    correct_keypoints, incorrect_keypoints = localize_extremas(keypoints, diff)

    # plot localization results
    make_two_plots(correct_keypoints,
                   incorrect_keypoints,
                   [f"Correct Keypoints {len(correct_keypoints)}",
                    f"Incorrect Keypoints {len(incorrect_keypoints)}"],
                   IMG_PATH, save=True)

    # before & after localization
    make_two_plots(keypoints,
                   correct_keypoints,
                   [f"Before Localization: {len(keypoints)}",
                    f"After Localization: {len(correct_keypoints)}"],
                   IMG_PATH, save=True)

    # orientation
    oriented_keypoints, magnitudes = keypoints_with_orientation(keypoints=correct_keypoints,
                                                                images=pyramid)

    # plot orientation results
    plot_orientations(IMG_PATH, oriented_keypoints, magnitudes, save=True)

    # plot blobs
    plot_blobs(IMG_PATH, oriented_keypoints, save=True)
