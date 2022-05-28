import numpy as np
import imageio as iio
import cv2
import sys
import matplotlib.pyplot as plt
import time

from rich.progress import Progress
from rich.table import Table
from rich.console import Console
from rich.prompt import Prompt, Confirm
from pprint import pprint

import sift
from ransac import ransac
from visualization import visualize_points, plot_DoG, plot_orientations

class Interface:
    def __init__(self):
        self._images = list()
        self._paths = list()
        self._oriented_kp = list()
        self._kp = list()
        self._dog = list()
        self._des = list()
        self._magnitudes = list()
        self._options = {
                "upload image": "/upload,<path>",
                "display number of images": "/displaynums",
                "display all images": "/displayall",
                "display difference of gaussian for set image": "/showdog,<imagenum>",
                "show all extremas": "/showkp,<imagenum>",
                "show oriented keypoints": "showort,<imagenum>",
                "draw matches between two pictures": "/drawmatches,<imagenum1>,<imagenum2>",
                "show picture": "/showpic,<imagenum>",
                "stitch two images": "/stitch,<imagenum1>,<imagenum2>",
                "display options": "/help",
                "exit application": "/exit",
            }
        self._commands = [
                "/upload",
                "/displaynums",
                "/displayall",
                "/showdog",
                "/showkp",
                "/showort",
                "/drawmatches",
                "/showpic",
                "/stitch",
                "/help",
                "/exit"
            ]

    def _display_image(self, idx):
        plt.imshow(self._images)

    def about(self):
        title =  "SIFT & RANSAC in Image Stitching"
        authors = "Taras Rodzin, Yaroslav Romanus, Mykhailo Kuzmyn"
        description = "Our application is aimed at helping YOU stitch multiple images into one high-rez image"
        to_start = "Here are some commands to help you navigate through our app:"
        print(title)
        print(f"created by {authors}")
        print(self.get_prompt(len(authors) + 11))
        print('\n' + description)
        print(self.get_prompt(len(description)))
        print(to_start)
        self.display_options()
        return True

    def display_options(self):
        for option, keyname in self._options.items():
            print(f"\t{option}: {keyname}")
        return True

    def process_option(self, option):
        opt = option.split(",")
        # print(opt)
        if opt[0] in self._commands:
            func_name = opt[0][1:]
            func = getattr(self, func_name)
            func(opt[1:])

    def get_prompt(self, length, ch = '_'):
        return ch * length

    def process_image(self, path, idx, idx2=None):
        img = sift.gen_base_image(path)
        pyramid = sift.construct_octave_pyramid(img)
        diff = sift.generate_dog(pyramid=pyramid)
        self._dog[idx] = diff
        print(f"for image at {path} dog generated...")

        octaves, images, keypoints = sift.detect_extrema(diff, debug=False)
        self._kp[idx] = keypoints
        print(f"for image at {path} keypoints detected...")

        correct_keypoints, incorrect_keypoints = sift.localize_extremas(keypoints, diff)
        oriented_keypoints, magnitudes = sift.keypoints_with_orientation(keypoints=correct_keypoints,                                                       images=pyramid)
        or_keypoints = sift.handle_duplicates(oriented_keypoints)
        self._magnitudes[idx] = magnitudes
        self._oriented_kp[idx] = or_keypoints
        print(f"for image at {path} keypoints oriented...")

        descriptors = sift.generate_descriptors(keypoints, pyramid)
        self._des[idx] = descriptors
        print(f"for image at {path} descriptors generated...")

        if idx2 is not None:
            self._des[idx2] = descriptors
            self._oriented_kp[idx2] = or_keypoints
            self._kp[idx2] = keypoints
            self._dog[idx2] = diff
            self._magnitudes[idx2] = magnitudes

    def upload(self, paths):
        print(f"paths specified: {len(paths)}")
        for path in paths:
            self._dog.append(None)
            self._kp.append(None)
            self._oriented_kp.append(None)
            self._des.append(None)
            self._magnitudes.append(None)
            # print(paths)
            if path not in self._paths:
                self._paths.append(path)
                try:
                    img = iio.v3.imread(path)
                    self._images.append(img)
                    print(f"file from {path} uploaded")
                except IOError:
                    print(f"{path} is not found or not an image")

                self.process_image(path, idx=len(self._dog) - 1)
                msg = f"image from {path} finished processing"
                print(msg)
                print(self.get_prompt(len(msg)))

    def displaynums(self, *args):
        lst = [x + 1 for x in range(len(self._images))]
        print(lst)
        print(str(lst).replace(", ", " ")[1:-1])

    def displayall(self, *args):
        for idx, img in enumerate(self._images):
            path = self._paths[idx]
            plt.imshow(img)
            plt.show()
            msg = f"image at {path} displayed, it's index is {idx}"
            print(msg)
            print(self.get_prompt(len(msg)))

    def showdog(self, indices):
        indices = [int(x) - 1 for x in indices]
        for i in indices:
            try:
                assert (i >= 0 and i < len(self._dog))
                path, dog = self._paths[i], self._dog[i]
                plot_DoG(dog)
                msg = f"for image at {path} difference of gaussian plotted..."
            except AssertionError:
                msg = f"{i + 1} is incorrect index"
            print(msg)
            print(self.get_prompt(len(msg)))

    def showkp(self, indices):
        indices = [int(x) for x in indices]
        for i in indices:
            try:
                assert (i > 0 and i <= len(self._images))
                idx = i - 1
                path, img, kp = self._paths[idx], self._images[idx], self._kp[idx]
                visualize_points(kp, path)
                msg = f"all extremas for image at {path} visualized..."
            except AssertionError:
                msg = f"{i} is incorrect index"
            print(msg)
            print(self.get_prompt(len(msg)))

    def showort(self, indices):
        indices = [int(x) for x in indices]
        for i in indices:
            try:
                assert (i > 0 and i <= len(self._images))
                idx = i - 1
                path, img, kp, magnitudes = self._paths[idx], self._images[idx], self._oriented_kp[idx], self._magnitudes[idx]
                plot_orientations(path, kp, magnitudes)
                msg = f"oriented keypoints for image at {path} visualized..."
            except AssertionError:
                msg = f"{i} is incorrect index"
            print(msg)
            print(self.get_prompt(len(msg)))

    def drawmatches(self, indices):
        indices = [int(x) - 1 for x in indices]
        try:
            assert (len(indices) == 2)
            assert (indices[0] in range(len(self._kp)))
            assert (indices[1] in range(len(self._kp)))
            i, j = indices
            path1, img1, kp1, des1 = self._paths[i], self._images[i], self._oriented_kp[i], self._des[i]
            path2, img2, kp2, des2 = self._paths[j], self._images[j], self._oriented_kp[j], self._des[j]
            matches = sift.match_feautures(des1, des2)
            sift.draw_matches(img1, img2, kp1, kp2, matches)
            msg = f"matches between image at\n {path1}\n" \
                  f"image at\n {path2}\n displayed"
        except AssertionError:
            msg = "one of the images does not exist..."
        print(msg)
        print(self.get_prompt(9))

    def showpic(self, indices):
        indices = [int(x) for x in indices]
        for i in indices:
            try:
                assert (i > 0 and i <= len(self._images))
                idx = i - 1
                path, img = self._paths[idx], self._images[idx]
                plt.imshow(img)
                plt.show()
                msg = f"image at {path} visualized..."
            except AssertionError:
                msg = f"{i} is incorrect index"
            print(msg)
            print(self.get_prompt(len(msg)))

    def stitch(self, indices):
        indices = [int(x) - 1 for x in indices]
        try:
            assert (len(indices) == 2)
            assert (indices[0] in range(len(self._kp)))
            assert (indices[1] in range(len(self._kp)))
            i, j = indices
            path1, img1, kp1, des1 = self._paths[i], self._images[i], self._oriented_kp[i], self._des[i]
            path2, img2, kp2, des2 = self._paths[j], self._images[j], self._oriented_kp[j], self._des[j]

            matches = sift.match_feautures(des1, des2)
            print(kp1.shape)
            print(des1.shape)
            print(matches.shape)

            sift.draw_matches(img1, img2, kp1, kp2, matches)
            src_pts = np.float32([kp1[m[0].queryIdx].pt for m in matches]).reshape(-1, 1, 2)
            dst_pts = np.float32([kp2[m[0].trainIdx].pt for m in matches]).reshape(-1, 1, 2)
            H = ransac(src_pts, dst_pts)

            dst = cv2.warpPerspective(img1, H, ((img1.shape[1] + img2.shape[1]), img2.shape[0]))  # wraped image
            dst[0:img2.shape[0], 0:img2.shape[1]] = img2  # stitched image
            plt.imshow(dst)
            plt.show()

            new_path = f"{path1[:-3]}_(1).png"
            cv2.imwrite(new_path, dst)
            combined_msg = f"image combined from images {i + 1} and {j + 1}"
            self._paths[i], self._paths[j] = combined_msg, combined_msg
            self._images[i], self._images[j] = dst, dst
            self._paths[i], self._paths[j] = new_path
            self.process_image(new_path, idx=i, idx2=j)

        except AssertionError:
            msg = "one of the images does not exist..."
        print(msg)
        print(self.get_prompt(9))

    def help(self, *args):
        self.display_options()

    def exit(self, *args):
        sys.exit()

if __name__ == "__main__":
    interface = Interface()
    interface.about()
    while True:
        option = input()
        interface.process_option(option)

