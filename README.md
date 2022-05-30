# Image-Stitching
## Topic & Objective
Image Stitching - is a project in terms of Linear Algebra Course at Ukrainian Catholic University. Its primary objective are python implementation of **Scale Invariant Feature Transform** and **Random Sample Consensus** methods.

## Testing & Visualization Guide
Testing algorithm using preuploaded images is possible with test.py
In order to create panoramic image using own photos run image_stitcher.py
Use following commands to navigate the program:
1. /upload,<path> to upload images, multiple path upload is supported, if separated by comma
  Please note that you should upload image in order from left to right using 

2. /displayall to display all uploaded images
3. /showdog,<imagenum> to display difference of gaussian for set image
  supports multiple indices separated by comma
4. /showkp,<imagenum> to show raw extramas for set image
  supports multiple indices separated by comma
5. /clear to clear all uploaded data
6. /showort,<imagenum> to show oriented keypoints
  supports multiple indices separated by comma
7. /drawmatches,<imagenum1>,<imagenum2> to show same features across two! pictures
8. /showpic,<imagenum> displays set image
  supports multiple indices separated by comma
9. /showres to stitch all uploaded images
10. /help to display the command list
11. /exit type when you are finished to safely terminate program's runtime

## Contributors
1. Taras Rodzin
2. Yaroslav Romanus
3. Mykhailo Kuzmyn
