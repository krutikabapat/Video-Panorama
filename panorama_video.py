import cv2 as cv
import numpy as np

modes = (cv.Stitcher_PANORAMA, cv.Stitcher_SCANS)

# enter filename
filename = 'sample-cut.mp4'

# init VideoCapture
cp = cv.VideoCapture(filename)

n_frames = int(cp.get(cv.CAP_PROP_FRAME_COUNT))
print("Number of frames in the input video: {}".format(n_frames))

# get properties of the video for output video
width = int(cp.get(cv.CAP_PROP_FRAME_WIDTH))
height = int(cp.get(cv.CAP_PROP_FRAME_HEIGHT))
fps = cp.get(cv.CAP_PROP_FPS)

# uncomment to print frames per seconds
print("FPS: {}".format(fps))

# read first frame
success, prev = cp.read()

# initialize ORB image descriptor
orb = cv.ORB_create()

count = 0
index = 0

stitcher = cv.Stitcher.create(cv.Stitcher_PANORAMA)
imgCount = 0

while 1:
    print("Processing frame: {}".format(int(cp.get(cv.CAP_PROP_POS_FRAMES))))
    cp.set(cv.CAP_PROP_POS_FRAMES, count)
    succ, curr = cp.read()
    if succ == False:
        break
    status, pano = stitcher.stitch([prev, curr])
    if(status != cv.Stitcher_OK):
        imgCount += 1
        count += 20
        prev = curr
        continue
    imgCount += 1
    prev = pano
    cv.imwrite("images/{}.png".format(str(imgCount).zfill(2)),pano)
    count += 20

cp.release()
