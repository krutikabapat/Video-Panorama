import cv2 as cv
import numpy as np

modes = (cv.Stitcher_PANORAMA, cv.Stitcher_SCANS)

# enter filename
filename = 'home.mp4'

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

# init video file
out_filename = 'sample.mp4'
# todo: is 2*width needed?
out = cv.VideoWriter(out_filename, 0x7634706d, fps/2, (2*width, height))

# read first frame
success, prev = cp.read()

# initialize SIFT image descriptor
sift = cv.xfeatures2d.SIFT_create()

count = 0
index = 0

stitcher = cv.Stitcher.create(cv.Stitcher_PANORAMA)

while(cp.isOpened()):
    frameId = int(round(cp.get(1)))
    # convert prev frame to GrayScale
    # prev_gray = cv.cvtColor(prev, cv.COLOR_BGR2GRAY)

    # find keypoints and descriptor for prev_gray
    # kp_prev, desc_prev = sift.detectAndCompute(prev_gray, mask=None)

    count += int(fps/2)
    cp.set(1, count)
    succ, curr = cp.read()
    status, pano = stitcher.stitch([prev, curr])
    print(status)
    if(status == 1):
        prev = curr
        continue
    if(status != cv.Stitcher_OK):
        prev = curr
        continue

    # dst = pano
    prev = pano
    if(pano.shape[1] >= 1200):
        pano = cv.resize(pano, (int(pano.shape[1]/2), int(pano.shape[0]/2)))

    cv.imshow("dst", pano)
    cv.waitKey(0)

    '''
    if(success):
        # only get frame after each second
        count += fps
        cp.set(1, count)

        # get next frame
        success, curr = cp.read()

        # convert to grayscale
        curr_gray = cv.cvtColor(curr, cv.COLOR_BGR2GRAY)

        # find keypoints and descriptors for curr_gray
        kp_curr, desc_curr = sift.detectAndCompute(curr_gray, mask=None)

        if(len(kp_curr) == 0):
            # in case no keypoints detected
            break

        # initialize brute force matcher
        bf = cv.BFMatcher()
        matches = bf.knnMatch(desc_prev, desc_curr, k=2)

        if(len(matches) == 0):
            print("Found no match for index: {}".format(index))
            prev = curr
            index += 1
            continue

        good = []
        for m in matches:
            if(len(m) == 2 and m[0].distance < 0.75 * m[1].distance):
                good.append((m[0].trainIdx, m[0].queryIdx))

        matches = good

        if(len(matches) > 4):
            kp1 = np.float32([kp_.pt for kp_ in kp_prev])
            kp2 = np.float32([kp_.pt for kp_ in kp_curr])

            src = np.float32([kp1[i] for (_, i) in matches])
            dst = np.float32([kp2[i] for (i, _) in matches])

            H, masked = cv.findHomography(src, dst, cv.RANSAC, 4.0)
        else:
            index += 1
            prev = curr
            continue

        if(index == 30):
            break

        dst = cv.warpPerspective(prev, H, (curr.shape[1] + prev.shape[1], prev.shape[0]))
        dst[0:curr.shape[0], 0:curr.shape[1]] = curr
        prev = dst

        if(dst.shape[1] > 1920):
            dst = cv.resize(dst, (int(dst.shape[1]/2), int(dst.shape[0]/2)))

        cv.imshow("dst", dst)
        cv.waitKey(5)
        print("Index: {}".format(index))
        index += 1
    else:
        cp.release()
        break
    '''

cp.release()

cv.imwrite("final.jpg", dst)
# stitcher = cv.Stitcher.create(cv.Stitcher_PANORAMA)
# status, pano = stitcher.stitch([prev, curr])

# dst = pano