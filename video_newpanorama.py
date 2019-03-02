import numpy as np
import cv2
import matplotlib.pyplot as plt
from random import randrange
import time

cp = cv2.VideoCapture('sample.mp4')

n_frames = int(cp.get(cv2.CAP_PROP_FRAME_COUNT))
print("N_frames: ", n_frames)
width = int(cp.get(cv2.CAP_PROP_FRAME_WIDTH)) 
height = int(cp.get(cv2.CAP_PROP_FRAME_HEIGHT))

fps = cp.get(cv2.CAP_PROP_FPS)


fourcc = cv2.VideoWriter_fourcc(*'MJPG')
#print(fourcc)

# Try doing 2*width
out = cv2.VideoWriter('video_out.mp4',0x7634706d, fps/2, (2*width, height))

_, prev = cp.read()
index = 0
prev_gray = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)

for i in range(n_frames - 2):
		begin = time.time()
		sift = cv2.xfeatures2d.SIFT_create()
		kp1, des1 = sift.detectAndCompute(prev_gray,None)

		succ, curr = cp.read()
		if ( index % 20 ==0):
			curr_gray = cv2.cvtColor(curr, cv2.COLOR_BGR2GRAY)
			prev_gray = curr_gray
			kp2, des2 = sift.detectAndCompute(curr_gray,None)
			bf = cv2.BFMatcher()
			matches = bf.knnMatch(des1,des2, k=2) 
			end = time.time()
			print("Time taken: ", end-begin)
			if(len(matches) == 0):
				continue
			#print(matches)
			good = []
			for m in matches:
				if m[0].distance < 1*m[1].distance:         
					good.append(m)
			matches = np.asarray(good)
			#print(matches.shape)
			#print(matches)


			if len(matches[:,0]) >= 4:
				src = np.float32([ kp1[m.queryIdx].pt for m in matches[:,0] ]).reshape(-1,1,2)
				dst = np.float32([ kp2[m.trainIdx].pt for m in matches[:,0] ]).reshape(-1,1,2)
				H, masked = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)
				print(H)
				print(masked)
			else:
				continue
			'''
			else:
				raise AssertionError("Can't find enough keypoints.")
			'''

			dst = cv2.warpPerspective(prev,H,(curr.shape[1] + prev.shape[1], curr.shape[0]))     	
			dst[0:curr.shape[0], 0:curr.shape[1]] = curr
			if(dst.shape[1] > 1920): 
				dst = cv2.resize(dst, (dst.shape[1]/2, dst.shape[0]/2))
		index = index +1

#cv2.imwrite('resultant_stitched_panorama_' + str(i) + ".jpg",dst)
cv2.imwrite('resultant_stitched_panorama_.jpg',dst)
#cv2.imshow("img1", img1)
cv2.imshow("dst", dst)
cv2.waitKey(0)
cv2.destroyAllWindows()   
