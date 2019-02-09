import cv2
cp = cv2.VideoCapture('sample.mp4')

n_frames = int(cp.get(cv2.CAP_PROP_FRAME_COUNT))
#print("N_frames: ", n_frames)
width = int(cp.get(cv2.CAP_PROP_FRAME_WIDTH)) 
height = int(cp.get(cv2.CAP_PROP_FRAME_HEIGHT))

fps = cp.get(cv2.CAP_PROP_FPS)

fourcc = cv2.VideoWriter_fourcc(*'MJPG')
#print(fourcc)

# Try doing 2*width
out = cv2.VideoWriter('video_out.mp4',0x7634706d, fps/2, (2*width, height))

_, prev = cp.read()
prev_gray = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)

for i in range(n_frames - 2):
    succ, curr = cp.read()
    


'''
img = cv2.imread("/home/krutika/Documents/Image_Stitching/1.jpg")
img1 = cv2.imread("/home/krutika/Documents/Image_Stitching/2.jpg")
cv2.ocl.setUseOpenCL(False)
stitcher = cv2.createStitcher()
status, result = stitcher.stitch(img, img1) 
print(result.shape)   
cv2.imwrite('/home/krutika/Documents/Image_Stitching/result.png',result)




OpenCV(3.4.1) Error: Assertion failed ((M0.type() == 5 || M0.type() == 6) && M0.rows == 3 && M0.cols == 3) in warpPerspective, file /home/krutika/OpenCV/opencv/modules/imgproc/src/imgwarp.cpp, line 3002
Traceback (most recent call last):
  File "video_panorama.py", line 62, in <module>
    dst = cv2.warpPerspective(prev,H,(curr.shape[1] + prev.shape[1], curr.shape[0]))     	
cv2.error: OpenCV(3.4.1) /home/krutika/OpenCV/opencv/modules/imgproc/src/imgwarp.cpp:3002: error: (-215) (M0.type() == 5 || M0.type() == 6) && M0.rows == 3 && M0.cols == 3 in function warpPerspective
'''

