# Video-Panorama
Video Panorama using OpenCV


## Reference paper:-

1. https://pdfs.semanticscholar.org/0dab/05ef70923ac0c0bf5aaff51f8db14901f487.pdf.  


## Steps to create video Panorama:-

1. Extract feature in individual frame.  
2. Match features using RANSAC algorithm.  
3. Match the images crossing the threshold value of matched features.  
4. Stitch the Image (frames) in the video.  


## Usage:-  

<code> python3 video_panorama.py  </code>
