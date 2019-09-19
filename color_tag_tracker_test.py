import numpy as np
import cv2
from color_tag_tracker import find_tag

frames = 5000

cam_mtx = np.load('mtx.npy')
cam_dist = np.load('dist.npy')

camera = cv2.VideoCapture(0)

print("Starting test")

for i in range(frames):
    if i % 50 is 0:
        print(i)
    _, image = camera.read()
    find_tag(image, cam_mtx, cam_dist, debug_txt=True, display_img=False)

print("Finished test")