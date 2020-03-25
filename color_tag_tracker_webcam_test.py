import numpy as np
import cv2
from color_tag_tracker import find_tags

frames = 5000

cam_mtx = np.load('mtx.npy')
cam_dist = np.load('dist.npy')

camera = cv2.VideoCapture(0)

print("Starting test")

for i in range(frames):
    if i % 50 is 0:
        print(i)
    _, image = camera.read()
    tags = find_tags(image, cam_mtx, cam_dist, display_img=True)
    if len(tags) == 0:
        continue
    tag_id, r_vec, t_vec = tags[0]
    print('Tag id: ' + str(tag_id))
    print('Rotation vector:')
    print(r_vec)
    print('Translation vector:')
    print(t_vec)

print("Finished test")
