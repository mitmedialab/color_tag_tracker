import numpy as np
import cv2
from color_tag_tracker import find_tag
import sys

frames = 5000

cam_mtx = np.load('mtx.npy')
cam_dist = np.load('dist.npy')

video = cv2.VideoCapture(sys.argv[1])

print("Starting test")

frame_count = 0
tags_found = 0

while video.isOpened():
    ret, image = video.read()

    if not ret:
        break

    res = find_tag(image, cam_mtx, cam_dist, display_img=True)

    frame_count += 1

    if res is None:
        continue

    tags_found += 1

    tag_id, r_vec, t_vec = res
    print('Tag id: ' + str(tag_id))
    print('Rotation vector:')
    print(r_vec)
    print('Translation vector:')
    print(t_vec)
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break


print("Finished test")
print(F'Found {tags_found} in {frame_count} frames.')
