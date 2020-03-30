import numpy as np
import cv2
from color_tag_tracker import find_tags
import sys
import time

frames = 5000

cam_mtx = np.load('mtx.npy')
cam_dist = np.load('dist.npy')

video = cv2.VideoCapture(sys.argv[1])

print("Starting test")

frame_count = 0
tags_found = 0

start_time = time.time()

while video.isOpened():
    # Load image from video file
    ret, image = video.read()

    if not ret:
        break

    # Attempt to find tag in image
    tags = find_tags(image, cam_mtx, cam_dist, display_img=True)

    frame_count += 1

    if len(tags) == 0:
        # OPTIONAL CODE, uncomment so can save images where it fails to find a tag
        # response = input()
        # if response != '':
        #     cv2.imwrite(F'test_images/test_image{response}.jpg', image)
        continue

    tags_found += 1

    # If successful, print pose of first tag
    tag_id, r_vec, t_vec = tags[0]
    print('Tag id: ' + str(tag_id))
    print('Rotation vector:')
    print(r_vec)
    print('Translation vector:')
    print(t_vec)
    if cv2.waitKey(100) & 0xFF == ord('q'):
        break


end_time = time.time()

print("Finished test")
print(F"Time taken: {end_time - start_time}s")
print(F'Found {tags_found} in {frame_count} frames.')
