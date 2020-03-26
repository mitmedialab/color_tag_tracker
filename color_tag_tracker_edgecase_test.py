import numpy as np
import cv2
from color_tag_tracker import find_tags
import glob

frames = 5000

cam_mtx = np.load('mtx.npy')
cam_dist = np.load('dist.npy')

print("Starting test")

for i, f in enumerate(glob.glob("test_images/*.jpg")):
    print(F"Frame{i}")

    image = cv2.imread(f)
    tags = find_tags(image, cam_mtx, cam_dist, debug_txt=True, display_img=True)
    if len(tags) == 0:
        continue
    tag_id, r_vec, t_vec = tags[0]
    print('Tag id: ' + str(tag_id))
    print('Rotation vector:')
    print(r_vec)
    print('Translation vector:')
    print(t_vec)
    if cv2.waitKey(100) & 0xFF == ord('q'):
        raise Exception("display cancelled 2")
    input()

print("Finished test")
