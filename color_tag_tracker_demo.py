import numpy as np
import cv2
from color_tag_tracker import find_tags

frames = 5000

cam_mtx = np.load('mtx.npy')
cam_dist = np.load('dist.npy')

# Coordinates of ends of axis in object space
axis = np.float32([[0, 0, 0], [2, 0, 0], [0, -2, 0], [0, 0, 2]])


# Draw 3 axis lines on image img using point axis_points - points of axis
def draw_axis(img, axis_points):
    origin = tuple(axis_points[0].ravel())
    img = cv2.line(img, origin, tuple(axis_points[1].ravel()), (255, 0, 0), 3)  # x-axis is blue
    img = cv2.line(img, origin, tuple(axis_points[3].ravel()), (0, 0, 255), 3)  # z-axis is red
    img = cv2.line(img, origin, tuple(axis_points[2].ravel()), (0, 255, 0), 3)  # y-axis is green
    return img


# Coordinates of cube vertices in object space
cube = np.float32([[2.5, 0, 0], [0, 0, 2.5], [-2.5, 0, 0], [0, 0, -2.5],
                   [2.5, -3.535, 0], [0, -3.535, 2.5], [-2.5, -3.535, 0], [0, -3.535, -2.5]])


# Draws cube on image img using
def draw_cube(img, cube_points):
    cube_points = np.int32([cp.ravel() for cp in cube_points])

    img = cv2.drawContours(img, [cube_points[:4]], -1, (0, 255, 0), -3)

    # draw pillars in blue color
    for i, j in zip(range(4), range(4, 8)):
        img = cv2.line(img, tuple(cube_points[i]), tuple(cube_points[j]), 255, 3)

    # draw top layer in red color
    img = cv2.drawContours(img, [cube_points[4:]], -1, (0, 0, 255), 3)

    return img


camera = cv2.VideoCapture(0)

print("Starting test")

for frame_num in range(frames):
    if frame_num % 50 is 0:
        print(frame_num)

    # Read image from camera
    _, image = camera.read()

    # Find tag in image

    tags = find_tags(image, cam_mtx, cam_dist, display_img=False)

    if len(tags) != 0:
        # If any tags are found, print the pose of the first tag
        tag_id, r_vec, t_vec = tags[0]
        print('Tag id: ' + str(tag_id))
        print('Rotation vector:')
        print(r_vec)
        print('Translation vector:')
        print(t_vec)

        if tag_id == 0:
            # If it has ID 0, draw axis on the image
            axis_proj_points, _ = cv2.projectPoints(axis, r_vec, t_vec, cam_mtx, cam_dist)
            image = draw_axis(image, axis_proj_points)
        else:
            # For any other ID, draw a cube
            cube_proj_points, _ = cv2.projectPoints(cube, r_vec, t_vec, cam_mtx, cam_dist)
            image = draw_cube(image, cube_proj_points)
    else:
        print('Tag not found')

    print()

    # Show image with axis/cube drawn on top
    cv2.imshow('Demo image', image)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        raise Exception("display cancelled 2")

print("Finished test")
