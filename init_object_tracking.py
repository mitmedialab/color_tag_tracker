import cv2
import numpy as np
from time import time
import math

num_of_prep_frames = 50

green_low = np.array([45, 75, 75])
green_high = np.array([75, 255, 255])

blue_low = np.array([75, 75, 128])
blue_high = np.array([105, 255, 255])


def prepare_camera(cam):
    print("Preparing camera")
    for _ in range(num_of_prep_frames):
        cam.read()
    return cam.read()[1]


def get_colour_contours(img, range_low, range_high):
    hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv_img, range_low, range_high)
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE,
                                           cv2.CHAIN_APPROX_SIMPLE)
    return contours


def get_largest_contours(contours):
    largest_area = 0
    largest_index = -1
    snd_largest_area = 0
    snd_largest_index = -1

    i = 0
    total_contours = len(contours)
    while i < total_contours:
        area = cv2.contourArea(contours[i])
        if area > largest_area:
            snd_largest_area = largest_area
            snd_largest_index = largest_index
            largest_area = area
            largest_index = i
        elif area > snd_largest_area:
            snd_largest_area = area
            snd_largest_index = i
        i += 1

    if snd_largest_index is -1:
        return contours[largest_index], None
    else:
        return contours[largest_index], contours[snd_largest_index]


def get_circle_pts(cnt):
    # Calculate extreme points of region
    ext_top = tuple(cnt[cnt[:, :, 1].argmin()][0])
    ext_right = tuple(cnt[cnt[:, :, 0].argmax()][0])
    ext_bot = tuple(cnt[cnt[:, :, 1].argmax()][0])
    ext_left = tuple(cnt[cnt[:, :, 0].argmin()][0])

    #  Calculate region centre using moments
    moments = cv2.moments(cnt)
    if moments["m00"]:
        cx = int(moments["m10"] / moments["m00"])
        cy = int(moments["m01"] / moments["m00"])
        centre = (cx, cy)
    else:
        centre = (0, 0)
        print("NO CENTRE???")
    
    return [centre, (centre[0], ext_top[1]), (ext_right[0], centre[1]),
            (centre[0], ext_bot[1]), (ext_left[0], centre[1])]


def circle_largest_green_region(img):
    contours = get_colour_contours(img, green_low, green_high)

    if not contours:
        return None, []
    main_contour, _ = get_largest_contours(contours)

    return main_contour, get_circle_pts(main_contour)


def draw_debug_circles(image, points):
    cv2.circle(image, points[0], 4, (0, 0, 0), -1)
    cv2.circle(image, points[1], 4, (0, 0, 255), -1)
    cv2.circle(image, points[2], 4, (255, 0, 255), -1)
    cv2.circle(image, points[3], 4, (255, 0, 0), -1)
    cv2.circle(image, points[4], 4, (255, 255, 0), -1)

    return image


def outline_blue_circles(img):
    contours = get_colour_contours(img, blue_low, blue_high)

    if not contours:
        return [], []

    circ_1, circ_2 = get_largest_contours(contours)

    if circ_2 is None:
        circs = [circ_1]
    else:
        circs = [circ_1, circ_2]

    circs_pts = list(map(get_circle_pts, circs))

    if len(circs_pts) is 2:
        # Compare y coordinates of centres
        if circs_pts[0][0][1] > circs_pts[1][0][1]:
            circs.reverse()
            circs_pts.reverse()

    return circs, circs_pts


def flatten(arr):
    new_arr = []
    for a in arr:
        for e in a:
            new_arr.append(e)
    return new_arr


def circle_solve_pnp(pxl_pts, cam_mat, cam_dist):
    object_points = [[0, 0, 0], [0, -3, 0], [3, 0, 0], [0, 3, 0], [-3, 0, 0]]

    _, _, obj_3d_coords = cv2.solvePnP(np.float32(object_points),
                                       np.float32(pxl_pts),
                                       cam_mat, cam_dist)
    return flatten(obj_3d_coords)


def display_images(img1, img2):
    cv2.imshow('cam input', img1)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        raise Exception("display cancelled 1")

    cv2.imshow('green area', img2)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        raise Exception("display cancelled 1")


def calc_obj_pos(obj_3d_coords, bc_3d_coords, debug=False):

    origin = bc_3d_coords[0][:2]

    y_axis = np.subtract(bc_3d_coords[1][:2], origin)
    distance_between_circles = math.sqrt(y_axis[0] ** 2 + y_axis[1] ** 2)

    y_axis = y_axis / distance_between_circles
    x_axis = [-y_axis[1], y_axis[0]]

    origin_to_obj = np.subtract(obj_3d_coords[:2], origin)

    obj_2d_x = np.inner(origin_to_obj, x_axis)
    obj_2d_y = np.inner(origin_to_obj, y_axis)
    obj_2d_coords = [obj_2d_x, obj_2d_y]

    if debug:
        print("distance between circles %.2f" % distance_between_circles)
        print("y axis")
        print(y_axis)
        print("x axis")
        print(x_axis)
        print("2d vec to obj")
        print(origin_to_obj)
        print("object coords")
        print(obj_2d_coords)
        print("")
        print("")

    return obj_2d_coords


def find_obj_coords(img, cam_mat, cam_dist, debug_txt=False, display_img=False):

    green_contour, pixel_points = circle_largest_green_region(img)

    if not pixel_points:
        green_frame = img.copy()

        cv2.drawContours(green_frame, [green_contour], -1, (0, 255, 0), 3)
        green_frame = draw_debug_circles(green_frame, pixel_points)

        if display_img:
            display_images(img, green_frame)
        return [-1, -1], green_frame

    obj_3d_coords = circle_solve_pnp(pixel_points, cam_mat, cam_dist)

    if debug_txt:
        print("circle vector")
        print(obj_3d_coords)

    blue_circs_conts, blue_circs_pts = outline_blue_circles(img)

    highlight_frame = img.copy()

    if green_contour is not None:
        cv2.drawContours(highlight_frame, [green_contour], -1, (0, 255, 0), 3)
        highlight_frame = draw_debug_circles(highlight_frame, pixel_points)

    cv2.drawContours(highlight_frame, blue_circs_conts, -1, (255, 255, 0), 3)

    blue_3d_coords = []
    for c in blue_circs_pts:
        cv2.circle(highlight_frame, c[0], 4, (0, 0, 0), -1)

        blue_3d_coords.append(circle_solve_pnp(np.float32(c), cam_mat, cam_dist))

    if display_img:
        display_images(img, highlight_frame)

    if len(blue_3d_coords) is not 2:
        print("Can't calculate object position")
        return [-1, -1], highlight_frame

    obj_pos = calc_obj_pos(obj_3d_coords, blue_3d_coords, debug_txt)
    return obj_pos, highlight_frame


def test(num_frames):
    video_capture = cv2.VideoCapture(0)
    mtx = np.load('mtx.npy')
    dist = np.load('dist.npy')

    # Check success
    if not video_capture.isOpened():
        raise Exception("Could not open video device")

    try:
        prep_frame = prepare_camera(video_capture)

        start = time()

        for _ in range(num_frames):
            ret, new_frame = video_capture.read()
            obj_coords, _ = find_obj_coords(new_frame, mtx, dist, display_img=True)
            print("Obj coords")
            print(obj_coords)
            print()
            print()

        print("%d frames took %d seconds" % (num_frames, (time() - start)))

    finally:
        # Close device
        video_capture.release()
