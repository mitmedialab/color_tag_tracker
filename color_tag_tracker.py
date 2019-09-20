import cv2
import numpy as np
from time import time
import math

# TODO generalise ranges later
green_low = np.array([35, 75, 75])
green_high = np.array([85, 255, 255])

blue_low = np.array([75, 75, 128])
blue_high = np.array([105, 255, 255])

ELLIPSE_TO_DOTS_SCALE = 1.125

CM_TO_DOT_CENTER = 4.5


# Given a HSV colour range, returns the list of contours
def get_colour_contours(img, range_low, range_high):
    hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv_img, range_low, range_high)
    _, contours, _ = cv2.findContours(mask, cv2.RETR_TREE,
                                           cv2.CHAIN_APPROX_SIMPLE)
    return contours


# Given a list of contours, returns that list in order of descending size
def get_largest_contours(contours):

    ordered_contours = list(map(lambda c: (c, cv2.contourArea(c)), contours))
    ordered_contours.sort(key=lambda p: p[1], reverse=True)
    return list(map(lambda p: p[0], ordered_contours))


def calc_point_coords(ellipse, theta_2, scale=1.):
    center, axes, theta_1 = ellipse
    min_axis_len, maj_axis_len = axes
    maj_axis_len /= 2
    min_axis_len /= 2
    theta_1 = math.radians(theta_1)
    theta_2 = math.radians(theta_2)

    sin_theta_1 = math.sin(theta_1)
    cos_theta_1 = math.cos(theta_1)

    maj_axis = np.array([sin_theta_1, -cos_theta_1]) * maj_axis_len * scale
    min_axis = np.array([cos_theta_1, sin_theta_1]) * min_axis_len * scale

    return tuple((np.array(center) + maj_axis * math.cos(theta_2) + min_axis * math.sin(theta_2)).astype(int))


def bgr_to_hsv(col):
    return cv2.cvtColor(np.array([[col]]), cv2.COLOR_BGR2HSV)[0, 0]


def black_at_angle(img, ellipse, theta):
    coords = calc_point_coords(ellipse, theta, ELLIPSE_TO_DOTS_SCALE)
    if coords[1] >= img.shape[0] or coords[1] < 0 or coords[0] >= img.shape[1] or coords[0] < 0:
        return False
    colour = img[coords[1], coords[0]]
    hsv_colour = bgr_to_hsv(colour)
    return hsv_colour[2] < 165


def find_first_dot(img, ellipse, debug_txt):

    init_angle = 0.
    while not black_at_angle(img, ellipse, init_angle) and init_angle < 360:
        init_angle += 5

    if init_angle >= 360:
        if debug_txt:
            print("No dot found")
        return None

    left_angle = init_angle
    while black_at_angle(img, ellipse, left_angle - 2) and init_angle - left_angle < 15:
        left_angle -= 2
    while black_at_angle(img, ellipse, left_angle - 1) and init_angle - left_angle < 15:
        left_angle -= 1
    while black_at_angle(img, ellipse, left_angle - 0.1) and init_angle - left_angle < 15:
        left_angle -= 0.1

    right_angle = init_angle
    while black_at_angle(img, ellipse, right_angle + 2) and right_angle - init_angle < 15:
        right_angle += 2
    while black_at_angle(img, ellipse, right_angle + 1) and right_angle - init_angle < 15:
        right_angle += 1
    while black_at_angle(img, ellipse, right_angle + 0.1) and right_angle - init_angle < 15:
        right_angle += 0.1

    if right_angle - left_angle > 13:
        if debug_txt:
            print("Dot too big")
        return None
    return (left_angle + right_angle) / 2


# Given a list of arrays, flattens them and returns a single array
def flatten(arr):
    new_arr = []
    for a in arr:
        for e in a:
            new_arr.append(e)
    return new_arr


def tag_solve_pnp(pxl_pts, cam_mat, cam_dist):
    object_points = [[0, 0, 0], [0, -CM_TO_DOT_CENTER, 0], [CM_TO_DOT_CENTER, 0, 0]]

    _, rot, obj_3d_coords = cv2.solvePnP(np.float32(object_points),
                                         np.float32(pxl_pts),
                                         cam_mat, cam_dist)
    return rot, flatten(obj_3d_coords)


def display_images(img1, img2):
    cv2.imshow('cam input', img1)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        raise Exception("display cancelled 1")

    cv2.imshow('cam input with highlighted tag', img2)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        raise Exception("display cancelled 2")


def find_tag(img, cam_mat, cam_dist, debug_txt=False, display_img=False):
    # Plan
    #   Find all contours
    #   Order by size, starting with largest
    #   Starting with the first contour try
    #       Fit an ellipse
    #       Decode pattern, find orientation
    #   If fails, repeat with next smallest contour
    #   When one succeeds
    #       solvePnP
    #       return result of solvePnP until have better idea
    contours = get_colour_contours(img, green_low, green_high)
    if not contours:
        if debug_txt:
            print("No contours found")
        return

    contours = get_largest_contours(contours)

    for contour in contours:
        if len(contour) < 5:
            if debug_txt:
                print("Contour too small")
            break
        ellipse = cv2.fitEllipse(contour)

        first_dot_angle = find_first_dot(img, ellipse, debug_txt)
        if first_dot_angle is None:
            continue

        top_dot_angle = first_dot_angle

        while not (black_at_angle(img, ellipse, top_dot_angle) and black_at_angle(img, ellipse, top_dot_angle + 90)) and top_dot_angle < first_dot_angle + 360:
            top_dot_angle += 15

        if top_dot_angle >= first_dot_angle + 360:
            if debug_txt:
                print("Failed to decode tag")
            continue

        if display_img:
            highlight = img.copy()

            cv2.ellipse(highlight, ellipse, (0, 0, 255))
            cv2.circle(highlight, calc_point_coords(ellipse, top_dot_angle, ELLIPSE_TO_DOTS_SCALE), 3, (0, 255, 0))
            cv2.circle(highlight, calc_point_coords(ellipse, top_dot_angle + 90, ELLIPSE_TO_DOTS_SCALE), 3, (0, 0, 255))

            display_images(img, highlight)

        return
        # TODO need >= 4 points, redesign tags

        pixel_points = [list(ellipse[0]),
                        calc_point_coords(ellipse, top_dot_angle, ELLIPSE_TO_DOTS_SCALE),
                        calc_point_coords(ellipse, top_dot_angle + 90, ELLIPSE_TO_DOTS_SCALE)]

        r_vec, t_vecs = tag_solve_pnp(pixel_points, cam_mat, cam_dist)

        print('r_vec: ', r_vec)
        print('t_vec', t_vecs)
        print()

    return
