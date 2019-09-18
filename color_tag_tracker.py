import cv2
import numpy as np
from time import time
import math

# TODO generalise ranges later
green_low = np.array([45, 75, 75])
green_high = np.array([75, 255, 255])

blue_low = np.array([75, 75, 128])
blue_high = np.array([105, 255, 255])


# Given a HSV colour range, returns the list of contours
def get_colour_contours(img, range_low, range_high):
    hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv_img, range_low, range_high)
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE,
                                           cv2.CHAIN_APPROX_SIMPLE)
    return contours


# Given a list of contours, returns that list in order of descending size
def get_largest_contours(contours):

    ordered_contours = list(map(lambda c: (c, cv2.contourArea(c)), contours))
    ordered_contours.sort(key=lambda p: p[1], reverse=True)

    return list(map(lambda p: p[0], contours))


# Given a list of arrays, flattens them and returns a single array
def flatten(arr):
    new_arr = []
    for a in arr:
        for e in a:
            new_arr.append(e)
    return new_arr


def tag_solve_pnp(pxl_pts, cam_mat, cam_dist):
    object_points = [[0, 0, 0], [0, -3, 0], [3, 0, 0]]

    _, _, obj_3d_coords = cv2.solvePnP(np.float32(object_points),
                                       np.float32(pxl_pts),
                                       cam_mat, cam_dist)
    return flatten(obj_3d_coords)


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


    return
