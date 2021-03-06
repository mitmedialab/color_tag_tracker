import cv2
import numpy as np
import math

white_low = np.array([0, 0, 190])
white_high = np.array([255, 100, 255])

green_low = np.array([35, 50, 75])
green_high = np.array([85, 255, 255])

ELLIPSE_TO_DOTS_SCALE = 1.25

CM_TO_DOT_CENTER = 5

ANGLE_BETWEEN_DOTS = 22.5

WIDTH_OF_DOT = 17.5


# Given a HSV colour range, returns the list of contours
def get_colour_contours(hsv_img, range_low, range_high):
    mask = cv2.inRange(hsv_img, range_low, range_high)
    major = cv2.__version__.split('.')[0]
    if major == '3':
        _, contours, _ = cv2.findContours(mask,
                                          cv2.RETR_TREE,
                                          cv2.CHAIN_APPROX_SIMPLE)
    else:
        contours, _ = cv2.findContours(mask,
                                       cv2.RETR_TREE,
                                       cv2.CHAIN_APPROX_SIMPLE)
    return contours


# Given a list of contours, returns that list in order of descending size
def sort_contours_on_size(contours):
    ordered_contours = list(map(lambda c: (c, cv2.contourArea(c)), contours))
    ordered_contours.sort(key=lambda p: p[1], reverse=True)
    return list(map(lambda p: p[0], ordered_contours))


# Given an image, returns a list of ellipses (opencv RotatedRect) of white regions in image
# Ignores small regions
def find_white_ellipses(hsv_img):
    white_contours = get_colour_contours(hsv_img, white_low, white_high)

    ellipses = []
    for c in white_contours:
        if len(c) >= 20:
            ellipse = cv2.fitEllipse(c)
            ellipses.append(ellipse)
    return ellipses


# Finds ellipse in possible_ellipses which best matches target_ellipse
def get_matching_ellipse(target_ellipse, possible_ellipses):
    best_ellipse = None
    best_ellipse_area = 0

    target_coords, target_axes, theta = target_ellipse

    c_x, c_y = target_coords
    maj_len, min_len = target_axes

    maj_len /= 2
    min_len /= 2

    maj_axis_len_sqr = maj_len * maj_len
    min_axis_len_sqr = min_len * min_len

    if maj_axis_len_sqr <= 0.0 or min_axis_len_sqr <= 0.0:
        return None

    theta_rad = math.radians(theta)
    cos_theta = math.cos(theta_rad)
    sin_theta = math.sin(theta_rad)

    for e in possible_ellipses:
        e_coords, e_axes, _ = e

        d_x = e_coords[0] - c_x
        d_y = e_coords[1] - c_y

        comp_1 = (cos_theta * d_x + sin_theta * d_y) ** 2
        comp_1 /= maj_axis_len_sqr
        # if comp_1 > 1:
        #     continue

        comp_2 = (sin_theta * d_x - cos_theta * d_y) ** 2
        comp_2 /= min_axis_len_sqr

        dist_from_ellipse = comp_1 + comp_2
        if dist_from_ellipse > 1:
            continue

        # Leave out constant factor of pi
        ellipse_area_approx = (e_axes[0] / 2) * (e_axes[1] / 2)

        if best_ellipse is None or ellipse_area_approx < best_ellipse_area:
            best_ellipse = e
            best_ellipse_area = ellipse_area_approx

    return best_ellipse


# Finds the (x, y) coordinates of the point on ellipse (scaled by
# factor of scale) which is theta_2 degrees from it's major axis
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


# Returns tuple of
#   + Boolean of whether in hsv_img the pixel at angle theta from
#     the major axis of ellipse, scaled by a factor of scale, is unsaturated
#   + The hsv colour at that position
def unsaturated_at_angle(hsv_img, ellipse, theta, scale):
    coords = calc_point_coords(ellipse, theta, scale)
    if coords[1] >= hsv_img.shape[0] or coords[1] < 0 or coords[0] >= hsv_img.shape[1] or coords[0] < 0:
        return False, False
    hsv_colour = hsv_img[coords[1], coords[0]]
    return hsv_colour[1] < 100, hsv_colour


# Returns boolean of whether in hsv_img the pixel at angle theta from
# the major axis of ellipse, scaled by a factor of scale, is coloured black
def black_at_angle(hsv_img, ellipse, theta, scale=ELLIPSE_TO_DOTS_SCALE):
    is_unsaturated, hsv_colour = unsaturated_at_angle(hsv_img, ellipse, theta, scale)
    return is_unsaturated and hsv_colour[2] < 180


# Returns boolean of whether in hsv_img the pixel at angle theta from
# the major axis of ellipse, scaled by a factor of scale, is coloured white
def white_at_angle(hsv_img, ellipse, theta, scale=ELLIPSE_TO_DOTS_SCALE):
    is_unsaturated, hsv_colour = unsaturated_at_angle(hsv_img, ellipse, theta, scale)
    return is_unsaturated and hsv_colour[2] > 200


# Given the image hsv_img, ellipse at the centre of the tag, initial estimate of angle to
# the centre of the marking and initial estimate of scale of the center of the marking,
# returns the true angle and scale of the centre of the marking on the edge of the tag
def find_dot_centre(hsv_img, ellipse, init_angle, init_scale, debug_txt):
    left_angle = init_angle
    while black_at_angle(hsv_img, ellipse, left_angle - 2) and init_angle - left_angle < WIDTH_OF_DOT:
        left_angle -= 2
    while black_at_angle(hsv_img, ellipse, left_angle - 1) and init_angle - left_angle < WIDTH_OF_DOT:
        left_angle -= 1
    while black_at_angle(hsv_img, ellipse, left_angle - 0.1) and init_angle - left_angle < WIDTH_OF_DOT:
        left_angle -= 0.1

    right_angle = init_angle
    while black_at_angle(hsv_img, ellipse, right_angle + 2) and right_angle - init_angle < WIDTH_OF_DOT:
        right_angle += 2
    while black_at_angle(hsv_img, ellipse, right_angle + 1) and right_angle - init_angle < WIDTH_OF_DOT:
        right_angle += 1
    while black_at_angle(hsv_img, ellipse, right_angle + 0.1) and right_angle - init_angle < WIDTH_OF_DOT:
        right_angle += 0.1

    if right_angle - left_angle > WIDTH_OF_DOT + 5:
        if debug_txt:
            print(F"Dot too big, angle: {right_angle - left_angle} wide, found at angle {init_angle}")
        return None, None
    true_angle = (left_angle + right_angle) / 2

    scale_to_top = init_scale
    while black_at_angle(hsv_img, ellipse, true_angle, scale_to_top + 0.05) and scale_to_top < 1.75:
        scale_to_top += 0.05
    while black_at_angle(hsv_img, ellipse, true_angle, scale_to_top + 0.02) and scale_to_top < 1.75:
        scale_to_top += 0.02
    while black_at_angle(hsv_img, ellipse, true_angle, scale_to_top + 0.01) and scale_to_top < 1.75:
        scale_to_top += 0.01

    scale_to_bottom = init_scale
    while black_at_angle(hsv_img, ellipse, true_angle, scale_to_bottom - 0.05) and scale_to_bottom > 1:
        scale_to_bottom -= 0.05
    while black_at_angle(hsv_img, ellipse, true_angle, scale_to_bottom - 0.02) and scale_to_bottom > 1:
        scale_to_bottom -= 0.02
    while black_at_angle(hsv_img, ellipse, true_angle, scale_to_bottom - 0.01) and scale_to_bottom > 1:
        scale_to_bottom -= 0.01

    true_scale = (scale_to_top + scale_to_bottom) / 2

    return true_angle, true_scale


# Searches around the ellipse in the image hsv_img, returning the angle and scale
# of the first marking on the tag it successfully finds
def find_first_dot(hsv_img, ellipse, debug_txt):
    init_angle = 0.
    while not black_at_angle(hsv_img, ellipse, init_angle) and init_angle < 360:
        init_angle += 5

    if init_angle >= 360:
        if debug_txt:
            print("No dot found")
        return None, None

    angle, scale = find_dot_centre(hsv_img, ellipse, init_angle, ELLIPSE_TO_DOTS_SCALE, debug_txt)

    if angle is None:
        if debug_txt:
            print("First dot had invalid size")
        return None, None
    return angle, scale


# Returns boolean of whether the bottom of the tag is valid/can be correctly decoded
def check_bottom_of_tag(hsv_img, ellipse, top_dot_angle, scale):
    return white_at_angle(hsv_img, ellipse, top_dot_angle + 112.5, scale) and \
           white_at_angle(hsv_img, ellipse, top_dot_angle + 135, scale) and \
           black_at_angle(hsv_img, ellipse, top_dot_angle + 157.5, scale) and \
           white_at_angle(hsv_img, ellipse, top_dot_angle + 180, scale) and \
           black_at_angle(hsv_img, ellipse, top_dot_angle + 202.5, scale) and \
           white_at_angle(hsv_img, ellipse, top_dot_angle + 225, scale) and \
           white_at_angle(hsv_img, ellipse, top_dot_angle + 247.5, scale)


# Given the image, ellipse at the top of the tag and the angle of the top of
# the tag relative to the major axis of the ellipse, returns the decoded id
# number of the tag
def decode_tag_id(hsv_img, ellipse, top_dot_angle):
    dot_id = 0
    if black_at_angle(hsv_img, ellipse, top_dot_angle + 67.25):
        dot_id += 1
    if black_at_angle(hsv_img, ellipse, top_dot_angle + 45):
        dot_id += 2
    if black_at_angle(hsv_img, ellipse, top_dot_angle - 45):
        dot_id += 4
    if black_at_angle(hsv_img, ellipse, top_dot_angle - 67.25):
        dot_id += 8
    return dot_id


# Finds the coordinates of the center of the marking on the outside of the tag,
# in the immage hsv_img, with ellipse being the centre of the tag, given an initial
# estimate of its position in terms of scale and angle from the major axis of the ellipse
def find_dot_coords(hsv_img, ellipse, estimated_dot_angle, estimated_dot_scale, debug_text):
    angle, scale = find_dot_centre(hsv_img, ellipse, estimated_dot_angle, estimated_dot_scale, debug_text)
    if angle is None:
        return None
    return calc_point_coords(ellipse, angle, scale)


# Given a set of pxl_pts which are the coordinates of the markings on the tag,
# the camera matrix and the vector of distortion coefficients,
# return the rotation and translation vector of the tag in 3d space
def tag_solve_pnp(pxl_pts, cam_mat, cam_dist):
    bottom_dots_x = CM_TO_DOT_CENTER * math.sin(math.pi / 8)
    bottom_dots_z = CM_TO_DOT_CENTER * math.cos(math.pi / 8)
    object_points = [[0, 0, CM_TO_DOT_CENTER],  # top dot
                     [CM_TO_DOT_CENTER, 0, 0],  # right dot
                     [bottom_dots_x, 0, -bottom_dots_z],  # bottom right dot
                     [-bottom_dots_x, 0, -bottom_dots_z],  # bottom left dot
                     [-CM_TO_DOT_CENTER, 0, 0]]  # left dot

    _, rot, obj_3d_coords = cv2.solvePnP(np.float32(object_points),
                                         np.float32(pxl_pts),
                                         cam_mat, cam_dist)
    return rot, obj_3d_coords


# Finds and returns rotation and translation vector of all visible tags
# See README for full description
def find_tags(img, cam_mat, cam_dist, debug_txt=False, display_img=False):
    hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    col_contours = get_colour_contours(hsv_img, green_low, green_high)
    if not col_contours:
        if debug_txt:
            print("No contours found")
        return

    col_contours = sort_contours_on_size(col_contours)

    if display_img:
        contour_img = img.copy()
        cv2.drawContours(contour_img, col_contours, -1, (255, 255, 0))
        cv2.imshow('cam input with highlighted contours', contour_img)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            raise Exception("display cancelled 2")

    white_ellipses = find_white_ellipses(hsv_img)

    found_tags = []
    for contour in col_contours:
        if len(contour) < 10:
            if debug_txt:
                print("Contour too small")
            break
        coloured_ellipse = cv2.fitEllipse(contour)

        matched_ellipse = get_matching_ellipse(coloured_ellipse, white_ellipses)

        if matched_ellipse is None:
            ellipses_to_test = [coloured_ellipse]
        else:
            ellipses_to_test = [matched_ellipse, coloured_ellipse]

        for e in ellipses_to_test:

            first_dot_angle, first_dot_scale = find_first_dot(hsv_img, e, debug_txt)
            if first_dot_angle is None:
                continue

            top_dot_angle = first_dot_angle

            while not (black_at_angle(hsv_img, e, top_dot_angle, scale=first_dot_scale) and
                       (black_at_angle(hsv_img, e, top_dot_angle + 180, scale=first_dot_scale) or
                        black_at_angle(hsv_img, e, top_dot_angle + 178, scale=first_dot_scale) or
                        black_at_angle(hsv_img, e, top_dot_angle + 182, scale=first_dot_scale))) \
                    and top_dot_angle < first_dot_angle + 360:
                top_dot_angle += ANGLE_BETWEEN_DOTS

            if top_dot_angle >= first_dot_angle + 360:
                if debug_txt:
                    print(F"Failed to find 2 markings opposite eachother, at scale {first_dot_scale}.")
                continue

            temp_angle, temp_scale = find_dot_centre(hsv_img, e, top_dot_angle, init_scale=first_dot_scale,
                                                     debug_txt=debug_txt)
            if temp_angle is not None:
                first_dot_scale = temp_scale
                top_dot_angle = temp_angle

            if black_at_angle(hsv_img, e, top_dot_angle + 90, scale=first_dot_scale):
                top_dot_angle += 90
            elif black_at_angle(hsv_img, e, top_dot_angle - 90, scale=first_dot_scale):
                top_dot_angle -= 90
            else:
                if debug_txt:
                    print("Failed to decode tag after finding dots on opposite sides of tag, using scale: ",
                          first_dot_scale)

                continue

            if not check_bottom_of_tag(hsv_img, e, top_dot_angle, first_dot_scale):

                top_dot_angle, first_dot_scale = find_dot_centre(hsv_img, e, top_dot_angle, init_scale=first_dot_scale,
                                                                 debug_txt=debug_txt)

                if top_dot_angle is None:
                    if debug_txt:
                        print("Failed to decode tag, bottom of tag not valid.")
                        print("Tried to recenter on top dot, but failed to find centre.")
                    continue

                if not check_bottom_of_tag(hsv_img, e, top_dot_angle, first_dot_scale):
                    if debug_txt:
                        print("Failed to decode tag, bottom of tag not valid.")
                        print("Found to be invalid before and after recentering top dot")
                    continue

            dot_id = decode_tag_id(hsv_img, e, top_dot_angle)

            if dot_id in [old_id for old_id, _, _ in found_tags]:
                if debug_txt:
                    print(F"Found tag {dot_id} twice somehow.")
                continue

            top_dot = find_dot_coords(hsv_img, e, top_dot_angle, first_dot_scale, debug_txt)
            right_dot = find_dot_coords(hsv_img, e, top_dot_angle + 90, first_dot_scale, debug_txt)
            bottom_right_dot = find_dot_coords(hsv_img, e, top_dot_angle + 157.5, first_dot_scale, debug_txt)
            bottom_left_dot = find_dot_coords(hsv_img, e, top_dot_angle + 202.5, first_dot_scale, debug_txt)
            left_dot = find_dot_coords(hsv_img, e, top_dot_angle + 270, first_dot_scale, debug_txt)

            if top_dot is None or right_dot is None or bottom_right_dot is None or \
                    bottom_left_dot is None or left_dot is None:
                if debug_txt:
                    print("Failed to calculate coords of all dots.")
                continue

            if display_img:
                highlight = img.copy()

                if e is not matched_ellipse and matched_ellipse is not None:
                    cv2.ellipse(highlight, matched_ellipse, (0, 0, 255))

                cv2.ellipse(highlight, e, (0, 255, 255))
                cv2.circle(highlight, top_dot, 3, (0, 255, 0))
                cv2.circle(highlight, right_dot, 3, (0, 0, 255))
                cv2.circle(highlight, left_dot, 3, (0, 0, 255))
                cv2.circle(highlight, bottom_right_dot, 3, (255, 255, 0))
                cv2.circle(highlight, bottom_left_dot, 3, (255, 255, 0))

                cv2.imshow('cam input with highlighted tag', highlight)
                if cv2.waitKey(25) & 0xFF == ord('q'):
                    raise Exception("display cancelled 2")

            pixel_points = [top_dot, right_dot, bottom_right_dot, bottom_left_dot, left_dot]

            r_vec, t_vecs = tag_solve_pnp(pixel_points, cam_mat, cam_dist)

            found_tags.append((dot_id, r_vec, t_vecs))
            break

    return found_tags
