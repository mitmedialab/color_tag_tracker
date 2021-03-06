# color_tag_tracker
Track color tags with ID codes.
Useful for tracking position of objects and AR.

## Requirements
+ Python3
+ OpenCV
+ Numpy
+ A calibrated camera
  - camera matrix and vector of distortion coefficients, as calculated by opencv's calibrateCamera function
  - see https://docs.opencv.org/2.4/modules/calib3d/doc/camera_calibration_and_3d_reconstruction.html
  
## Usage
Import `find_tags` method from `color_tag_tracker`. Use `find_tags` to find the pose of all visible tags in 3D space.

Parameters:
  + img
    - Image for tag to be found in, encoded with BGR colour representation (opencv's default)
  + cam_mat
    - Camera matrix of the camera which took the image
  + cam_dist
    - Vector of distortion coefficients for camera
  + debug_txt
    - Optional, default `False`
    - If `True` additional text is printed, useful for debugging.
  + display_img
    + Optional, default `False`
    + If `True`, 2 windows are displayed
      - img with contours of green area's overlayed
      - img with markings overlayed showing position of tag (central ellipse and markings around)

Returns:
  + List of tag descriptors as 3-tuples, including:
      + tag_id
        - ID of decoded tag
      + r_vec
        - Rotational vector of tag, as returned by opencv's `solvePnP` function.
      + t_vec
        - Translation vector of tag, as returned by opencv's `solvePnP` function.

## Examples of usage
All test scripts require replacing `mtx.npy` and `dist.npy` with the camera matrix and vector of distortion coefficients for camera in use. Current `mtx.npy` and `dist.npy` are parameters for camera used to film `tag_test_video.webm`.

+ `color_tag_tracker_webcam_test.py`
  - Takes input from webcam, prints results of `find_tags` for first tag.
+ `color_tag_tracker_demo.py`
  - Takes input from webcame, prints results of `find_tags` for first tag.
  - If tag_id is 0, projects axes onto first tag in image.
  - Otherwise, projects cube onto first tag in image.
+ `color_tag_tracker_playback_test.py`
  - Takes input from video file, prints results of `find_tags`.
  - Video file path given as command line argument.

## Possible future development
+ Use scale of found markings when decoding id of tag - minor
+ Incorporate undistorted image into tag finding process - major
    + It's possible to get an undistorted version of the input image using the camera matrix and distortion coefficents
    + Currently the decoding fails sometimes as due to distortion in image an ellipse cannot be properly fit. Using the
    undistorted image may be able to fix this.
    + Possible solution outline:
        + Undistort the image with cam_mat and cam_dist
        + Fit ellipse in undistorted image
        + Find centre of markings on tag in undistorted image
        + Map coordinates of markings in undistorted image to coordinates in distorted image
        + Use distorted coordinates with tag_solve_pnp as normal
    + Possibly useful opencv functions:
        + remap
        + initUndistortRectifyMap
        + undistort
