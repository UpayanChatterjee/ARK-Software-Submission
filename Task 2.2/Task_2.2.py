import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import patches


img_left = cv.imread("./left.png")[:, :, ::-1]

img_right = cv.imread("./right.png")[:, :, ::-1]

p_left = np.array([[640.0,   0.0, 640.0, 2176.0],
                   [0.0, 480.0, 480.0,  552.0],
                   [0.0,   0.0,   1.0,    1.4]])

p_right = np.array([[640.0,   0.0, 640.0, 2176.0],
                    [0.0, 480.0, 480.0,  792.0],
                    [0.0,   0.0,   1.0,    1.4]])

window_size = 6
params = {
    'minDisparity': 0,
    'numDisparities': 6 * 16,
    'blockSize': 11,
    'P1': 8 * 3 * window_size**2,
    'P2': 32 * 3 * window_size**2,
    'disp12MaxDiff': 0,
    'preFilterCap': 0,
    'uniquenessRatio': 0,
    'speckleWindowSize': 0,
    'speckleRange': 0,
    'mode': cv.STEREO_SGBM_MODE_SGBM_3WAY
}
matcher_type = 'sgbm'  # or 'sgbm'


def left_disparity_map(img_left, img_right, params, matcher_type='bm'):

    img_left_gray = cv.cvtColor(img_left, cv.COLOR_BGR2GRAY)
    img_right_gray = cv.cvtColor(img_right, cv.COLOR_BGR2GRAY)

    if matcher_type == 'bm':
        matcher = cv.StereoBM_create(
            numDisparities=6*16, blockSize=11)
    else:
        matcher = cv.StereoSGBM_create(**params)

    disp_left = matcher.compute(
        img_left_gray, img_right_gray).astype(np.float32) / 16

    return disp_left


def decompose_projection_matrix(p):

    k, r, t = cv.decomposeProjectionMatrix(p)[:3]
    t /= t[3]

    return k, r, t


def calc_depth_map(disp_left, k_left, t_left, t_right):

    f = k_left[0, 0]
    b = t_left[1, 0] - t_right[1, 0]
    d = disp_left.copy()
    d[d == 0] = 0.1
    d[d == -1] = 0.1
    depth_map = f * b / d

    return depth_map


def locate_obstacle_in_image(image, obstacle_image):

    cross_corr_map = cv.matchTemplate(
        image, obstacle_image, method=cv.TM_CCOEFF)
    obs_loc = cv.minMaxLoc(cross_corr_map)[3]

    return obs_loc


def calculate_nearest_point(depth_map, obs_loc, obstacle_img):

    obstacle_height, obstacle_width, _ = obstacle_img.shape
    obs_min_x = obs_loc[1]
    obs_min_y = obs_loc[0]
    distances = depth_map[obs_min_y:obs_min_y + obstacle_height,
                          obs_min_x:obs_min_x + obstacle_width]
    closest_point_depth = distances.min()

    # Create the obstacle bounding box
    obs_window = patches.Rectangle((obs_min_y, obs_min_x), obstacle_width, obstacle_height,
                                   linewidth=1, edgecolor='r', facecolor='none')

    return closest_point_depth, obs_window


def main():
    # plt.figure(figsize=(10, 10))
    # plt.gcf().canvas.set_window_title('Left')
    # plt.imshow(img_left)
    # plt.show()

    # plt.figure(figsize=(10, 10))
    # plt.gcf().canvas.set_window_title('Right')
    # plt.imshow(img_right)
    # plt.show()

    # print("p_left: ", p_left)
    # print("p_right: ", p_right)

    # Compute the disparity map
    disp_left = left_disparity_map(
        img_left, img_right, params, matcher_type=matcher_type)
    # Show the left disparity map
    plt.figure(figsize=(10, 10))
    plt.gcf().canvas.set_window_title('Left disp map')
    plt.imshow(disp_left)
    plt.show()

    obstacle_image = cv.imread("./bike.png")[:, :, ::-1]

    obs_loc = locate_obstacle_in_image(img_left, obstacle_image)

    print("Obstacle Location: ", obs_loc)

    # Estimating Depth
    disp_left = left_disparity_map(
        img_left, img_right, params, matcher_type=matcher_type)
    k_left, r_left, t_left = decompose_projection_matrix(p_left)
    k_right, r_right, t_right = decompose_projection_matrix(p_right)
    depth_map_left = calc_depth_map(disp_left, k_left, t_left, t_right)
    # Display the depth map
    plt.figure(figsize=(8, 8), dpi=100)
    plt.gcf().canvas.set_window_title('Depth map')
    plt.imshow(depth_map_left, cmap='flag')
    plt.show()

    # Finding the distance to collision
    obs_loc = locate_obstacle_in_image(
        img_left, obstacle_image)
    closest_point_depth, obs_window = calculate_nearest_point(
        depth_map_left, obs_loc, obstacle_image)

    # Display the image with the bounding box displayed
    fig, ax = plt.subplots(1, figsize=(10, 10))
    plt.gcf().canvas.set_window_title('Obstacle detected')
    ax.imshow(img_left)
    ax.add_patch(obs_window)
    plt.show()

    print(f"Closest point depth (meters): {closest_point_depth}")


if __name__ == "__main__":
    main()
