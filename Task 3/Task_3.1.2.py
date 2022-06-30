import cv2 as cv
import numpy as np

# Step 1: find and identify any squares in the image
#       1a: find contours in the image(since findContours takes input a binary image, we need to convert the image to binary)
# we notice that global thresholding would not be that good, so we resort to adaptive thresholding

# Step 1.5: Make sure the corners are in correct order

# Step 2 : detect signature of the square and compare it to a list of predefined signatures
#         if the signature is found, then the square is a signature
# Step 2.1: perspective transform the square to a square with the same aspect ratio as the predefined signature
# Step 2.2: compare the signature to the predefined signature
# Step 2.3: if the signature is found, then the square is a signature
# Step 2.4: if the signature is not found, then the square is not a signature

TERM_CRIT = criteria = (cv.TERM_CRITERIA_EPS +
                        cv.TERM_CRITERIA_MAX_ITER, 100, 0.001)


class ArucoDict:
    def __init__(self):
        self.sig = []
        self.world_coord = []

    def add_signature(self, sig):
        self.sig.append(sig)

    def add_world_coordinates(self, coord):
        self.world_coord.append(coord)


class Aruco:
    def __init__(self, corners, ids):
        self.corners = corners
        self.ids = ids


def draw_aruco(frame, aruco_list):
    for aruco in aruco_list:
        cv.drawContours(frame, aruco.corners, -1, (0, 0, 255), 3)
        cv.putText(frame, str(aruco.ids), (aruco.corners[0][0][0], aruco.corners[0][0][1]),
                   cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)


def order_contour(contours):
    contours = contours.reshape(4, 2)
    cx = np.average(contours[:, 0])
    cy = np.average(contours[:, 1])
    if(cx >= contours[0][0] and cy >= contours[0][1]):
        contours[[1, 3]] = contours[[3, 1]]
    else:
        contours[[0, 1]] = contours[[1, 0]]
        contours[[2, 3]] = contours[[3, 2]]
    pass


def find_squares(gray_frame):
    color_frame = cv.cvtColor(gray_frame, cv.COLOR_GRAY2BGR)
    # thresholded = cv.threshold(gray_frame, 127, 255, cv.THRESH_BINARY)[1]
    thresholded = cv.adaptiveThreshold(
        gray_frame, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY, 11, 2)

    # since we want to detect black blobs, we need to invert the image
    thresholded = cv.bitwise_not(thresholded)
    contours, hierarchy = cv.findContours(
        image=thresholded, mode=cv.RETR_EXTERNAL, method=cv.CHAIN_APPROX_SIMPLE)  # RETR_EXTERNAL returns only the outermost contour

    # draw all contours in green on gray_frame
    # cv.drawContours(gray_frame, contours, -1, (0, 255, 0), 3)

    candidates = []
    for contour in contours:

        # approximate contour to a polygon
        approx = cv.approxPolyDP(
            contour, epsilon=0.01 * cv.arcLength(contour, True), closed=True)

        # if the approximated contour has four points and it is convex, then it is safe to be assumed a square
        if (len(approx) != 4) or (not cv.isContourConvex(approx)) or (cv.contourArea(approx) < 100):
            continue
        # print(approx.ndim)
        # refine the position of the approx corners
        # corners = cv.cornerSubPix(
        #     gray_frame, approx, (5, 5), (-1, -1), TERM_CRIT)

        order_contour(approx)
        cv.circle(color_frame, (approx[0][0][0],
                  approx[0][0][1]), 10, (0, 0, 255), -1)
        cv.circle(color_frame, (approx[1][0][0],
                  approx[1][0][1]), 10, (0, 255, 0), -1)
        cv.circle(color_frame, (approx[2][0][0],
                  approx[2][0][1]), 10, (255, 0, 0), -1)
        cv.circle(color_frame, (approx[3][0][0],
                  approx[3][0][1]), 10, (255, 255, 0), -1)

        candidates.append(approx)

    show = cv.drawContours(
        color_frame, candidates, -1, (0, 0, 255), 3)
    # cv.imshow("frame", show)
    # cv.waitKey(1)

    return candidates


def get_contour_signature(gray_frame, contour, size):

    pixelLength = np.sqrt(size)
    corners = np.float32([
        [0, 0],
        [size, 0],
        [size, size],
        [0, size]
    ])
    contour = np.float32(contour)
    persp = cv.getPerspectiveTransform(contour, corners)
    persp_corr = cv.warpPerspective(
        gray_frame, persp, (int(size), int(size)))

    # now we otsu threshold the image
    ret, thresh = cv.threshold(
        persp_corr, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)

    # a good practice now is to erode the image
    kernel = np.ones((3, 3), np.uint8)
    eroded = cv.erode(thresh, kernel, iterations=1)

    # now we determine the signature of the contour
    # the 64x64 final prespective corrected otsu thresholded and eroded image that
    # we obtained can be divided into 8 blocks with each block containing 8x8 pixels
    # we will sample the color from the centre of each block and create a signature
    sig = []
    for row in range(0, int(pixelLength)):
        for col in range(0, int(pixelLength)):
            x = int(row * pixelLength)+int(pixelLength/2)
            y = int(col * pixelLength)+int(pixelLength/2)
            if(eroded[x][y] >= 128):
                sig.append(1)
            else:
                sig.append(0)

    # print(sig)
    # cv.imshow('eroded', eroded)
    # cv.imshow("thresh", thresh)
    # cv.imshow("persp_corr", persp_corr)
    # cv.waitKey(1)

    return sig


def detectAruco(gray_frames, aruco_dict):

    candidates = find_squares(gray_frames)
    res = []
    for candidate in candidates:
        sig = get_contour_signature(gray_frames, candidate, 64)
        for _ in range(4):
            if sig == aruco_dict.sig[_]:
                res.append(Aruco(candidate, _))
                break

    return res


def load_aruco_dictionary(marker, size):
    aruco_dict = ArucoDict()
    height, width = marker.shape
    corners = np.array([
        [0, 0],
        [width, 0],
        [width, height],
        [0, height]
    ])
    world = np.array([
        [0, 0, 0],
        [1, 0, 0],
        [1, 1, 0],
        [0, 1, 0]
    ])
    for _ in range(4):  # for all the four orientations of the marker
        sig = get_contour_signature(marker, corners, size)
        aruco_dict.add_signature(sig)

        aruco_dict.add_world_coordinates(world)

        marker = cv.rotate(marker, cv.ROTATE_90_CLOCKWISE)  # rotate the marker

        world = np.insert(world, [0], world[3], axis=0)
        world = np.delete(world, 4, 0)  # rotate the world coordinates

    return aruco_dict


def main():
    aruco_marker = cv.imread("./aruco.png")
    aruco_marker_gray = cv.cvtColor(aruco_marker, cv.COLOR_BGR2GRAY)
    aruco_dict = load_aruco_dictionary(aruco_marker_gray, 64)
    cap = cv.VideoCapture(0)
    while True:
        _, frame = cap.read()
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

        result = detectAruco(gray, aruco_dict)

        draw_aruco(frame, result)

        cv.imshow('frame', frame)
        if cv.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv.destroyAllWindows()


if __name__ == "__main__":
    main()
