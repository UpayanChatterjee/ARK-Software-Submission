import cv2 as cv
import numpy as np
import cv2.aruco as aruco


def track(frame, matrix_coefficients, distortion_coefficients):

    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    aruco_dict = aruco.Dictionary_get(aruco.DICT_6X6_250)
    parameters = aruco.DetectorParameters_create()
    corners, ids, rejectedImgPoints = aruco.detectMarkers(
        gray, aruco_dict, parameters=parameters)
    # print(ids)
    if ids is not None:
        aruco.drawDetectedMarkers(frame, corners, ids)
        rvecs, tvecs, _ = aruco.estimatePoseSingleMarkers(
            corners, 0.02, matrix_coefficients, distortion_coefficients)
        for rvec, tvec in zip(rvecs, tvecs):
            cv.drawFrameAxes(frame, matrix_coefficients,
                             distortion_coefficients, rvec, tvec, 0.1)
    return frame


if __name__ == "__main__":
    matrix_coeff = np.array([[6.9542619962191930e+02, 0., 3.3595878994582023e+02],
                            [0., 7.0123291039058915e+02, 2.0328731927191703e+02],
                            [0., 0., 1.]])

    distortion_coeff = np.array([2.2201115943292554e-01, -9.0652859085691029e-01,
                                 -3.0401185881237146e-02, 1.2054235882663385e-02,
                                 2.0127215854109610e+00])

    cap = cv.VideoCapture(1)

    while True:
        ret, frame = cap.read()

        track(frame, matrix_coeff, distortion_coeff)

        cv.imshow('frame', frame)
        if cv.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv.destroyAllWindows()
