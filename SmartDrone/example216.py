import numpy as np
from djitellopy import tello
import cv2

markerRange = [10000, 12000]
pid = [0.2, 0.2, 0]
udpid = [0.2, 0.2, 0]
pError = 0
udPError = 0
n = 3
rvec = [[[0 for k in range(n)] for j in range(n)] for i in range(n)]
tvec = [[[0 for k in range(n)] for j in range(n)] for i in range(n)]

ARUCO_DICT = {
    "DICT_4X4_50": cv2.aruco.DICT_4X4_50,
    "DICT_4X4_100": cv2.aruco.DICT_4X4_100,
    "DICT_4X4_250": cv2.aruco.DICT_4X4_250,
    "DICT_4X4_1000": cv2.aruco.DICT_4X4_1000,
    "DICT_5X5_50": cv2.aruco.DICT_5X5_50,
    "DICT_5X5_100": cv2.aruco.DICT_5X5_100,
    "DICT_5X5_250": cv2.aruco.DICT_5X5_250,
    "DICT_5X5_1000": cv2.aruco.DICT_5X5_1000,
    "DICT_6X6_50": cv2.aruco.DICT_6X6_50,
    "DICT_6X6_100": cv2.aruco.DICT_6X6_100,
    "DICT_6X6_250": cv2.aruco.DICT_6X6_250,
    "DICT_6X6_1000": cv2.aruco.DICT_6X6_1000,
    "DICT_7X7_50": cv2.aruco.DICT_7X7_50,
    "DICT_7X7_100": cv2.aruco.DICT_7X7_100,
    "DICT_7X7_250": cv2.aruco.DICT_7X7_250,
    "DICT_7X7_1000": cv2.aruco.DICT_7X7_1000,
    "DICT_ARUCO_ORIGINAL": cv2.aruco.DICT_ARUCO_ORIGINAL,
    "DICT_APRILTAG_16h5": cv2.aruco.DICT_APRILTAG_16h5,
    "DICT_APRILTAG_25h9": cv2.aruco.DICT_APRILTAG_25h9,
    "DICT_APRILTAG_36h10": cv2.aruco.DICT_APRILTAG_36h10,
    "DICT_APRILTAG_36h11": cv2.aruco.DICT_APRILTAG_36h11
}

def aruco_display(corners, ids, rejected, image):
    markerList = []
    markerListArea = []

    if len(corners) > 0:

        ids = ids.flatten()

        for (markerCorner, markerID) in zip(corners, ids):
            
            corners = markerCorner.reshape((4, 2))
            (topLeft, topRight, bottomRight, bottomLeft) = corners

            topRight = (int(topRight[0]), int(topRight[1]))
            bottomRight = (int(bottomRight[0]), int(bottomRight[1]))
            bottomLeft = (int(bottomLeft[0]), int(bottomLeft[1]))
            topLeft = (int(topLeft[0]), int(topLeft[1]))

            cv2.line(image, topLeft, topRight, (0, 255, 0), 2)
            cv2.line(image, topRight, bottomRight, (0, 255, 0), 2)
            cv2.line(image, bottomRight, bottomLeft, (0, 255, 0), 2)
            cv2.line(image, bottomLeft, topLeft, (0, 255, 0), 2)

            cX = int((topLeft[0] + bottomRight[0]) / 2.0)
            cY = int((topLeft[1] + bottomRight[1]) / 2.0)
            cv2.circle(image, (cX, cY), 4, (0, 0, 255), -1)

            markerList.append([cX, cY])
            markerListArea.append(cv2.norm(topLeft, topRight) * cv2.norm(topLeft, bottomLeft))

            cv2.putText(image, str(markerID), (topLeft[0], topLeft[1] - 10), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (0, 255, 0), 2)
            print("[Inference] ArUco marker ID: {}".format(markerID))

    if len(markerListArea) != 0:
        i = markerListArea.index(max(markerListArea))
        return image, [markerList[i], markerListArea[i]]
    else:
        return image, [[0, 0], 0]

def trackMarker(frame, drone, info, w, h, pid, pError, udPError):
    area = info[1]
    x, y = info[0]
    fb = 0  # front back
    lr = 0

    error = x - w // 2
    udError = y - h // 2
    speed = pid[0] * error + pid[1] * (error - pError)
    speed = int(np.clip(speed, -100, 100))
    udSpeed = udpid[0] * udError + udpid[1] * (udError - udPError)
    udSpeed = int(np.clip(udSpeed, -100, 100))

    if udSpeed > 0:  # for tello, negative is up & positive is down
        udSpeed = -abs(udSpeed)
    else:
        udSpeed = abs(udSpeed)

    if markerRange[0] < area < markerRange[1]:
        fb = 0
    elif area > markerRange[1]:
        fb = -20
    elif area < markerRange[0] and area != 0:
        fb = 20
    if x == 0:
        speed = 0
        error = 0
    if y == 0:
        udSpeed = 0
        udError = 0
    if 0.0 <= abs(float(rvec[0][0][2])) < 0.1:
        cv2.circle(frame, (490, 240), 60, color=(255, 0, 0), thickness=3)
        fb = 0
        udSpeed = 0
        speed = 0
        lr = 0
    elif (tvec[0][0][0]>0.01):
        lr = 50
        cv2.arrowedLine(frame, (490, 240), (590, 240), (138,43,226), 3)
    elif (tvec[0][0][0]<-0.01):
        lr = -50
        cv2.arrowedLine(frame, (150, 240), (50, 240), (138,43,226), 3)

    drone.send_rc_control(lr, fb, udSpeed, speed)
    return error, udError

def pose_estimation(frame, aruco_dict_type, matrix_coefficients, distortion_coefficients, dictionary, arucoParams, detector):

    #gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    #cv.aruco_dict = cv2.aruco.Dictionary_get(aruco_dict_type)
    #parameters = cv2.aruco.DetectorParameters_create()

    corners, ids, rejected_img_points = detector.detectMarkers(frame, dictionary, parameters = arucoParams)
    
    if len(corners) > 0:
        for i in range(0, len(ids)):

            rvec, tvec, markerPoints = cv2.aruco.estimatePoseSingleMarkers(corners[i], 0.02, matrix_coefficients,
                                                                        distortion_coefficients)
            
            cv2.aruco.drawDetectedMarkers(frame, corners)
            print( 'tvecs[{}] {}'.format( i, tvec ))
            cv2.drawFrameAxes(frame, matrix_coefficients, distortion_coefficients, rvec, tvec, 0.01)

    return frame

aruco_type = "DICT_5X5_100"

dictionary = cv2.aruco.getPredefinedDictionary(ARUCO_DICT[aruco_type])
arucoParams = cv2.aruco.DetectorParameters()
detector = cv2.aruco.ArucoDetector(dictionary, arucoParams)

intrinsic_camera = np.array(((933.15867, 0, 657.59), (0, 933.1586, 400.36993), (0, 0, 1)))
distortion = np.array((-0.43948, 0.18514, 0, 0))

drone = tello.Tello()
drone.connect()

print('battery volume:', drone.get_battery())
drone.streamon()
frame_read = drone.get_frame_read()
drone.takeoff()

while True:
    img = frame_read.frame

    h, w, _ = img.shape

    width = 1000
    height = int(width * (h / w))
    img = cv2.resize(img, (width, height), interpolation=cv2.INTER_CUBIC)

    corners, ids, rejected = detector.detectMarkers(img)

    detected_markers, info = aruco_display(corners, ids, rejected, img)

    #cv2.imshow("Image", detected_markers)
    # print("Center", info[0], "Area", info[1])

    output = pose_estimation(img, ARUCO_DICT[aruco_type], intrinsic_camera, distortion, dictionary, arucoParams, detector)

    pError, udPError = trackMarker(img, drone, info, w, h, pid, pError, udPError)

    cv2.imshow('Estimated Pose', output)

    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        drone.land()
        drone.streamoff()
        break

cv2.destroyAllWindows()