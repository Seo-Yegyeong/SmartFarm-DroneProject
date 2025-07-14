import numpy as np
import cv2
import sys
import time
import math

def isRotationMatrix(R) : 
    Rt = np.transpose(R) 
    shouldBeIdentity = np.dot(Rt, R) 
    I = np.identity(3, dtype = R.dtype) 
    n = np.linalg.norm(I - shouldBeIdentity) 
    return n < 1e-6

# https://stackoverflow.com/questions/53808503/how-to-test-if-a-matrix-is-a-rotation-matrix
# def isRotationMatrix(M):
#     tag = False
#     I = np.identity(M.shape[0])
#     if np.all((np.matmul(M, M.T)) == I) and (np.linalg.det(M)==1): tag = True
#     return tag

def rotationMatrixToEulerAngles(R): 
    assert(isRotationMatrix(R))
    sy = math.sqrt(R[0,0] * R[0,0] + R[1,0] * R[1,0])

    singular = sy < 1e-6

    if not singular :
        x = math.atan2(R[2,1] , R[2,2])
        y = math.atan2(-R[2,0], sy)
        z = math.atan2(R[1,0], R[0,0])
        # z = math.atan2(R[0,1], -R[1,1])
    else:
        x = math.atan2(-R[1,2], R[1,1])
        y = math.atan2(-R[2,0], sy)
        z = 0

    return np.array([x, y, z]) * (180/np.pi) # move from radians to degrees

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

    if len(corners) > 0:

        ids = ids.flatten()

        for(markerCorner, markerID) in zip(corners, ids):

            corners = markerCorner.reshape((4, 2))
            (topLeft, topRight, bottomRight, bottomLeft)

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

            cv2.putText(image, str(markerID), (topLeft[0], topLeft[1] - 10), cv2.FONT_HERSHEY_SIMPLEX,
                0.5, (0, 255, 0), 2)
            print("[Inference] ArUco marker ID: {}".format(markerID))

    return image

def pose_estimation(frame, aruco_dict_type, matrix_coefficients, distortion_coefficients):

    #gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    #cv.aruco_dict = cv2.aruco.Dictionary_get(aruco_dict_type)
    #parameters = cv2.aruco.DetectorParameters_create()

    dictionary = cv2.aruco.getPredefinedDictionary(ARUCO_DICT[aruco_type])
    arucoParams = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(dictionary, arucoParams)

    corners, ids, rejected_img_points = cv2.aruco.detectMarkers(frame, dictionary, parameters = arucoParams)
    
    if len(corners) > 0:
        for i in range(0, len(ids)):

            rvec, tvec, markerPoints = cv2.aruco.estimatePoseSingleMarkers(corners[i], 0.02, matrix_coefficients,
                                                                        distortion_coefficients)
            
            cv2.aruco.drawDetectedMarkers(frame, corners)

            R, jacobian = cv2.Rodrigues(rvec)

            theta_x = np.arctan2(R[2, 1], R[2, 2])
            theta_y = np.arctan2(-R[2, 0], np.sqrt(R[2, 1]**2 + R[2, 2]**2))    
            theta_z = np.arctan2(R[1, 0], R[0, 0])

            print("Rotation angles (in degrees):")
            print("  x-axis: {:.2f}".format(np.degrees(theta_x)))
            print("  y-axis: {:.2f}".format(np.degrees(theta_y)))
            print("  z-axis: {:.2f}".format(np.degrees(theta_z)))

            # print( 'rvecs[{}] {}'.format( i, rvec ))
            # print(rvec[0][0][2])
            # print( 'tvecs[{}] {}'.format( i, tvec ))
            if 0.0 <= abs(float(tvec[0][0][0])) < 0.021:
                cv2.circle(frame, (490, 240), 60, color=(255, 0, 0), thickness=3)
            
            elif (tvec[0][0][0]>0.03):
                cv2.arrowedLine(frame, (490, 240), (590, 240), (138,43,226), 3)
                lr = 50
            elif (tvec[0][0][0]<-0.02):
                cv2.arrowedLine(frame, (150, 240), (50, 240), (138,43,226), 3)
                lr = -50
            cv2.drawFrameAxes(frame, matrix_coefficients, distortion_coefficients, rvec, tvec, 0.01)
        
    return frame

aruco_type = "DICT_5X5_100"

#arucoDict = cv2.aruco.Dictionary_get(ARUCO_DICT[aruco_type])
#arucoParams = cv2.aruco.DetectorParameters_create()

calibration_matrix_path = "calibration_matrix.npy"
distortion_coefficients_path = "distortion_coefficients.npy"

k = np.load(calibration_matrix_path)
d = np.load(distortion_coefficients_path)

cap = cv2.VideoCapture(0)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

while cap.isOpened():

    ret, img = cap.read()

    h, w, _ = img.shape

    width = 1000
    height = int(width * (h / w))
    img = cv2.resize(img, (width, height), interpolation=cv2.INTER_CUBIC)
    
    output = pose_estimation(img, ARUCO_DICT[aruco_type], k, d)

    cv2.imshow('Estimated Pose', output)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()