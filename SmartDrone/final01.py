import numpy as np
from djitellopy import tello
import cv2
import time
import math
import utils
from filterpy.kalman import ExtendedKalmanFilter
import matplotlib.pyplot as plt
from PyQt5.QtWidgets import *
from PyQt5 import *
from PyQt5.QtCore import *
import sys

flight_time = []
pos_x = [0]
pos_y = [0]
pos_z = [10]
vgx = [0]
vgy = [0]
yaw = []
# ekf_X = []
# ekf_Y = []
eX = []
eY = []

markerRange = [12, 15]
pid = [0.2, 0.2, 0]
udpid = [0.2, 0.2, 0]
pError = 0
udPError = 0
checkStep = 1
n = 3
rvec = [[[0 for k in range(n)] for j in range(n)] for i in range(n)]
tvec = [[[0 for k in range(n)] for j in range(n)] for i in range(n)]
myvec = [0 for k in range(n)]
markerNum = 1
prevMarkerNum = 1
isDetected = 0
start_time = 0
index = 0

ax = plt.figure().add_subplot(projection='3d')

# Define measurement noise covariance matrix
R = np.diag([0.1, 0.1])

# Define process noise covariance matrix
q = 0.01
Q = np.array([[0.25*q**2, 0, 0.5*q**2, 0],
              [0, 0.25*q**2, 0, 0.5*q**2],
              [0.5*q**2, 0, q**2, 0],
              [0, 0.5*q**2, 0, q**2]])

# Define initial state and covariance matrix
x0 = np.array([0, 0, 0, 0])
P0 = np.diag([5, 5, 1, 1])

aruco_type = "DICT_5X5_100"

calibration_matrix_path = "calibration_matrix.npy"
distortion_coefficients_path = "distortion_coefficients.npy"

k = np.load(calibration_matrix_path)
d = np.load(distortion_coefficients_path)

# Initialize the Extended Kalman Filter object
# ekf = ExtendedKalmanFilter(dim_x=4, dim_z=2)
# ekf.x = x0
# ekf.P = P0
# ekf.Q = Q
# ekf.R = R

# Define time step
dt = 0
i = 0

fov = 63.5
frame_width = 1000
focal_length = (frame_width / 2) / np.tan(fov * np.pi / 360)

dictionary = cv2.aruco.getPredefinedDictionary(utils.ARUCO_DICT["DICT_5X5_100"])
arucoParams = cv2.aruco.DetectorParameters()
detector = cv2.aruco.ArucoDetector(dictionary, arucoParams)

drone = tello.Tello()

class Thread(QThread):
    img = None
    img_h = None
    img_w = None
    gray = None

    def __init__(self, parent):
        super().__init__(parent)
        self.parent = parent

    def run(self):
        while True:
            img = drone.get_frame_read().frame

            img_h, img_w, _ = img.shape

            width = 1000
            height = int(width * (img_h / img_w))
            img = cv2.resize(img, (width, height), interpolation=cv2.INTER_CUBIC)
            Thread.img_h = img_h
            Thread.img_w = img_w
            Thread.img = img

            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            Thread.gray = gray

class WindowClass(QMainWindow):
    def __init__(self):
        super().__init__()

        self.start_code()

    # Define state transition function
    # def f(self, x, dt):
    #     F = np.array([[1, 0, dt, 0],
    #                 [0, 1, 0, dt],
    #                 [0, 0, 1, 0],
    #                 [0, 0, 0, 1]])
    #     return np.dot(F, x)

    # # Define measurement function
    # def h(self, x):
    #     return np.array([x[2], x[3]])

    # # Define Jacobian of state transition function with respect to state
    # def Fx(self, dt):
    #     return np.array([[1, 0, dt, 0],
    #                     [0, 1, 0, dt],
    #                     [0, 0, 1, 0],
    #                     [0, 0, 0, 1]])

    # # Define Jacobian of measurement function with respect to state
    # def Hx(self):
    #     return np.array([[0, 0, 1, 0],
    #                     [0, 0, 0, 1],
    #                     [0, 0, 0, 0],
    #                      [0, 0, 0, 0],
    #                     ])

    # ax = plt.figure().add_subplot(projection='3d')
    # def animate(self, i):
    #     global index
    #     # eX.append(ekf_X[index])
    #     # eY.append(ekf_Y[index])
    #     plt.cla()
    #     ax.plot(eX, eY, label = "EKF")
    #     ax.set_xlabel('X')
    #     ax.set_ylabel('Y')
    #     ax.set_zlabel('Z')
    #     ax.set_title('Odometry')
    #     index+=1

    def calculate_distance(self, marker_size, focal_length, pixel_width):
        
        marker_actual_size = 0.05
        distance = (marker_actual_size * focal_length) / marker_size
        distance_v = (distance * pixel_width[0]) / marker_size
        distance_h = (distance * pixel_width[1]) / marker_size

        return distance

    def aruco_display(self, corners, ids, distance, image, centerX, centerY,):
        markerList = []
        markerListArea = []

        if len(corners) > 0:

            ids = ids.flatten()

            for (markerCorner, markerID) in zip(corners, ids):
                if markerID ==  markerNum:
                    corners = markerCorner.reshape((4, 2))
                    (topLeft, topRight, bottomRight, bottomLeft) = corners

                    topRight = (int(topRight[0]), int(topRight[1]))
                    bottomRight = (int(bottomRight[0]), int(bottomRight[1]))
                    bottomLeft = (int(bottomLeft[0]), int(bottomLeft[1]))
                    topLeft = (int(topLeft[0]), int(topLeft[1]))

                    distance = round(distance * 100, 1)

                    cv2.line(image, topLeft, topRight, (0, 255, 0), 2)
                    cv2.line(image, topRight, bottomRight, (0, 255, 0), 2)
                    cv2.line(image, bottomRight, bottomLeft, (0, 255, 0), 2)
                    cv2.line(image, bottomLeft, topLeft, (0, 255, 0), 2)

                    cX = int((topLeft[0] + bottomRight[0]) / 2.0)
                    cY = int((topLeft[1] + bottomRight[1]) / 2.0)
                    cv2.circle(image, (cX, cY), 4, (0, 0, 255), -1)

                    centerX = int(centerX)
                    centerY = int(centerY)
                    cv2.circle(image, (centerX, centerY), 4, (255, 0, 0), -1)
                    cv2.line(image, (centerX, centerY), (cX, cY), (255, 255, 255), 4)

                    markerList.append([cX, cY])
                    markerListArea.append(distance)

                    cv2.putText(image, str(markerID), (topLeft[0], topLeft[1] - 10), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (0, 255, 0), 2)
                    cv2.putText(image, str(distance), (topLeft[0] + 20, topLeft[1] - 10), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (255, 0, 0), 2)
                    #print("[Inference] ArUco marker ID: {}".format(markerID))

                    marker_size = (abs(topLeft[1]-bottomLeft[1]) + abs(topRight[1]-bottomRight[1]))/2

        if len(markerListArea) != 0:
            i = markerListArea.index(max(markerListArea))
            start_time = time.perf_counter()
            return image, [markerList[i], markerListArea[i]], marker_size
        else:
            return image, [[0, 0], 0], 1

    def trackMarker(self, frame, drone, info, w, h, pid, pError, udPError, distance):
        global markerNum
        global isDetected
        global start_time
        area = info[1]
        x, y = info[0]
        fb = 0  # front back

        error = x - w // 2
        udError = y - h // 2
        speed = pid[0] * error + pid[1] * (error - pError)
        speed = int(np.clip(speed, -15, 15))
        udSpeed = udpid[0] * udError + udpid[1] * (udError - udPError)
        udSpeed = int(np.clip(udSpeed, -15, 15))
        if udSpeed > 0:  # for tello, negative is up & positive is down
            udSpeed = -abs(udSpeed)
        else:
            udSpeed = abs(udSpeed)

        # Rodrigues function: Converts a rotation matrix to a rotation vector
        R, _ = cv2.Rodrigues(rvec)

        # Extract the rotation angle from the rotation matrix
        theta_x = np.degrees(np.arctan2(R[2, 1], R[2, 2]))
        theta_y = np.degrees(np.arctan2(-R[2, 0], np.sqrt(R[2, 1]**2 + R[2, 2]**2)))
        theta_z = np.degrees(np.arctan2(R[1, 0], R[0, 0]))

        # Calculate the length(move_distance) using angle and distance
        move_distance = math.sin(math.radians(abs(theta_y))) * distance

        #print("move_distance: ", move_distance)      
        # left and right
        if (x not in range(int((w/2) - 20), int((w/2) + 20))):
            drone.send_rc_control(0, 0, 0, speed)
            #print(x)
        # up and down
        if (y not in range(int((h/2) - 25), int((h/2) + 25))):
            drone.send_rc_control(0, 0, udSpeed, 0)
            #print(y)
        # front and back
        if (area not in range(markerRange[0], markerRange[1])):
            if area > markerRange[1]:
                fb = 10
            elif area < markerRange[0] and area != 0:
                fb = -10
            else:
                fb = 0
            drone.send_rc_control(0, fb, 0, 0)
        
        # At first, rotate drone for centering
        if(1 < int(theta_y) < 90):
            drone.rotate_clockwise(abs(int(theta_y)+5))
        elif (-90 < int(theta_y) < -1):
            drone.rotate_counter_clockwise(abs(int(theta_y)+5))

        # Secondly, let the drone move from side to side
        move_distance = move_distance * 100 + 20
        if (theta_y < -5 and move_distance >= 20):
            drone.move_right(int(abs(move_distance)))
            #drone.move_left(int(abs(move_distance))) 
        elif (theta_y > 5 and move_distance >= 20):
            drone.move_left(int(abs(move_distance))) 
            #drone.move_right(int(abs(move_distance))) 
        drone.send_rc_control(0, 0, 0, 0)
        drone.send_rc_control(0, 0, 0, 0)
        drone.send_rc_control(0, 0, 0, 0)
        # if markerNum == 4:
        #     drone.land()
        #     drone.streamoff()
        markerNum += 1
        print("markerNum: " , markerNum)
        isDetected = 0
        # start_time = time.perf_counter()
        
        return error, udError

    def pose_estimation(self, frame, aruco_dict_type, matrix_coefficients, distortion_coefficients, img_w, img_h, gray):

        cv2.aruco_dict = cv2.aruco.getPredefinedDictionary(aruco_dict_type)
        parameters = cv2.aruco.DetectorParameters()
        detector = cv2.aruco.ArucoDetector(cv2.aruco_dict, parameters)
             
        corners, ids, rejected_img_points = detector.detectMarkers(gray)
        global rvec, tvec
        global pError, udPError
        global isDetected

        if len(corners) > 0 and markerNum in ids:
            isDetected = 1
            rvec, tvec, markerPoints = cv2.aruco.estimatePoseSingleMarkers(corners[0], 0.02, matrix_coefficients,
                                                                        distortion_coefficients)
            detected_markers, info, marker_size = self.aruco_display(corners, ids, tvec[0][0][2], Thread.img, img_w/2, img_h/2)
            distance = self.calculate_distance(marker_size, focal_length, (1936, 2592))  # distance between tello and AR marker.
            #print("distance", distance)
            #cv2.aruco.drawDetectedMarkers(frame, corners)
            pError, udPError = self.trackMarker(frame, drone, info, img_w, img_h, pid, pError, udPError, distance)

            # if(0.0 <= abs(float(rvec[0][0][2])) < 0.1):
            #     cv2.circle(frame, (490, 240), 60, color=(255, 0, 0), thickness=3)
            
            cv2.drawFrameAxes(frame, matrix_coefficients, distortion_coefficients, rvec, tvec, 0.01)
        else:
            if markerNum - 1 == 0:
                drone.send_rc_control(0, 0, 20, 0)
                print("go up to 1")
            elif markerNum - 1 == 1: # go up to find first marker
                if isDetected == 1:
                    drone.send_rc_control(0, 0, 0, 0)
                else:
                    drone.send_rc_control(20, 0, 0, 0)
                print("go right to 2")
            elif markerNum - 1 == 2: # go right to second marker
                # if time.perf_counter() - start_time > 20:
                #     drone.send_rc_control(0, 0, 0, -15)
                # else:
                if isDetected == 1:
                    drone.send_rc_control(0, 0, 0, 0)
                else:
                    drone.send_rc_control(0, 0, -20, 0)
                print("go down to 3")
            elif markerNum - 1 == 3: # go down to third marker
                if isDetected == 1:
                    drone.send_rc_control(0, 0, 0, 0)
                else:
                    drone.send_rc_control(-20, 0, 0, 0)
                print("go left to 4")
            elif markerNum - 1 == 4: # go left to fourth marker
                # drone.send_rc_control(-20, 0, 0, 0)
                drone.land()
                drone.streamoff()   
                print("go down to 5")
            elif markerNum - 1 == 5: # go down to fifth marker
                drone.send_rc_control(0, 0, -20, 0)
                print("go right to 6")
        return frame

    def start_code(self):
        
        flight_time.append(time.time())
        drone.connect()

        drone.streamon()
        img = drone.get_frame_read().frame

        img_h, img_w, _ = img.shape

        width = 1000
        height = int(width * (img_h / img_w))
        img = cv2.resize(img, (width, height), interpolation=cv2.INTER_CUBIC)

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        myThread = Thread(self)
        myThread.start()
        print('battery volume:', drone.get_battery())
        
        yaw.append(drone.get_yaw())
        drone.takeoff()
        i = 1
        while True:
            flight_time.append(time.time())
            vgx.append(drone.get_speed_x())
            vgy.append(drone.get_speed_y())
            pos_z.append(drone.get_distance_tof())
            yaw.append(drone.get_yaw())

            output = self.pose_estimation(Thread.img, utils.ARUCO_DICT["DICT_5X5_100"], k, d, Thread.img_w, Thread.img_h, Thread.gray)

            # Get drone speed measurements
            v = np.array([vgx[i], vgy[i]])
            if i == 0:
                dt = 0
            else:
                dt = flight_time[i] - flight_time[i-1]
            
            # Update the state transition and measurement Jacobians
            # ekf.F = self.Fx(dt)
            # ekf.x = self.Hx()
            
            # # Predict state and state covariance
            # ekf.predict()
            
            # # Update state and state covariance
            # ekf.update(v, HJacobian=ekf.H, Hx=self.h, R=ekf.R)
            
            # # Print estimated position
            # ekf_X.append(ekf.x[0])
            # ekf_Y.append(ekf.x[1])
            # #print("Estimated position:", ekf.x[0], ekf.x[1])
            # self.animate(index)
            # plt.pause(0.001)
            # i+=1

            cv2.imshow('Estimated Pose', output)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                drone.land()
                drone.streamoff()
                myThread.quit()
                break
        # plt.legend()
        # plt.show()

        cv2.destroyAllWindows()

if __name__ == "__main__":
    app = QApplication(sys.argv)

    myWindow = WindowClass()

    # Set the state transition function and Jacobian
    # ekf.f = myWindow.f
    # ekf.F = myWindow.Fx

    # Set the measurement function and Jacobian
    # ekf.h = myWindow.h
    # ekf.H = myWindow.Hx

    sys.exit(app.exec_())
