import socket
import cv2 as cv
import numpy as np
import mediapipe as mp
import math
import time

UDP_IP = "192.168.4.1"
UDP_PORT = 8888

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)  # UDP

RIGHT_IRIS = [474, 475, 476, 477]
LEFT_IRIS = [469, 470, 471, 472]

L_H_TOP = [223]  # right eye upper landmark
L_H_BOTTOM = [230]  # right eye lower landmark
L_H_LEFT = [33]  # right eye right most landmark
L_H_RIGHT = [133]  # right eye left most landmark
L_H_EYELID_TOP = [159]  # right eyelid upper landmark
L_H_EYELID_BOTTOM = [145]  # right eyelid lower landmark

R_H_TOP = [443]  # left eye upper landmark
R_H_BOTTOM = [450]  # left eye lower landmark
R_H_LEFT = [362]  # left eye right most landmark
R_H_RIGHT = [263]  # left eye left most landmark
R_H_EYELID_TOP = [386]  # left eyelid upper landmark
R_H_EYELID_BOTTOM = [374]  # left eyelid lower landmark


COMMANDS = {"top": "1", "bottom": "2", "right": "3", "left": "4", "stop": "5"}


def euclidean_distance(point1, point2):
    x1, y1 = point1.ravel()
    x2, y2 = point2.ravel()
    distance = math.sqrt((x2-x1)**2 + (y2-y1)**2)
    return distance


def iris_position(r_iris_center, l_iris_center, right_eye_landmarks, left_eye_landmarks):
    # right eye
    r_top_point = right_eye_landmarks[0]
    r_bottom_point = right_eye_landmarks[1]
    r_right_point = right_eye_landmarks[2]
    r_left_point = right_eye_landmarks[3]
    r_eyelid_top_point = right_eye_landmarks[4]
    r_eyelid_bottom_point = right_eye_landmarks[5]

    # left eye
    l_top_point = left_eye_landmarks[0]
    l_bottom_point = left_eye_landmarks[1]
    l_right_point = left_eye_landmarks[2]
    l_left_point = left_eye_landmarks[3]
    l_eyelid_top_point = left_eye_landmarks[4]
    l_eyelid_bottom_point = left_eye_landmarks[5]

    r_vertical_pos = euclidean_distance(
        r_iris_center, r_top_point) / euclidean_distance(r_top_point, r_bottom_point)

    l_vertical_pos = euclidean_distance(
        l_iris_center, l_top_point) / euclidean_distance(l_top_point, l_bottom_point)

    vertical_pos = (r_vertical_pos + l_vertical_pos) / 2

    r_horizontal_pos = euclidean_distance(
        r_iris_center, r_right_point) / euclidean_distance(r_right_point, r_left_point)

    l_horizontal_pos = euclidean_distance(
        l_iris_center, l_right_point) / euclidean_distance(l_right_point, l_left_point)

    horizontal_pos = (r_horizontal_pos + l_horizontal_pos) / 2

    r_blink = euclidean_distance(r_eyelid_top_point, r_eyelid_bottom_point)

    l_blink = euclidean_distance(l_eyelid_top_point, l_eyelid_bottom_point)

    blink = (r_blink + l_blink) / 2

    iris_position = "center"
    if blink < 9.0:
        iris_position = "blink"
    elif abs(vertical_pos - 0.5) < abs(horizontal_pos - 0.5):
        if horizontal_pos < 0.42:
            iris_position = "right"
        elif horizontal_pos > 0.57:
            iris_position = "left"
    else:
        if vertical_pos < 0.48:
            iris_position = "top"
        elif vertical_pos > 0.57:
            iris_position = "bottom"

    return iris_position


mp_face_mesh = mp.solutions.face_mesh

cap = cv.VideoCapture(0)

counter = {"top": 0, "bottom": 0, "left": 0,
           "right": 0, "center": 0, "blink": 0}

power = False
first_blink = False
check_second_blink = False

with mp_face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
) as face_mesh:
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv.flip(frame, 1)
        rgb_frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        img_h, img_w = frame.shape[:2]
        results = face_mesh.process(rgb_frame)

        if results.multi_face_landmarks:
            mesh_points = np.array([
                np.multiply([p.x, p.y], [img_w, img_h]).astype(int)
                for p in results.multi_face_landmarks[0].landmark])

            (l_cx, l_cy), l_radius = cv.minEnclosingCircle(
                mesh_points[LEFT_IRIS])
            (r_cx, r_cy), r_radius = cv.minEnclosingCircle(
                mesh_points[RIGHT_IRIS])

            center_left = np.array([l_cx, l_cy], dtype=np.int32)
            center_right = np.array([r_cx, r_cy], dtype=np.int32)

            mesh_points[443][0] = int(
                (mesh_points[443][0] + mesh_points[257][0]) / 2)
            mesh_points[443][1] = int(
                (mesh_points[443][1] + mesh_points[257][1]) / 2)

            mesh_points[223][0] = int(
                (mesh_points[223][0] + mesh_points[27][0]) / 2)
            mesh_points[223][1] = int(
                (mesh_points[223][1] + mesh_points[27][1]) / 2)

            cv.circle(frame, center_left, int(l_radius),
                      (255, 0, 255), 1, cv.LINE_AA)
            cv.circle(frame, center_right, int(r_radius),
                      (255, 0, 255), 1, cv.LINE_AA)

            cv.circle(frame, mesh_points[R_H_TOP]
                      [0], 3, (0, 0, 255), -1, cv.LINE_AA)
            cv.circle(frame, mesh_points[R_H_BOTTOM]
                      [0], 3, (0, 255, 0), -1, cv.LINE_AA)
            cv.circle(frame, mesh_points[R_H_RIGHT]
                      [0], 3, (0, 255, 255), -1, cv.LINE_AA)
            cv.circle(frame, mesh_points[R_H_LEFT]
                      [0], 3, (255, 0, 0), -1, cv.LINE_AA)

            cv.circle(frame, mesh_points[L_H_TOP]
                      [0], 3, (0, 0, 255), -1, cv.LINE_AA)
            cv.circle(frame, mesh_points[L_H_BOTTOM]
                      [0], 3, (0, 255, 0), -1, cv.LINE_AA)
            cv.circle(frame, mesh_points[L_H_RIGHT]
                      [0], 3, (0, 255, 255), -1, cv.LINE_AA)
            cv.circle(frame, mesh_points[L_H_LEFT]
                      [0], 3, (255, 0, 0), -1, cv.LINE_AA)

            right_eye_landmarks = [mesh_points[R_H_TOP][0],
                                   mesh_points[R_H_BOTTOM][0],
                                   mesh_points[R_H_RIGHT][0],
                                   mesh_points[R_H_LEFT][0],
                                   mesh_points[R_H_EYELID_TOP][0],
                                   mesh_points[R_H_EYELID_BOTTOM][0]]

            left_eye_landmarks = [mesh_points[L_H_TOP][0],
                                  mesh_points[L_H_BOTTOM][0],
                                  mesh_points[L_H_RIGHT][0],
                                  mesh_points[L_H_LEFT][0],
                                  mesh_points[L_H_EYELID_TOP][0],
                                  mesh_points[L_H_EYELID_BOTTOM][0]]

            iris_pos = iris_position(
                center_right,
                center_left,
                right_eye_landmarks,
                left_eye_landmarks
            )

            # if counter is bigger than 10 set the command
            counter[iris_pos] += 1
            if counter[iris_pos] > 10:
                command = iris_pos
                counter = {"top": 0, "bottom": 0,
                           "left": 0, "right": 0, "center": 0, "blink": 0}

                # detect 2 blinks commands in a row
                if command == "blink" and not first_blink:
                    first_blink = True
                    first_blink_time = time.time()
                elif command != "blink" and first_blink:
                    check_second_blink = True
                elif command == "blink" and check_second_blink:
                    power = not power
                    first_blink = False
                    check_second_blink = False

                # if the first blink is too old reset the first blink
                if first_blink and time.time() - first_blink_time > 5:
                    first_blink = False
                    check_second_blink = False

                print(command)

                if power:
                    if command != "blink" and command != "center":
                        sock.sendto(
                            bytes(COMMANDS[command], "utf-8"), (UDP_IP, UDP_PORT))
                else:
                    sock.sendto(
                        bytes(COMMANDS["stop"], "utf-8"), (UDP_IP, UDP_PORT))

            if power:
                cv.putText(frame, "POWER ON", (50, 50),
                           cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv.putText(frame, command, (10, 30),
                           cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv.LINE_AA)
            else:
                cv.putText(frame, "POWER OFF", (50, 50),
                           cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        cv.imshow("img", frame)
        key = cv.waitKey(100)
        if key == ord("q"):
            break
cap.release()
cv.destroyAllWindows()
