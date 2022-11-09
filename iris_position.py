import cv2 as cv
import numpy as np
import mediapipe as mp
import math


RIGHT_IRIS = [474, 475, 476, 477]
LEFT_IRIS = [469, 470, 471, 472]

L_H_TOP = [159]  # right eye upper landmark
L_H_BOTTOM = [145]  # right eye lower landmark
L_H_LEFT = [33]  # right eye right most landmark
L_H_RIGHT = [133]  # right eye left most landmark

R_H_TOP = [257]  # left eye upper landmark
R_H_BOTTOM = [253]  # left eye lower landmark
R_H_LEFT = [463]  # left eye right most landmark
R_H_RIGHT = [359]  # left eye left most landmark


def euclidean_distance(point1, point2):
    x1, y1 = point1.ravel()
    x2, y2 = point2.ravel()
    distance = math.sqrt((x2-x1)**2 + (y2-y1)**2)
    return distance


def iris_position(iris_center, right_point, left_point):
    center_to_right_dist = euclidean_distance(iris_center, right_point)
    total_distance = euclidean_distance(right_point, left_point)
    ratio = center_to_right_dist/total_distance
    iris_position = ""
    if ratio <= 0.42:
        iris_position = "right"
    elif ratio > 0.42 and ratio <= 0.57:
        iris_position = "center"
    else:
        iris_position = "left"
    return iris_position, ratio


mp_face_mesh = mp.solutions.face_mesh

cap = cv.VideoCapture(0)

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

            iris_pos, ratio = iris_position(
                center_right, mesh_points[R_H_RIGHT][0], mesh_points[R_H_LEFT][0])

            print(iris_pos)
        cv.imshow("img", frame)
        key = cv.waitKey(100)
        if key == ord("q"):
            break
cap.release()
cv.destroyAllWindows()
