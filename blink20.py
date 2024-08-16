# import cv2
# import dlib
# import imutils
# from scipy.spatial import distance as dist
# from imutils import face_utils
# import numpy as np


# def calculate_EAR(eye):
#     y1 = dist.euclidean(eye[1], eye[5])
#     y2 = dist.euclidean(eye[2], eye[4])
#     x1 = dist.euclidean(eye[0], eye[3])
#     EAR = (y1 + y2) / x1
#     return EAR

# class BlinkDetector:
#     def __init__(self, blink_thresh=0.45, succ_frame=2):
#         self.blink_thresh = blink_thresh
#         self.succ_frame = succ_frame
#         self.count_frame = 0
#         self.blink_count = 0
#         self.detector = dlib.get_frontal_face_detector()
#         self.landmark_predict = dlib.shape_predictor('eyeModel/shape_predictor_68_face_landmarks.dat')
#         (self.L_start, self.L_end) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
#         (self.R_start, self.R_end) = face_utils.FACIAL_LANDMARKS_IDXS['right_eye']

#     def detect_blink(self, frame):
#         frame = imutils.resize(frame, width=640)
#         img_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#         faces = self.detector(img_gray)
        
#         for face in faces:
#             shape = self.landmark_predict(img_gray, face)
#             shape = face_utils.shape_to_np(shape)
#             lefteye = shape[self.L_start: self.L_end]
#             righteye = shape[self.R_start:self.R_end]
#             left_EAR = calculate_EAR(lefteye)
#             right_EAR = calculate_EAR(righteye)
#             avg = (left_EAR + right_EAR) / 2

#             if avg < self.blink_thresh:
#                 self.count_frame += 1
#             else:
#                 if self.count_frame >= self.succ_frame:
#                     self.blink_count += 1
#                     self.count_frame = 0

#         return self.blink_count, frame

# import cv2
# import dlib
# import imutils
# from scipy.spatial import distance as dist
# from imutils import face_utils
# from fastapi import FastAPI, WebSocket
# import asyncio

# # EAR calculation function
# def calculate_EAR(eye):
#     y1 = dist.euclidean(eye[1], eye[5])
#     y2 = dist.euclidean(eye[2], eye[4])
#     x1 = dist.euclidean(eye[0], eye[3])
#     EAR = (y1 + y2) / x1
#     return EAR

# # BlinkDetector class
# class BlinkDetector:
#     def __init__(self, blink_thresh=0.45, succ_frame=2):
#         self.blink_thresh = blink_thresh
#         self.succ_frame = succ_frame
#         self.count_frame = 0
#         self.blink_count = 0
#         self.detector = dlib.get_frontal_face_detector()
#         self.landmark_predict = dlib.shape_predictor('eyeModel/shape_predictor_68_face_landmarks.dat')
#         (self.L_start, self.L_end) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
#         (self.R_start, self.R_end) = face_utils.FACIAL_LANDMARKS_IDXS['right_eye']

#     def detect_blink(self, frame):
#         frame = imutils.resize(frame, width=640)
#         img_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#         faces = self.detector(img_gray)
        
#         for face in faces:
#             shape = self.landmark_predict(img_gray, face)
#             shape = face_utils.shape_to_np(shape)
#             lefteye = shape[self.L_start: self.L_end]
#             righteye = shape[self.R_start:self.R_end]
#             left_EAR = calculate_EAR(lefteye)
#             right_EAR = calculate_EAR(righteye)
#             avg = (left_EAR + right_EAR) / 2

#             if avg < self.blink_thresh:
#                 self.count_frame += 1
#             else:
#                 if self.count_frame >= self.succ_frame:
#                     self.blink_count += 1
#                     self.count_frame = 0

#         return self.blink_count, frame

import cv2
import dlib
import imutils
from scipy.spatial import distance as dist
from imutils import face_utils
from fastapi import FastAPI, WebSocket
import asyncio

# EAR calculation function
def calculate_EAR(eye):
    y1 = dist.euclidean(eye[1], eye[5])
    y2 = dist.euclidean(eye[2], eye[4])
    x1 = dist.euclidean(eye[0], eye[3])
    EAR = (y1 + y2) / x1
    return EAR

# BlinkDetector class
class BlinkDetector:
    def __init__(self, blink_thresh=0.45, succ_frame=1):
        self.blink_thresh = blink_thresh
        self.succ_frame = succ_frame
        self.count_frame = 0
        self.detector = dlib.get_frontal_face_detector()
        self.landmark_predict = dlib.shape_predictor('eyeModel/shape_predictor_68_face_landmarks.dat')
        (self.L_start, self.L_end) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
        (self.R_start, self.R_end) = face_utils.FACIAL_LANDMARKS_IDXS['right_eye']

    def detect_blink(self, frame):
        frame = imutils.resize(frame, width=640)
        img_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.detector(img_gray)
        
        for face in faces:
            shape = self.landmark_predict(img_gray, face)
            shape = face_utils.shape_to_np(shape)
            lefteye = shape[self.L_start: self.L_end]
            righteye = shape[self.R_start:self.R_end]
            left_EAR = calculate_EAR(lefteye)
            right_EAR = calculate_EAR(righteye)
            avg = (left_EAR + right_EAR) / 2

            if avg < self.blink_thresh:
                self.count_frame += 1
            else:
                if self.count_frame >= self.succ_frame:
                    self.count_frame = 0
                    return True, frame

        return False, frame