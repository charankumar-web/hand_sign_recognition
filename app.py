#!/usr/bin/env python
# -*- coding: utf-8 -*-
import csv
import copy
import itertools

import cv2 as cv
import numpy as np
import mediapipe as mp

######
import urllib.request
import os
from mediapipe.tasks.python import vision

from utils import CvFpsCalc
from model import KeyPointClassifier


def main():
    cap_device = 0  # Default camera device
    cap_width = 960  # Default camera width
    cap_height = 540  # Default camera height
    use_brect = True
    #Download model from internet
    model_url = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
    model_path = "hand_landmarker.task"
    if not os.path.exists(model_path):
        print("Downloading model...")
        urllib.request.urlretrieve(model_url, model_path)
        print("Model downloaded.")
    #Setup MediaPipe HandLandmarker
    BaseOptions = mp.tasks.BaseOptions
    VisionRunningMode = vision.RunningMode

    options = vision.HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=model_path),
        running_mode=VisionRunningMode.VIDEO,
        num_hands=2,
    )


    # Camera preparation
    cap = cv.VideoCapture(cap_device)
    cap.set(cv.CAP_PROP_FRAME_WIDTH, cap_width)
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, cap_height)

    # Models Loading
    detector = vision.HandLandmarker.create_from_options(options)
    
    keypoint_classifier = KeyPointClassifier()

    # Read labels 
    with open('model/keypoint_classifier/keypoint_classifier_label.csv',
              encoding='utf-8-sig') as f:
        keypoint_classifier_labels = csv.reader(f)
        keypoint_classifier_labels = [
            row[0] for row in keypoint_classifier_labels
        ]
    # FPS Measurement 
    cvFpsCalc = CvFpsCalc(buffer_len=10)

    mode = 0

    while True:
        fps = cvFpsCalc.get()

        # Process Key (ESC: end)
        key = cv.waitKey(10)
        if key == 27:  # ESC
            break
        number, mode = select_mode(key, mode)

        # Camera capture 
        ret, image = cap.read()
        if not ret:
            break
        image = cv.flip(image, 1)  # Mirror display
        debug_image = copy.deepcopy(image)

        # Detection implementation 
        image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image)
        timestamp = int(cap.get(cv.CAP_PROP_POS_MSEC))

        image.flags.writeable = False
        results = detector.detect_for_video(mp_image, timestamp) #For clean detection we first set it to false and change it into true
        image.flags.writeable = True
          
        if results.hand_landmarks:
            for hand_landmarks,handedness in zip(results.hand_landmarks,results.handedness):
                
                #Reverse handedness
                handedness = handness(handedness)
                
                # Bounding box calculation
                brect = calc_bounding_rect(debug_image, hand_landmarks)
                # Landmark calculation #image coordinates
                landmark_list = calc_landmark_list(debug_image, hand_landmarks)

                # Conversion to relative coordinates / normalized coordinates
                pre_processed_landmark_list = pre_process_landmark(
                    landmark_list)
                
                # Write to the dataset file
                logging_csv(number, mode, pre_processed_landmark_list)

                # Hand sign classification
                hand_sign_id = keypoint_classifier(pre_processed_landmark_list)

                # Drawing part
                debug_image = draw_bounding_rect(use_brect, debug_image, brect)
                debug_image = draw_landmarks(debug_image, landmark_list)
                debug_image = draw_info_text(
                    debug_image,
                    brect,
                    handedness,
                    keypoint_classifier_labels[hand_sign_id]
                )
                
        debug_image = draw_info(debug_image, fps, mode, number)

        # Screen reflection #############################################################
        cv.imshow('Hand Gesture Recognition', debug_image)

    cap.release()
    cv.destroyAllWindows()


def select_mode(key, mode):
    number = -1
    if 48 <= key <= 57:  # 0 ~ 9
        number = key - 48
    if key == 110:  # n
        mode = 0
    if key == 107:  # k
        mode = 1
    return number, mode

def handness(handedness):
    if handedness[0].category_name == 'Right':
        return 'Left'
    elif handedness[0].category_name == 'Left':
        return 'Right'

def calc_bounding_rect(image, landmarks):
    image_width, image_height = image.shape[1], image.shape[0]

    landmark_array = np.empty((0, 2), int)

    for lm in landmarks:
        landmark_x = min(int(lm.x * image_width), image_width - 1)
        landmark_y = min(int(lm.y * image_height), image_height - 1)

        landmark_point = [np.array((landmark_x, landmark_y))]

        landmark_array = np.append(landmark_array, landmark_point, axis=0)

    x, y, w, h = cv.boundingRect(landmark_array)

    return [x, y, x + w, y + h]


def calc_landmark_list(image, landmarks):
    image_width, image_height = image.shape[1], image.shape[0]

    landmark_point = []

    # Keypoint
    for lm in landmarks:
        landmark_x = int(lm.x*image_width)
        landmark_y = int(lm.y*image_height)
        # landmark_z = landmark.z

        landmark_point.append([landmark_x, landmark_y])
    return landmark_point

def pre_process_landmark(landmark_list):
    temp_landmark_list = copy.deepcopy(landmark_list)

    # Convert to relative coordinates
    base_x, base_y = 0, 0
    for index, landmark_point in enumerate(temp_landmark_list):
        if index == 0:
            base_x, base_y = landmark_point[0], landmark_point[1]

        temp_landmark_list[index][0] = temp_landmark_list[index][0] - base_x
        temp_landmark_list[index][1] = temp_landmark_list[index][1] - base_y

    # Convert to a one-dimensional list
    temp_landmark_list = list(
        itertools.chain.from_iterable(temp_landmark_list))

    # Normalization
    max_value = max(list(map(abs, temp_landmark_list)))

    def normalize_(n):
        return n / max_value

    temp_landmark_list = list(map(normalize_, temp_landmark_list))

    return temp_landmark_list

def logging_csv(number, mode, landmark_list):
    if mode == 0:
        pass
    if mode == 1 and (0 <= number <= 9):
        csv_path = 'model/keypoint_classifier/keypoint.csv'
        with open(csv_path, 'a', newline="") as f:
            writer = csv.writer(f)
            writer.writerow([number, *landmark_list])
    return

def draw_landmarks(image, landmark_point):
    if len(landmark_point) == 0:
        return image

    # Define connections based on MediaPipe hand topology
    connections = [
        # Thumb
        (2, 3), (3, 4),
        # Index
        (5, 6), (6, 7), (7, 8),
        # Middle
        (9, 10), (10, 11), (11, 12),
        # Ring
        (13, 14), (14, 15), (15, 16),
        # Little
        (17, 18), (18, 19), (19, 20),
        # Palm connections
        (0, 1), (1, 2), (2, 5), (5, 9),
        (9, 13), (13, 17), (17, 0)
    ]

    # Draw all bone connections
    for start_idx, end_idx in connections:
        pt1 = tuple(landmark_point[start_idx])
        pt2 = tuple(landmark_point[end_idx])
        cv.line(image, pt1, pt2, (0, 0, 0), 6)
        cv.line(image, pt1, pt2, (255, 255, 255), 2)

    # Draw key points
    for idx, point in enumerate(landmark_point):
        radius = 8 if idx in [4, 8, 12, 16, 20] else 5  # fingertips bigger
        cv.circle(image, (point[0], point[1]), radius, (255, 255, 255), -1)
        cv.circle(image, (point[0], point[1]), radius, (0, 0, 0), 1)

    return image


def draw_bounding_rect(use_brect, image, brect):
    if use_brect:
        # Outer rectangle
        cv.rectangle(image, (brect[0], brect[1]), (brect[2], brect[3]),
                     (0, 0, 0), 1)

    return image


def draw_info_text(image, brect, handedness, hand_sign_text):
    cv.rectangle(image, (brect[0], brect[1]), (brect[2], brect[1] - 22),
                 (0, 0, 0), -1)

    info_text = handedness
    if hand_sign_text != "":
        info_text = info_text + ':' + hand_sign_text if hand_sign_text else handedness
    cv.putText(image, info_text, (brect[0] + 5, brect[1] - 4),
               cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv.LINE_AA)
    return image


def draw_info(image, fps, mode, number):
    cv.putText(image, "FPS:" + str(fps), (10, 30), cv.FONT_HERSHEY_SIMPLEX,
               1.0, (0, 0, 0), 4, cv.LINE_AA)
    cv.putText(image, "FPS:" + str(fps), (10, 30), cv.FONT_HERSHEY_SIMPLEX,
               1.0, (255, 255, 255), 2, cv.LINE_AA)

    mode_string = ['Logging Key Point']
    if mode==1:
        cv.putText(image, "MODE:" + mode_string[0], (10, 60),
                   cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1,
                   cv.LINE_AA)
        if 0 <= number <= 9:
            cv.putText(image, "NUM:" + str(number), (10, 90),
                       cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1,
                       cv.LINE_AA)
    return image


if __name__ == '__main__':
    main()
