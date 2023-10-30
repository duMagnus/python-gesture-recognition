import math

import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import cv2
import numpy as np

cap = cv2.VideoCapture(0)

model_path = 'model/gesture_recognizer.task'

BaseOptions = mp.tasks.BaseOptions
GestureRecognizer = mp.tasks.vision.GestureRecognizer
GestureRecognizerOptions = mp.tasks.vision.GestureRecognizerOptions
GestureRecognizerResult = mp.tasks.vision.GestureRecognizerResult
VisionRunningMode = vision.RunningMode

base_options = BaseOptions(model_asset_path=model_path)

landmarks = []

connections = [(0, 1), (1, 2), (2, 3), (3, 4),  # Thumb
               (0, 5), (5, 6), (6, 7), (7, 8),  # Index finger
               (0, 9), (9, 10), (10, 11), (11, 12),  # Middle finger
               (0, 13), (13, 14), (14, 15), (15, 16),  # Ring finger
               (0, 17), (17, 18), (18, 19), (19, 20),  # Little finger
               (1, 5), (5, 9), (9, 13), (13, 17)]  # Palm

colors = [(111, 71, 239), (102, 209, 255), (160, 214, 6), (178, 138, 17)]


def num_to_range(num, inMin, inMax, outMin, outMax):
    return outMin + (float(num - inMin) / float(inMax - inMin) * (outMax - outMin))


def handleResult(result: GestureRecognizerResult, output_image: mp.Image, timestamp_ms: int):
    output_ndarray = output_image.numpy_view()
    resultFrame = cv2.cvtColor(output_ndarray, cv2.COLOR_BGR2RGB)
    frameY, frameX, c = resultFrame.shape

    landmarks.clear()
    if result.hand_landmarks:
        for handslms in result.hand_landmarks:
            handLandmarks = []
            for lm in handslms:
                lmx = int(lm.x * frameX)
                lmy = int(lm.y * frameY)
                radius = num_to_range(abs(lm.z), 0, 0.7, 5, 100)
                handLandmarks.append([lmx, lmy, radius])
            landmarks.append(handLandmarks)


def drawLandmarks(drawFrame):
    for handLandmarks in landmarks:
        for connection in connections:
            x1, y1, radius1 = handLandmarks[connection[0]]
            x2, y2, radius2 = handLandmarks[connection[1]]
            cv2.line(drawFrame, (x1 * 2, y1 * 2), (x2 * 2, y2 * 2), (255, 255, 255), 3)

        for i in range(len(handLandmarks)):
            cv2.circle(drawFrame, (handLandmarks[i][0] * 2, handLandmarks[i][1] * 2),
                       int(handLandmarks[i][2]), colors[i % 4], cv2.FILLED)


options = GestureRecognizerOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    running_mode=VisionRunningMode.LIVE_STREAM,
    result_callback=handleResult,
    num_hands=4)

recognizer = GestureRecognizer.create_from_options(options)

while True:
    _, frame = cap.read()
    frame = cv2.flip(frame, 1)
    framergb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    framergb = cv2.resize(framergb, (640, 360))
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=framergb)

    black_frame = np.full_like(frame, 51)

    frame_timestamp_ms = math.floor(cap.get(cv2.CAP_PROP_POS_MSEC))

    recognizer.recognize_async(mp_image, frame_timestamp_ms)

    try:
        if landmarks: drawLandmarks(black_frame)
    except Exception as e:
        print(e)

    cv2.imshow('Output Image', black_frame)
    cv2.waitKey(1)

    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
