import cv2
import numpy as np


class PoseAvatarWindow:
    def __init__(self, window_title="Pose Avatar", width=640, height=360):
        self.window_title = window_title
        self.width = width
        self.height = height
        cv2.namedWindow(self.window_title, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.window_title, self.width, self.height)

    def render(self, landmarks, connections):
        canvas = np.zeros((self.height, self.width, 3), dtype=np.uint8)

        for connection in connections:
            p1 = landmarks[connection.start]
            p2 = landmarks[connection.end]
            x1, y1 = int(p1.x * self.width), int(p1.y * self.height)
            x2, y2 = int(p2.x * self.width), int(p2.y * self.height)
            cv2.line(canvas, (x1, y1), (x2, y2), (255, 200, 0), 1)

        for landmark in landmarks:
            x, y = int(landmark.x * self.width), int(landmark.y * self.height)
            cv2.circle(canvas, (x, y), 2, (0, 255, 0), -1)

        cv2.imshow(self.window_title, canvas)
