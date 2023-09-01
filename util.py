import cv2
import numpy as np
import threading
import time
import traceback

last_update = 0
period = 500

def empty():
    return np.zeros((100, 100, 3), dtype=np.uint8)


def nothing(x):
    pass


def controller(layer, draw_image_results):
    new_layer = {}
    for k, img in layer.items():
        new_layer[k] = img.copy()

    def apply(x):
        try:
            lh = cv2.getTrackbarPos("low_h", "Trackbars")
            ls = cv2.getTrackbarPos("low_s", "Trackbars")
            lv = cv2.getTrackbarPos("low_v", "Trackbars")
            uh = cv2.getTrackbarPos("up_h", "Trackbars")
            us = cv2.getTrackbarPos("up_s", "Trackbars")
            uv = cv2.getTrackbarPos("up_v", "Trackbars")
            for key, frame in layer.items():
                hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
                l_blue = np.array([lh, ls, lv])
                u_blue = np.array([uh, us, uv])
                mask = cv2.inRange(hsv, l_blue, u_blue)
                result = cv2.bitwise_and(frame, frame, mask=mask)
                new_layer[key] = mask
            draw_image_results()
        except:
            # traceback.print_exc()
            return

    cv2.namedWindow("Trackbars",)
    cv2.createTrackbar("low_h", "Trackbars", 100, 179, apply)
    cv2.createTrackbar("low_s", "Trackbars", 0, 255, apply)
    cv2.createTrackbar("low_v", "Trackbars", 146, 255, apply)
    cv2.createTrackbar("up_h", "Trackbars", 179, 179, apply)
    cv2.createTrackbar("up_s", "Trackbars", 255, 255, apply)
    cv2.createTrackbar("up_v", "Trackbars", 255, 255, apply)

    # control_thread(new_layer)
    # t = threading.Thread(target=control_thread, args=(new_layer,))
    # t.start()
    return new_layer


def control_thread(layer):
    while True:
        lh = cv2.getTrackbarPos("lh", "Trackbars")
        ls = cv2.getTrackbarPos("ls", "Trackbars")
        lv = cv2.getTrackbarPos("lv", "Trackbars")
        uh = cv2.getTrackbarPos("uh", "Trackbars")
        us = cv2.getTrackbarPos("us", "Trackbars")
        uv = cv2.getTrackbarPos("uv", "Trackbars")

        for key, frame in layer.items():
            # cap.read(frame)
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

            l_blue = np.array([lh, ls, lv])
            u_blue = np.array([uh, us, uv])
            mask = cv2.inRange(hsv, l_blue, u_blue)
            result = cv2.bitwise_or(frame, frame, mask=mask)
            layer[key] = result
