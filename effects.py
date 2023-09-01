import cv2
import numpy as np


def null_image():
    return np.zeros((100, 100, 3), dtype=np.uint8)


def hsv(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2HSV)


def rgb(img):
    return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)


def mask(img, low, high):
    lower = np.array(low)
    upper = np.array(high)
    mask = cv2.inRange(img, lower, upper)
    return mask


def inverted_mask(img, low, high):
    lower = np.array(low)
    upper = np.array(high)
    mask = cv2.inRange(img, lower, upper)
    return 255 - mask


def double_mask(img, low1, high1, low2, high2):
    lower1 = np.array(low1)
    upper1 = np.array(high1)
    lower2 = np.array(low2)
    upper2 = np.array(high2)
    mask1 = cv2.inRange(img, lower1, upper1)
    mask2 = cv2.inRange(img, lower2, upper2)
    mask = mask1 + mask2
    return cv2.bitwise_and(img, img, mask=mask)


def grayscale(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def threshold(img, val, flag):
    _, result = cv2.threshold(img, val, 255, flag)
    return result


def adaptive_threshold(img, block, C):
    gray = grayscale(img)
    real_block = block * 2 + 1
    return cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, real_block, C)


def blur(img, size):
    block = 2 * size + 1
    return cv2.GaussianBlur(img, (block, block), 0)


def canny(img):
    result = cv2.Canny(img, 150, 200)
    return cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)


def remove_dark(img, thr):
    new_img = img.copy()
    gray = grayscale(img)
    mask = gray < thr
    for i in range(3):  # B, G, R
        new_img[:, :, i][mask] = 255  # 흰색으로 변경
    return new_img


def draw_contours(img, width):
    contours, _ = cv2.findContours(img.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) == 0:
        return img.copy()
    largest_contour = max(contours, key=cv2.contourArea)
    new_img = img.copy()
    cv2.drawContours(new_img, contours, -1, (0, 255, 0), width)
    # cv2.drawContours(new_img, [largest_contour], 0, (0, 255, 0), width)
    return new_img
