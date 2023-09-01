import cv2
from effects import *


def bitwise_and(img, requested):
    [original] = requested
    return cv2.bitwise_and(original, original, mask=img)


def highlight_white(img, requested):
    [original] = requested
    new_img = img.copy()
    img_hsv = hsv(img)
    sub_mask = mask(img_hsv, [100, 0, 0], [179, 255, 255])
    return sub_mask
    new_img = cv2.bitwise_and(new_img, new_img, mask=sub_mask)
    return new_img


def draw_all_contours(img, width, requested):
    [original] = requested
    contours, _ = cv2.findContours(img.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    new_img = original.copy()
    new_img = rgb(new_img)
    cv2.drawContours(new_img, contours, -1, (0, 0, 255), width)
    return new_img


def draw_largest_contours(img, width, requested):
    [original] = requested
    contours, _ = cv2.findContours(img.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) == 0:
        return img.copy()
    largest_contour = max(contours, key=cv2.contourArea)
    new_img = original.copy()
    cv2.drawContours(new_img, [largest_contour], 0, (0, 255, 0), width)
    return new_img


def draw_approx_contours(img, width, requested):
    [original] = requested
    contours, _ = cv2.findContours(img.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) == 0:
        return img.copy()
    largest_contour = max(contours, key=cv2.contourArea)
    new_img = original.copy()

    epsilon = 0.03 * cv2.arcLength(largest_contour, True)
    approx = cv2.approxPolyDP(largest_contour, epsilon, True)
    cv2.drawContours(new_img, [approx], 0, (0, 255, 0), width)
    return new_img


def draw_hull(img, width, requested):
    [original] = requested
    new_img = original.copy()

    contours, _ = cv2.findContours(img.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) == 0:
        return img.copy()
    largest_contour = max(contours, key=cv2.contourArea)

    epsilon = 0.03 * cv2.arcLength(largest_contour, True)
    approx = cv2.approxPolyDP(largest_contour, epsilon, True)

    x, y, w, h = cv2.boundingRect(approx)
    hull = cv2.convexHull(approx)
    cv2.drawContours(new_img, [hull], 0, (0, 255, 0), width)
    return new_img


def draw_cropped_result(img, requested):
    [original] = requested
    new_img = original.copy()

    contours, _ = cv2.findContours(img.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) == 0:
        return img.copy()
    largest_contour = max(contours, key=cv2.contourArea)

    epsilon = 0.03 * cv2.arcLength(largest_contour, True)
    approx = cv2.approxPolyDP(largest_contour, epsilon, True)

    x, y, w, h = cv2.boundingRect(approx)
    mask = np.zeros((h, w, 3), dtype=np.uint8)
    cv2.drawContours(mask, [approx], 0, (255, 255, 255), -1)

    hull = cv2.convexHull(approx)
    rect_pts = np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], dtype="float32")

    # 극단점을 찾기 위한 좌표 변환
    sums = hull.sum(axis=2)
    diffs = np.diff(hull, axis=2)

    # 극단점 찾기
    left_top_most = hull[np.argmin(sums)][0]
    right_bottom_most = hull[np.argmax(sums)][0]
    right_top_most = hull[np.argmax(diffs)][0]
    left_bottom_most = hull[np.argmin(diffs)][0]

    extreme_points = np.array([left_top_most, left_bottom_most, right_bottom_most, right_top_most, ], dtype="float32")

    matrix = cv2.getPerspectiveTransform(extreme_points, rect_pts)
    warped = cv2.warpPerspective(new_img, matrix, (w, h))
    return warped
