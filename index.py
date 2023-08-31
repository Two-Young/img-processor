import cv2
import uuid
import os
import shutil
import numpy as np
import traceback

SOURCE_DIR = os.path.relpath("assets")
FMT_DIR = os.path.relpath("formatted")

original_images = {}
image_layers = []
box_width, box_height = 100, 90

# truncate all files from fmt
shutil.rmtree(FMT_DIR)
os.mkdir(FMT_DIR)

# iterate images
for file in os.listdir(SOURCE_DIR):
    filename = os.fsdecode(file)
    filepath = SOURCE_DIR + "/" + filename

    new_filename = str(uuid.uuid4())
    new_filepath = FMT_DIR + "/" + new_filename + ".png"
    shutil.copyfile(filepath, new_filepath)

for file in os.listdir(FMT_DIR):
    filename = os.fsdecode(file)
    filepath = FMT_DIR + "/" + filename

    image = cv2.imread(filepath)
    original_images[filename] = image

# image_layers.append(original_images)
print(f"loaded {len(original_images)} images.",)


def is_8bit(img):
    return len(img.shape) == 2


def assert_8bit(img):
    new_img = img.copy()
    if is_8bit(new_img):
        return new_img
    return cv2.cvtColor(new_img, cv2.COLOR_BGR2GRAY)


def assert_24bit(img):
    new_img = img.copy()
    if not is_8bit(new_img):
        return new_img
    return cv2.cvtColor(new_img, cv2.COLOR_GRAY2BGR)


def draw_image_results():
    layer_count = len(image_layers)
    image_count = len(original_images)

    win_height = image_count * box_height
    win_width = layer_count * box_width
    print(win_width, win_height, box_width, box_height)

    canvas = np.zeros((win_height, win_width, 3), dtype=np.uint8)
    for i in range(layer_count):
        image_bundle = image_layers[i]
        for j, item in enumerate(image_bundle.items()):
            image = item[1]
            if image is not None:
                try:
                    resized_image = cv2.resize(image, (box_width, box_height))
                    result_image = assert_24bit(resized_image)
                    canvas[j*box_height:(j+1)*box_height, i*box_width:(i+1)*box_width] = result_image
                except:
                    print("exception on", i, j)
                    traceback.print_exc()
                    # exit(-1)

    cv2.imshow('result', canvas)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def apply_effect(effect_func, *args):
    last_layer = image_layers[-1]
    new_layer = {}
    for key, image in last_layer.items():
        new_image = effect_func(image, *args)
        new_layer[key] = new_image
    image_layers.append(new_layer)


def apply_original():
    image_layers.append(original_images)


def hsv(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2HSV)


def mask(img, r1, g1, b1, r2, g2, b2):
    low = np.array([r1, g1, b1])
    high = np.array([r2, g2, b2])
    mask = cv2.inRange(img, low, high)
    return cv2.bitwise_and(img, img, None, mask)


def grayscale(img):
    result = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)


def threshold(img, val, flag):
    _, result = cv2.threshold(img, val, 255, flag)
    return result


def adaptive_threshold(img, block, C):
    new_img = assert_8bit(img)
    return cv2.adaptiveThreshold(new_img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, block, C)


def blur(img):
    return cv2.GaussianBlur(img, (7, 7), 0)


def canny(img):
    result = cv2.Canny(img, 150, 200)
    return cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)


def remove_dark(img, thr):
    new_img = assert_24bit(img)
    mask = (image[:, :, 0] < thr) & (image[:, :, 1] < thr) & (image[:, :, 2] < thr)
    new_img[mask] = [255, 255, 255]
    return new_img


def draw_contours(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    contours, _ = cv2.findContours(gray.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) == 0:
        return img.copy()
    largest_contour = max(contours, key=cv2.contourArea)
    new_img = img.copy()
    cv2.drawContours(new_img, [largest_contour], 0, (0, 255, 0), 3)
    return new_img


# apply effects
apply_original()
apply_effect(blur)
# apply_effect(threshold, 40, cv2.THRESH_TOZERO)
# apply_effect(mask, 0, 0, 10, 180, 255, 255)
# apply_effect(remove_dark, 15)
apply_effect(hsv)
# apply_effect(mask, )
# apply_effect(grayscale)
# apply_effect(threshold)
apply_effect(adaptive_threshold, 27, 3)
apply_effect(canny)
apply_effect(draw_contours)

draw_image_results()
