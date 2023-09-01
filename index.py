import uuid
import os
import shutil
import traceback
from effects import *
import controller
import time

SOURCE_DIR = os.path.relpath("assets")
FMT_DIR = os.path.relpath("formatted")
RESULT_DIR = os.path.relpath("result")

original_images = {}
image_layers = []
final_layers = []
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


def null_image():
    return np.zeros((100, 100, 3), dtype=np.uint8)


def draw_image_results(layers):
    global final_layers
    layer_count = len(layers)
    image_count = len(original_images)

    win_height = max(image_count * box_height, 100)
    win_width = max(layer_count * box_width, 100)
    # print(win_width, win_height, box_width, box_height)

    canvas = np.zeros((win_height, win_width, 3), dtype=np.uint8)
    for i in range(layer_count):
        image_bundle = layers[i]
        for j, item in enumerate(image_bundle.items()):
            image = item[1]
            if image is not None:
                try:
                    copied = image.copy()
                    resized_image = cv2.resize(copied, (box_width, box_height))
                    result_image = assert_24bit(resized_image)
                    canvas[j*box_height:(j+1)*box_height, i*box_width:(i+1)*box_width] = result_image
                except:
                    print("exception on", i, j)
                    traceback.print_exc()

    cv2.imshow('result', canvas)
    final_layers = layers


def apply_effect(effect_func, *args, target=-1):
    last_layer = image_layers[target]
    new_layer = {}
    for key, image in last_layer.items():
        try:
            new_image = effect_func(image, *args)
            new_layer[key] = new_image
        except:
            traceback.print_exc()
    image_layers.append(new_layer)


def add_layer(layer_func, *args, target=-1):
    last_layer = image_layers[target]
    layer = layer_func(last_layer, *args)
    image_layers.append(layer)


controller.controller(original_images, draw_image_results)
cv2.waitKey(0)
cv2.destroyAllWindows()

# save file
result_layer = final_layers[-1]
for key, image in result_layer.items():
    curtime = time.localtime()
    timestr = time.strftime("%Y%m%d-%H%M%S", curtime)
    dirpath = RESULT_DIR + "/" + timestr
    filename = key
    if not os.path.isdir(dirpath):
        os.mkdir(dirpath)
    filepath = dirpath + "/" + filename
    cv2.imwrite(filepath, image)