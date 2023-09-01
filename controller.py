import cv2
import traceback
from effects import *
from layer_effects import *

layers = []

def set_prop(key, current, max_value, callback):
    cv2.createTrackbar(key, "Trackbars", current, max_value, callback)


def set_current(key, value):
    cv2.setTrackbarPos(key, "Trackbars", value)


def get_prop(key):
    return cv2.getTrackbarPos(key, "Trackbars")


def get_props(key_list):
    values = []
    for key in key_list:
        value = get_prop(key)
        values.append(value)
    return values


def controller(original_image_layer, draw_image_results):
    global layers
    layers = [original_image_layer]

    def apply_effect(effect_func, *args, target=-1):
        global layers
        last_layer = layers[target]
        new_layer = {}
        for key, image in last_layer.items():
            try:
                new_image = effect_func(image, *args)
                new_layer[key] = new_image
            except:
                traceback.print_exc()
                new_layer[key] = null_image()
        layers.append(new_layer)
        return new_layer

    def apply_layer_effect(effect_func, *args, target=-1, targets):
        global layers
        target_layers = []
        target_layer = layers[target]
        new_layer = {}
        for t in targets:
            target_layers.append(layers[t])
        for key, image in target_layer.items():
            try:
                target_images = []
                for layer in target_layers:
                    target_images.append(layer[key])

                new_image = effect_func(image, *args, requested=target_images)
                new_layer[key] = new_image
            except:
                traceback.print_exc()
                new_layer[key] = null_image()
        layers.append(new_layer)
        return new_layer

    def apply(x):
        try:
            global layers
            layers = [original_image_layer]

            # blur
            blur_block = get_prop('blur')

            # threshold
            thr = get_prop('threshold')

            # hsv
            [lh, ls, lv] = get_props(['low_h', 'low_s', 'low_v'])
            [uh, us, uv] = get_props(['up_h', 'up_s', 'up_v'])

            # adaptive_threshold
            [adapthr_block, adapthr_C] = get_props(['adaptive_threshold_block', 'adaptive_threshold_C'])

            # effects
            apply_effect(blur, blur_block)
            apply_effect(threshold, thr, cv2.THRESH_TOZERO)
            apply_effect(adaptive_threshold, adapthr_block, adapthr_C)
            apply_layer_effect(bitwise_and, targets=[-3])
            apply_effect(hsv)
            apply_effect(mask, [lh, ls, lv], [uh, us, uv])
            apply_layer_effect(draw_all_contours, 2, targets=[-1])
            apply_layer_effect(draw_largest_contours, 5, target=-2, targets=[0])
            apply_layer_effect(draw_approx_contours, 5, target=-3, targets=[0])
            apply_layer_effect(draw_hull, 5, target=-4, targets=[0])
            apply_layer_effect(draw_cropped_result, target=-5, targets=[0])
            draw_image_results(layers)
        except:
            # traceback.print_exc()
            pass

    cv2.namedWindow("Trackbars", cv2.WINDOW_GUI_EXPANDED)

    # blur
    set_prop('blur', 1, 50, apply)

    # threshold
    set_prop('threshold', 0, 300, apply)

    # adaptive_threshold
    set_prop('adaptive_threshold_block', 27, 50, apply)
    set_prop('adaptive_threshold_C', 7, 30, apply)

    # hsv
    set_prop("low_h", 10, 179, apply)
    set_prop("up_h", 160, 179, apply)
    set_prop("low_s", 0, 255, apply)
    set_prop("up_s", 255, 255, apply)
    set_prop("low_v", 100, 255, apply)
    set_prop("up_v", 255, 255, apply)
