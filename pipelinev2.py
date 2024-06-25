import argparse
import os
import logging
import time
import json

import cv2
import numpy as np
from PIL import Image

from plots import RUN_PATH, Plotter
from gauge_detection.detection_inference import detection_gauge_face
from ocr.ocr_inference import ocr, ocr_rotations, ocr_single_rotation, ocr_warp
from key_point_detection.key_point_inference import KeyPointInference, detect_key_points
from geometry.ellipse import fit_ellipse, cart_to_pol, get_line_ellipse_point, \
    get_point_from_angle, get_polar_angle, get_theta_middle, get_ellipse_error
from angle_reading_fit.angle_converter import AngleConverter
from angle_reading_fit.line_fit import line_fit, line_fit_ransac
from segmentation.segmenation_inference import get_start_end_line, segment_gauge_needle, \
    get_fitted_line, cut_off_line
# pylint: disable=no-name-in-module
# pylint: disable=no-member
from evaluation import constants

OCR_THRESHOLD = 0.7
RESOLUTION = (448, 448)  # Make sure both dimensions are multiples of 14 for keypoint detection

# Flags for pipeline customization
WRAP_AROUND_FIX = True
RANSAC = True
WARP_OCR = True
RANDOM_ROTATIONS = False
ZERO_POINT_ROTATION = True
OCR_ROTATION = RANDOM_ROTATIONS or ZERO_POINT_ROTATION

def crop_image(img, box, flag=False, two_dimensional=False):
    """
    Crop image to bounding box and pad to maintain aspect ratio
    """
    img = np.copy(img)
    if two_dimensional:
        cropped_img = img[box[1]:box[3], box[0]:box[2]]
    else:
        cropped_img = img[box[1]:box[3], box[0]:box[2], :]

    height = box[3] - box[1]
    width = box[2] - box[0]

    delta = height - width if height > width else width - height
    pad_color = [0, 0, 0]

    if height > width:
        left, right = delta // 2, delta - (delta // 2)
        top = bottom = 0
    else:
        top, bottom = delta // 2, delta - (delta // 2)
        left = right = 0

    new_img = cv2.copyMakeBorder(cropped_img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=pad_color)
    return (new_img, (top, bottom, left, right)) if flag else new_img

def move_point_resize(point, original_resolution, resized_resolution):
    new_point_x = point[0] * resized_resolution[0] / original_resolution[0]
    new_point_y = point[1] * resized_resolution[1] / original_resolution[1]
    return new_point_x, new_point_y

def rescale_ellipse_resize(ellipse_params, original_resolution, resized_resolution):
    x0, y0, ap, bp, phi = ellipse_params
    x0_new, y0_new = move_point_resize((x0, y0), original_resolution, resized_resolution)
    scaling_factor = resized_resolution[0] / original_resolution[0]
    ap_x_new = scaling_factor * ap
    bp_x_new = scaling_factor * bp
    return x0_new, y0_new, ap_x_new, bp_x_new, phi

def detect_gauge(image, detection_model_path, plotter, debug):
    box, all_boxes = detection_gauge_face(image, detection_model_path)
    if debug:
        plotter.plot_bounding_box_img(all_boxes)
    return box

def keypoint_detection(cropped_resized_img, key_point_model_path, plotter, debug, eval_mode):
    key_point_inferencer = KeyPointInference(key_point_model_path)
    heatmaps = key_point_inferencer.predict_heatmaps(cropped_resized_img)
    key_point_list = detect_key_points(heatmaps)
    key_points = key_point_list[1]
    start_point = key_point_list[0]
    end_point = key_point_list[2]

    if debug:
        plotter.plot_heatmaps(heatmaps)
        plotter.plot_key_points(key_point_list)

    return key_points, start_point, end_point

def fit_ellipse_params(key_points, debug, plotter):
    coeffs = fit_ellipse(key_points[:, 0], key_points[:, 1])
    try:
        ellipse_params = cart_to_pol(coeffs)
    except ValueError:
        raise ValueError("Ellipse parameters not an ellipse.")
    if debug:
        plotter.plot_ellipse(key_points, ellipse_params, 'key_points')
    return ellipse_params

def find_zero_point(start_point, end_point, ellipse_params):
    if WRAP_AROUND_FIX and start_point.shape == (1, 2) and end_point.shape == (1, 2):
        theta_start = get_polar_angle(start_point.flatten(), ellipse_params)
        theta_end = get_polar_angle(end_point.flatten(), ellipse_params)
        theta_zero = get_theta_middle(theta_start, theta_end)
    else:
        bottom_middle = np.array((RESOLUTION[0] / 2, RESOLUTION[1]))
        theta_zero = get_polar_angle(bottom_middle, ellipse_params)
    zero_point = get_point_from_angle(theta_zero, ellipse_params)
    return zero_point

def perform_ocr(cropped_img, plotter, debug):
    if RANDOM_ROTATIONS:
        ocr_readings, ocr_visualization, degree = ocr_rotations(cropped_img, plotter, debug)
    elif WARP_OCR or OCR_ROTATION:
        # Implemented later in the refactored code
        pass
    else:
        if debug:
            ocr_readings, ocr_visualization = ocr(cropped_img, debug)
        else:
            ocr_readings = ocr(cropped_img, debug)
    return ocr_readings

def process_image(image, detection_model_path, key_point_model_path, segmentation_model_path, run_path, debug, eval_mode, image_is_raw=False):
    result = []
    errors = {}
    result_full = {}

    if not image_is_raw:
        logging.info("Start processing image at path %s", image)
        image = Image.open(image).convert("RGB")
        image = np.asarray(image)
    else:
        logging.info("Start processing image")

    plotter = Plotter(run_path, image)
    if eval_mode:
        result_full[constants.IMG_SIZE_KEY] = {'width': image.shape[1], 'height': image.shape[0]}

    if debug:
        plotter.save_img()

    box = detect_gauge(image, detection_model_path, plotter, debug)
    cropped_img = crop_image(image, box)
    cropped_resized_img = cv2.resize(cropped_img, dsize=RESOLUTION, interpolation=cv2.INTER_CUBIC)

    if eval_mode:
        result_full[constants.GAUGE_DET_KEY] = {'x': box[0].item(), 'y': box[1].item(), 'width': box[2].item() - box[0].item(), 'height': box[3].item() - box[1].item()}

    if debug:
        plotter.set_image(cropped_resized_img)
        plotter.plot_image('cropped')

    key_points, start_point, end_point = keypoint_detection(cropped_resized_img, key_point_model_path, plotter, debug, eval_mode)
    if eval_mode:
        if start_point.shape == (1, 2):
            result_full[constants.KEYPOINT_START_KEY] = {'x': start_point[0][0], 'y': start_point[0][1]}
        else:
            result_full[constants.KEYPOINT_START_KEY] = constants.FAILED
        if end_point.shape == (1, 2):
            result_full[constants.KEYPOINT_END_KEY] = {'x': end_point[0][0], 'y': end_point[0][1]}
        else:
            result_full[constants.KEYPOINT_END_KEY] = constants.FAILED
        result_full[constants.KEYPOINT_NOTCH_KEY] = [{'x': point[0], 'y': point[1]} for point in key_points]

    ellipse_params = fit_ellipse_params(key_points, debug, plotter)
    zero_point = find_zero_point(start_point, end_point, ellipse_params)
    if debug:
        plotter.plot_zero_point_ellipse(np.array(zero_point), np.vstack((start_point, end_point)), ellipse_params)

    cropped_img_resolution = (cropped_img.shape[1], cropped_img.shape[0])
    if WARP_OCR or OCR_ROTATION:
        res_zero_point = list(move_point_resize(zero_point, RESOLUTION, cropped_img_resolution))
        res_ellipse_params = rescale_ellipse_resize(ellipse_params, RESOLUTION, cropped_img_resolution)
        if OCR_ROTATION:
            ocr_readings, ocr_visualization, degree = ocr_warp(cropped_img, res_zero_point, res_ellipse_params, plotter, debug, RANDOM_ROTATIONS, ZERO_POINT_ROTATION)
            if eval_mode:
                result_full[constants.OCR_ROTATION_KEY] = degree
        else:
            ocr_readings, ocr_visualization = ocr_warp(cropped_img, res_zero_point, res_ellipse_params, plotter, debug, RANDOM_ROTATIONS, ZERO_POINT_ROTATION)
    else:
        ocr_readings = perform_ocr(cropped_img, plotter, debug)

    if debug:
        plotter.plot_ocr_result(ocr_visualization)

    ocr_readings = [reading for reading in ocr_readings if reading['conf'] > OCR_THRESHOLD]
    angles, readings = [], []
    for reading in ocr_readings:
        point = (reading['cx'], reading['cy'])
        theta = get_polar_angle(point, res_ellipse_params)
        if theta is not None:
            angles.append(theta)
            readings.append(reading['text'])

    result.extend(list(zip(readings, angles)))
    if eval_mode:
        result_full[constants.OCR_KEY] = [{'value': reading['text'], 'conf': reading['conf'], 'cx': reading['cx'], 'cy': reading['cy']} for reading in ocr_readings]
    if debug:
        plotter.plot_ocr_result(ocr_visualization, plot_final=True)

    start_line, end_line = get_start_end_line(cropped_img, segmentation_model_path)
    segment_gauge_needle(start_line, end_line, cropped_img, plotter, debug)

    if debug:
        plotter.plot_image('final')
    return result, errors, result_full

def main(args):
    start_time = time.time()
    detection_model_path = args.detection_model_path
    key_point_model_path = args.key_point_model_path
    segmentation_model_path = args.segmentation_model_path
    images = args.images
    print(images)
    run_path = RUN_PATH
    debug = args.debug
    eval_mode = args.eval_mode
    raw = args.raw

    results = []
    for image in images:

        result, errors, result_full = process_image(
            image, detection_model_path, key_point_model_path, segmentation_model_path, run_path, debug, eval_mode, raw)
        results.append({'result': result, 'errors': errors, 'result_full': result_full})

    with open('output.json', 'w') as f:
        json.dump(results, f)

    logging.info('Total Time: %s seconds', str(time.time() - start_time))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process images to detect gauges, key points, and perform OCR")
    parser.add_argument("--detection_model_path", type=str, required=True, help="Path to the gauge detection model")
    parser.add_argument("--key_point_model_path", type=str, required=True, help="Path to the key point detection model")
    parser.add_argument("--segmentation_model_path", type=str, required=True, help="Path to the segmentation model")
    parser.add_argument("--images", nargs='+', required=True, help="List of images to process")
    parser.add_argument("--debug", action="store_true", help="Flag to enable debug mode")
    parser.add_argument("--eval_mode", action="store_true", help="Flag to enable evaluation mode")
    parser.add_argument("--raw", action="store_true", help="Flag to indicate images are raw (in-memory)")
    args = parser.parse_args()
    main(args)
