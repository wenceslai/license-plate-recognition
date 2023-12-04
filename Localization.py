import cv2
import numpy as np


def plate_detection(image):
    """
    In this file, you need to define plate_detection function.
    To do:
        1. Localize the plates and crop the plates
        2. Adjust the cropped plate images
    Inputs:(One)
        1. image: captured frame in CaptureFrame_Process.CaptureFrame_Process function
        type: Numpy array (imread by OpenCV package)
    Outputs:(One)
        1. plate_imgs: cropped and adjusted plate images
        type: list, each element in 'plate_imgs' is the cropped image(Numpy array)
    Hints:
        1. You may need to define other functions, such as crop and adjust function
        2. You may need to define two ways for localizing plates(yellow or other colors)
    """

    # Convert the img to hsv spectrum
    img_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Apply Gaussian filter to each channel
    h, s, v = cv2.split(img_hsv)
    blurred_v = cv2.GaussianBlur(v, (5, 5), 0)
    img_hsv = cv2.merge([h, s, blurred_v])

    # Colour segmentation
    colorMin = np.array([15, 50, 50])
    colorMax = np.array([30, 256, 256])
    mask = cv2.inRange(img_hsv, colorMin, colorMax)

    # TODO: apply contour search on the result of edge detection
    # TODO: add morphological techniques to make the edge more robust?

    # Find the contour clusters
    contours, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    # Filter out contours that are too small
    plates = [c for c in contours if cv2.contourArea(c) > 2000]  # TODO: Tune the area parameter

    if len(plates) == 0:
        return None

    # Fit the bounding box
    c = plates[0]
    x = c[:, :, 0]
    y = c[:, :, 1]
    min_x, max_x, min_y, max_y = int(min(x)), int(max(x)), int(min(y)), int(max(y))

    # Check the aspect ratio
    if 4 >= (max_x - min_x) / (max_y - min_y) >= 6:
        return None

    img_cropped = image[min_y:max_y, min_x:max_x]

    return img_cropped
