import copy
import os
import time

import cv2
import numpy as np
from scipy import ndimage

from utils import is_close


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

    # Saving a copy to make an unblurred crop
    image_original = copy.deepcopy(image)

    # Preprocess and binarize
    mask = preprocess(image)

    #cv2.imwrite("debug-images/masksss.png", mask)

    # Find the contour clusters
    contours, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    # Filter out contours that are too small
    plates = [c for c in contours if cv2.contourArea(c) > 2000]  # TODO: Tune the area parameter

    # If no plate cluster were found terminate
    if len(plates) == 0:
        return None

    # Iterate over top 5 biggest clusters and try to find one with a matching aspect ratio
    tries = 0
    found_plates = 0
    bbs = []
    last_contour_size = None

    # Going over the 3 biggest contours, finding the first 1 or 2 valid ones
    while tries < 3 and tries < len(plates):
        # Fit the bounding box
        c = plates[tries]
        x = c[:, :, 0]
        y = c[:, :, 1]

        min_x, max_x, min_y, max_y = int(min(x)), int(max(x)), int(min(y)), int(max(y))

        aspect_ratio = (max_x - min_x) / (max_y - min_y)

        # Check the aspect ratio
        if aspect_ratio < 2 or aspect_ratio > 8:
            tries += 1
        else:
            if found_plates == 0:  # If no plates found yet save the size
                last_contour_size = cv2.contourArea(c)

            if found_plates == 1 and not is_close(last_contour_size, cv2.contourArea(c), 50):  # The license plates should be similar to each other
                break

            bbs.append([min_x, min_y, max_x, max_y])
            found_plates += 1

            if found_plates == 2:  # We found two license plates so break
                break

            tries += 1

    # If no plates found return None
    if len(bbs) == 0:
        return None

    results = []
    for bb in bbs:
        # Cropping out the license plate
        min_x, min_y, max_x, max_y = bb
        img_cropped = image_original[min_y:max_y, min_x:max_x]

        # Rotating the license plate
        mask_cropped = mask[min_y:max_y, min_x:max_x]
        angle = find_rotation_angle(mask_cropped)
        angle = angle - 90 if angle >= 0 else angle + 90
        img_rotated = ndimage.rotate(img_cropped, angle, reshape=False)

       #cv2.imwrite(f"debug-images/test1{time.time()}.png", img_rotated)

        if angle > 10:
            img_rotated = crop_after_rotation(img_rotated)

        results.append(img_rotated)

        #cv2.imwrite(f"debug-images/test2{time.time()}.png", img_rotated)

    # TODO: Sharpen in the image?

    return results


def crop_after_rotation(img):
    mask = preprocess(img)

    contours, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    plate = contours[0]
    x = plate[:, :, 0]
    y = plate[:, :, 1]
    min_x, max_x, min_y, max_y = int(min(x)), int(max(x)), int(min(y)), int(max(y))
    img_cropped = img[min_y:max_y, min_x:max_x]

    # Create a mask to identify black pixels and replace with orange
    mask = np.all(img_cropped == np.array([0, 0, 0]), axis=-1)
    img_cropped[mask] = [50, 165, 255]  # BGR values for orange

    return img_cropped


def preprocess(image):
    # Removing noise
    image = cv2.GaussianBlur(image, (5, 5), 0)
    image = cv2.medianBlur(image, 5)

    # Convert the img to hsv spectrum
    img_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Colour segmentation
    colorMin = np.array([15, 50, 50])
    colorMax = np.array([30, 256, 256])
    mask = cv2.inRange(img_hsv, colorMin, colorMax)

    # Apply erosion and dilation to remove noise
    kernel_size = 5
    erosion_kernel = np.ones((kernel_size, kernel_size), np.uint8)
    mask = cv2.erode(mask, erosion_kernel, iterations=1)
    dilation_kernel = np.ones((kernel_size, kernel_size), np.uint8)
    mask = cv2.dilate(mask, dilation_kernel, iterations=1)

    return mask


def find_rotation_angle(binarized_image):
    # Perform Hough Line Transform
    lines = cv2.HoughLines(binarized_image, 1, np.pi / 180, threshold=100)

    # Calculate the average angle of detected lines
    angles = []
    if lines is None:
        print("no lines found")
        return 1
    for line in lines:
        rho, theta = line[0]
        angles.append(theta)

    lines_image = cv2.cvtColor(binarized_image, cv2.COLOR_GRAY2BGR)
    for line in lines:
        rho, theta = line[0]
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = int(a * rho)
        y0 = int(b * rho)
        x1 = int(x0 + 1000 * (-b))
        y1 = int(y0 + 1000 * (a))
        x2 = int(x0 - 1000 * (-b))
        y2 = int(y0 - 1000 * (a))
        cv2.line(lines_image, (x1, y1), (x2, y2), (0, 0, 255), 2)

    #cv2.imwrite("debug-images/test3.png", binarized_image)
    #cv2.imwrite("debug-images/test4.png", lines_image)

    if len(angles) == 0:
        print("No lines detected. Unable to determine rotation angle.")
        return None

    average_angle = np.degrees(np.mean(angles))
    return average_angle


if __name__ == "__main__":
    f = "cat1img8.png"
    img = cv2.imread(os.path.join("training-images/Categorie1", f), cv2.IMREAD_UNCHANGED)

    plate_detection(img)