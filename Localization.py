import copy
import time
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

    # Saving a copy to make an unblurred crop
    image_original = copy.deepcopy(image)

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

    #t = time.time()
    #cv2.imwrite(f"debug-images/test{t}.png", mask)

    # TODO: apply contour search on the result of edge detection

    # Find the contour clusters
    contours, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    # Filter out contours that are too small
    plates = [c for c in contours if cv2.contourArea(c) > 2000]  # TODO: Tune the area parameter

    #print([cv2.contourArea(c) for c in plates])

    # TODO: deal with rotations

    # If no plate cluster were found terminate
    if len(plates) == 0:
        #cv2.imwrite(f"debug-images/test{t}FAILABOVEnothing.png", np.ones((5, 5)))
        return None

    # Iterate over top 3 biggest clusters and try to find one with a matching aspect ratio
    tries = 0
    found_plate = False
    while tries < 3 and tries < len(plates):
        # Fit the bounding box
        c = plates[tries]
        x = c[:, :, 0]
        y = c[:, :, 1]
        min_x, max_x, min_y, max_y = int(min(x)), int(max(x)), int(min(y)), int(max(y))

        aspect_ratio=(max_x - min_x) / (max_y - min_y)

        # Check the aspect ratio
        if aspect_ratio < 2 or aspect_ratio > 8:  # TODO: first rotate the plate, then check the ratio
            #cv2.imwrite(f"debug-images/test{t}FAILABOVE.png", np.ones((5, 5)))
            tries += 1
        else:
            found_plate = True
            break

    if not found_plate:
        return None

    img_cropped = image_original[min_y:max_y, min_x:max_x]

    bbs = [[min_x, min_y, max_x, max_y]]

    # TODO: add a majority classifier in combination with scene change

    return img_cropped, bbs
