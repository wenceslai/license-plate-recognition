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

    # TODO: Replace the below lines with your code.
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    colorMin = np.array([15, 50, 50])
    colorMax = np.array([30, 256, 256])

    mask = cv2.inRange(hsv, colorMin, colorMax)

    c, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    plates = []
    marked = image.copy()
    marked = cv2.cvtColor(marked, cv2.COLOR_BGR2RGB)
    img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    for contour in c:
        area = cv2.contourArea(contour)
        if area > 2000:
            plates.append(contour)
    if len(plates)>0:
        c = plates[0]

        x = c[:, :, 0]
        y = c[:, :, 1]

        minx = int(min(x))
        maxx = int(max(x))
        miny = int(min(y))
        maxy = int(max(y))

        cropimg = image[miny:maxy, minx:maxx]


        return cropimg
    return None
