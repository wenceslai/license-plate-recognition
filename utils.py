import os
import re

import matplotlib.pyplot as plt

def intersection_over_union(box_a, box_b):
    """
    computers intersection over union the input format of each box is
    [x1, y1, x2, y2]
    """
    # determine the (x, y)-coordinates of the intersection rectangle
    x_a = max(box_a[0], box_b[0])
    y_a = max(box_a[1], box_b[1])
    x_b = min(box_a[2], box_b[2])
    y_b = min(box_a[3], box_b[3])

    # compute the area of intersection rectangle
    inter_area = max(0, x_b - x_a + 1) * max(0, y_b - y_a + 1)

    # compute the area of both the prediction and ground-truth rectangles
    box_a_area = (box_a[2] - box_a[0] + 1) * (box_a[3] - box_a[1] + 1)
    box_b_area = (box_b[2] - box_b[0] + 1) * (box_b[3] - box_b[1] + 1)

    # compute the intersection over union by taking the intersection area
    # and dividing it by the sum of prediction + ground-truth
    # areas - the intersection area
    iou = inter_area / float(box_a_area + box_b_area - inter_area)

    # return the intersection over union value
    return iou


def get_filenames(directory):
    filenames = []
    for filename in os.listdir(directory):
        if os.path.isfile(os.path.join(directory, filename)):
            filenames.append(filename)
    return filenames


def crop_by_percentage(img, scale_width, scale_height):
    center_x, center_y = img.shape[1] / 2, img.shape[0] / 2
    width_scaled, height_scaled = img.shape[1] * scale_width, img.shape[0] * scale_height
    left_x, right_x = center_x - width_scaled / 2, center_x + width_scaled / 2
    top_y, bottom_y = center_y - height_scaled / 2, center_y + height_scaled / 2
    img_cropped = img[int(top_y):int(bottom_y), int(left_x):int(right_x)]

    return img_cropped

def plotImage(img, title=""):
    # Display image
    plt.imshow(img, cmap=plt.cm.gray, vmin=0, vmax=255)
    plt.title(title)
    plt.show()

def fill_dashes(s):
    if len(s) != 6: return None

    if re.search(r'\d{2}[A-Z]{4}', s):
        return s[0] + s[1] + "-" + s[2] + s[3] + "-" + s[4] + s[5]

    elif re.search(r'\d{2}[A-Z]{2}\d{2}', s):
        return s[0] + s[1] + "-" + s[2] + s[3] + "-" + s[4] + s[5]

    elif re.search(r'\d{2}[A-Z]{3}\d{1}', s):
        return s[0] + s[1] + "-" + s[2] + s[3] + s[4] + "-" + s[5]

    elif re.search(r'\d{1}[A-Z]{3}\d{2}', s):
        return s[0] + "-" + s[1] + s[2] + s[3] + "-" + s[4] + s[5]

    elif re.search(r'[A-Z]{2}\d{3}[A-Z]', s):
        return s[0] + s[1] + "-" + s[2] + s[3] + s[4] + "-" + s[5]

    elif re.search(r'[A-Z]\d{3}[A-Z]{2}', s):
        return s[0] + "-" + s[1] + s[2] + s[3] + "-" + s[4] + s[5]

    elif re.search(r'[A-Z]{3}\d{2}[A-Z]', s):
        return s[0] + s[1] + s[2] + "-" + s[3] + s[4] + "-" + s[5]

    elif re.search(r'[A-Z]{1}\d{2}[A-Z]{3}', s):
        return s[0] + "-" + s[1] + s[2] + "-" + s[3] + s[4] + s[5]

    elif re.search(r'\d{1}[A-Z]{2}\d{3}', s):
        return s[0] + "-" + s[1] + s[2] + "-" + s[3] + s[4] + s[5]

    elif re.search(r'\d{3}[A-Z]{2}\d{1}', s):
        return s[0] + s[1] + s[2] + "-" + s[3] + s[4] + "-" + s[5]

    elif re.search(r'[A-Z]{2}\d{2}[A-Z]{2}', s):
        return s[0] + s[1] + "-" + s[2] + s[3] + "-" + s[4] + s[5]

    elif re.search(r'[A-Z]{4}\d{2}', s):
        return s[0] + s[1] + "-" + s[2] + s[3] + "-" + s[4] + s[5]

    else:
        return None


def is_close(value1, value2, margin_percent=20):
    margin = margin_percent / 100.0
    lower_bound = (1 - margin) * value2
    upper_bound = (1 + margin) * value2
    return lower_bound <= value1 <= upper_bound

def strip_of_dashes(plate):
    return plate.replace('-', '')

def similar_strings(str1, str2):
    # Check if the lengths are different by more than 1
    if abs(len(str1) - len(str2)) > 1:
        return False

    # Make sure the longer string is str1
    if len(str1) < len(str2):
        str1, str2 = str2, str1

    differences = 0

    # Compare each character in the strings
    for char1, char2 in zip(str1, str2):
        if char1 != char2:
            differences += 1
            if differences > 2:
                return False  # More than 2 difference, not similar

    # Check if the length difference is exactly 2
    if len(str1) > len(str2):
        differences += 1

    return differences <= 2  # True if at most 2 difference
