import time

import cv2
import numpy as np
import os

import utils
from utils import crop_by_percentage, plotImage, fill_dashes, isodata_thresholding


def segment_and_recognize(plate_images):
    """
    In this file, you will define your own segment_and_recognize function.
    To do:
        1. Segment the plates character by character
        2. Compute the distances between character images and reference character images(in the folder of 'SameSizeLetters' and 'SameSizeNumbers')
        3. Recognize the character by comparing the distances
    Inputs:(One)
        1. plate_imgs: cropped plate images by Localization.plate_detection function
        type: list, each element in 'plate_imgs' is the cropped image(Numpy array)
    Outputs:(One)
        1. recognized_plates: recognized plate characters
        type: list, each element in recognized_plates is a list of string(Hints: the element may be None type)
    Hints:
        You may need to define other functions.
    """
    recognized_plates = []

    for image in plate_images:
        #cv2.imwrite(f"debug-images/test1.png", image)
        image = preprocess(image)
        #cv2.imwrite(f"debug-images/test2.png",image)
        #plotImage(image)
        char_images = crop(image)
        plate_string = recognise(char_images)
        #plate_string_dashes = fill_dashes(plate_string)

        #if plate_string_dashes is None:
            #plate_string_dashes = plate_string# + "(none)"

        #print(plate_string)
        if(utils.is_correct_format(plate_string)):
            recognized_plates.append(plate_string)


    return recognized_plates


def preprocess(image):

    #cv2.imwrite(f"debug-images/test1.png", image)

    # Zoom in to remove borders of the plate
    image = crop_by_percentage(image, 0.90, 0.65)

    #cv2.imwrite(f"debug-images/test2.png", image)

    # Remove noise
    #image = cv2.GaussianBlur(image, (3, 3), 0)


    # Resize to height of x pixels
    desired_height = 70
    aspect_ratio = image.shape[1] / image.shape[0]
    desired_width = int(aspect_ratio * desired_height)
    image = cv2.resize(image, (desired_width, desired_height))

    #cv2.imwrite(f"debug-images/test3.png", image)

    # Convert to grayscale
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Histogram equalisation
    image = cv2.equalizeHist(image)

    #cv2.imwrite(f"debug-images/test4.png", image)
    #plotImage(image)

    # Threshold to binarize the image
    #mask = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 11, 2)  # INV because letters are black
    _, mask = cv2.threshold(image, 80, 255, cv2.THRESH_BINARY_INV)  # was 80

    #cv2.imwrite("debug-images/maskrecog.png", mask)

    #cv2.imwrite(f"debug-images/test5.png", mask)
    #plotImage(mask)

    # Apply erosion and dilation to remove noise
    kernel_size = 5
    erosion_kernel = np.ones((kernel_size, kernel_size), np.uint8)
    mask = cv2.erode(mask, erosion_kernel, iterations=1)
    dilation_kernel = np.ones((kernel_size, kernel_size), np.uint8)
    mask = cv2.dilate(mask, dilation_kernel, iterations=1)


    #cv2.imwrite(f"debug-images/test6.png", mask)

    return mask


def crop(image):
    # Horizontal boundaries for each character
    char_index_ranges = []
    for j in range(image.shape[1]):
        pixel_sum = np.sum(image[:, j])

        if pixel_sum >= 4 and (len(char_index_ranges) == 0 or len(char_index_ranges[-1]) == 2):
            char_index_ranges.append([])
            char_index_ranges[-1].append(j)
        elif pixel_sum < 4:
            if len(char_index_ranges) > 0 and len(char_index_ranges[-1]) == 1:
                char_index_ranges[-1].append(j - 1)

    # if no last column with no white pixels found use the last one
    if len(char_index_ranges[-1]) == 1:
        char_index_ranges[-1].append(image.shape[1] - 1)

    # Crop characters
    chars = []
    for char_range in char_index_ranges:
        l, r = char_range
        chars.append(image[:, l: r + 1])

    # Vertical boundary for each character
    chars_cropped = []
    for char in chars:
        top_index, bottom_index = None, None
        for i in range(char.shape[0]):
            pixel_sum = np.sum(char[i, :])

            if pixel_sum >= 8 and top_index is None:
                top_index = i
            elif pixel_sum < 8 and top_index is not None:
                bottom_index = i - 1
                break
        # if no last column with no white pixels found use the last one
        if bottom_index is None:
            bottom_index = char.shape[0] - 1

        # Filter out dashes
        if (bottom_index - top_index) / char.shape[0] < 0.3:
            continue

        chars_cropped.append(char[top_index: bottom_index + 1, :])

        char_index_ranges[-1].append(image.shape[1] - 1)

    #for i, img in enumerate(chars_cropped):
        #cv2.imwrite(f"debug-images/croppedchar{i}_{time.time()}.png", img)

    return chars_cropped


def recognise(images):
    result = ""
    for image in images:
        letter = recogniseletter(image)
        result = result + letter
    return result


def lowest_score(test_image, character_set, reference_characters):
    # Get the difference score with each of the reference characters
    # (or only keep track of the lowest score)
    mini = 9999999
    ans = None
    for char in character_set:
        resized_reference=cv2.resize(reference_characters[char], (70, 85))
        kernel_size = 3
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        resized_reference = cv2.dilate( resized_reference, kernel, iterations=1)
        score = difference_score(test_image,resized_reference)
        if score < mini:
            mini = score
            ans = char
    # Return a single character based on the lowest score
    return ans


def difference_score(test_image, reference_character):
    # XOR images
    res = cv2.bitwise_xor(test_image, reference_character)
    # Return the number of non-zero pixels
    return np.count_nonzero(res)


def recogniseletter(image):
    resized_image = cv2.resize(image, (70, 85))
    #plotImage(image)
    path = 'dataset\CharactersDifferentSizes'

    total_set = {'B', 'D', 'F', 'G', 'H', 'J', 'K', 'L', 'M', 'N', 'P', 'R', 'S', 'T', 'V', 'X', 'Z', '0', '1', '2',
                 '3', '4', '5', '6', '7', '8', '9'}

    reference_characters = {}
    for char in total_set:
        reference_characters[char] = cv2.imread(os.path.join("dataset", "CharactersDifferentSizes", char + ".bmp"), cv2.IMREAD_GRAYSCALE)

    result = lowest_score(resized_image, total_set, reference_characters)

    #resized_reference = cv2.resize(reference_characters[result], (70, 85))

    #plotImage(resized_reference)
    return result


if __name__ == "__main__":
    # This block will be executed only if the script is run directly
    plate = "56-JTT-5"
    #plate = "5-SXB-74"
    #plate = "01-XJ-ND"

    img = cv2.imread(os.path.join("test-set-recognition", plate + ".jpg"), cv2.IMREAD_UNCHANGED)

    res = segment_and_recognize([img])

    print(res)
