import pandas as pd
import os
import cv2
from utils import intersection_over_union, get_filenames
from Recognize import segment_and_recognize

files = get_filenames("test-set-recognition")

images = []
labels = []

for file in files:
    label = file.split(".")[0]
    labels.append(label)
    images.append(cv2.imread(os.path.join("test-set-recognition", file), cv2.IMREAD_UNCHANGED))

preds = segment_and_recognize(images)

correct = 0

for y_hat, y in zip(preds, labels):
    if y_hat == y:
        correct += 1
    print(y_hat, y)

print(correct / len(preds))




