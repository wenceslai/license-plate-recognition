import pandas as pd
import os
import cv2
from Localization import plate_detection
from utils import intersection_over_union, get_filenames
import json


images = get_filenames("test-images/Categorie1")

nans = 0
ious = []

for filename in val_images:
    row = annot.loc[annot["image"] == filename].iloc[0]

    if row["image"].startswith("cat1"):
        path = os.path.join("validation-images", "Categorie1", row["image"])
    else:
        path = os.path.join("validation-images", "Categorie2", row["image"])

    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)

    out = plate_detection(img)

    if out is None:
        nans += 1
        continue

    plate, bbs = out

    h, w, _ = img.shape

    pred_bb = bbs[0]  # for cat1 and 2 only 1 license plate possible
    label = json.loads(row["label"])[0]
    gt_bb = [label["x"] / 100 * w, label["y"] / 100 * h,
             (label["x"] + label["width"]) / 100 * w, (label["y"] + label["height"]) / 100 * h]

    iou = intersection_over_union(pred_bb, gt_bb)
    cv2.imwrite(os.path.join("output-images", row["image"]), plate)

    ious.append(iou)

ious = [iou for iou in ious if iou > 0.6]

print(f"accuracy: {len(ious) / len(val_images)}, computed on {len(val_images)} examples, number of 0 plates detected {nans}")


