import cv2
import os
import matplotlib.pyplot as plt

folder_path = "dataset/TrainingSet/Categorie III"

video_files = [f for f in os.listdir(folder_path) if f.endswith('.avi')]

all_video_images = []

for video_file in video_files:
    video_path = os.path.join(folder_path, video_file)
    print(f"Processing video: {video_path}")

    cap = cv2.VideoCapture(video_path)

    video_images = []

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        # Convert the frame to grayscale if needed
        # gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        video_images.append(frame)

    cap.release()

    all_video_images.append(video_images[7])

print(len(all_video_images))


from Localization import plate_detection
from Recognize import segment_and_recognize

image = all_video_images[4]

plt.imshow(image)
plt.show()

results = plate_detection(image)

if results is not None:
    print("plates", len(results))
    print(segment_and_recognize(results))
else:
    print("None")
