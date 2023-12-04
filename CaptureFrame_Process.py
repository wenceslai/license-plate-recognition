import cv2
import os
import pandas as pd
import Localization
import Recognize


def CaptureFrame_Process(file_path, sample_frequency, save_path):
    """
    In this file, you will define your own CaptureFrame_Process funtion. In this function,
    you need three arguments: file_path(str type, the video file), sample_frequency(second), save_path(final results saving path).
    To do:
        1. Capture the frames for the whole video by your sample_frequency, record the frame number and timestamp(seconds).
        2. Localize and recognize the plates in the frame.(Hints: need to use 'Localization.plate_detection' and 'Recognize.segmetn_and_recognize' functions)
        3. If recognizing any plates, save them into a .csv file.(Hints: may need to use 'pandas' package)
    Inputs:(three)
        1. file_path: video path
        2. sample_frequency: second
        3. save_path: final .csv file path
    Output: None
    """
    print(file_path)
    directory = r'testimages'
    # TODO: Read frames from the video (saved at `file_path`) by making use of `sample_frequency`
    video = cv2.VideoCapture(file_path)

    fps = video.get(cv2.CAP_PROP_FPS)

    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

    frame_interval = int(fps / sample_frequency)

    frames = []

    for frame_count in range(0, total_frames, frame_interval):
        # Set the frame position
        video.set(cv2.CAP_PROP_POS_FRAMES, frame_count)

        ret, frame = video.read()

        if ret:
            frames.append(frame)

    # TODO: Implement actual algorithms for Localizing Plates
    i=-1
    for frame in frames:
        i+=1
        plate = Localization.plate_detection(frame)

        # TODO: Implement actual algorithms for Recognizing Characters
        if plate is not None:
            cv2.imwrite(os.path.join(directory, 'im' + str(i)+'.jpg'), plate)

        output = open(save_path, "w")
        output.write("License plate,Frame no.,Timestamp(seconds)\n")

        # TODO: REMOVE THESE (below) and write the actual values in `output`
        output.write("XS-NB-23,34,1.822\n")
        # output.write("YOUR,STUFF,HERE\n")
        # TODO: REMOVE THESE (above) and write the actual values in `output`

    pass
