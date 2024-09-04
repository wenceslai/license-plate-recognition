# License plate recognition template

This is the final submission for the CSE2250 Image Procesing project at TU Delft.

You can find the poster with detailed description of the algorithms and results in ``poster.pdf``.

Given an input video you should recognize Dutch license plates. No modern machine learnig techniques that learn from data are used. Instead, older computer vision techniques such as color segmentation, Hough transform, morphology and other heuristics are used to process, localize, straighten and recognize the license plates. Majority voting is used to improve accuracy.

## Project setup
The shell script ``evaluator.sh`` is used for running the project and calculating scores.  
This file initially runs ``main.py`` file, followed by ``evaluation.py``. 

Rest of the project:
- ``CaptureFrame_Process.py`` for reading the input video
- ``Localization.py`` for figuring out the location of the plate in a frame
- ``Recognize.py`` for figuring out what characters are in a plate
- ``helpers/`` additional methods
- ``.gitlab-ci.yml`` Gitlab pipeline file