# SW-NIDS-Net

This is a try on improving NIDS architecture, as the paper mentions the model has very low AP values for the small objects in the images, so after some research we found that in many cases the gDino is not able to pick up the small objects from the image, hence in this method we divided the image into windows using the sliding window approach and then passed the windows through NIDS, which retured the detected objects.

## Comparision on Results:
1. Image 002 from HighRes Dataset
["res_002"]("https://github.com/divyam-prajapati/NIDS-Net/blob/main/results/Screenshot%202024-09-18%20191120.png?raw=true")
2. Image 039 from HighRes Dataset
["res_039"]("https://github.com/divyam-prajapati/NIDS-Net/blob/main/results/Screenshot%202024-09-18%20191154.png?raw=true")



- `utils/sliding_window.py`: contains the code for creating windows from images/ frames.
- `utils/changes_fn.py`: has some of the functions that are used in NIDs but have been tweaked a little bit.
- `2_sliding_high_res_inference.ipynb`: Running NIDS over every window in a image. (Single Image Test)
- `2_sliding_single_image_test`: Running NIDS over every window in the image. (Whole Dataset Test, currently for High Res Dataset)

