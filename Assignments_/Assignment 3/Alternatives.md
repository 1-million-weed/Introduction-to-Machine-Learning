---
tags:
  - Marinus
  - _FirstPass
  - Assignment
  - Notes
Created: 2025-01-24
---


YOLOv7 is optimised for real-time object recognition. A study on the performance of this model in real world agents can be found here: [[Real-time_Object_Detection_Performance_Analysis_Using_YOLOv7_on_Edge_Devices.pdf]].

This article from [HiTech](https://www.hitechbpo.com/blog/top-object-detection-models.php?utm_source=chatgpt.com) about `Top Object Detection Models in 2025` describes some of the following:
- Use Fast R-CNN to generate region proposals to refine object locations
- YOLO classify and localize objects in one pass
- **Anchor Boxes:** Provide reference points of varied shapes and sizes for  detection.
- **Feature Pyramid Network (FPN):** Reconstructs high-resolution layers, allowing detection across multiple scales.
- **Data Augmentation:** Increases the size of training datasets by artificially changing images.
	- I need to write more about this later.
	- I want to find a way to use the current ai models to expand and diversify my datasets
- **Combining Datasets**: We'll want to combine a couple different datasets (that of course were scraped by us to ensure quality)  to have a more diverse range of data.
	- more isnt often better but more good should result in a better model. 
	- remember, garbage in is garbage out
- Another issue is the fact that images will only have a couple of main objects and the rest of the image is background, we need to somehow shift the focus of the detection models to only the objects we want to detect and not the background.
	- **Focal Loss:** Reduces the impact of class imbalance by diminishing the loss for well-classified examples.
	- **Hard Negative Mining:** Selects a subset of hard-to-detect negative examples to ensure the majority class did not n’t overwhelm the model.
- A decent alternative model that we can consider is the Detectron2
	- It features a vast and comprehesive set of innovative algorithms. its modular and and can be tailaired to specific needs.
- If we wanted to make the models run locally on user devices (like possibly making more use of the new apple hardware specks)
	- EfficientDet might be the way to go for that
- 