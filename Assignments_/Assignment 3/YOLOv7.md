---
tags:
  - Marinus
  - _FirstPass
  - Assignment
  - Notes
Created: 2025-01-24
---
> [!read] 
> The research paper on YOLOv7: 
> [[YOLOv7_Trainable_Bag-of-Freebies_Sets_New_State-of-the-Art_for_Real-Time_Object_Detectors.pdf]]
> 
> LearnOpenCV review and explanation of the model:
> [YOLOv7 Object Detection Paper Explanation & Inference](https://learnopencv.com/yolov7-object-detection-paper-explanation-and-inference/)

Yolo has three parts:
- Backbone
	- Extracts essential features
	- feeds to neck
- Neck 
	- Uses extracted features
	- makes feature pyramids
- Head
	- output layers with final detection mechanisms
![[Pasted image 20250122153447.png]]