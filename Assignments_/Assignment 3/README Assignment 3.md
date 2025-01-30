---
tags:
  - Marinus
  - Assignment
  - Notes
  - LLM
Created: 2024-12-13
DateReviewed1: 2025-01-24
---
# Introduction
Chatgpt summary taken from [[Intro_to_ML_Assignment_3_Proposal.pdf]]

### Summary for Assignment ReadMe File

This project focuses on fine-tuning YOLO v7, an object detection model, to classify items in images of household fridges or grocery cabinets. The primary objective is to adapt YOLO v7, currently trained on a diverse set of objects, to recognize common supermarket products.

#### Key Steps:

1. **Dataset Curation**: Utilize a combination of externally sourced and in-house annotated datasets, ensuring high-quality bounding box annotations aligned with best practices.
    
2. **Model Fine-Tuning**: Employ PyTorch for fine-tuning YOLO v7, drawing inspiration from existing projects and research.
    
3. **Evaluation Metrics**: Assess performance using:
    
    - **Intersection over Union (IoU)**: Measures overlap between predicted and ground truth bounding boxes.
        
    - **Average Precision (AP)**: Calculates the area under the precision-recall curve for class-wise evaluation.
        

This approach aims to optimize YOLO v7 for real-world applications in grocery inventory management, leveraging established methodologies and cutting-edge tools.

# Dataset Curation

## Tools

For curating the datasets I will be using [CVAT](https://www.cvat.ai)
Although, I went ahead and installed CVAT `locally` by following their online guide. 
- WSL 2
- Docker Compose
- Django backend
- I also got CUDA running for GPU accelerated tasks.
	- But I'm in the library, so maybe I shouldn't fire that up haha

Then how about a BIG FAT ✨NO✨

We used Roboflow, here, have a look at our dataset [here](https://universe.roboflow.com/endexspace/supermarket-items-yolov7/dataset/2)

Our dataset is a combination of two datasets that we found fitting for the task on hand. The first was made a group doing this exact same project two years ago (they trained YOLOv8) but we didn't find any report on their findings. The second is one about supermarket items we found off Kaggle.
Datasets:
1.  [Smart Refrigirator](https://universe.roboflow.com/northumbria-university-newcastle/smart-refrigerator-zryjr/dataset/2)
2. [Refrigerator Contents](https://www.kaggle.com/datasets/surendraallam/refrigerator-contents?resource=download)

I think I found the prefect dataset on Hugging Face:[UniDataPro/grocery-shelves](https://huggingface.co/datasets/UniDataPro/grocery-shelves/blob/main/README.md)
But you need to pay for it so no. 

> [!TODO] 
> Make a page describing how CVAT works.
> I spent wayyy too much time getting this running locally. what a hassle
> Now im going to spend some more time making it use GPU acceleration
> And half an hour later GPU acceleration works haha


# Model

Have a look at some general YOLOv7 information: [[YOLOv7]]

Yo, what about this? [YOLO11](https://learnopencv.com/yolo11/) Released in 2024, brand new and apparently super fast as well.
### Alternative options
I had some doubts about using YOLOv7 so here are some alternatives and general comparison information: [[Alternatives]]
# Fine-Tuning

For the fine-tuning part of the model I used [Google Colab](https://colab.research.google.com) 
To fine-tune the model, we ran it 

[Fine-tune YOLOv7](https://learnopencv.com/fine-tuning-yolov7-on-custom-dataset/)




