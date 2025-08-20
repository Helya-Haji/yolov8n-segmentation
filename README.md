# Car Segmentation with YOLOv8 

This repository contains code for training and using YOLOv8 segmentation to segment cars in images. The model is trained on a custom dataset and is able to generate segmentation masks that highlight the car region.

It has two types of outputs:

* Saving model’s raw output (blue polygon mask + bounding box, similar to YOLO’s default visualizations) ----> saving models output.py

* Saving results on a white background (the segmented car cropped and placed on a plain white canvas) ----> saving results.py

original image:

![pexels-alexgtacar-745150-1592384_jpg rf 18c8f8199a00c3035d6f2f8f636c7f96](https://github.com/user-attachments/assets/07fe9bb6-7140-423e-acd4-5a3cff4e829a)

model’s raw output:

![pexels-alexgtacar-745150-1592384_jpg rf 18c8f8199a00c3035d6f2f8f636c7f96](https://github.com/user-attachments/assets/17852473-1126-4803-88e7-a7caa8e599b1)

results on a white background:

![pexels-alexgtacar-745150-1592384_jpg rf 18c8f8199a00c3035d6f2f8f636c7f96](https://github.com/user-attachments/assets/f1652e4b-d8ea-45f3-9eb0-f10a29c26b63)
