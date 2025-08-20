# Car Segmentation with YOLOv8 

This repository contains code for training and using YOLOv8 segmentation to segment whole cars in images. The model is trained on a custom dataset and is able to generate segmentation masks that highlight the car region.

It has two types of outputs:

* Saving model’s raw output (blue polygon mask + bounding box, similar to YOLO’s default visualizations) ----> saving models output.py

* Saving results on a white background (the segmented car cropped and placed on a plain white canvas) ----> saving results.py
