import os
import sys
import json
import numpy as np
import tensorflow as tf
from mrcnn.config import Config
from mrcnn import model as modellib
from dataset_converter import SampahDataset

# Konfigurasi Model
class SampahConfig(Config):
    NAME = "sampah"
    NUM_CLASSES = 1 + 2  # Background + sampah_organik + sampah_anorganik
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    STEPS_PER_EPOCH = 100
    DETECTION_MIN_CONFIDENCE = 0.9
    IMAGE_MIN_DIM = 512
    IMAGE_MAX_DIM = 512

# Load Dataset
train_dataset = SampahDataset()
train_dataset.load_dataset(json_path="D:/testmatterport/Mask-RCNN-TF2/datasetbaru/train/_annotations.coco.json", images_dir="D:/testmatterport/Mask-RCNN-TF2/datasetbaru/train")
train_dataset.prepare()

val_dataset = SampahDataset()
val_dataset.load_dataset(json_path="D:/testmatterport/Mask-RCNN-TF2/datasetbaru/valid/_annotations.coco.json", images_dir="D:/testmatterport/Mask-RCNN-TF2/datasetbaru/valid")
val_dataset.prepare()

# Inisialisasi Model
model = modellib.MaskRCNN(mode="training", config=SampahConfig(), model_dir="logs/")
model.load_weights("mask_rcnn_coco.h5", by_name=True, exclude=["mrcnn_class_logits", "mrcnn_bbox_fc", "mrcnn_bbox", "mrcnn_mask"])

# Training Model
model.train(train_dataset, val_dataset, learning_rate=SampahConfig().LEARNING_RATE, epochs=10, layers="heads")
