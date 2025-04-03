import os
import numpy as np
import json
import cv2
import tensorflow as tf
from mrcnn.config import Config
from mrcnn import model as modellib, utils

# Konfigurasi untuk dataset sampah
class TrashConfig(Config):
    NAME = "trash"
    NUM_CLASSES = 3 # Background + Organik + Anorganik
    IMAGES_PER_GPU = 1
    STEPS_PER_EPOCH = 100
    DETECTION_MIN_CONFIDENCE = 0.7
    
config = TrashConfig()

# Dataset Sampah
class TrashDataset(utils.Dataset):

    def __init__(self, coco_json_path):
        super().__init__()  # Pastikan memanggil konstruktor induk
        self.categories = self.load_categories(coco_json_path)

    def load_trash(self, dataset_dir, annotations_file):
        # Menambahkan kelas background dengan ID 0
        self.add_class("trash", 0, "background")
        self.add_class("trash", 1, "organik")
        self.add_class("trash", 2, "anorganik")

        with open(annotations_file) as f:
            data = json.load(f)

        images = {img["id"]: img for img in data["images"]}  # Mapping ID ke info gambar
        
        for ann in data["annotations"]:
            image_id = ann["image_id"]
            image_info = images[image_id]
            image_path = os.path.join(dataset_dir, image_info["file_name"])
            width = image_info["width"]
            height = image_info["height"]

            # Menentukan class ID
            category_id = ann["category_id"]
            class_id = category_id  # Sesuai dengan category_id di JSON COCO

            # Mengambil koordinat bbox dan segmentasi
            if "segmentation" in ann and ann["segmentation"]:
                polygons = ann["segmentation"]
            else:
                x, y, w, h = ann["bbox"]
                polygons = [[
                    [x, y], [x + w, y], [x + w, y + h], [x, y + h]
                ]]  # Default bounding box jika tidak ada segmentasi

            self.add_image(
                "trash", image_id=image_id, path=image_path,
                width=width, height=height,
                polygons=polygons, class_ids=[class_id]
            )

    def load_categories(self, coco_json_path):
        # Load the categories from the COCO JSON file
        with open(coco_json_path, 'r') as f:
            coco_data = json.load(f)
        return coco_data["categories"]        
  

    def load_mask(self, image_id):
        info = self.image_info[image_id]
        height, width = info["height"], info["width"]
        
        masks = np.zeros((height, width, len(info["polygons"])), dtype=np.uint8)
        class_ids = []

        for i, polygon in enumerate(info["polygons"]):
            pts = np.array(polygon, np.int32).reshape((-1, 1, 2))  
            cv2.fillPoly(masks[:, :, i], [pts], 1)
            
            # Get the class ID directly from the image info
            class_id = info["class_ids"][i]  # This should be the class ID for the current polygon
            class_ids.append(class_id)

        return masks.astype(np.bool8), np.array(class_ids, dtype=np.int32)




# Load dataset
# Load dataset
dataset_train = TrashDataset(coco_json_path="D:/testmatterport/Mask-RCNN-TF2/datasetbaru/train/_annotations.coco.json")
dataset_train.load_trash("D:/testmatterport/Mask-RCNN-TF2/datasetbaru/train", "D:/testmatterport/Mask-RCNN-TF2/datasetbaru/train/_annotations.coco.json")
dataset_train.prepare()

dataset_val = TrashDataset(coco_json_path="D:/testmatterport/Mask-RCNN-TF2/datasetbaru/valid/_annotations.coco.json")
dataset_val.load_trash("D:/testmatterport/Mask-RCNN-TF2/datasetbaru/valid", "D:/testmatterport/Mask-RCNN-TF2/datasetbaru/valid/_annotations.coco.json")
dataset_val.prepare()

# Buat model Mask R-CNN
model = modellib.MaskRCNN(mode="training", config=config, model_dir="logs")

# Load COCO weights dan mulai training
model.load_weights("mask_rcnn_coco.h5", by_name=True, exclude=["mrcnn_class_logits", "mrcnn_bbox_fc", "mrcnn_bbox", "mrcnn_mask"])

model.train(dataset_train, dataset_val, 
            learning_rate=config.LEARNING_RATE, 
            epochs=30, 
            layers='heads')

# Simpan model
model_path = os.path.join("logs", "mask_rcnn_trash.h5")
model.keras_model.save_weights(model_path)