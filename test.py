import os
import sys
import json
import numpy as np
import skimage.draw

# Tambahkan path ke Matterport Mask R-CNN
ROOT_DIR = os.path.abspath("D:/testmatterport/Mask-RCNN-TF2/kangaroo-transfer-learning")
sys.path.append(ROOT_DIR)

from mrcnn.config import Config
from mrcnn import model as modellib, utils

# Konfigurasi dataset sampah
class TrashConfig(Config):
    NAME = "trash"
    IMAGES_PER_GPU = 2
    NUM_CLASSES = 1 + 2  # Background + organik + anorganik
    STEPS_PER_EPOCH = 100
    DETECTION_MIN_CONFIDENCE = 0.9

class TrashDataset(utils.Dataset):
    def load_trash(self, dataset_dir, json_path):
        with open(json_path) as f:
            dataset = json.load(f)
        
        for category in dataset["categories"]:
            if category["id"] in [3, 4]:  # Hanya anorganik dan organik
                self.add_class("trash", category["id"], category["name"])
        
        for image in dataset["images"]:
            image_path = os.path.join(dataset_dir, image["file_name"])
            self.add_image(
                "trash",
                image_id=image["id"],
                path=image_path,
                width=image["width"],
                height=image["height"],
                annotations=[anno for anno in dataset["annotations"] if anno["image_id"] == image["id"]]
            )

    def load_mask(self, image_id):
        info = self.image_info[image_id]
        annotations = info["annotations"]
        masks = np.zeros([info["height"], info["width"], len(annotations)], dtype=np.uint8)
        class_ids = []
        
        for i, annotation in enumerate(annotations):
            segmentation = annotation["segmentation"]
            rr, cc = skimage.draw.polygon(segmentation[1::2], segmentation[0::2])
            masks[rr, cc, i] = 1
            class_ids.append(annotation["category_id"])
        
        return masks.astype(np.bool_), np.array(class_ids, dtype=np.int32)


# Load dataset
dataset_train = TrashDataset()
dataset_train.load_trash("D:/testmatterport/Mask-RCNN-TF2/datasetbaru/train", "D:/testmatterport/Mask-RCNN-TF2/datasetbaru/train/_annotations.coco.json")
dataset_train.prepare()

dataset_val = TrashDataset()
dataset_val.load_trash("D:/testmatterport/Mask-RCNN-TF2/datasetbaru/val", "D:/testmatterport/Mask-RCNN-TF2/datasetbaru/valid/_annotations.coco.json")
dataset_val.prepare()

# Buat model
config = TrashConfig()
model = modellib.MaskRCNN(mode="training", config=config, model_dir="logs")

# Load model COCO pre-trained sebagai base
model.load_weights("D:/testmatterport/Mask-RCNN-TF2/mask_rcnn_coco.h5", by_name=True, exclude=["mrcnn_class_logits", "mrcnn_bbox_fc", "mrcnn_bbox", "mrcnn_mask"])

# Train model
model.train(dataset_train, dataset_val,
            learning_rate=config.LEARNING_RATE,
            epochs=30,
            layers="heads")
