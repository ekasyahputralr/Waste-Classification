import os
import sys
import json
import numpy as np
import skimage.io
from skimage.draw import polygon
from mrcnn.config import Config
from mrcnn import model as modellib, utils
from mrcnn.model import log
from pycocotools.coco import COCO

# === Path Config ===
ROOT_DIR = os.path.abspath("D:/testmatterport/Mask-RCNN-TF2")
DATASET_DIR = os.path.join(ROOT_DIR, "dataset2", "waste")
ANNOTATION_FILE = os.path.join(DATASET_DIR, "train", "_annotations.coco.json")
IMAGE_DIR = os.path.join(DATASET_DIR, "train")
LOGS_DIR = os.path.join(ROOT_DIR, "logs")

# Cek image file jika diperlukan
# Contoh hanya jika ingin ngecek ada/tidak
# img_path = os.path.join(IMAGE_DIR, "some_image.jpg")
# if not os.path.exists(img_path):
#     print(f"Image file not found: {img_path}")

# === Config for Training ===
class TrashConfig(Config):
    NAME = "trash_segmentation"
    NUM_CLASSES = 1 + 5  # background + 5 class sampah
    GPU_COUNT = 1
    IMAGES_PER_GPU = 2
    STEPS_PER_EPOCH = 100
    DETECTION_MIN_CONFIDENCE = 0.7
    IMAGE_MIN_DIM = 512
    IMAGE_MAX_DIM = 512

config = TrashConfig()

# === Custom Dataset Loader ===
class TrashDataset(utils.Dataset):
    def load_trash(self, annotation_json, images_dir):
        coco = COCO(annotation_json)
        class_ids = sorted(coco.getCatIds())
        image_ids = list(coco.imgs.keys())

        # Register classes
        for cid in class_ids:
            cat = coco.loadCats(cid)[0]
            self.add_class("trash", cid, cat["name"])

        # Register images
        for image_id in image_ids:
            image_info = coco.loadImgs(image_id)[0]
            annotations = coco.loadAnns(coco.getAnnIds(imgIds=image_id, catIds=class_ids, iscrowd=None))

            self.add_image(
                source="trash",
                image_id=image_id,
                path=os.path.join(images_dir, image_info["file_name"]),
                width=image_info["width"],
                height=image_info["height"],
                annotations=annotations
            )

        def load_mask(self, image_id):
            image_info = self.image_info[image_id]
            annotations = image_info["annotations"]
            instance_masks = []
            class_ids = []

            for annotation in annotations:
                class_id = annotation["category_id"]
                if class_id:
                    m = np.zeros((image_info["height"], image_info["width"]), dtype=np.uint8)
                    for seg in annotation["segmentation"]:
                        # seg = list of x, y alternating
                        rr, cc = polygon(np.array(seg[1::2]), np.array(seg[0::2]))
                        m[rr, cc] = 1
                    instance_masks.append(m)
                    class_ids.append(class_id)

            if class_ids:
                mask = np.stack(instance_masks, axis=-1)
                class_ids = np.array(class_ids, dtype=np.int32)
                return mask, class_ids
            else:
                return super().load_mask(image_id)

    def image_reference(self, image_id):
        info = self.image_info[image_id]
        if info["source"] == "trash":
            return info["path"]
        else:
            return super(self.__class__, self).image_reference(image_id)
# === Load Dataset ===
dataset_train = TrashDataset()
dataset_train.load_trash(ANNOTATION_FILE, IMAGE_DIR)
dataset_train.prepare()

dataset_val = TrashDataset()
dataset_val.load_trash(ANNOTATION_FILE, IMAGE_DIR)  # Bisa diganti jika punya val sendiri
dataset_val.prepare()

# === Build Model ===
model = modellib.MaskRCNN(mode="training", config=config, model_dir=LOGS_DIR)

# Load COCO weights (pretrained)
COCO_WEIGHTS_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")
model.load_weights(COCO_WEIGHTS_PATH, by_name=True,
                   exclude=["mrcnn_class_logits", "mrcnn_bbox_fc",
                            "mrcnn_bbox", "mrcnn_mask"])

# === Train the model ===
model.train(dataset_train, dataset_val,
            learning_rate=config.LEARNING_RATE,
            epochs=30,
            layers='heads')
