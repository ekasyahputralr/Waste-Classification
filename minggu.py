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
ROOT_DIR = os.path.abspath("/content/Waste-Classification")
DATASET_DIR = os.path.join(ROOT_DIR, "dataset2", "waste")
ANNOTATION_FILE = os.path.join(DATASET_DIR, "train", "_annotations.coco.json")
IMAGE_DIR = os.path.join(DATASET_DIR, "train")
LOGS_DIR = os.path.join(ROOT_DIR, "logs")

COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")
# Download COCO trained weights from Releases if needed
if not os.path.exists(COCO_MODEL_PATH):
    utils.download_trained_weights(COCO_MODEL_PATH)

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
        """Load dataset from COCO-format JSON"""
        with open(annotation_json) as f:
            coco_json = json.load(f)

        # Add classes
        for category in coco_json["categories"]:
            self.add_class("dataset", category["id"], category["name"])

        # Add images
        annotations = coco_json["annotations"]
        images = {image["id"]: image for image in coco_json["images"]}

        # Map image ID to its annotations
        image_id_to_annotations = {}
        for annotation in annotations:
            img_id = annotation["image_id"]
            if img_id not in image_id_to_annotations:
                image_id_to_annotations[img_id] = []
            image_id_to_annotations[img_id].append(annotation)

        # Add images to dataset
        for image_id, image in images.items():
            file_name = image["file_name"]
            width = image["width"]
            height = image["height"]
            anns = image_id_to_annotations.get(image_id, [])
            self.add_image(
                source="dataset",
                image_id=image_id,
                path=os.path.join(images_dir, file_name),
                width=width,
                height=height,
                annotations=anns
            )

    def load_mask(self, image_id):
        """Generate instance masks for an image."""
        image_info = self.image_info[image_id]
        if image_info["source"] != "dataset":
            return super().load_mask(image_id)

        annotations = image_info["annotations"]
        instance_masks = []
        class_ids = []

        for annotation in annotations:
            class_id = annotation["category_id"]
            mask = np.zeros((image_info["height"], image_info["width"]), dtype=np.uint8)

            if annotation["iscrowd"]:
                continue

            # COCO format: polygons are in segmentation key
            segs = annotation["segmentation"]
            for seg in segs:
                if len(seg) < 6:
                    continue
                poly = np.array(seg).reshape((-1, 2))
                rr, cc = polygon(poly[:, 1], poly[:, 0])
                rr = np.clip(rr, 0, mask.shape[0] - 1)
                cc = np.clip(cc, 0, mask.shape[1] - 1)
                mask[rr, cc] = 1

            instance_masks.append(mask)
            class_ids.append(class_id)

        if class_ids:
            mask = np.stack(instance_masks, axis=-1)
            class_ids = np.array(class_ids, dtype=np.int32)
            return mask, class_ids
        else:
            # No valid annotation
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
dataset_val.load_trash(
    os.path.join(DATASET_DIR, "valid", "_annotations.coco.json"),
    os.path.join(DATASET_DIR, "valid")
)
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
