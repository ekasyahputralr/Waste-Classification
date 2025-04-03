import json
import os
import numpy as np
import skimage.draw
from mrcnn.utils import Dataset

class SampahDataset(Dataset):
    def load_dataset(self, json_path, images_dir):
        with open(json_path, "r") as f:
            data = json.load(f)

        self.add_class("dataset", 1, "sampah_organik")
        self.add_class("dataset", 2, "sampah_anorganik")

        for image in data["images"]:
            image_id = image["id"]
            image_path = os.path.join(images_dir, image["file_name"])  # Ganti "key" ke "file_name"

            self.add_image(
                "dataset",
                image_id=image_id,
                path=image_path,
                width=image["width"],
                height=image["height"],
                annotations=[ann for ann in data["annotations"] if ann["image_id"] == image_id]
            )
            image_id += 1

    def load_mask(self, image_id):
        info = self.image_info[image_id]
        annotations = info["annotations"]
        masks = []
        class_ids = []

        for annot in annotations:
            if "type" in annot and annot["type"] == "polygon":
                poly = np.array(annot["points"])
                mask = np.zeros((info["height"], info["width"]), dtype=np.uint8)
                rr, cc = skimage.draw.polygon(poly[:, 1], poly[:, 0])
                mask[rr, cc] = 1
                masks.append(mask)
                class_ids.append(2 if "anorganik" in annot["label"].lower() else 1)

        masks = np.stack(masks, axis=-1) if masks else np.zeros((info["height"], info["width"], 0), dtype=np.uint8)
        return masks, np.array(class_ids, dtype=np.int32)
 