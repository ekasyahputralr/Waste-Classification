import os
import sys
import json
import numpy as np
import skimage.io
import skimage.draw
import matplotlib.pyplot as plt
from mrcnn.config import Config
from mrcnn import model as modellib
from mrcnn import visualize
from mrcnn import utils
from pycocotools.coco import COCO
from pycocotools import mask as maskUtils
import tensorflow as tf

# Root directory of the project
ROOT_DIR = os.path.abspath("D:/testmatterport/Mask-RCNN-TF2")

# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs")

# Directory to the dataset
DATASET_DIR = os.path.join(ROOT_DIR, "dataset")

# Local path to trained weights file
COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")

# Download COCO trained weights if not present
if not os.path.exists(COCO_MODEL_PATH):
    utils.download_trained_weights(COCO_MODEL_PATH)

class TrashConfig(Config):
    """Configuration for training on the trash dataset.
    Derives from the base Config class and overrides specific values.
    """
    # Give the configuration a recognizable name
    NAME = "trash"

    # Train on 1 GPU and 2 images per GPU
    GPU_COUNT = 1
    IMAGES_PER_GPU = 2

    # Number of classes (including background)
    # Background + Anorganik_mask + Organik_mask + anorganik + organik
    NUM_CLASSES = 1 + 4  

    # Number of training steps per epoch
    STEPS_PER_EPOCH = 100

    # Skip detections with < 90% confidence
    DETECTION_MIN_CONFIDENCE = 0.9
    
    # Set batch size and other parameters as needed
    LEARNING_RATE = 0.001
    BACKBONE = "resnet50"

    
    # Use smaller anchors because trash objects are often smaller
    RPN_ANCHOR_SCALES = (8, 16, 32, 64, 128)


class TrashDataset(utils.Dataset):
    def load_trash(self, dataset_dir, subset):
        """Load a subset of the Trash dataset.
        dataset_dir: Root directory of the dataset.
        subset: Subset to load: train or val
        """
        # Add classes. We have background + 4 trash classes
        self.add_class("trash", 1, "Anorganik_mask")
        self.add_class("trash", 2, "Organik_mask")
        self.add_class("trash", 3, "anorganik")
        self.add_class("trash", 4, "organik")
        
        # Path to the annotations JSON file
        annotations_path = os.path.join(dataset_dir, f"{subset}/annotations.json")
        
        # Load the annotations file
        with open(annotations_path, 'r') as f:
            annotations = json.load(f)
        
        # Create a mapping from image_id to image file path
        image_paths = {}
        for image in annotations['images']:
            image_paths[image['id']] = os.path.join(
                dataset_dir, subset, image['file_name'])
        
        # Add images and annotations
        for annotation in annotations['annotations']:
            image_id = annotation['image_id']
            
            # Skip if we don't have the image
            if image_id not in image_paths:
                continue
                
            # Get the path to the image file
            image_path = image_paths[image_id]
            
            # Get the width and height
            for img in annotations['images']:
                if img['id'] == image_id:
                    width = img['width']
                    height = img['height']
                    break
            
            # Add the image
            self.add_image(
                "trash",
                image_id=image_id,
                path=image_path,
                width=width,
                height=height)
    
    def load_mask(self, image_id):
        """Generate instance masks for an image.
        Returns:
        masks: A bool array of shape [height, width, instance count] with
            one mask per instance.
        class_ids: a 1D array of class IDs of the instance masks.
        """
        # Get the image info
        info = self.image_info[image_id]
        
        # Path to the annotations JSON file
        annotations_path = os.path.join(os.path.dirname(os.path.dirname(info["path"])), 
                                        "annotations.json")
        
        # Load the annotations file
        with open(annotations_path, 'r') as f:
            annotations = json.load(f)
        
        # Find all annotations for this image
        annotations_for_image = []
        for annotation in annotations['annotations']:
            if annotation['image_id'] == info['id']:
                annotations_for_image.append(annotation)
        
        # If there are no annotations, return empty arrays
        if not annotations_for_image:
            return np.zeros([info["height"], info["width"], 0], dtype=np.bool), np.zeros([0], dtype=np.int32)
        
        # Initialize masks array
        masks = np.zeros([info["height"], info["width"], len(annotations_for_image)], dtype=np.bool)
        
        # Initialize class_ids array
        class_ids = np.zeros([len(annotations_for_image)], dtype=np.int32)
        
        # Generate masks
        for i, annotation in enumerate(annotations_for_image):
            # Get class ID
            class_id = annotation['category_id']
            
            # Get segmentation
            segmentation = annotation['segmentation']
            
            # Create binary mask
            mask = np.zeros([info["height"], info["width"]], dtype=np.bool)
            
            # Handle different segmentation formats
            if isinstance(segmentation, list):
                # Polygon
                for polygon in segmentation:
                    # Polygon is a list of points [x1, y1, x2, y2, ...]
                    # Convert to arrays of x and y coordinates
                    rr, cc = skimage.draw.polygon(
                        np.asarray(polygon[1::2]),  # y coordinates
                        np.asarray(polygon[0::2])   # x coordinates
                    )
                    
                    # Make sure the polygon points are within the image bounds
                    rr = np.clip(rr, 0, info["height"] - 1)
                    cc = np.clip(cc, 0, info["width"] - 1)
                    
                    # Set the mask to True for these pixels
                    mask[rr, cc] = True
            elif isinstance(segmentation, dict):
                # COCO RLE
                binary_mask = maskUtils.decode(segmentation)
                mask = binary_mask.astype(np.bool)
            
            # Add mask to the array
            masks[:, :, i] = mask
            
            # Add class ID
            class_ids[i] = class_id
        
        return masks, class_ids

    def image_reference(self, image_id):
        """Return the path of the image."""
        info = self.image_info[image_id]
        return info["path"]


def train_model():
    config = TrashConfig()
    config.display()
    
    # Create model in training mode
    model = modellib.MaskRCNN(mode="training", config=config, model_dir=MODEL_DIR)
    
    # Which weights to start with?
    init_with = "coco"  # Options: "imagenet", "coco", "last", or "path/to/weights"
    
    if init_with == "imagenet":
        model.load_weights(model.get_imagenet_weights(), by_name=True)
    elif init_with == "coco":
        # Load weights trained on MS COCO, but skip layers that
        # are different due to the different number of classes
        model.load_weights(COCO_MODEL_PATH, by_name=True,
                         exclude=["mrcnn_class_logits", "mrcnn_bbox_fc", 
                                  "mrcnn_bbox", "mrcnn_mask"])
    elif init_with == "last":
        # Load the last model you trained and continue training
        checkpoint_file = model.find_last()
        model.load_weights(checkpoint_file, by_name=True)
    
    # Training dataset
    dataset_train = TrashDataset()
    dataset_train.load_trash(DATASET_DIR, "train")
    dataset_train.prepare()
    
    # Validation dataset
    dataset_val = TrashDataset()
    dataset_val.load_trash(DATASET_DIR, "val")
    dataset_val.prepare()
    
    # Train the head layers
    print("Training network heads")
    model.train(dataset_train, dataset_val,
              learning_rate=config.LEARNING_RATE,
              epochs=30,
              layers='heads')
    
    # Fine tune all layers
    print("Fine-tuning all layers")
    model.train(dataset_train, dataset_val,
              learning_rate=config.LEARNING_RATE / 10,
              epochs=100,
              layers="all")
    
    return model

def prepare_dataset(annotations_file, output_dir):
    """
    Prepares the dataset by splitting it into train and validation sets
    
    Args:
        annotations_file: Path to the COCO format annotations JSON file
        output_dir: Directory to save the prepared dataset
    """
    # Create output directories
    train_dir = os.path.join(output_dir, "train")
    val_dir = os.path.join(output_dir, "val")
    
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)
    
    # Load annotations
    with open(annotations_file, 'r') as f:
        annotations = json.load(f)
    
    # Get all image IDs
    image_ids = [img["id"] for img in annotations["images"]]
    
    # Shuffle and split into train and validation sets (80% train, 20% validation)
    np.random.shuffle(image_ids)
    split_idx = int(len(image_ids) * 0.8)
    train_image_ids = image_ids[:split_idx]
    val_image_ids = image_ids[split_idx:]
    
    # Create train and validation annotation files
    train_annotations = {
        "info": annotations["info"],
        "licenses": annotations["licenses"],
        "categories": annotations["categories"],
        "images": [],
        "annotations": []
    }
    
    val_annotations = {
        "info": annotations["info"],
        "licenses": annotations["licenses"],
        "categories": annotations["categories"],
        "images": [],
        "annotations": []
    }
    
    # Split images
    for img in annotations["images"]:
        img_dir = os.path.dirname(os.path.dirname(annotations_file))
        src_path = os.path.join(img_dir, img["file_name"])
        
        if img["id"] in train_image_ids:
            dst_path = os.path.join(train_dir, img["file_name"])
            train_annotations["images"].append(img)
        else:
            dst_path = os.path.join(val_dir, img["file_name"])
            val_annotations["images"].append(img)
        
        # Copy the image file
        if os.path.exists(src_path):
            import shutil
            os.makedirs(os.path.dirname(dst_path), exist_ok=True)
            shutil.copy(src_path, dst_path)
    
    # Split annotations
    for ann in annotations["annotations"]:
        if ann["image_id"] in train_image_ids:
            train_annotations["annotations"].append(ann)
        else:
            val_annotations["annotations"].append(ann)
    
    # Save the split annotation files
    with open(os.path.join(train_dir, "annotations.json"), 'w') as f:
        json.dump(train_annotations, f)
    
    with open(os.path.join(val_dir, "annotations.json"), 'w') as f:
        json.dump(val_annotations, f)
    
    print(f"Dataset prepared: {len(train_image_ids)} training images and {len(val_image_ids)} validation images")

def detect_and_visualize(model, image_path, output_folder=None):
    """Run detection on an image and visualize the results
    
    Args:
        model: The trained Mask R-CNN model
        image_path: Path to the image to detect
        output_folder: Folder to save the visualization
    """
    # Create the output folder if it doesn't exist
    if output_folder:
        os.makedirs(output_folder, exist_ok=True)
    
    # Read the image
    image = skimage.io.imread(image_path)
    
    # Detect objects
    r = model.detect([image], verbose=1)[0]
    
    # Map class IDs to class names
    class_names = ['BG', 'Anorganik_mask', 'Organik_mask', 'anorganik', 'organik']
    
    # Visualize results
    plt.figure(figsize=(12, 12))
    visualize.display_instances(
        image, r['rois'], r['masks'], r['class_ids'], 
        class_names, r['scores']
    )
    
    # Save the visualization if an output folder is specified
    if output_folder:
        plt.savefig(os.path.join(output_folder, os.path.basename(image_path)))
    
    plt.show()
    
    return r

def save_model(model, model_path):
    """Save the trained model"""
    model.keras_model.save_weights(model_path)
    print(f"Model saved to {model_path}")

def main():
    # Prepare the dataset
    annotations_file = os.path.join(ROOT_DIR, 'datasets', 'waste', 'train', "_annotations.coco.json")
    prepare_dataset(annotations_file, DATASET_DIR)
    
    # Train the model
    model = train_model()
    
    # Save the trained model
    model_path = os.path.join(ROOT_DIR, "trash_maskrcnn_model.h5")
    save_model(model, model_path)
    
    # Switch to inference mode
    inference_config = TrashConfig()
    inference_config.BATCH_SIZE = 1
    inference_config.IMAGES_PER_GPU = 1
    inference_config.GPU_COUNT = 1
    
    model_inference = modellib.MaskRCNN(
        mode="inference", config=inference_config, model_dir=MODEL_DIR
    )
    model_inference.load_weights(model_path, by_name=True)
    
    # Test the model on a sample image (can be modified to process multiple images)
    sample_image = os.path.join(DATASET_DIR, "val", "botol_minum-7-_jpg.rf.10c25e960b646ca7bd518042a611be07.jpg")
    if os.path.exists(sample_image):
        results = detect_and_visualize(model_inference, sample_image, "output_images")
        
        # Print out the detection results
        class_names = ['BG', 'Anorganik_mask', 'Organik_mask', 'anorganik', 'organik']
        for i, class_id in enumerate(results['class_ids']):
            print(f"Detected {class_names[class_id]} with confidence {results['scores'][i]:.3f}")
            y1, x1, y2, x2 = results['rois'][i]
            print(f"Bounding box: ({x1}, {y1}), ({x2}, {y2})")
    else:
        print(f"Sample image not found: {sample_image}")
        print("Please modify the sample image path or run inference on your own images.")

if __name__ == "__main__":
    main()