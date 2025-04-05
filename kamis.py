import os
import sys
import json
import numpy as np
import skimage.draw
import skimage.io
import matplotlib.pyplot as plt
from mrcnn.config import Config
from mrcnn import model as modellib, utils
from mrcnn import visualize

# Root directory of the project
ROOT_DIR = os.path.abspath("D:/testmatterport/Mask-RCNN-TF2")

# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs")

# Local path to trained weights file
COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")

# Download COCO trained weights if needed
if not os.path.exists(COCO_MODEL_PATH):
    utils.download_trained_weights(COCO_MODEL_PATH)

class WasteConfig(Config):
    """Configuration for training on the waste dataset."""
    # Give the configuration a recognizable name
    NAME = "waste"

    # Train on 1 GPU and 2 images per GPU. Batch size is 2 (GPUs * images/GPU).
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

    # Number of classes (including background)
    # Background + Anorganik_mask + Organik_mask + anorganik + organik
    NUM_CLASSES = 1 + 4  

    # Number of training steps per epoch
    STEPS_PER_EPOCH = 100

    # Skip detections with < 90% confidence
    DETECTION_MIN_CONFIDENCE = 0.9

    # Input image resizing
    IMAGE_RESIZE_MODE = "square"
    IMAGE_MIN_DIM = 640
    IMAGE_MAX_DIM = 640
    BACKBONE = "resnet50"

class WasteDataset(utils.Dataset):
    def load_waste(self, dataset_dir, subset):
        """Load a subset of the waste dataset.
        dataset_dir: Root directory of the dataset.
        subset: Subset to load: train or val
        """
        # Add classes - make sure the class IDs match the ones in the annotation file
        self.add_class("waste", 1, "Anorganik_mask")
        self.add_class("waste", 2, "Organik_mask")
        self.add_class("waste", 3, "anorganik")
        self.add_class("waste", 4, "organik")

        # Train or validation dataset?
        assert subset in ["train", "val"]
        dataset_dir = os.path.join(dataset_dir, subset)
        
        # Load annotations
        annotations_path = os.path.join(dataset_dir, "_annotations.coco.json")
        with open(annotations_path, 'r') as f:
            annotations = json.load(f)
        
        # Create image_id to annotation mapping for faster access
        image_annotations = {}
        for annotation in annotations['annotations']:
            image_id = annotation['image_id']
            if image_id not in image_annotations:
                image_annotations[image_id] = []
            image_annotations[image_id].append(annotation)
        
        # Create image_id to image info mapping
        images_info = {}
        for image_info in annotations['images']:
            images_info[image_info['id']] = image_info
        
        # Add images
        for image_id, image_info in images_info.items():
            # Skip images without annotations
            if image_id not in image_annotations:
                continue
                
            # Get the image path
            image_path = os.path.join(dataset_dir, image_info['file_name'])
            
            # Add the image
            self.add_image(
                "waste",
                image_id=image_id,
                path=image_path,
                width=image_info['width'],
                height=image_info['height'],
                annotations=image_annotations[image_id]
            )

    def load_mask(self, image_id):
        """Generate instance masks for an image.
        Returns:
        masks: A bool array of shape [height, width, instance count] with
            one mask per instance.
        class_ids: a 1D array of class IDs of the instance masks.
        """
        # Get the image info
        image_info = self.image_info[image_id]
        annotations = image_info['annotations']
        
        # Get image dimensions
        height = image_info['height']
        width = image_info['width']
        
        # Initialize masks array and class_ids list
        masks = np.zeros([height, width, len(annotations)], dtype=np.uint8)
        class_ids = []
        
        # Loop through annotations
        for i, annotation in enumerate(annotations):
            # Get class ID
            class_id = annotation['category_id']
            class_ids.append(class_id)
            
            # Get segmentation
            segmentation = annotation.get('segmentation', [])
            if len(segmentation) > 0:
                # Handle different segmentation formats
                if isinstance(segmentation[0], list):  # Polygon format
                    # Convert to array and reshape
                    poly = np.array(segmentation[0]).reshape(-1, 2)
                    rr, cc = skimage.draw.polygon(poly[:, 1], poly[:, 0])
                    
                    # Clip to image boundaries
                    rr = np.clip(rr, 0, height - 1)
                    cc = np.clip(cc, 0, width - 1)
                    
                    # Set mask
                    masks[rr, cc, i] = 1
                else:  # RLE format
                    masks[:, :, i] = utils.unmold_mask(segmentation, height, width)
            else:
                # If no segmentation, use bounding box
                bbox = annotation['bbox']  # [x, y, width, height]
                x, y, w, h = [int(coord) for coord in bbox]
                x2, y2 = x + w, y + h
                
                # Clip to image boundaries
                x = max(0, x)
                y = max(0, y)
                x2 = min(width, x2)
                y2 = min(height, y2)
                
                # Set mask using bounding box
                masks[y:y2, x:x2, i] = 1
        
        # Ensure class_ids is a numpy array
        class_ids = np.array(class_ids, dtype=np.int32)
        
        return masks, class_ids

    def image_reference(self, image_id):
        """Return the path of the image."""
        info = self.image_info[image_id]
        return info["path"]


def train_model():
    # Configuration
    config = WasteConfig()
    config.display()
    
    # Create model in training mode
    model = modellib.MaskRCNN(mode="training", config=config, model_dir=MODEL_DIR)
    
    # Which weights to start with?
    init_with = "coco"  # Options: "coco", "last", "imagenet", or ""
    
    if init_with == "coco":
        # Load weights trained on MS COCO
        model.load_weights(COCO_MODEL_PATH, by_name=True,
                           exclude=["mrcnn_class_logits", "mrcnn_bbox_fc", 
                                    "mrcnn_bbox", "mrcnn_mask"])
    elif init_with == "last":
        # Load the last model you trained
        model.load_weights(model.find_last(), by_name=True)
    elif init_with == "imagenet":
        # Start from ImageNet trained weights
        model.load_weights(model.get_imagenet_weights(), by_name=True)
    
    # Dataset paths
    dataset_dir = os.path.join(ROOT_DIR, "datasets", "waste")
    
    # Training dataset
    dataset_train = WasteDataset()
    dataset_train.load_waste(dataset_dir, "train")
    dataset_train.prepare()
    
    # Validation dataset
    dataset_val = WasteDataset()
    dataset_val.load_waste(dataset_dir, "val")
    dataset_val.prepare()
    
    # Training - Stage 1
    # Train the head branches
    print("Training network heads")
    model.train(dataset_train, dataset_val,
                learning_rate=config.LEARNING_RATE,
                epochs=20,
                layers='heads')
    
    # Training - Stage 2
    # Finetune layers from ResNet stage 4 and up
    print("Fine-tuning network stage 4 and up")
    model.train(dataset_train, dataset_val,
                learning_rate=config.LEARNING_RATE / 10,
                epochs=40,
                layers='4+')
    
    # Training - Stage 3
    # Fine-tune all layers
    print("Fine-tuning all network layers")
    model.train(dataset_train, dataset_val,
                learning_rate=config.LEARNING_RATE / 100,
                epochs=60,
                layers='all')
    
    # Save weights
    model_path = os.path.join(MODEL_DIR, "waste_final.h5")
    model.keras_model.save_weights(model_path)
    print(f"Model weights saved to {model_path}")
    
    return model


class InferenceConfig(WasteConfig):
    # Set batch size to 1 since we'll be running inference on one image at a time
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1


def detect_and_visualize(model, image_path):
    # Load image
    image = skimage.io.imread(image_path)
    
    # Run detection
    results = model.detect([image], verbose=1)
    r = results[0]
    
    # Visualize results
    visualize.display_instances(
        image, r['rois'], r['masks'], r['class_ids'], 
        ['BG', 'Anorganik_mask', 'Organik_mask', 'anorganik', 'organik'], r['scores'],
        title="Waste Detection Results",
        figsize=(8, 8))
    
    return r


def prepare_dataset():
    """Function to prepare the dataset structure for training"""
    # Create necessary directories
    os.makedirs(os.path.join(ROOT_DIR, "dataset2", "waste", "train"), exist_ok=True)
    os.makedirs(os.path.join(ROOT_DIR, "dataset2", "waste", "val"), exist_ok=True)
    
    print("Dataset directories created. Please place your images and annotations as follows:")
    print("- Images for training: datasets/waste/train/")
    print("- Images for validation: datasets/waste/val/")
    print("- Annotations for training: datasets/waste/train/annotations.json")
    print("- Annotations for validation: datasets/waste/val/annotations.json")
    
    print("\nMake sure your annotations follow the COCO format as shown in your example.")
    print("You may need to split your dataset into training and validation sets.")


def process_annotations_from_roboflow(annotations_file, output_dir, split_ratio=0.8):
    """Process annotations from Roboflow format and split into train/val sets"""
    with open(annotations_file, 'r') as f:
        data = json.load(f)
    
    # Create train/val split
    image_ids = [img['id'] for img in data['images']]
    np.random.shuffle(image_ids)
    split_idx = int(len(image_ids) * split_ratio)
    train_ids = set(image_ids[:split_idx])
    val_ids = set(image_ids[split_idx:])
    
    # Create train annotations
    train_data = {
        'info': data['info'],
        'licenses': data['licenses'],
        'categories': data['categories'],
        'images': [img for img in data['images'] if img['id'] in train_ids],
        'annotations': [ann for ann in data['annotations'] if ann['image_id'] in train_ids]
    }
    
    # Create val annotations
    val_data = {
        'info': data['info'],
        'licenses': data['licenses'],
        'categories': data['categories'],
        'images': [img for img in data['images'] if img['id'] in val_ids],
        'annotations': [ann for ann in data['annotations'] if ann['image_id'] in val_ids]
    }
    
    # Save annotations
    os.makedirs(os.path.join(output_dir, "train"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "val"), exist_ok=True)
    
    with open(os.path.join(output_dir, "train", "annotations.json"), 'w') as f:
        json.dump(train_data, f)
    
    with open(os.path.join(output_dir, "val", "annotations.json"), 'w') as f:
        json.dump(val_data, f)
    
    print(f"Dataset split into {len(train_data['images'])} training images and {len(val_data['images'])} validation images.")
    print(f"Please copy the images to their respective directories:")
    print(f"- Training images: {os.path.join(output_dir, 'train')}")
    print(f"- Validation images: {os.path.join(output_dir, 'val')}")


def visualize_dataset_examples(dataset, num_examples=3):
    """Visualize some examples from the dataset to verify proper loading"""
    for i in range(min(num_examples, len(dataset.image_ids))):
        image_id = dataset.image_ids[i]
        
        # Load image and masks
        image = dataset.load_image(image_id)
        masks, class_ids = dataset.load_mask(image_id)
        
        # Display image and masks
        visualize.display_instances(
            image, 
            boxes=utils.extract_bboxes(masks), 
            masks=masks, 
            class_ids=class_ids, 
            class_names=['BG', 'Anorganik_mask', 'Organik_mask', 'anorganik', 'organik'], 
            title=f"Image ID: {image_id}")


def main():
    # Step 1: Set up dataset structure
    prepare_dataset()
    
    # Process annotations from Roboflow (uncomment if needed)
    # process_annotations_from_roboflow("annotations.json", os.path.join(ROOT_DIR, "datasets", "waste"))
    
    # Step 2: Load the dataset and visualize examples to verify it's working
    print("Loading dataset to verify it's working properly...")
    try:
        dataset_dir = os.path.join(ROOT_DIR, "datasets", "waste")
        dataset = WasteDataset()
        dataset.load_waste(dataset_dir, "train")
        dataset.prepare()
        
        # Visualize some examples
        if len(dataset.image_ids) > 0:
            print(f"Dataset loaded successfully with {len(dataset.image_ids)} images.")
            print("Visualizing dataset examples...")
            visualize_dataset_examples(dataset)
        else:
            print("No images found in the dataset. Please check your data.")
    except Exception as e:
        print(f"Error loading dataset: {e}")
        print("Please ensure your dataset is properly set up before training.")
    
    # Step 3: Train the model
    model_weights_path = os.path.join("logs", "waste_final.h5")
    if os.path.exists(model_weights_path):
        print("✅ Model weights found! Skipping training.")
    else:
        train_choice = input("No model weights found. Proceed with training? (y/n): ")
        if train_choice.lower() == 'y':
            model = train_model()
            print("Training completed successfully!")
        else:
            print("Training skipped.")
            return
    
    # Step 4: Set up inference
    inference_config = InferenceConfig()
    
    # Create model in inference mode
    model = modellib.MaskRCNN(mode="inference", config=inference_config, model_dir=MODEL_DIR)
    
    # Load the last model you trained
    model_path = model_weights_path  # langsung pakai path ke 'waste_final.h5'
    print("Loading weights from ", model_path)
    model.load_weights(model_path, by_name=True)
    
    # Test on an image
    while True:
        image_path = input("Enter path to test image (or 'q' to quit): ")
        if image_path.lower() == 'q':
            break
            
        if os.path.exists(image_path):
            results = detect_and_visualize(model, image_path)
            print(f"Detected {len(results['class_ids'])} objects:")
            class_names = ['BG', 'Anorganik_mask', 'Organik_mask', 'anorganik', 'organik']
            
            for i in range(len(results['class_ids'])):
                class_id = results['class_ids'][i]
                class_name = class_names[class_id]
                score = results['scores'][i]
                bbox = results['rois'][i]  # [y1, x1, y2, x2] format
                
                # Convert to [x, y, width, height] format for output
                x = bbox[1]
                y = bbox[0]
                width = bbox[3] - bbox[1]
                height = bbox[2] - bbox[0]
                
                print(f"Object {i+1}: {class_name}, Score: {score:.2f}")
                print(f"  Bounding Box [x,y,w,h]: [{x}, {y}, {width}, {height}]")
                
                # If you want to save mask for this instance
                mask = results['masks'][:, :, i]
                # You can save this mask or use it as needed
        else:
            print("Image not found. Please provide a valid image path.")


if __name__ == "__main__":
    main()