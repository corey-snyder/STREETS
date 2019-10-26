"""
Mask R-CNN
Train on STREETS dataset.

Edited from the following source:
*    Title: Matterport, Inc.
*    Author: Waleed Abdulla
*    Date: 2018
*    Code version: 2.0
*    Availability: https://github.com/matterport/Mask_RCNN
*
------------------------------------------------------------

Usage: import the module (see Jupyter notebooks for examples), or run from
       the command line as such:

    # Train a new model starting from pre-trained COCO weights
    python3 travel.py train --dataset=/path/to/travel/dataset --weights=coco

    # Resume training a model that you had trained earlier
    python3 travel.py train --dataset=/path/to/travel/dataset --weights=last

    # Train a new model starting from ImageNet weights
    python3 travel.py train --dataset=/path/to/travel/dataset --weights=imagenet

"""

import os
import sys
import json
import datetime
import numpy as np
import skimage.draw
import warnings

warnings.filterwarnings("ignore",category=DeprecationWarning)
warnings.filterwarnings("ignore",category=UserWarning)
# Root directory of the project
ROOT_DIR = os.path.abspath("../../")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn.config import Config
from mrcnn import model as modellib, utils

# Path to trained weights file
COCO_WEIGHTS_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")

# Directory to save logs and model checkpoints, if not provided
# through the command line argument --logs
DEFAULT_LOGS_DIR = os.path.join(ROOT_DIR, "logs")

# Classes from TravelMidwest dataset.
travel_classes = ['car']
############################################################
#  Configurations
############################################################


class TravelConfig(Config):
    """Configuration for training on the TravelMidwest dataset.
    Derives from the base Config class and overrides some values.
    """
    # Give the configuration a recognizable name
    NAME = "travel"

    # We use a GPU with 12GB memory, which can fit two images.
    # Adjust down if you use a smaller GPU.
    IMAGES_PER_GPU = 1

    # Number of classes (including background)
    NUM_CLASSES = 1 + 1  # Background + (car)

    # Skip detections with < 90% confidence
    DETECTION_MIN_CONFIDENCE = 0.7 

    # Length of square anchor side in pixels
    RPN_ANCHOR_SCALES = (16, 32, 64, 128, 256)

############################################################
#  Dataset
############################################################

class TravelDataset(utils.Dataset):

    def load_travel(self, dataset_dir, subset):
        """Load a subset of the TravelMidwest dataset.
        dataset_dir: Root directory of the dataset.
        subset: Subset to load: train or val
        """
        # Add classes.
        class_id = 1 # id 0 is for BG
        for class_name in travel_classes:
            self.add_class("travel", class_id, class_name)
            class_id += 1

        # Load road polygons from json before overwriting dataset_dir
        road_annotations = json.load(open(os.path.join(dataset_dir, "road_via_region_data.json")))
        road_annotations = list(road_annotations.values())
        
        road_dict = {}
        for r in road_annotations:
            view = self.get_view_name(r['filename'])
            road_dict[view] = [d['shape_attributes'] for d in r['regions'].values()]

        # Train or validation dataset?
        assert subset in ["train", "val"]
        dataset_dir = os.path.join(dataset_dir, subset)

        # Load annotations
        # VGG Image Annotator saves each image in the form:
        # { 'filename': '28503151_5b5b7ec140_b.jpg',
        #   'regions': {
        #       '0': {
        #           'region_attributes': {},
        #           'shape_attributes': {
        #               'all_points_x': [...],
        #               'all_points_y': [...],
        #               'name': 'polygon'}},
        #       ... more regions ...
        #   },
        #   'size': 100202
        # }
        # We mostly care about the x and y coordinates of each region
        annotations = json.load(open(os.path.join(dataset_dir, "via_region_data.json")))
        annotations = list(annotations.values())  # don't need the dict keys

        # The VIA tool saves images in the JSON even if they don't have any
        # annotations. Skip unannotated images.
        annotations = [a for a in annotations if a['regions']]
        
        # Add images
        for a in annotations:
            # Get the x, y coordinates of points of the polygons that make up
            # the outline of each object instance. There are stores in the
            # shape_attributes (see json format above)
            polygons = [r['shape_attributes'] for r in a['regions'].values()]

            # load_mask() needs the image size to convert polygons to masks.
            # Unfortunately, VIA doesn't include it in JSON, so we must read
            # the image. This is only managable since the dataset is tiny.

            image_path = os.path.join(dataset_dir, a['filename'])
            image = skimage.io.imread(image_path)
            height, width = image.shape[:2]

            # Get view_name from image_name
            view_name = self.get_view_name(a['filename'])
            road_polygons = road_dict[view_name]
            polygons = self.get_polygons_on_road(polygons, road_polygons, width, height)
        
            self.add_image(
                "travel",
                image_id=a['filename'],  # use file name as a unique image id
                path=image_path,
                width=width, height=height,
                polygons=polygons,
                road_polygons=road_dict[view_name],
                view_id=view_name)

    def load_mask(self, image_id):
        """Generate instance masks for an image.
       Returns:
        masks: A bool array of shape [height, width, instance count] with
            one mask per instance.
        class_ids: a 1D array of class IDs of the instance masks.
        """
        # If not a TravelMidwest dataset image, delegate to parent class.
        image_info = self.image_info[image_id]
        if image_info["source"] != "travel":
            return super(self.__class__, self).load_mask(image_id)

        # Convert polygons to a bitmap mask of shape
        # [height, width, instance_count]
        info = self.image_info[image_id]
        mask = np.zeros([info["height"], info["width"], len(info["polygons"])],
                        dtype=np.uint8)
        for i, p in enumerate(info["polygons"]):
            # Get indexes of pixels inside the polygon and set them to 1
            rr, cc = skimage.draw.polygon(p['all_points_y'], p['all_points_x'])
            mask[rr, cc, i] = 1

        # Return mask, and array of class IDs of each instance. Since we have
        # one class ID only, we return an array of 1s
        return mask.astype(np.bool), np.ones([mask.shape[-1]], dtype=np.int32)
    
    def load_road_mask(self, image_id):
        """Generate instance masks for an image.
       Returns:
        masks: A bool array of shape [height, width] with
            one mask consisting of inbound and outbound roads.
        """
        # Convert polygons to a bitmap mask of shape
        # [height, width, instance_count]
        info = self.image_info[image_id]
        road_mask = np.zeros([info["height"], info["width"], len(info["road_polygons"])],
                        dtype=np.uint8)
        for i, p in enumerate(info["road_polygons"]):
            # Get indexes of pixels inside the polygon and set them to 1
            rr, cc = skimage.draw.polygon(p['all_points_y'], p['all_points_x'])
            road_mask[rr, cc, i] = 1

        # Combine the inbound and outbound road masks
        combined_mask = np.zeros([info["height"], info["width"]],
                        dtype=np.uint8)
        for i in range(road_mask.shape[2]):
            combined_mask = np.maximum(combined_mask,road_mask[:,:,i])

        return combined_mask.astype(np.bool) #road_mask.astype(np.bool)

    def get_polygons_on_road(self, polygons, road_polygons, width, height):
        # Get masks
        mask = np.zeros([height, width, len(polygons)], dtype=np.uint8)
        for i, p in enumerate(polygons):
            # Get indexes of pixels inside the polygon and set them to 1
            rr, cc = skimage.draw.polygon(p['all_points_y'], p['all_points_x'])
            mask[rr, cc, i] = 1

        # Get road masks
        road_mask = np.zeros([height, width, len(road_polygons)], dtype=np.uint8)
        for i, p in enumerate(road_polygons):
            # Get indexes of pixels inside the polygon and set them to 1
            rr, cc = skimage.draw.polygon(p['all_points_y'], p['all_points_x'])
            road_mask[rr, cc, i] = 1

        # Combine the inbound and outbound road masks
        combined_mask = np.zeros([height, width], dtype=np.uint8)
        for i in range(road_mask.shape[2]):
            combined_mask = np.maximum(combined_mask,road_mask[:,:,i])
        on_road_indices = []
        for i in range(len(polygons)):
            m = mask[:,:,i]
            intersect = np.add(m, combined_mask)

            if(len(np.where(intersect == 2)[0]) > 0):
                on_road_indices.append(i)

        
        polygons = np.asarray(polygons)
        on_road_indices = np.asarray(on_road_indices, dtype=np.int)
        #print(np.shape(on_road_indices))
        #print(on_road_indices.dtype)
        new_polygons = polygons[on_road_indices]
        new_polygons = new_polygons.tolist()

        return new_polygons

    def image_reference(self, image_id):
        """Return the path of the image."""
        info = self.image_info[image_id]
        if info["source"] == "travel":
            return info["path"]
        else:
            super(self.__class__, self).image_reference(image_id)

    def get_view_name(self, image_id):
        hyphen_split = image_id.split('-')
        view_name = image_id.replace('-'+hyphen_split[-1],'')
        return view_name


def train(model):
    """Train the model."""
    # Training dataset.
    dataset_train = TravelDataset()
    dataset_train.load_travel(args.dataset, "train")
    dataset_train.prepare()

    # Validation dataset
    dataset_val = TravelDataset()
    dataset_val.load_travel(args.dataset, "val")
    dataset_val.prepare()

    # *** This training schedule is an example. Update to your needs ***
    # Since we're using a very small dataset, and starting from
    # COCO trained weights, we don't need to train too long. Also,
    # no need to train all layers, just the heads should do it.
    print("Training network heads")
    model.train(dataset_train, dataset_val,
                learning_rate=config.LEARNING_RATE,
                epochs=30,
                layers='all')



############################################################
#  Training
############################################################

if __name__ == '__main__':
    import argparse

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Train Mask R-CNN to detect TM objects.')
    parser.add_argument("command",
                        metavar="<command>",
                        help="'train'")
    parser.add_argument('--dataset', required=False,
                        metavar="/path/to/travel/dataset/",
                        help='Directory of the TravelMidwest dataset')
    parser.add_argument('--weights', required=True,
                        metavar="/path/to/weights.h5",
                        help="Path to weights .h5 file or 'coco'")
    parser.add_argument('--logs', required=False,
                        default=DEFAULT_LOGS_DIR,
                        metavar="/path/to/logs/",
                        help='Logs and checkpoints directory (default=logs/)')

    args = parser.parse_args()

    # Validate arguments
    if args.command == "train":
        assert args.dataset, "Argument --dataset is required for training"

    print("Weights: ", args.weights)
    print("Dataset: ", args.dataset)
    print("Logs: ", args.logs)

    # Configurations
    if args.command == "train":
        config = TravelConfig()
    else:
        class InferenceConfig(TravelConfig):
            # Set batch size to 1 since we'll be running inference on
            # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
            GPU_COUNT = 1
            IMAGES_PER_GPU = 1
        config = InferenceConfig()
    config.display()

    # Create model
    if args.command == "train":
        model = modellib.MaskRCNN(mode="training", config=config,
                                  model_dir=args.logs)
    else:
        model = modellib.MaskRCNN(mode="inference", config=config,
                                  model_dir=args.logs)

    # Select weights file to load
    if args.weights.lower() == "coco":
        weights_path = COCO_WEIGHTS_PATH
        # Download weights file
        if not os.path.exists(weights_path):
            utils.download_trained_weights(weights_path)
    elif args.weights.lower() == "last":
        # Find last trained weights
        weights_path = model.find_last()[1]
    elif args.weights.lower() == "imagenet":
        # Start from ImageNet trained weights
        weights_path = model.get_imagenet_weights()
    else:
        weights_path = args.weights

    # Load weights
    print("Loading weights ", weights_path)
    if args.weights.lower() == "coco":
        # Exclude the last layers because they require a matching
        # number of classes
        model.load_weights(weights_path, by_name=True, exclude=[
            "mrcnn_class_logits", "mrcnn_bbox_fc",
            "mrcnn_bbox", "mrcnn_mask"])
    else:
        model.load_weights(weights_path, by_name=True)

    # Train
    if args.command == "train":
        train(model)
    else:
        print("'{}' is not recognized. "
              "Use 'train'".format(args.command))
