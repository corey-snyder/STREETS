import os
import sys
import random
import math
import numpy as np
import warnings
import json

from skimage.io import imread
from tqdm import tqdm
from shapely.geometry import Polygon
from shapely.geometry import Point
warnings.filterwarnings("ignore",category=DeprecationWarning)
warnings.filterwarnings("ignore",category=UserWarning)

# Root directory of the project
ROOT_DIR = os.path.abspath("Mask_RCNN/")
# Import Mask_RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
MODEL_DIR = os.path.join(ROOT_DIR, "logs")
RETRAINED_MODEL_PATH = os.path.join(ROOT_DIR, "detectorweights.h5")
DATA_PATH = '../../data'

from mrcnn import utils
import mrcnn.model as modellib
import coco

class InferenceConfig(coco.CocoConfig):
    # Set batch size to 1 since we'll be running inference on
    # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    DETECTION_MIN_CONFIDENCE = 0.7
    NUM_CLASSES = 1+1
    RPN_ANCHOR_SCALES = (16, 32, 64, 128, 256)
def extract_json_points_polygons(label_dict,image_name):
    label_polygons = [] #shapely.geometry.Polygon objects
    label_polygon_points = [] #List of (x_pts,y_pts) tuple for each polygon
    label_key = None
    for k in label_dict:
        if image_name in k:
            label_key = k
    label_dict_entry = label_dict[label_key]
    region_dict = label_dict_entry['regions']
    for region_key in region_dict:
        reg_shape = region_dict[region_key]['shape_attributes']
        if reg_shape['name'] == 'polygon':
            n_pts = len(reg_shape['all_points_x'])
            if n_pts > 2:
                polygon_points = [(reg_shape['all_points_x'][i],reg_shape['all_points_y'][i]) for i in range(n_pts)]
                label_polygon_points.append((reg_shape['all_points_x'],reg_shape['all_points_y']))
                label_polygons.append(Polygon(polygon_points))
        else:
            print('Unknown shape encountered.')
    return label_polygons,label_polygon_points

def polygon_to_ndarray(polygon,n_rows,n_cols):
    ret_array = np.zeros((n_rows,n_cols),dtype=bool)
    min_c,min_r,max_c,max_r = polygon.bounds
    for r in range(int(min_r),int(max_r)):
        for c in range(int(min_c),int(max_c)):
            curr_pt = Point(c,r) #(x,y)
            if polygon.contains(curr_pt):
                ret_array[r,c] = True
            else:
                ret_array[r,c] = False
    return ret_array

config = InferenceConfig()
config.display()

# Create model object in inference mode.
model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=config)

# Load weights trained on MS-COCO
model.load_weights(RETRAINED_MODEL_PATH, by_name=True)
assert len(sys.argv) == 3,"Missing arguments"
mode = sys.argv[1]
tau = float(sys.argv[2]) #IoU threshold

with open(os.path.join(DATA_PATH, 'vehicleannotations', 'annotations', 'vehicle-annotations.json'), 'r') as f:
    data_dict = json.load(f)
with open(os.path.join(DATA_PATH, 'train-val-split.json'), 'r') as f:
    train_val_split = json.load(f)

image_dir = os.path.join(DATA_PATH, 'vehicleannotations', 'images')

AE = []
APE = []
ADE = []
ADPE = []
n_tp = 0
total_detect = 0
total_gt = 0

for f in tqdm(os.listdir(image_dir)):
    if f in train_val_split[mode]:
        image_name = f
        label_polygons, label_polygon_points = extract_json_points_polygons(data_dict, image_name)
        view = image_name.replace('-'+image_name.split('-')[-1],'')
        road_mask = imread(os.path.join(DATA_PATH, 'roadmasks', '2018', view + '.png'))
        n_rows, n_cols = road_mask.shape
        binary_mask = road_mask > 0
        gt_masks = [polygon_to_ndarray(lp, n_rows, n_cols) for lp in label_polygons]
        n_gt = 0
        check_gt = []
        for k in range(len(gt_masks)):
            gt_mask = gt_masks[k]
            gt_size = np.sum(gt_mask)
            if gt_size > 0:
                inter = np.sum(np.logical_and(gt_mask, binary_mask))
                if inter / gt_size:
                    n_gt += 1
                    check_gt.append(k)
        total_gt += n_gt
        image = imread(os.path.join(image_dir, image_name))
        results = model.detect([image], verbose=0)[0]
        masks = results['masks'] #row,col,mask_idx
        found_idx = []
        #check each detected mask, only consider masks on roadway
        n_detected = 0
        for j in range(masks.shape[2]):
            curr_mask = masks[:,:,j]
            curr_size = np.sum(curr_mask)
            if curr_size > 0:
                inter = np.sum(np.logical_and(curr_mask, binary_mask))
                if inter/curr_size:
                    n_detected += 1
                    for k in check_gt:
                        if k not in found_idx:
                            curr_gt = gt_masks[k]
                            inter = np.sum(np.logical_and(curr_mask, curr_gt))
                            union = np.sum(np.logical_or(curr_mask, curr_gt))
                            if inter/union > tau:
                                n_tp += 1
                                found_idx.append(k)
        total_detect += n_detected
        n_correct = len(found_idx)
        AE.append(abs(n_detected - n_gt))
        ADE.append(abs(n_correct-n_gt))
        if n_gt:
            APE.append(abs(n_detected-n_gt)/n_gt)
            ADPE.append(abs(n_correct-n_gt)/n_gt)
print('MAE: {}'.format(np.mean(ADE)))
print('MAPE: {}'.format(np.mean(ADPE)))
print('Precision: {}'.format(n_tp/total_detect))
print('Recall: {}'.format(n_tp/total_gt))
print('Total Vehicles: {}'.format(total_gt))
