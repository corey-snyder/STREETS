import os
import sys
import random
import math
import numpy as np
import warnings
import gzip
import shutil
import pickle
import json

from PIL import Image
from sklearn.externals import joblib
from skimage.io import imread
from tqdm import tqdm
from skimage.transform import resize
from shapely.geometry import Polygon
from shapely.geometry import Point
warnings.filterwarnings("ignore",category=DeprecationWarning)
warnings.filterwarnings("ignore",category=UserWarning)

# Root directory of the project
ROOT_DIR = os.path.abspath("Mask_RCNN/")
# Import Mask_RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn import utils
import travel
import mrcnn.model as modellib
TRAVEL_CLASSES = ['BG','car']
# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs")
# Local path to trained weights file
TRAFFIC_MODEL_PATH = os.path.join(ROOT_DIR, "detectorweights.h5")
DATA_PATH = '/mnt/data0-nfs/shared-datasets/STREETS/mask_rcnn_training'
sys.path.append(os.path.join(ROOT_DIR, "samples/coco/"))
import coco
COCO_MODEL_PATH = os.path.join(ROOT_DIR,"mask_rcnn_coco.h5")
if not os.path.exists(COCO_MODEL_PATH):
    utils.download_trained_weights(COCO_MODEL_PATH)
TAU = 0.2
class InferenceConfig(coco.CocoConfig): #travel.TravelConfig
    # Set batch size to 1 since we'll be running inference on
    # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    DETECTION_MIN_CONFIDENCE = 0.7
    NUM_CLASSES = 1+1
    RPN_ANCHOR_SCALES = (16, 32, 64, 128, 256)
def load_view_classifier(view):
    direction = view.split(' ')[-1]
    location = view.replace(' '+direction,'')
    if view == 'IL 43 at Northpoint North':
        location = 'IL 43 at Northpoint'
    clf_path = os.path.join('ViewClassifiers',location+'.pkl')
    clf = joblib.load(clf_path)
    return clf
def classify_images(image_names,view_dir,view_clf):
    clf_images = []
    true_shape = BAD_IMAGE.shape
    for name in image_names:
        image_path = os.path.join(view_dir,name)
        image = imread(image_path)
        if not np.array_equal(image,BAD_IMAGE):
            true_shape = image.shape
    for name in image_names:
        image_path = os.path.join(view_dir,name)
        image = imread(image_path)
        if not np.array_equal(image,BAD_IMAGE) and np.array_equal(np.array(image.shape),np.array(true_shape)):
            img_arr = np.array(Image.open(image_path).convert('L'))
            resized = resize(img_arr, output_shape=(int(img_arr.shape[0]/DF), int(img_arr.shape[1]/DF)), anti_aliasing=True, mode='reflect')
            flattened = resized.flatten()
            clf_images.append(flattened)
        else:
            clf_images.append(np.zeros(int(true_shape[0]/DF)*int(true_shape[1]/DF)))
    view_predictions = view_clf.predict(clf_images)
    return view_predictions
def timestamp_to_hour(name):
    k = name.split('-')
    return int(k[3])
def is_inbound(vehicle_mask,in_mask,out_mask):
    occupied = []
    #identify rows that the vehicle mask occupies
    for i,row in enumerate(vehicle_mask):
        if np.sum(row):
            occupied.append(i)
    occupied = np.array(occupied)
    top_row,bottom_row = np.min(occupied),np.max(occupied) #find top and bottom rows
    w = np.zeros(vehicle_mask.shape[0]) #weight vector for each row
    #populate weights linearly from top (weight = 0) to bottom (weight = 1)
    for r in range(top_row,bottom_row+1):
        w[r] = (r-top_row)/(bottom_row-top_row+1)
    #get overlaps with respect to inbound and outbound road masks
    in_inter,out_inter = np.logical_and(vehicle_mask,in_mask), np.logical_and(vehicle_mask,out_mask)
    #weigh each rows pixels appropriately and sum their weights (votes)
    in_votes,out_votes = np.sum(np.matmul(in_inter.T,w)),np.sum(np.matmul(out_inter.T,w))
    if in_votes >= out_votes:
        return True
    else:
        return False
def masks_to_density(masks):
    D = np.zeros(masks[0].shape)
    for mask in masks:
        size = np.sum(mask)
        rows = np.any(mask, axis=1)
        cols = np.any(mask, axis=0)
        rmin, rmax = np.where(rows)[0][[0, -1]]
        cmin, cmax = np.where(cols)[0][[0, -1]]
        for r in range(rmin,rmax+1):
            for c in range(cmin,cmax+1):
                if mask[r,c]:
                    D[r,c] += 1/size
    return D

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
class_names = ['BG', 'person', 'bicycle', 'car', 'motorcycle', 'airplane',
               'bus', 'train', 'truck', 'boat', 'traffic light',
               'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird',
               'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear',
               'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie',
               'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
               'kite', 'baseball bat', 'baseball glove', 'skateboard',
               'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
               'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
               'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
               'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed',
               'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
               'keyboard', 'cell phone', 'microwave', 'oven', 'toaster',
               'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors',
               'teddy bear', 'hair drier', 'toothbrush']
traffic_ids = [2,3,4,6,8]
# Create model object in inference mode.
model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=config)

# Load weights trained on MS-COCO
model.load_weights(TRAFFIC_MODEL_PATH, by_name=True)
assert len(sys.argv) == 2,"Missing arguments"
mode = sys.argv[1]
with open(os.path.join(DATA_PATH, mode, 'via_region_data.json'),'r') as f:
    val_dict = json.load(f)
img_dir = os.path.join(DATA_PATH, mode)
AE = []
APE = []
ADE = []
ADPE = []
n_tp = 0
total_detect = 0
total_gt = 0
for f in tqdm(os.listdir(img_dir)):
    if '.jpg' in f:
        image_name = f
        label_polygons, label_polygon_points = extract_json_points_polygons(val_dict,image_name)
        view = image_name.replace('-'+image_name.split('-')[-1],'')
        road_mask = np.load(os.path.join('RoadMasks', view + '.npy'))
        n_rows,n_cols = road_mask.shape
        in_mask = road_mask > 1
        out_mask = road_mask == 1
        binary_mask = road_mask > 0
        gt_masks = [polygon_to_ndarray(lp,n_rows,n_cols) for lp in label_polygons]
        n_gt = 0
        check_gt = []
        for k in range(len(gt_masks)):
            gt_mask = gt_masks[k]
            gt_size = np.sum(gt_mask)
            if gt_size > 0:
                inter = np.sum(np.logical_and(gt_mask, binary_mask))
                if inter / gt_size > 0.1:
                    n_gt += 1
                    check_gt.append(k)
        total_gt += n_gt
        image = imread(os.path.join(img_dir,image_name))
        results = model.detect([image], verbose=0)[0]
        masks = results['masks'] #row,col,mask_idx
        found_idx = []
        #check each detected mask, only consider masks on roadway
        n_detected = 0
        for j in range(masks.shape[2]):
            curr_mask = masks[:,:,j]
            curr_size = np.sum(curr_mask)
            if curr_size > 0: #and results['class_ids'][j] in traffic_ids:
                inter = np.sum(np.logical_and(curr_mask,binary_mask))
                if inter/curr_size > 0.1:
                    n_detected += 1
                    for k in check_gt:
                        if k not in found_idx:
                            curr_gt = gt_masks[k]
                            inter = np.sum(np.logical_and(curr_mask,curr_gt))
                            union = np.sum(np.logical_or(curr_mask,curr_gt))
                            if inter/union > TAU:
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
