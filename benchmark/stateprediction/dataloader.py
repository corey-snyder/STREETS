import numpy as np
import os
import torch
import json

from skimage.io import imread
from skimage.transform import resize
from torch.utils.data import Dataset

SHAPE = (480, 720)

class FVQDataset(Dataset):
    """Dataset for loading free-flow (f) vs. queue (q) images."""
    def __init__(self, path):
        self.path = path
        self.label_dict = {'f': 0, 'q': 1}
        image_names = np.array([name for name in os.listdir(self.path) if '.json' not in name])
        with open(os.path.join(self.path, 'traffic_state_labels.json'), 'r') as f:
            raw_json = json.load(f)
        self.labels = {}
        self.image_names = []
        for name in image_names:
            for key in raw_json:
                file_atts = raw_json[key]['file_attributes']
                if name in key and 'state' in file_atts:
                    if file_atts['state'] in self.label_dict:
                        self.labels[name] = self.label_dict[raw_json[key]['file_attributes']['state']]
                        self.image_names.append(name)
        self.image_names = np.array(self.image_names)
        self.roadmask_dict = {}
        for name in self.image_names:
            view = name.split('.jpg-')[1].replace('.jpg', '')
            year = name.split('-')[0]
            roadmask_path = os.path.join('/mnt/data0-nfs/shared-datasets/STREETS/roadmasks/', year, view+'.png')
            self.roadmask_dict[view] = (imread(roadmask_path)/MASK_GAIN).astype(np.uint8)
    def __len__(self):
        return len(self.labels)
    def __getitem__(self, idx):
        name = self.image_names[idx]
        view = name.split('.jpg-')[1].replace('.jpg', '')
        image = resize(imread(os.path.join(self.path, self.image_names[idx])),
                       output_shape=SHAPE,
                       mode='reflect',
                       anti_aliasing=True)
        image_tensor = torch.from_numpy(image.transpose(2, 0, 1)).float()
        label = self.labels[name]
        return image_tensor, label
