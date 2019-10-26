import numpy as np
import torch
import os
import sys
import argparse
import json

from tqdm import tqdm
from skimage.transform import resize
from skimage.io import imread
from torch import nn
from torchvision import models

'''
Inference code for traffic state prediction. Class 0
is "free-flow", class 1 is "queue".
'''

class TrafficStateClassifier(nn.Module):
    def __init__(self, n_channels, n_classes=2):
        super(TrafficStateClassifier, self).__init__()
        self.clf = models.resnet50(pretrained=True)
        self.clf.conv1 = nn.Conv2d(n_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        n_features = self.clf.fc.in_features
        self.clf.fc = nn.Linear(n_features, n_classes)
    
    def forward(self, x):
        x = self.clf(x)
        return x

def load_image(image_path):
    return resize(imread(image_path),
                  output_shape=(480,720),
                  mode='reflect',
                  anti_aliasing=True)

def image_is_bad(image):
    bad_image_2018 = load_image('bad-image-2018.jpg')
    bad_image_2019 = load_image('bad-image-2019.jpg')
    if np.allclose(image, bad_image_2018) or np.allclose(image, bad_image_2019):
        return True
    else:
        return False

def image_is_valid(image, name):
    hour = image_hour(name)
    if hour < 5 or hour >= 23:
        return False
    if image_is_bad(image):
        return False
    return True

def image_to_tensor(image):
    image_tensor = torch.from_numpy(image.transpose(2, 0, 1)).unsqueeze(0).float()
    return image_tensor    
       
def image_hour(image_name):
    elems = image_name.split('-') #year, month, day, hour, minute
    hour = int(elems[3])
    return hour

def load_model(model_path):
    clf = TrafficStateClassifier(3)
    clf.load_state_dict(torch.load(model_path))
    return clf

def main():
    #parse input date
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--date', required=True, type=str, help='date to process images')
    parser.add_argument('-b', '--batch', required=True, type=int, help='batch size for inference')
    args = parser.parse_args()
    date = args.date
    batch_size = args.batch
    print('Processing {} with batch size {}'.format(date, batch_size))
    device = 'cuda:0' #specify gpu with CUDA_VISIBLE_DEVICES
    #load classifier, establish file paths
    streets_path = '/mnt/data0-nfs/shared-datasets/STREETS'
    if '2018' in date:
        date_path = os.path.join(streets_path, 'ImageData2018', date)
    else:
        date_path = os.path.join(streets_path, 'ImageData2019', date)
    model_path = os.path.join(streets_path, 'TrafficStateModel', 'fvq-model.pth')
    clf = load_model(model_path)
    clf.to(device)
    clf.eval()

    #dictionary for results, sd[view][image_name] = traffic_state (0 or 1).
    save_dict = {}
    #iterate through each view
    views = os.listdir(date_path)
    with torch.no_grad():
        for view in tqdm(views):
            save_dict[view] = {}
            view_path = os.path.join(date_path, view)
            image_names = os.listdir(view_path)
            images = [load_image(os.path.join(view_path, name)) for name in image_names]
            data_list = [(images[i], image_names[i]) for i in range(len(image_names)) if image_is_valid(images[i], image_names[i])]
            n_batches = int(len(data_list)/batch_size)
            for n in range(n_batches):
                curr_slice = data_list[n*batch_size:(n+1)*batch_size]
                curr_images = torch.cat([image_to_tensor(s[0]) for s in curr_slice]).to(device)
                curr_names = [s[1] for s in curr_slice]
                preds = torch.max(clf(curr_images), dim=1)[1]
                for i, p in enumerate(preds):
                    save_dict[view][curr_names[i]] = preds[i].item()
            last_slice = data_list[n_batches*batch_size:]
            if len(last_slice):
                curr_images = torch.cat([image_to_tensor(s[0]) for s in curr_slice]).to(device)
                curr_names = [s[1] for s in curr_slice]
                preds = torch.max(clf(curr_images), dim=1)[1]
                for i, p in enumerate(preds):
                    save_dict[view][curr_names[i]] = preds[i].item()

    #save results
    with open(os.path.join('data/states', '{}-states.json'.format(date)), 'w') as f:
        json.dump(save_dict, f)
            
if __name__ == '__main__':
    main()
