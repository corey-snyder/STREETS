# STREETS
## A Novel Camera Network Dataset for Traffic Flow
This is the code repository for "Corey Snyder and Minh N. Do, Streets: A novel camera  network  dataset  for  traffic  flow, Accepted to Thirty-third Conference on Neural Information Processing Systems, 2019".

## Requirements
This work was implemented with the following packages and others included in `requirements.txt`.
+ tensorflow 1.13.1
+ Keras 2.2.4
+ torch 1.0.0
+ torchvision 0.2.1

## STREETS
The data for the STREETS dataset may be found on the [Illinois Data Bank](https://databank.illinois.edu/datasets/IDB-3671567). Each element of the dataset may be downloaded separately from this page. Each part of the dataset is explained in detail with the accompanying readme files on Illinois Data Bank. Place all the data you download in the `data/` folder of this repository.

## Benchmarking
We provide our code to benchmark three tasks presented in the paper: vehicle counting, traffic state inference, and single-step traffic prediction. The usage of each folder is as follows.

### benchmark/stateprediction
This code concerns the task of predicting whether traffic is in a "free-flow" or "queue" state on the roadway of interest for a given image. The model training code is implemented in PyTorch and may be found in `train.py`. The default settings are 20 epochs, batch size of 8, learning rate of 1e-3, training-testing split of 80%-20%, and device set to cuda:0. These may be changed in `train.py` if you wish. The relevant data afor this task is found in `trafficstate.zip`. Please download and unzip this folder into the `data/` folder of this repository. Training is performed on a random training-testing split and the training code may be run with 

    python3 train.py
    
### benchmark/trafficprediction
This code reproduces the single-step traffic prediction results presented in the paper. Various models are built in files like `linear.py`, `rfr.py`, etc. Several utility functions for tasks like loading traffic data, K-hop neighborhoods, and training data are provided in `benchmarkutils.py`. The benchmarking code itself is contained in `benchmark.py` and the usage is given by

    python3 benchmarking.py -m [ha, rfr, svr, ann, linear, linear-lasso, linear-ridge]
    
Simply choose the model string for which you would like the test. The model-specific settings, like number of trees in Random Forest Regression or kernel in Support Vector Regressor, may be easily modified in the `load_model` function. Furthermore, you can easily change the community graph (`community`), training dates (`training_dates`), testing dates (`testing_dates`), and sampling period (`T`) in the main function. The graphs and preprocessed traffic data are provided within `trafficprediction` for convenience. Simply unzip the .json files in `trafficdata` to get started. Note: we make heavy used of the `tqdm` package to carefully track the progress of our experiments. You may remove this from each for loop to reduce the amount of printing during testing.

### benchmark/vehiclecounting
This code allows testing of our retrained Mask R-CNN vehicle detector. This code is implemented in Tensorflow and the Mask R-CNN code is taken from the Matterport implementation found [here](https://github.com/matterport/Mask_RCNN). The files `mrcnn_coco_benchmarking.py` and `mrcnn_retrained_benchmarking.py` provide the benchmarking code for assessing the MAE, MAPE, Precision, and Recall of the Mask R-CNN with pretrained COCO weights and our retrained weights, respectively. To test the COCO weights model, run

    python3 mrcnn_coco_benchmarking.py [mode] [IoU threshold]
    
The mode should be either train or val and the IoU threshold can be any number between 0 and 1. An example train-val split file is provided in the `data/` folder. To test our retrained model weights, please download them from Illinois Data Bank under `detectorweights.h5` and place them in the `Mask_RCNN/` folder. Similarly, the usage for testing the retrained model weights is

    python3 mrcnn_retrained_benchmarking.py [mode] [IoU threshold]
    
Note that the results you will obtain differ slightly from those reported in the paper due to the use of a random training-validation split.

### Future Work
We will continue to update this repository with further code for easily loading and visualizing other parts of the dataset. Please let us know if you have any issues or suggestions and we would be happy to help.
