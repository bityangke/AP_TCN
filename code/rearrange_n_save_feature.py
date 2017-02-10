import os
import glob
from collections import OrderedDict

import numpy as np
from scipy import io as sio

# TCN imports
import ap_datasets, utils
from utils import imshow_

# ---------- Directories & User inputs --------------
# Location of data/features folder
base_dir = os.path.expanduser("../")

# Set dataset and action label granularity (if applicable)
dataset = ["50Salads", "JIGSAWS", "MERL", "GTEA", "UCF101"][4]
# Set model and parameters
model_type = ["AP-TCN", "AP-TCN-SanityCheck"][0]
# causal or acausal? (If acausal use Bidirectional LSTM)
causal = [False, True][0]

# How many latent states/nodes per layer of network
# Only applicable to the TCNs. The ECCV and LSTM  model suses the first element from this list.
n_nodes = [64, 96]
nb_epoch = 40 #50
conv = {'50Salads':25, "JIGSAWS":20, "MERL":5, "GTEA":25, 'UCF101': 25}[dataset]

# Which features for the given dataset
features = "SpatialCNN"

if dataset == "UCF101":
    base_dir = "/home/jinchoi/src/rehab/dataset/action/UCF101/"
    feature_type = 'relu7_feat'
    video_rate = 10

feature_type='pool5_feat'
data = ap_datasets.Dataset(dataset, base_dir, feature_type='pool5')
for video_rate in [25, 10]:
    for split in data.splits:
        if dataset != 'UCF101':
            feature_type = "A" if model_type != "SVM" else "X"

        # Load the feature files
        print("Loading data split...")
        # If there exist .npy files, load them
        if ( os.path.exists(base_dir + 'AlexNet-pool5-npy/X_train_ucf_' + split + '_' + str(video_rate) + 'f_to1f' +'.npy') and
             os.path.exists(base_dir + 'AlexNet-pool5-npy/y_train_ucf_' + split + '_' + str(video_rate) + 'f_to1f' +'.npy') and
             os.path.exists(base_dir + 'AlexNet-pool5-npy/X_test_ucf_' + split + '_' + str(video_rate) + 'f_to1f' +'.npy') and
             os.path.exists(base_dir + 'AlexNet-pool5-npy/y_test_ucf_' + split + '_' + str(video_rate) + 'f_to1f' +'.npy') ):
            print("npy files found.")
            X_train = np.load(base_dir + 'AlexNet-pool5-npy/X_train_ucf_' + split + '_' + str(video_rate) + 'f_to1f' +'.npy');
            y_train = np.load(base_dir + 'AlexNet-pool5-npy/y_train_ucf_' + split + '_' + str(video_rate) + 'f_to1f' +'.npy');
            X_test = np.load(base_dir + 'AlexNet-pool5-npy/X_test_ucf_' + split + '_' + str(video_rate) + 'f_to1f' +'.npy');
            y_test = np.load(base_dir + 'AlexNet-pool5-npy/y_test_ucf_' + split + '_' + str(video_rate) + 'f_to1f' +'.npy');
        # If there are no .npy files, load .mat files and generate the numpy features
        else:
            print("npy files not found. Loading from mat files. This would take a while...")
            X_train, y_train, X_test, y_test = data.load_split(features, split=split,
                                                                sample_rate=video_rate,
                                                                feature_type=feature_type)
            np.save(base_dir + 'AlexNet-pool5-npy/X_train_ucf_' + split + '_' + str(video_rate) + 'f_to1f' +'.npy', X_train);
            np.save(base_dir + 'AlexNet-pool5-npy/y_train_ucf_' + split + '_' + str(video_rate) + 'f_to1f' +'.npy', y_train);
            np.save(base_dir + 'AlexNet-pool5-npy/X_test_ucf_' + split + '_' + str(video_rate) + 'f_to1f' +'.npy', X_test);
            np.save(base_dir + 'AlexNet-pool5-npy/y_test_ucf_' + split + '_' + str(video_rate) + 'f_to1f' +'.npy', y_test);
        print("Loading done.")
