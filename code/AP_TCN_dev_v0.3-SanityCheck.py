import os
import glob
from collections import OrderedDict

import numpy as np

from scipy import io as sio
import sklearn.metrics as sm
from sklearn.svm import LinearSVC
from sklearn.decomposition import PCA

from keras.utils import np_utils

# TCN imports 
import tf_models, ap_datasets, utils, metrics
from utils import imshow_

# ---------- Directories & User inputs --------------
# Location of data/features folder
base_dir = os.path.expanduser("../")

save_predictions = [False, True][1]
viz_predictions = [False, True][1]
viz_weights = [False, True][0]

# Set dataset and action label granularity (if applicable)
dataset = ["50Salads", "JIGSAWS", "MERL", "GTEA", "UCF101"][4]
granularity = ["eval", "mid"][1]
sensor_type = ["video", "sensors"][0]

# Set model and parameters
model_type = ["ED-TCN","AP-TCN"][1]
# causal or acausal? (If acausal use Bidirectional LSTM)
causal = [False, True][0]

# How many latent states/nodes per layer of network
# Only applicable to the TCNs. The ECCV and LSTM  model suses the first element from this list.
n_nodes = [64, 96]
nb_epoch = 40 #50
video_rate = 3
conv = {'50Salads':25, "JIGSAWS":20, "MERL":5, "GTEA":25, 'UCF101': 25}[dataset]

# Which features for the given dataset
features = "SpatialCNN"
bg_class = 0 if dataset is not "JIGSAWS" else None

if dataset == "50Salads":
    features = "SpatialCNN_" + granularity
if dataset == "UCF101":
    base_dir = "/home/jinchoi/src/rehab/dataset/action/UCF101/"
    feature_type = 'relu7_feat'
    video_rate = 25
    architecture = 'sanity_check1'

data = ap_datasets.Dataset(dataset, base_dir, feature_type='fc7')
trial_metrics = metrics.ComputeMetrics(overlap=.1, bg_class=bg_class)


# In[ ]:

#from IPython.core.debugger import Pdb # for debugging, jinchoi@vt.edu
#pdb = Pdb()

# ------------------------------------------------------------------
# Evaluate using different filter lengths
if 1:
# for conv in [5, 10, 15, 20]:
    # Initialize dataset loader & metrics
    if dataset == 'UCF101':
        data = ap_datasets.Dataset(dataset, base_dir, feature_type='fc7')
    else:
        data = ap_datasets.Dataset(dataset, base_dir)

    split_cnt = 0
    accuracies = np.zeros(len(data.splits))
    train_accuracies = np.zeros(len(data.splits))
    
    # Load data for each split
    for split in data.splits:
        if dataset != 'UCF101':
            if sensor_type=="video":
                feature_type = "A" if model_type != "SVM" else "X"
            else:
                feature_type = "S"
        
        print("Loading data split...")
        if ( os.path.exists(base_dir + 'AlexNet-fc7-npy/X_train_ucf_' + split + '_' + architecture +'.npy') and 
             os.path.exists(base_dir + 'AlexNet-fc7-npy/y_train_ucf_' + split + '_' + architecture +'.npy') and
             os.path.exists(base_dir + 'AlexNet-fc7-npy/X_test_ucf_' + split + '_' + architecture +'.npy') and
             os.path.exists(base_dir + 'AlexNet-fc7-npy/y_test_ucf_' + split + '_' + architecture +'.npy') ):
            print("npy files found.")
            X_train = np.load(base_dir + 'AlexNet-fc7-npy/X_train_ucf_' + split + '_' + architecture +'.npy');
            y_train = np.load(base_dir + 'AlexNet-fc7-npy/y_train_ucf_' + split + '_' + architecture + '.npy');
            X_test = np.load(base_dir + 'AlexNet-fc7-npy/X_test_ucf_' + split + '_' + architecture +'.npy');
            y_test = np.load(base_dir + 'AlexNet-fc7-npy/y_test_ucf_' + split + '_' + architecture +'.npy');
        else:
            print("npy files not found. Loading from mat files. This would take a while...")
            X_train, y_train, X_test, y_test = data.load_split(features, split=split, 
                                                                sample_rate=video_rate, 
                                                                feature_type=feature_type)
            np.save(base_dir + 'AlexNet-fc7-npy/X_train_ucf_' + split + '_' + architecture +'.npy', X_train);
            np.save(base_dir + 'AlexNet-fc7-npy/y_train_ucf_' + split + '_' + architecture +'.npy', y_train);
            np.save(base_dir + 'AlexNet-fc7-npy/X_test_ucf_' + split + '_' + architecture +'.npy', X_test);
            np.save(base_dir + 'AlexNet-fc7-npy/y_test_ucf_' + split + '_' + architecture +'.npy', y_test);
        print("Loading done.")
        
#         pca = PCA(n_components = 32)
#         pca.fit(X_train[0])

        n_classes = data.n_classes
        n_train = len(X_train)
        n_test = len(X_test)

        n_feat = data.n_features = X_train[0].shape[1]

        # --------- CVPR model ----------
        if model_type in ["ED-TCN","AP-TCN"]:
            # for train a softmax classfier, we need one-hot encoded labels, from 0 to 100
            Y_train = [np_utils.to_categorical(y-1, n_classes) for y in y_train]            
            y_train_ = [np.array([y_train[i]]) for i in range(len(y_train))]
            
            # for train a softmax classfier, we need one-hot encoded labels, from 0 to 100
            y_test_ = [np.array([y_test[i]]) for i in range(len(y_test))]
            
            # In order process batches simultaneously all data needs to be of the same length
            # So make all same length and mask out the ends of each.
            n_layers = len(n_nodes)
            max_len = 90 # this should be elaborated
            
            print("Start data masking...")
            X_train_m, M_train = utils.mask_data_one_tensor(X_train, max_len, mask_value=-1)
            X_test_m, M_test = utils.mask_data_one_tensor(X_test, max_len, mask_value=-1)
            Y_train_ = np.array(Y_train)            
            print("Data masking done.")
      
            if model_type == "ED-TCN":
                model, param_str = tf_models.ED_TCN(n_nodes, conv, n_classes, n_feat, max_len, causal=causal, activation='norm_relu', return_param_str=True) 
            elif model_type == "AP-TCN":
                print 'Training AP TCN...'
                model, param_str = tf_models.AP_TCN(n_nodes, conv, n_classes, n_feat, max_len, causal=causal, 
                                        activation='norm_relu', return_param_str=True)               
            
            # summarize the model connection and compilation
            model.summary()
            M_train2 = np.array([[1] for i in range(len(X_train))])
                  
            ### Random shuffle of the training sequence and corresponding labels
#             rand_ind = np.random.permutation(len(X_train_m))
#             X_train_m_shuffle = X_train_m[rand_ind,:,:]
#             Y_train_shuffle = np.array( [Y_train[rand_ind[i]] for i in range(len(Y_train))] )
            ###
                       
            # fitting a model
#             model.fit(X_train_m_shuffle, Y_train_shuffle, nb_epoch=nb_epoch, batch_size=8, verbose=1, shuffle=True, sample_weight=M_train2) 
            model.fit(X_train_m, Y_train_, nb_epoch=nb_epoch, batch_size=8, verbose=1, shuffle=True, sample_weight=M_train2) 
            
            # predict on the dataset
            print("Prediction on training data...")            
            AP_train = model.predict(X_train_m, verbose=0)
            print("Prediction on test data...")
            AP_test = model.predict(X_test_m, verbose=0)
            print("All predictions done...")
            
            # prediction outputs from 0 to 100 while GT contains labels from 1 to 101
            P_train = [p.argmax(1)+1 for p in AP_train] 
            P_test = [p.argmax(1)+1 for p in AP_test]
 
        else:
            print("Model not available:", model_type)

        print(param_str)

        # get the test accuracy 
        cnt = 0
        for i in range(len(P_test)):
            if P_test[i][0] == y_test_[i][0]:
                cnt += 1
        accuracy = float(cnt)/len(P_test)
        print 'Test accuracy on {0}: {1}'.format(split, accuracy)
        accuracies[split_cnt] = accuracy
        
        # get the train accuracy 
        cnt = 0
        for i in range(len(P_train)):
            if P_train[i][0] == y_train_[i][0]:
                cnt += 1
        train_accuracy = float(cnt)/len(P_train)
        print 'Training accuracy on {0}: {1}'.format(split, train_accuracy)
        train_accuracies[split_cnt] = train_accuracy
        
        # save the model
        model.save('../models/AP-TCN_v0.3_{0}_{1}_{2}.h5'.format(dataset,split,architecture))
        
        split_cnt += 1
        
    print 'Mean test accuracy: {0}'.format(np.mean(accuracies))

