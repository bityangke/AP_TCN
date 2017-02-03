
# coding: utf-8

# In[1]:

import os
from collections import OrderedDict

import numpy as np
#import matplotlib.pylab as plt
#get_ipython().magic(u'matplotlib inline')

from scipy import io as sio
import sklearn.metrics as sm
from sklearn.svm import LinearSVC

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
dataset = ["50Salads", "JIGSAWS", "MERL", "GTEA"][0]
granularity = ["eval", "mid"][1]
sensor_type = ["video", "sensors"][0]

# Set model and parameters
model_type = ["ED-TCN","AP-TCN"][1]
# causal or acausal? (If acausal use Bidirectional LSTM)
causal = [False, True][0]

# How many latent states/nodes per layer of network
# Only applicable to the TCNs. The ECCV and LSTM  model suses the first element from this list.
n_nodes = [64, 96]
nb_epoch = 200
video_rate = 3
conv = {'50Salads':25, "JIGSAWS":20, "MERL":5, "GTEA":25}[dataset]

# Which features for the given dataset
features = "SpatialCNN"
bg_class = 0 if dataset is not "JIGSAWS" else None

if dataset == "50Salads":
    features = "SpatialCNN_" + granularity

data = ap_datasets.Dataset(dataset, base_dir)
trial_metrics = metrics.ComputeMetrics(overlap=.1, bg_class=bg_class)

device_type = "/gpu:1"


# In[12]:

#from IPython.core.debugger import Pdb # for debugging, jinchoi@vt.edu
#pdb = Pdb()
import traceback, pdb
pdb.pm()

# ------------------------------------------------------------------
# Evaluate using different filter lengths
if 1:
# for conv in [5, 10, 15, 20]:
    # Initialize dataset loader & metrics
    data = ap_datasets.Dataset(dataset, base_dir)
    trial_metrics = metrics.ComputeMetrics(overlap=.1, bg_class=bg_class)

    # Load data for each split
    for split in data.splits:
        if sensor_type=="video":
            feature_type = "A" if model_type != "SVM" else "X"
        else:
            feature_type = "S"

        X_train, y_train, X_test, y_test = data.load_split(features, split=split,
                                                            sample_rate=video_rate,
                                                            feature_type=feature_type)

        if trial_metrics.n_classes is None:
            trial_metrics.set_classes(data.n_classes)

        n_classes = data.n_classes
        print '(n_classes:{0})'.format(n_classes)
        train_lengths = [x.shape[0] for x in X_train]
        test_lengths = [x.shape[0] for x in X_test]
        n_train = len(X_train)
        n_test = len(X_test)

        n_feat = data.n_features
        print '(# Feat:{0})'.format(n_feat)

        # --------- CVPR model ----------
        if model_type in ["tCNN", "ED-TCN", "DilatedTCN", "TDNN", "AP-TCN"]:
            # Go from y_t = {from 1 to C} to one-hot vector (e.g. y_t = [0, 0, 1, 0])
            #Y_train = [np_utils.to_categorical(y, n_classes) for y in y_train]
            #Y_test = [np_utils.to_categorical(y, n_classes) for y in y_test]

            labels = list()
            X_train_f = list()
            num_frames = 0
            num_actions = 0
            for i in range(len(y_train)):
                #print 'data {0}'.format(i+1)
                tmp = np.array(data.split_actions(y_train[i]))

                if i >= 1:
                    labels = np.vstack([labels,tmp])
                else:
                    labels = tmp;
                #labels.append(tmp)
                for j in range(len(tmp)):
                    X_train_f.append(X_train[i][tmp[j,1]:tmp[j,2]+1])
                    num_frames += len(X_train[i][tmp[j,1]:tmp[j,2]+1])
                    num_actions += 1

            Y_train = [np_utils.to_categorical(y, n_classes) for y in labels[:,0]]

            labels = list()
            X_test_f = list()
            num_frames = 0
            num_actions = 0
            for i in range(len(y_test)):
                #print 'data {0}'.format(i+1)
                tmp = np.array(data.split_actions(y_test[i]))

                if i >= 1:
                    labels = np.vstack([labels,tmp])
                else:
                    labels = tmp;
                #labels.append(tmp)
                for j in range(len(tmp)):
                    X_test_f.append(X_test[i][tmp[j,1]:tmp[j,2]+1])
                    num_frames += len(X_test[i][tmp[j,1]:tmp[j,2]+1])
                    num_actions += 1

            Y_test = [np_utils.to_categorical(y, n_classes) for y in labels[:,0]]

            # In order process batches simultaneously all data needs to be of the same length
            # So make all same length and mask out the ends of each.
            n_layers = len(n_nodes)
            max_len = max(np.max(train_lengths), np.max(test_lengths))
            max_len = int(np.ceil(max_len / (2**n_layers)))*2**n_layers

            max_len = 128 # this should be elaborated

            X_train_m, Y_train_, M_train = utils.mask_data(X_train_f, Y_train, max_len, mask_value=-1)
            X_test_m, Y_test_, M_test = utils.mask_data(X_test_f, Y_test, max_len, mask_value=-1)

            Y_train_ = np.array(Y_train)
            #Y_train_ = Y_train_.reshape(Y_train_.shape[0],Y_train_.shape[2])

            Y_test_ = np.array(Y_test)
            #Y_test_ = Y_test_.reshape(Y_test_.shape[0],Y_test_.shape[2])

            if model_type == "ED-TCN":
                model, param_str = tf_models.ED_TCN(n_nodes, conv, n_classes, n_feat, max_len, causal=causal, activation='norm_relu', return_param_str=True)
                #model, param_str = tf_models.ED_TCN_atrous(n_nodes, conv, n_classes, n_feat, max_len, causal=causal, activation='norm_relu', return_param_str=True)
                model.summary()
            elif model_type == "AP-TCN":
                model, param_str = tf_models.AP_TCN(n_nodes, conv, n_classes, n_feat, max_len, causal=causal,
                                        activation='norm_relu', return_param_str=True)

            print("before fit")
            print "shape of Y_train_:{0}".format(Y_train_.shape)
            model.fit(X_train_m, Y_train_, nb_epoch=nb_epoch, batch_size=8, verbose=1, sample_weight=M_train[:,:,0])
            print("after fit")

            print("1")
            AP_train = model.predict(X_train_m, verbose=0)
            print("2")
            AP_test = model.predict(X_test_m, verbose=0)
            print("3")
            AP_train = utils.unmask(AP_train, M_train)
            print("4")
            AP_test = utils.unmask(AP_test, M_test)
            print("5")

            P_train = [p.argmax(1) for p in AP_train]
            P_test = [p.argmax(1) for p in AP_test]

        else:
            print("Model not available:", model_type)

        param_str = "_".join([granularity, sensor_type, param_str])
        print(param_str)

        # --------- Metrics ----------
        trial_metrics.add_predictions(split, P_test, y_test)
        trial_metrics.print_trials()
        print()

        # ----- Save predictions -----
        if save_predictions:
            dir_out = os.path.expanduser(base_dir+"/predictions/{}/{}/".format(dataset,param_str))

            # Make sure folder exists
            if not os.path.isdir(dir_out):
                os.makedirs(dir_out)

            out = {"P":P_test, "Y":y_test, "S":AP_test}
            sio.savemat( dir_out+"/{}.mat".format(split), out)

        # ---- Viz predictions -----
        if viz_predictions:
            max_classes = data.n_classes - 1
            # # Output all truth/prediction pairs
            plt.figure(split, figsize=(20,10))
            P_test_ = np.array(P_test)/float(n_classes-1)
            y_test_ = np.array(y_test)/float(n_classes-1)
            for i in range(len(y_test)):
                P_tmp = np.vstack([y_test_[i], P_test_[i]])
                plt.subplot(n_test,1,i+1); imshow_(P_tmp, vmin=0, vmax=1)
                plt.xticks([])
                plt.yticks([])
                acc = np.mean(y_test[i]==P_test[i])*100
                plt.ylabel("{:.01f}".format(acc))
                # plt.title("Acc: {:.03}%".format(100*np.mean(P_test[i]==y_test[i])))

        # ---- Viz weights -----
        if viz_weights and model_type is "TCN":
            # Output weights at the first layer
            plt.figure(2, figsize=(15,15))
            ws = model.get_weights()[0]
            for i in range(min(36, len(ws.T))):
                plt.subplot(6,6,i+1)
                # imshow_(model.get_weights()[0][i][:,:,0]+model.get_weights()[1][i])
                imshow_(np.squeeze(ws[:,:,:,i]).T)
                # Output weights at the first layer

            for l in range(2*n_layers):
                plt.figure(l+1, figsize=(15,15))
                ws = model.get_weights()[l*2]
                for i in range(min(36, len(ws.T))):
                    plt.subplot(6,6,i+1)
                    # imshow_(model.get_weights()[0][i][:,:,0]+model.get_weights()[1][i])
                    imshow_(np.squeeze(ws[:,:,:,i]).T)
                    # Output weights at the first layer

    print()
    trial_metrics.print_scores()
    trial_metrics.print_trials()
    print()
