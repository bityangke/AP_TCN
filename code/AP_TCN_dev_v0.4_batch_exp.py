import os
import glob
from collections import OrderedDict

import numpy as np

from scipy import io as sio
from sklearn.decomposition import PCA

from keras.utils import np_utils

# TCN imports
import tf_models, ap_datasets, utils, metrics
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

for model_type in ["AP-TCN", "AP-TCN-SanityCheck"]:
    for video_rate in [25, 10]:
        if model_type == "AP-TCN" and video_rate == 25 :
            continue
        for max_len in [90, 225]:
            if video_rate == 10 and max_len == 90:
                continue
            print('=================================================================================')
            print('=================================================================================')
            print('Training... model_type={0}, video_rate={1}, max_len={2}'.format(model_type, video_rate, max_len))
            # Initialize dataset loader & metrics
            if dataset == 'UCF101':
                data = ap_datasets.Dataset(dataset, base_dir, feature_type='fc7')
            else:
                data = ap_datasets.Dataset(dataset, base_dir)

            n_classes = data.n_classes
            test_accuracies = list()
            test_losses = list()
            split_cnt = 0

            # Load data for each split
            for split in data.splits:
                if dataset != 'UCF101':
                    feature_type = "A" if model_type != "SVM" else "X"

                # Load the feature files
                print("Loading data split...")
                # If there exist .npy files, load them
                if ( os.path.exists(base_dir + 'AlexNet-fc7-npy/X_train_ucf_' + split + '_' + str(video_rate) + 'f_to1f' +'.npy') and
                     os.path.exists(base_dir + 'AlexNet-fc7-npy/y_train_ucf_' + split + '_' + str(video_rate) + 'f_to1f' +'.npy') and
                     os.path.exists(base_dir + 'AlexNet-fc7-npy/X_test_ucf_' + split + '_' + str(video_rate) + 'f_to1f' +'.npy') and
                     os.path.exists(base_dir + 'AlexNet-fc7-npy/y_test_ucf_' + split + '_' + str(video_rate) + 'f_to1f' +'.npy') ):
                    print("npy files found.")
                    X_train = np.load(base_dir + 'AlexNet-fc7-npy/X_train_ucf_' + split + '_' + str(video_rate) + 'f_to1f' +'.npy');
                    y_train = np.load(base_dir + 'AlexNet-fc7-npy/y_train_ucf_' + split + '_' + str(video_rate) + 'f_to1f' +'.npy');
                    X_test = np.load(base_dir + 'AlexNet-fc7-npy/X_test_ucf_' + split + '_' + str(video_rate) + 'f_to1f' +'.npy');
                    y_test = np.load(base_dir + 'AlexNet-fc7-npy/y_test_ucf_' + split + '_' + str(video_rate) + 'f_to1f' +'.npy');
                # If there are no .npy files, load .mat files and generate the numpy features
                else:
                    print("npy files not found. Loading from mat files. This would take a while...")
                    X_train, y_train, X_test, y_test = data.load_split(features, split=split,
                                                                        sample_rate=video_rate,
                                                                        feature_type=feature_type)
                    np.save(base_dir + 'AlexNet-fc7-npy/X_train_ucf_' + split + '_' + str(video_rate) + 'f_to1f' +'.npy', X_train);
                    np.save(base_dir + 'AlexNet-fc7-npy/y_train_ucf_' + split + '_' + str(video_rate) + 'f_to1f' +'.npy', y_train);
                    np.save(base_dir + 'AlexNet-fc7-npy/X_test_ucf_' + split + '_' + str(video_rate) + 'f_to1f' +'.npy', X_test);
                    np.save(base_dir + 'AlexNet-fc7-npy/y_test_ucf_' + split + '_' + str(video_rate) + 'f_to1f' +'.npy', y_test);
                print("Loading done.")

                n_feat = data.n_features = X_train[0].shape[1]

                # --------- ICCV model ----------
                # for train a softmax classfier, we need one-hot encoded labels, from 0 to 100
                Y_train = [np_utils.to_categorical(y-1, n_classes) for y in y_train]
                Y_train_ = np.array(Y_train)
                y_train_ = [np.array([y_train[i]]) for i in range(len(y_train))] # no need

                # for train a softmax classfier, we need one-hot encoded labels, from 0 to 100
                Y_test = [np_utils.to_categorical(y-1, n_classes) for y in y_test]
                Y_test_ = np.array(Y_test)
                y_test_ = [np.array([y_test[i]]) for i in range(len(y_test))] # no need

                # In order process batches simultaneously all data needs to be of the same length
                # So make all same length and mask out the ends of each.
                n_layers = len(n_nodes)

                print("Start data masking...")
                X_train_m, _ = utils.mask_data_one_tensor(X_train, max_len, mask_value=-1)
                X_test_m,  _ = utils.mask_data_one_tensor(X_test,  max_len, mask_value=-1)
                print("Data masking done.")

                ###  Random shuffle of the training data and corresponding labels
                ###  This is important since the Keras model.fit function does shuffle
                ###  after sampling the last portion of the training data !!!
                print("Shuffling the training data...")
                rand_ind = np.random.permutation(len(X_train_m))
                X_train_m_shuffle = X_train_m[rand_ind,:,:]
                Y_train_shuffle   = np.array( [Y_train[rand_ind[i]] for i in range(len(Y_train))] )
                print("Shuffling done")

                if model_type == "AP-TCN":
                    print('Training AP-TCN...')
                    model, param_str = tf_models.AP_TCN(n_nodes, conv, n_classes, n_feat, max_len, causal=causal, activation='norm_relu', return_param_str=True)
                elif model_type == "AP-TCN-SanityCheck":
                    model, param_str = tf_models.AP_TCN_SanityCheck(n_nodes, conv, n_classes, n_feat, max_len, causal=causal,
                                                        activation='norm_relu', return_param_str=True)

                #model.summary()
                M_train2 = np.array([[1] for i in range(len(X_train))])

                # fitting a model
                print('model_type={0}, video_rate={1}, max_len={2}, nb_epoch={3}, split={4}'.format(model_type,video_rate,max_len,nb_epoch,split))
                model.fit(X_train_m_shuffle, Y_train_shuffle, nb_epoch=nb_epoch, batch_size=8, verbose=1, shuffle=True, sample_weight=M_train2)

                print(param_str)

                print('Evaluation on blind test dataset')
                [test_loss, test_accuracy] = model.evaluate(X_test_m, Y_test_)
                test_accuracies.append(test_accuracy)
                test_losses.append(test_accuracy)
                print('Test Accuracy: {0}, Test Loss:{1}'.format(test_accuracy, test_loss))

                # save the model
                model.save('../models/{0}_r{1}_l{2}_{3}_{4}_epoch{5}.h5'.format(model_type, video_rate, max_len, dataset, split, nb_epoch))

                split_cnt += 1
                if split_cnt >= 1:
                    break;

            print 'Mean test accuracy: {0}'.format(np.mean(test_accuracies))
            print('=================================================================================')
            print('=================================================================================')
