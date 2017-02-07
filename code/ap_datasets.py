import os
import glob
import numpy as np
import scipy.ndimage as nd
import scipy.io as sio
import utils
from itertools import groupby
import h5py
from IPython.core.debugger import Pdb # for debugging, jinchoi@vt.edu
pdb = Pdb()

def closest_file(fid, extension=".mat"):
    # Fix occasional issues with extensions (e.g. X.mp4.mat)
    basename = os.path.basename(fid)
    dirname = os.path.dirname(fid)
    dirfiles = os.listdir(dirname)
    
    if basename in dirfiles:
        return fid
    else:
        basename = basename.split(".")[0]
        files = [f for f in dirfiles if basename in f]
        if extension is not None:
            files = [f for f in files if extension in f]
        if len(files) > 0:
            return dirname+"/"+files[0]
        else:
            print("Error: can't find file")


def remove_exts(name, exts):
    for ext in exts:
        name = name.replace(ext, "")
    return name

class Dataset:
    name = ""
    n_classes = None
    n_features = None
    activity = None

    def __init__(self, name="", base_dir="", activity=None, feature_type=None):
        self.name = name
        self.base_dir = os.path.expanduser(base_dir)
        self.feature_type = feature_type

        #Find the number of splits
        if name != "UCF101":
            split_folders = os.listdir(self.base_dir+"splits/{}/".format(self.name))
            self.splits = np.sort([s for s in split_folders if "Split" in s])
        else:
            self.n_classes = 101
            split_files = glob.glob(base_dir+"ucfTrainTestlist/"+"train*.txt")
            index_raw_data = open(self.base_dir+"ucfTrainTestlist/"+"classInd.txt").readlines()
            self.class_index = dict()
            for s in range(len(index_raw_data)):
                tmp = index_raw_data[s].split('\r')[0].split(' ')
                self.class_index[tmp[1].lower()] = int(tmp[0])
            self.splits = list()
            for s in range(len(split_files)):
                split_num = int(split_files[s].split('/')[-1].split('list')[-1].split('.')[0])
                self.splits.append('Split_0' + str(split_num))
        self.n_splits = len(self.splits)

    def feature_path(self, features):
        return os.path.expanduser(self.base_dir+"features/{}/{}/".format(self.name, features))

    def get_files(self, dir_features, split=None):
        if "Split_1" in os.listdir(dir_features):
            files_features = np.sort(os.listdir(dir_features+"/{}/".format(split)))
        else:
            files_features = np.sort(os.listdir(dir_features))
            
        files_features = [f for f in files_features if f.find(".mat")>=0]
        return files_features
     
    def get_files_ucf(self, file_list, dir_features, split=None):
        files_features = list()
        for i in range(len(file_list)):
            subdir_name = file_list[i].split('/')[0]
            filename = file_list[i].split()[0].split('/')[-1].split('.')[0] + '.mat'
            filename = subdir_name + '/alex-{0}_'.format(self.feature_type) + filename
            files_features.append(filename)
            
        files_features = [f for f in files_features if f.find(".mat")>=0]
        return files_features

    def fid2idx(self, files_features, extensions=[".mov", ".mat", ".avi", "rgb-"]):
        return {remove_exts(files_features[i], extensions):i for i in range(len(files_features))}

    def load_split(self, features, split, feature_type="X", sample_rate=1):
        # Setup directory and filenames
        dir_features = self.feature_path(features)
        
        # Get splits for this partion of data
        if self.activity==None:
            if self.name != "UCF101":
                file_train = open(self.base_dir+"splits/{}/{}/train.txt".format(self.name, split)).readlines()
                file_test = open( self.base_dir+"splits/{}/{}/test.txt".format(self.name, split)).readlines()
            else:
                file_train = open(self.base_dir+"ucfTrainTestlist/"+"trainlist{}.txt".format(split.split('_')[-1])).readlines()
                file_test = open(self.base_dir+"ucfTrainTestlist/"+"testlist{}.txt".format(split.split('_')[-1])).readlines()
                dir_features = self.base_dir+"UCF101_AlexNet-{0}-features".format(self.feature_type)
        else:
            file_train = open(self.base_dir+"splits/{}/{}/{}/train.txt".format(self.name, self.activity, split)).readlines()
            file_test = open( self.base_dir+"splits/{}/{}/{}/test.txt".format(self.name, self.activity, split)).readlines()        
            
        file_train = [f.strip() for f in file_train]
        file_test = [f.strip() for f in file_test]     

        # Remove extension
        if  "." in file_train[0]:
            file_train = [".".join(f.split(".")[:-1]) for f in file_train]
            file_test = [".".join(f.split(".")[:-1]) for f in file_test]

        self.trials_train = file_train
        self.trials_test = file_test

        # Get all features
        if self.name != "UCF101":
            files_features = self.get_files(dir_features, split)
        else:
            files_features = self.get_files_ucf(file_train, dir_features, split)
            files_features_test = self.get_files_ucf(file_test, dir_features, split)
            
        X_all, Y_all, X_train, y_train = [], [], [], []
        cnt = 0
        for f in files_features: # loop over samples
            if self.name != "UCF101":
                if "Split_" in os.listdir(dir_features)[-1]:
                    data_tmp = sio.loadmat( closest_file("{}{}/{}".format(dir_features,split, f)) )
                else:
                    data_tmp = sio.loadmat( closest_file("{}/{}".format(dir_features, f)) )
                X_all += [ data_tmp[feature_type].astype(np.float32) ]
                Y_all += [ np.squeeze(data_tmp["Y"]) ]
            else:
                data_tmp = sio.loadmat( closest_file("{}/{}".format(dir_features, f)) ) 
                tmp_list = list()
                for i in range(len(data_tmp[feature_type])): # loop over time steps
                    # tmp_list.append(np.array(data_tmp[feature_type][i][0].reshape(4096).tolist()))
                    
                    temp = data_tmp[feature_type][i][0].astype(np.float32).reshape(4096).tolist()
                    tmp_list.append( np.array(temp) )
                    # tmp_list.append( data_tmp[feature_type][i][0].astype(np.float32).reshape(4096).tolist() )
                    
                X_train += [ np.array(tmp_list).astype(np.float32) ] 
                # X_train += [ tmp_list ] 
                if cnt%100 == 0:
                    print(cnt)
                cnt += 1
                y_train += [ self.class_index[f.split('_')[2].lower()] ]
                # if cnt >= 300:
                #     break
                #y_train += [ np.array(self.class_index[f.split('_')[2].lower()]) ]
        
        if self.name == "UCF101":
            X_test, y_test = [], []
            cnt = 0
            for f in files_features_test:
                data_tmp = sio.loadmat( closest_file("{}/{}".format(dir_features, f)) ) 
                tmp_list = list()
                for i in range(len(data_tmp[feature_type])):
                    #tmp_list.append(np.array(data_tmp[feature_type][i][0].reshape(4096).tolist()))
                    temp = data_tmp[feature_type][i][0].astype(np.float32).reshape(4096).tolist()
                    tmp_list.append( np.array(temp) )
                    # tmp_list.append(data_tmp[feature_type][i][0].astype(np.float32).reshape(4096).tolist())
                #X_test += [ np.array(tmp_list).astype(np.float32) ] 
                X_test += [ np.array(tmp_list).astype(np.float32) ] 
                if cnt%100 == 0:
                    print(cnt)
                cnt += 1
                y_test += [ self.class_index[f.split('_')[2].lower()] ]
                # if cnt >= 100:
                #     break
                
        # Make sure axes are correct (TxF not FxT for F=feat, T=time)
        if self.name != "UCF101":
            if X_all[0].shape[0]!=Y_all[0].shape[0]:
                X_all = [x.T for x in X_all]
            self.n_features = X_all[0].shape[1]
            self.n_classes = len(np.unique(np.hstack(Y_all)))

            # Make sure labels are sequential
            if self.n_classes != np.hstack(Y_all).max()+1:
                Y_all = utils.remap_labels(Y_all)
                print("Reordered class labels")
            # Subsample the data
            if sample_rate > 1:
                X_all, Y_all = utils.subsample(X_all, Y_all, sample_rate, dim=0)

            # ------------Train/test Splits---------------------------
            # Split data/labels into train/test splits
            fid2idx = self.fid2idx(files_features)
            X_train = [X_all[fid2idx[f]] for f in file_train if f in fid2idx]
            X_test = [X_all[fid2idx[f]] for f in file_test if f in fid2idx]

            y_train = [Y_all[fid2idx[f]] for f in file_train if f in fid2idx]
            y_test = [Y_all[fid2idx[f]] for f in file_test if f in fid2idx]
        else:
            self.n_features = X_train[0].shape[1]
            # self.n_features = len(X_train[0][0]) #X_train[0]
            self.n_classes = len(np.unique(np.hstack(y_train)))
            
             # Subsample the data
            if sample_rate > 1:
                X_train = utils.subsample_one_vector(X_train, sample_rate, dim=0)
                X_test = utils.subsample_one_vector(X_test, sample_rate, dim=0)

        if len(X_train)==0:
            print("Error loading data")

        return X_train, y_train, X_test, y_test
    
    # a function for trimming video features
    def split_actions(self, seq):
        start = 0
        duration = 0
        action_labels = list()
        
        for k,g in groupby(seq):
            this_group = list(g)
            label = k
            start += duration
            duration = len(this_group)
            end = start + duration - 1
            #print 'label:{0}, start:{1}, end:{2} duration:{3}'.format(label, start, end, duration)
            action_labels.append([label,start,end,duration])
            
        return action_labels