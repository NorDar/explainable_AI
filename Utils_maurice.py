import os 
import pandas as pd
import numpy as np
import h5py
import pickle as pkl

#import functions_read_data as rdat
# Tensorflow/Keras
import tensorflow as tf
import time
from tensorflow import keras
from tensorflow.keras import layers
import pickle as pkl

from keras.utils import to_categorical
from keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard

from tensorflow.keras.callbacks import ModelCheckpoint

def read_and_split_img_data_andrea_maurice(path_img, path_tab, path_splits, split, check_print = True):   
    # path_img: path to image data
    # path_tab: path to tabular data
    # path_splits: path to splitting definition
    # split: which split to use (1,2,3,4,5,6)
    # check_print: print shapes of data
     
    ## read image data
    with h5py.File(path_img, "r") as h5:
    # with h5py.File(IMG_DIR2 + 'dicom-3d.h5', "r") as h5:
    # both images are the same
        X_in = h5["X"][:]
        Y_img = h5["Y_img"][:]
        Y_pat = h5["Y_pat"][:]
        pat = h5["pat"][:]
    
    X_in = np.expand_dims(X_in, axis = 4)
    ## read tabular data
    dat = pd.read_csv(path_tab, sep=",")       

    ## read splitting file
    andrea_splits = pd.read_csv(path_splits, 
                                sep='\,', header = None, engine = 'python', 
                                usecols = [1,2,3]).apply(lambda x: x.str.replace(r"\"",""))
    andrea_splits.columns = andrea_splits.iloc[0]
    andrea_splits.drop(index=0, inplace=True)
    andrea_splits = andrea_splits.astype({'idx': 'int32', 'spl': 'int32'})
    splitx = andrea_splits.loc[andrea_splits['spl']==split]        

    
    ## extract X and Y and split into train, val, test
    n = []
    for p in pat:
        if p in dat.p_id.values:
            n.append(p)
    n = len(n)

    # match image and tabular data
    X = np.zeros((n, X_in.shape[1], X_in.shape[2], X_in.shape[3], X_in.shape[4]))
    X_tab = np.zeros((n, 13))
    Y_mrs = np.zeros((n))
    Y_eventtia = np.zeros((n))
    p_id = np.zeros((n))

    i = 0
    for j, p in enumerate(pat):
        if p in dat.p_id.values:
            k = np.where(dat.p_id.values == p)[0]
            X_tab[i,:] = dat.loc[k,["age", "sexm", "nihss_baseline", "mrs_before",
                                   "stroke_beforey", "tia_beforey", "ich_beforey", 
                                   "rf_hypertoniay", "rf_diabetesy", "rf_hypercholesterolemiay", 
                                   "rf_smokery", "rf_atrial_fibrillationy", "rf_chdy"]]
            X[i] = X_in[j]
            p_id[i] = pat[j]
            Y_eventtia[i] = Y_pat[j]
            Y_mrs[i] = dat.loc[k, "mrs3"]
            i += 1        
        
    ## all mrs <= 2 are favorable all higher unfavorable
    Y_new = []
    for element in Y_mrs:
        if element in [0,1,2]:
            Y_new.append(0)
        else:
            Y_new.append(1)
    Y_new = np.array(Y_new)
   
    X = np.squeeze(X)
    X = np.float32(X)

    train_idx = splitx["idx"][splitx['type'] == "train"].to_numpy() -1 
    valid_idx = splitx["idx"][splitx['type'] == "val"].to_numpy() - 1 
    test_idx = splitx["idx"][splitx['type'] == "test"].to_numpy() - 1 

    X_train = X[train_idx]
    X_valid = X[valid_idx]
    X_test = X[test_idx]
    
    Y_train = Y_new[train_idx]
    Y_valid = Y_new[valid_idx]
    Y_test = Y_new[test_idx]
         
    X_tab_train = X_tab[train_idx]
    X_tab_valid = X_tab[valid_idx]
    X_tab_test = X_tab[test_idx]
        
        
    ## safe data in table
    results = pd.DataFrame(
        {"p_idx": test_idx+1,
         "p_id": p_id[test_idx],
         "mrs": Y_mrs[test_idx],
         "unfavorable": Y_test
        }
    )
    
    return (X_train, X_valid, X_test, X_tab_train, X_tab_valid, X_tab_test), (Y_train, Y_valid, Y_test), results

# Model for the intercept function: C = number of classes
def mod_baseline(C):
    mod = keras.Sequential(name = "mod_baseline")
    mod.add(keras.Input(shape = (1, )))
    mod.add(keras.layers.Dense(C - 1, activation = "linear", use_bias = False))
    return mod

# Model for linear shift terms
def mod_linear_shift(x):
    mod = keras.Sequential(name = "mod_linear_shift")
    mod.add(keras.Input(shape = (x, )))
    mod.add(keras.layers.Dense(1, activation = "linear", use_bias = False))
    return mod

# Model for complex shift terms
def mod_complex_shift(x):
    mod = keras.Sequential(name = "mod_complex_shift")
    mod.add(keras.Input(shape = (x, )))
    mod.add(keras.layers.Dense(8, activation = "relu"))
    mod.add(keras.layers.Dense(8, activation = "relu"))
    mod.add(keras.layers.Dense(1, activation = "linear", use_bias = False))
    return mod  

def img_model_linear(input_shape, output_shape, activation = "linear"):
    initializer = keras.initializers.he_normal(seed = 2202)
    in_ = keras.Input(shape = input_shape)
    x = keras.layers.Convolution3D(32, kernel_size=(3, 3, 3), padding = 'same', activation = 'relu')(in_)
    x = keras.layers.MaxPooling3D(pool_size=(2, 2, 2))(x)
    x = keras.layers.Convolution3D(32, kernel_size=(3, 3, 3), padding = 'same', activation = 'relu')(x)
    x = keras.layers.MaxPooling3D(pool_size=(2, 2, 2))(x)
    x = keras.layers.Convolution3D(64, kernel_size=(3, 3, 3), padding = 'same', activation = 'relu')(x)
    x = keras.layers.MaxPooling3D(pool_size=(2, 2, 2))(x)
    x = keras.layers.Convolution3D(64, kernel_size=(3, 3, 3), padding = 'same', activation = 'relu')(x)
    x = keras.layers.MaxPooling3D(pool_size=(2, 2, 2))(x)
    x = keras.layers.Flatten()(x)
    x = keras.layers.Dense(128, activation = 'relu')(x)
    x = keras.layers.Dropout(0.3)(x)
    x = keras.layers.Dense(128, activation = 'relu')(x)
    x = keras.layers.Dropout(0.3)(x)
    out_ = keras.layers.Dense(output_shape, activation = activation, use_bias = False)(x) 
    nn_im = keras.Model(inputs = in_, outputs = out_)
    return nn_im

def img_model_linear_final(input_shape, output_shape, activation = "linear"):
    initializer = keras.initializers.he_normal(seed = 2202)
    in_ = keras.Input(shape = input_shape)

    # conv block 0
    x = keras.layers.Convolution3D(32, kernel_size=(3, 3, 3), padding = 'same', activation = 'relu')(in_)
    x = keras.layers.MaxPooling3D(pool_size=(2, 2, 2))(x)
    # conv block 1
    x = keras.layers.Convolution3D(32, kernel_size=(3, 3, 3), padding = 'same', activation = 'relu')(x)
    x = keras.layers.MaxPooling3D(pool_size=(2, 2, 2))(x)
    # conv block 2
    x = keras.layers.Convolution3D(64, kernel_size=(3, 3, 3), padding = 'same', activation = 'relu')(x)
    x = keras.layers.MaxPooling3D(pool_size=(2, 2, 2))(x)
    # conv block 3
    x = keras.layers.Convolution3D(64, kernel_size=(3, 3, 3), padding = 'same', activation = 'relu')(x)
    x = keras.layers.MaxPooling3D(pool_size=(2, 2, 2))(x)

    # cnn to flat connection
    x = layers.GlobalAveragePooling3D()(x) 
    
    # flat block
    x = keras.layers.Dense(128, activation = 'relu')(x)
    x = keras.layers.Dropout(0.3)(x)
    x = keras.layers.Dense(128, activation = 'relu')(x)
    x = keras.layers.Dropout(0.3)(x)
    out_ = keras.layers.Dense(output_shape, activation = activation, use_bias = False)(x) 
    nn_im = keras.Model(inputs = in_, outputs = out_)
    return nn_im


def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def inv_sigmoid(x):
    return np.log(x / (1 - x))




def read_and_split_img_data_andrea_maurice2(path_img, path_tab, path_splits, split, check_print = True):   
    # path_img: path to image data
    # path_tab: path to tabular data
    # path_splits: path to splitting definition
    # split: which split to use (1,2,3,4,5,6)
    # check_print: print shapes of data
     
    ## read image data
    with h5py.File(path_img, "r") as h5:
    # with h5py.File(IMG_DIR2 + 'dicom-3d.h5', "r") as h5:
    # both images are the same
        X_in = h5["X"][:]
        Y_img = h5["Y_img"][:]
        Y_pat = h5["Y_pat"][:]
        pat = h5["pat"][:]
    
    X_in = np.expand_dims(X_in, axis = 4)
    ## read tabular data
    dat = pd.read_csv(path_tab, sep=",")       

    ## read splitting file
    andrea_splits = pd.read_csv(path_splits, 
                                sep='\,', header = None, engine = 'python', 
                                usecols = [1,2,3]).apply(lambda x: x.str.replace(r"\"",""))
    andrea_splits.columns = andrea_splits.iloc[0]
    andrea_splits.drop(index=0, inplace=True)
    andrea_splits = andrea_splits.astype({'idx': 'int32', 'spl': 'int32'})
    splitx = andrea_splits.loc[andrea_splits['spl']==split]        

    
    ## extract X and Y and split into train, val, test
    n = []
    for p in pat:
        if p in dat.p_id.values:
            n.append(p)
    n = len(n)

    # match image and tabular data
    X = np.zeros((n, X_in.shape[1], X_in.shape[2], X_in.shape[3], X_in.shape[4]))
    X_tab = np.zeros((n, 13))
    Y_mrs = np.zeros((n))
    Y_eventtia = np.zeros((n))
    p_id = np.zeros((n))

    i = 0
    for j, p in enumerate(pat):
        if p in dat.p_id.values:
            k = np.where(dat.p_id.values == p)[0]
            X_tab[i,:] = dat.loc[k,["age", "sexm", "nihss_baseline", "mrs_before",
                                   "stroke_beforey", "tia_beforey", "ich_beforey", 
                                   "rf_hypertoniay", "rf_diabetesy", "rf_hypercholesterolemiay", 
                                   "rf_smokery", "rf_atrial_fibrillationy", "rf_chdy"]]
            X[i] = X_in[j]
            p_id[i] = pat[j]
            Y_eventtia[i] = Y_pat[j]
            Y_mrs[i] = dat.loc[k, "mrs3"]
            i += 1        
        
    ## all mrs <= 2 are favorable all higher unfavorable
    Y_new = []
    for element in Y_mrs:
        if element in [0,1,2]:
            Y_new.append(0)
        else:
            Y_new.append(1)
    Y_new = np.array(Y_new)
    p_idxx = np.arange(0, len(p_id))+1
   
    X = np.squeeze(X)
    X = np.float32(X)

    train_idx = splitx["idx"][splitx['type'] == "train"].to_numpy() -1 
    valid_idx = splitx["idx"][splitx['type'] == "val"].to_numpy() - 1 
    test_idx = splitx["idx"][splitx['type'] == "test"].to_numpy() - 1 

    train_idxx = splitx["idx"][splitx['type'] == "train"].to_numpy() 
    valid_idxx = splitx["idx"][splitx['type'] == "val"].to_numpy()
    test_idxx = splitx["idx"][splitx['type'] == "test"].to_numpy()

    X_train = X[train_idx]
    X_valid = X[valid_idx]
    X_test = X[test_idx]
    
    Y_train = Y_new[train_idx]
    Y_valid = Y_new[valid_idx]
    Y_test = Y_new[test_idx]
         
    X_tab_train = X_tab[train_idx]
    X_tab_valid = X_tab[valid_idx]
    X_tab_test = X_tab[test_idx]
        
        
    ## safe data in table
    results2 = pd.DataFrame(
        {"p_idx": test_idx+1,
         "p_id": p_id[test_idx],
         "mrs": Y_mrs[test_idx],
         "unfavorable": Y_test
       }
    )
    
    results = pd.DataFrame(
        {"p_idx": p_idxx,
        "p_id": p_id,
         "mrs": Y_mrs,
         "unfavorable": Y_new 
        }
    )

    results2["status"] = np.where(results2["p_idx"].isin(train_idxx), "train",
                   np.where(results2["p_idx"].isin(valid_idxx), "valid",
                   np.where(results2["p_idx"].isin(test_idxx), "test", None)))

      
    return (train_idx, valid_idx, test_idx, X, X_train, X_valid, X_test, X_tab_train, X_tab_valid, X_tab_test), (Y_train, Y_valid, Y_test), results, results2