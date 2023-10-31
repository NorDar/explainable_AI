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


def ontram(mod_baseline, mod_shift = None):
    # mod_baseline: keras model for the intercept term
    # mod_shift: list of keras models for the shift terms

    mod_baseline_input = mod_baseline.input
    mod_baseline_output = mod_baseline.output

    if mod_shift == None:
        mod_input = [mod_baseline_input]
        mod_output = [mod_baseline_output]
    else:
        if(not isinstance(mod_shift, list)):
            mod_shift = [mod_shift]
        n_shift = len(mod_shift)
        mod_shift_input = [m.input for m in mod_shift]
        mod_shift_output = [m.output for m in mod_shift]
        if n_shift == 1:
            mod_input = [mod_baseline_input] + mod_shift_input
            mod_output = [mod_baseline_output] + mod_shift_output
        elif n_shift >= 2:
            mod_input = [mod_baseline_input] + mod_shift_input
            mod_shift_output = keras.layers.add(mod_shift_output)
            mod_output = [mod_baseline_output, mod_shift_output]
        mod_output = keras.layers.concatenate(mod_output)
        
    mod = keras.Model(inputs = mod_input, outputs = mod_output)
    mod.mod_baseline = mod_baseline
    mod.mod_shift = mod_shift
    
    return mod

# Transform the raw intercept function to ensure that the transformation function is increasing
def to_theta(gamma, batch_size):
    theta0 = tf.constant(-np.inf, shape = (batch_size, 1))
    theta1 = tf.reshape(gamma[:, 0], shape = (batch_size, 1))
    thetaK = tf.constant(np.inf, shape = (batch_size, 1))
    thetak = tf.math.exp(gamma[:, 1:])
    thetak = tf.math.cumsum(thetak, axis = 1)
    thetas = tf.concat([theta0, theta1, theta1 + thetak, thetaK], axis = 1)
    return thetas

# Compute the negative log likelihood
def nll(y_true, y_pred, C, batch_size):
    mod_baseline_pred = y_pred[:, :(C - 1)]
    mod_baseline_pred = to_theta(mod_baseline_pred, batch_size)
    mod_shift_pred = y_pred[:, (C - 1):]
    
    yu = tf.math.argmax(y_true, axis = 1, output_type = tf.int32) + 1 #labels of class k
    yl = yu -1 #labels of class k-1
    yu = tf.reshape(yu, shape = (batch_size, 1))
    yl = tf.reshape(yl, shape = (batch_size, 1))
    idx = tf.range(batch_size, dtype = tf.int32) #index to get the theta values
    idx = tf.reshape(idx, shape = (batch_size, 1))
    idx_yu = tf.concat((idx, yu), axis = 1)
    idx_yl = tf.concat((idx, yl), axis = 1)
    thetau = tf.gather_nd(mod_baseline_pred, indices = idx_yu)
    thetal = tf.gather_nd(mod_baseline_pred, indices = idx_yl)
    
    if(mod_shift_pred.shape[1] == 0):
        lli = tf.sigmoid(thetau) - tf.sigmoid(thetal)
    else:
        mod_shift_pred = tf.reshape(mod_shift_pred, shape = (batch_size,))
        lli = tf.sigmoid(thetau - mod_shift_pred) - tf.sigmoid(thetal - mod_shift_pred)
    nll = -tf.reduce_mean(tf.math.log(lli + 1e-16)) # epsilon to make sure to get no 0 in the log function
    return nll

def ontram_loss(C, batch_size):
    def loss(y_true, y_pred):
        return nll(y_true, y_pred, C, batch_size)
    return loss


# metrics
def ontram_acc(C, batch_size):
    def acc(y_true, y_pred):
        mod_baseline_pred = y_pred[:, :(C - 1)]
        mod_baseline_pred = to_theta(mod_baseline_pred, batch_size)
        mod_shift_pred = y_pred[:, (C - 1):]
        if(mod_shift_pred.shape[1] == 0):
            cdf = tf.sigmoid(mod_baseline_pred)
        else:
            cdf = tf.sigmoid(mod_baseline_pred - mod_shift_pred)
        dens = cdf[:, 1:] - cdf[:, :-1]
        return tf.keras.metrics.categorical_accuracy(y_true, dens)
    return acc

def ontram_auc(C, batch_size):
    k_auc = tf.keras.metrics.AUC()
    def auc(y_true, y_pred):
        mod_baseline_pred = y_pred[:, :(C - 1)]
        mod_baseline_pred = to_theta(mod_baseline_pred, batch_size)
        mod_shift_pred = y_pred[:, (C - 1):]
        if(mod_shift_pred.shape[1] == 0):
            cdf = tf.sigmoid(mod_baseline_pred)
        else:
            cdf = tf.sigmoid(mod_baseline_pred - mod_shift_pred)
        dens = cdf[:, 1:] - cdf[:, :-1]
        y_true = tf.argmax(y_true, axis = 1)
        dens = tf.argmax(dens, axis = 1)
        return k_auc(y_true, dens)
    return auc


def predict_ontram(mod, data):
    # data = tuple or tf dataset
    if(type(data) == tuple):
        x, y_true = data
        y_pred = mod.predict(x)
    else:
        y_true = np.concatenate([y for x, y in data], axis=0)
        y_pred = mod.predict(data)
    
    batch_size = y_true.shape[0]
    nclasses = y_true.shape[1]
    
    mod_baseline_pred = y_pred[:, :(C - 1)]
    mod_baseline_pred = to_theta(mod_baseline_pred, batch_size)
    mod_shift_pred = y_pred[:, (C - 1):]
    
    if(mod_shift_pred.shape[1] == 0):
        cdf = tf.sigmoid(mod_baseline_pred)
    else:
        cdf = tf.sigmoid(mod_baseline_pred - mod_shift_pred)
    pdf = cdf[:, 1:] - cdf[:, :-1]
    pred_class = tf.argmax(pdf, axis = 1)
    nll_ = nll(y_true, y_pred, C, batch_size)
    return {"cdf": cdf.numpy(), "pdf": pdf.numpy(), "pred_class": pred_class.numpy(), "nll": nll_.numpy()}


def get_parameters(mod):
    # intercept
    w_intercept = [layer.get_weights() for layer in mod.mod_baseline.layers]
    # shift parameters
    w_shift_list = []
    if(mod.mod_shift != None):
        for model in mod.mod_shift:
            w_shift = [layer.get_weights() for layer in mod.mod_shift.layers]
            w_shift_list.append(w_shift)
    return {"intercept": w_intercept, "shift": w_shift_list}


def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def inv_sigmoid(x):
    return np.log(x / (1 - x))




