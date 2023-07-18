import os
import h5py
# import zipfile
import pandas as pd
import numpy as np


# not efficient when data is big
def read_and_split_img_data_andrea(path_img, path_tab, path_splits, split, check_print = True):     
     
    ## read image data
    with h5py.File(path_img, "r") as h5:
    # with h5py.File(IMG_DIR2 + 'dicom-3d.h5', "r") as h5:
    # both images are the same
        X_in = h5["X"][:]
        Y_img = h5["Y_img"][:]
        Y_pat = h5["Y_pat"][:]
        pat = h5["pat"][:]
    
    X_in = np.expand_dims(X_in, axis = 4)
    if check_print:
        print("image shape in: ", X_in.shape)
        print("image min, max, mean, std: ", X_in.min(), X_in.max(), X_in.mean(), X_in.std())
        
    
    ## read tabular data
    dat = pd.read_csv(path_tab, sep=",")
    if check_print:
        print("tabular shape in: ", dat.shape)
       

    ## read splitting file
    andrea_splits = pd.read_csv(path_splits, 
                                sep='\,', header = None, engine = 'python', 
                                usecols = [1,2,3]).apply(lambda x: x.str.replace(r"\"",""))
    andrea_splits.columns = andrea_splits.iloc[0]
    andrea_splits.drop(index=0, inplace=True)
    andrea_splits = andrea_splits.astype({'idx': 'int32', 'spl': 'int32'})
    splitx = andrea_splits.loc[andrea_splits['spl']==split]        
    if check_print:
        print("split file shape in: ", splitx.shape)
        
    
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
    if check_print:
        print("X tab out shape: ", X_tab.shape)
        print("Y mrs out shape: ", Y_mrs.shape)
        
        
    ## all mrs <= 2 are favorable all higher unfavorable
    Y_new = []
    for element in Y_mrs:
        if element in [0,1,2]:
            Y_new.append(0)
        else:
            Y_new.append(1)
    Y_new = np.array(Y_new)
    
    
    # # Split data into training set and test set "old"
    # X = np.squeeze(X)
    # X = np.float32(X)

    # rng = check_random_state(42)
    # X_train, X_test, y_train, y_test = train_test_split(X, Y_eventtia, train_size=0.8, random_state=rng)
    # X_test, X_valid, y_test, y_valid = train_test_split(X_test, y_test, train_size=0.5, random_state=rng)

    # print(X_train.shape, X_valid.shape, X_test.shape)
    # print(y_train.shape, y_valid.shape, y_test.shape)
    
    
    ## Split data into training set and test set "split" as defined by function
    X = np.squeeze(X)
    X = np.float32(X)

    train_idx = splitx["idx"][splitx['type'] == "train"].to_numpy() -1 
    valid_idx = splitx["idx"][splitx['type'] == "val"].to_numpy() - 1 
    test_idx = splitx["idx"][splitx['type'] == "test"].to_numpy() - 1 

    X_train = X[train_idx]
    # y_train = Y_eventtia[train_idx]
    y_train = Y_new[train_idx]
    X_valid = X[valid_idx]
    # y_valid = Y_eventtia[valid_idx]
    y_valid = Y_new[valid_idx]
    X_test = X[test_idx]
    # y_test = Y_eventtia[test_idx]
    y_test = Y_new[test_idx]

    if check_print:
        print("End shapes X (train, val, test): ", X_train.shape, X_valid.shape, X_test.shape)
        print("End shapes y (train, val, test): ", y_train.shape, y_valid.shape, y_test.shape)
        
        
    ## safe data in table
    results = pd.DataFrame(
        {"p_idx": test_idx+1,
         "p_id": p_id[test_idx],
         "mrs": Y_mrs[test_idx],
         "unfavorable": y_test
        }
    )
    
    return (X_train, X_valid, X_test), (y_train, y_valid, y_test), results


def split_data(id_tab, X, fold):
    
    # define indices of train, val, test
    train_idx_tab = id_tab[id_tab["fold" + str(fold)] == "train"]
    valid_idx_tab = id_tab[id_tab["fold" + str(fold)] == "val"]
    test_idx_tab = id_tab[id_tab["fold" + str(fold)] == "test"]
    
    # for X and y it is not the same, because X is defined for all valid patients,
    # but id_tab is only defined for patients with a stroke (no tia) in V3.
    # In V0, V1 and V2 X and id_tab are the same.
    
    # define data
    X_train = X[train_idx_tab.p_idx.to_numpy() - 1]
    y_train = id_tab["unfavorable"].to_numpy()[train_idx_tab.index.to_numpy()]
    X_valid = X[valid_idx_tab.p_idx.to_numpy() - 1]
    y_valid = id_tab["unfavorable"].to_numpy()[valid_idx_tab.index.to_numpy()]
    X_test = X[test_idx_tab.p_idx.to_numpy() - 1]
    y_test = id_tab["unfavorable"].to_numpy()[test_idx_tab.index.to_numpy()]
    
    return (X_train, X_valid, X_test), (y_train, y_valid, y_test)


# def normalize(volume):
#     """Normalize the volume"""
#     min = np.min(volume)
#     max = np.max(volume) 
#     volume = (volume - min) / (max - min)
#     volume = volume.astype("float32")
#     return volume

# X_in = np.array([normalize(img) for img in X_in])
# print(X_in.shape, X_in.min(), X_in.max(), X_in.mean(), X_in.std())
