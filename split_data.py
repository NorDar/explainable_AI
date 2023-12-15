import logging
import h5py
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import StratifiedKFold


def load_transform_dataset(path_img:str, 
                 path_tab:str, 
                 only_non_tia:bool, 
                 output_dir:str,
                 relevant_features:list,
                 favorable_mrs:list,
                 n_splits:int,
                 split_version:str = None):
    
    # Reading the images
    with h5py.File(path_img, "r") as h5:
        X_in = h5["X"][:]
        Y_img = h5["Y_img"][:]
        Y_pat = h5["Y_pat"][:]
        pat = h5["pat"][:]

    X_in = np.expand_dims(X_in, axis = 4)
    logging.info("image shape in: ", X_in.shape)
    logging.info(f"image min: {X_in.min()}, max: {X_in.max()}, mean: {X_in.mean()}, std: {X_in.std()}")

    # Read tabular data
    dat = pd.read_csv(path_tab, sep=",")
    logging.info("tabular shape in: ", dat.shape)

    # Get original data    
    n = []
    for p in pat:
        if p in dat.p_id.values:
            n.append(p)
    n = len(n)

    # Match image and tabular data
    # TODO: Can be optimized: preinitialization neccessary?
    X = np.zeros((n, X_in.shape[1], X_in.shape[2], X_in.shape[3], X_in.shape[4]))
    X_tab = np.zeros((n, len(relevant_features)))
    Y_mrs = np.zeros((n))
    Y_eventtia = np.zeros((n))
    p_id = np.zeros((n))


    non_matched_patients = []
    i = 0
    for j, p in enumerate(pat):
        if p in dat.p_id.values:
            k = np.where(dat.p_id.values == p)[0]
            X_tab[i,:] = dat.loc[k, relevant_features]
            X[i] = X_in[j]
            p_id[i] = pat[j]
            Y_eventtia[i] = Y_pat[j]
            Y_mrs[i] = dat.loc[k, "mrs3"]
            i += 1
        else:
            non_matched_patients.append(p)
    p_id = p_id.astype("int")
            
    logging.info("X img out shape: ", X.shape)
    logging.info("X tab out shape: ", X_tab.shape)
    logging.info("Y mrs out shape: ", Y_mrs.shape)
    if non_matched_patients:
        logging.info(f"{len(non_matched_patients)} patients were in the images but not in the tabular data and thus excluded.")
        logging.debug("Patients that were not added:")
        logging.debug(non_matched_patients)



    ## all mrs <= 2 are favorable all higher unfavorable
    Y_new = []
    for element in Y_mrs:
        if element in favorable_mrs:
            Y_new.append(0)
        else:
            Y_new.append(1)
    Y_new = np.array(Y_new)

    # Replace original patient id with 1:len(p_id)+1
    p_idx = np.arange(0, len(p_id))+1

    # plot_count_mrs_binned_mrs(Y_mrs, Y_new) # TODO: Removed plot HERE

    # reduce the data to only non-TIA patients if desired
    if only_non_tia:
        logging.info("Reducing the data to only non-TIA patients.")
        p_idx = p_idx[Y_eventtia == 1]
        X_tab = X_tab[Y_eventtia == 1]
        Y_mrs = Y_mrs[Y_eventtia == 1]
        p_id = p_id[Y_eventtia == 1]
        Y_new = Y_new[Y_eventtia == 1]
        Y_eventtia = Y_eventtia[Y_eventtia == 1]
        # plot_count_mrs_binned_mrs(Y_mrs, Y_new) # TODO: Removed plot HERE


    # Safe id translation in pandas df
    id_tab = pd.DataFrame(
        {"p_idx": p_idx,
        "p_id": p_id,
        "mrs": Y_mrs,
        "unfavorable": Y_new
        }
    )

    # Create StratifiedKFold object.
    # 10 Fold V0 random_state 100
    # 10 Fold V1 random_state 999
    # 10 Fold V2 random_stat3 500
    # 10 Fold V3 random_state 200
    split_dict = {
        "V0": 100,
        "V1": 999,
        "V2": 500,
        "V3": 200
    }

    if not split_version:
        logging.warning("Split version is not set so default value for random seed of split 'V3' is used")
    elif split_version not in split_dict:
        logging.warning(f"Split version '{split_version}' is not in {split_dict}. Using default value of 'V3'")
    else:
        logging.info(f"Using random seed for split '{split_version}: {split_version.get(split_version)}'")
    
    
    skf = StratifiedKFold(n_splits=n_splits, 
                          shuffle=True, 
                          random_state=split_dict.get(split_version, 200))
    folds = []
    
    # TODO: Here we only spit out info for binary... Ordinal output (Y_mrs) not handled.
    logging.info("Splitting by binned target variable Y_new:") 
    for train_index, test_index in skf.split(p_id, Y_new): # 10 Fold sigmoid stratified with Outcome Good/Bad (V0, V2, V3)
    # for train_index, test_index in skf.split(p_id, Y_mrs): # 10 Fold sigmoid stratified with Outcome MRS (V1)
        folds.append(p_id[test_index])
        logging.info(f"Current split has {sum(Y_new[train_index])} bad outcomes in the train samples and {sum(Y_new[test_index])} bad outcomes in the test samples.")


    for i, fold in enumerate(folds):
        logging.info(f"{len(fold)} samples in fold {i}")
        id_tab["fold" + str(i)] = "train" 
        
        # increment for val (+5 so that no fold has only 40 in train & test)
        j = i+5
        if j >= len(folds):
            j = j-10

        id_tab.loc[id_tab["p_id"].isin(fold), "fold"+str(i)] = "test"
        id_tab.loc[id_tab["p_id"].isin(folds[j]), "fold"+str(i)] = "val"



    # Save Data
        
    # id_tab.to_csv(OUTPUT_DIR + "10Fold_ids_V0.csv",  index=False)
    # id_tab.to_csv(OUTPUT_DIR + "10Fold_ids_V1.csv",  index=False)
    # id_tab.to_csv(OUTPUT_DIR + "10Fold_ids_V2.csv",  index=False)
    id_tab.to_csv(OUTPUT_DIR + "10Fold_ids_V3.csv",  index=False)


    # Is the same for all versions but is generated nonetheless, could als be generated only once
    X = X.squeeze()
    X = np.float32(X)

    np.save(OUTPUT_DIR + "prepocessed_dicom_3d.npy", X)



    # Analyze Data
    # id_tab = pd.read_csv(OUTPUT_DIR + "10Fold_ids_V0.csv", sep=",")
    # id_tab = pd.read_csv(OUTPUT_DIR + "10Fold_ids_V1.csv", sep=",")
    # id_tab = pd.read_csv(OUTPUT_DIR + "10Fold_ids_V2.csv", sep=",")
    id_tab = pd.read_csv(OUTPUT_DIR + "10Fold_ids_V3.csv", sep=",")
    X = np.load(OUTPUT_DIR + "prepocessed_dicom_3d.npy")


    id_tab["unfavorable"].value_counts()

    for i in range(10):
        fig, (ax1, ax2) = plt.subplots(1,2)
        sns.countplot(x = id_tab[id_tab["fold"+str(i)]=="test"].mrs, ax = ax1)
        sns.countplot(x = id_tab[id_tab["fold"+str(i)]=="test"].unfavorable, ax = ax2)

    # Check Images
        

    patient = 460
    index1 = id_tab[id_tab.p_id == patient].p_idx.values[0] -1
    # index1 = id_tab[id_tab.p_id == patient].index
    print(index1)
    index2 = np.argwhere(pat == patient).squeeze()
    print(index2)



    im1 = X[index1].astype("float64")
    im2 = X_in.squeeze()[index2].astype("float64")
    np.allclose(im1, im2)

def plot_count_mrs_binned_mrs(Y_mrs, Y_new):
    fig, (ax1, ax2) = plt.subplots(1,2, figsize=(15, 8))
    sns.countplot(x = Y_mrs, ax = ax1)
    sns.countplot(x = Y_new, ax = ax2)

    print("Left: Distribution of the MRS score")
    print("Right: Distribution of the binary outcome (MRS>=2)")
    print(f"Binary count: Good outcome: {sum(Y_new == 0)}, Bad outcome{sum(Y_new == 1)}")

def main():
    IMG_DIR = "/tf/notebooks/hezo/stroke_perfusion/data/"
    path_img = IMG_DIR + 'dicom_2d_192x192x3_clean_interpolated_18_02_2021_preprocessed2.h5'
    path_tab = IMG_DIR + 'baseline_data_zurich_prepared.csv'

    OUTPUT_DIR = "/tf/notebooks/schnemau/xAI_stroke_3d/data/"

    # should only non TIA (transient ischemic attack) patients be included?
    only_non_tia = True

    relevant_features = ["age", "sexm", "nihss_baseline", "mrs_before",
                        "stroke_beforey", "tia_beforey", "ich_beforey", 
                        "rf_hypertoniay", "rf_diabetesy", "rf_hypercholesterolemiay", 
                        "rf_smokery", "rf_atrial_fibrillationy", "rf_chdy"]

    d = load_transform_dataset(path_img=path_img, 
                     path_tab=path_tab, 
                     only_non_tia=only_non_tia, 
                     output_dir=OUTPUT_DIR,
                     relevant_features=relevant_features,
                     favorable_mrs=[0,1,2],
                     n_splits=10,
                     split_version='V3')
    

    

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    main()