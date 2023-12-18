from typing import List, Tuple
import logging
import h5py
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import StratifiedKFold


def load_transform_dataset(
        tabular_input_path:str,
        image_input_path:str, 
        tabular_output_path:str,
        image_output_path:str,
        only_non_tia:bool,
        relevant_features:list,
        favorable_mrs:list,
        n_splits:int,
        split_version:str,
        ) -> None:
    
    # Reading the images
    with h5py.File(image_input_path, "r") as h5:
        X_in = h5["X"][:]
        Y_img = h5["Y_img"][:]
        Y_pat = h5["Y_pat"][:]
        pat = h5["pat"][:]

    X_in = np.expand_dims(X_in, axis = 4)
    logging.info(f"image shape in: {X_in.shape}")
    logging.info(f"image min: {X_in.min()}, max: {X_in.max()}, mean: {X_in.mean()}, std: {X_in.std()}")

    # Read tabular data
    dat = pd.read_csv(tabular_input_path, sep=",")
    logging.info(f"tabular shape in: {dat.shape}")

    # Get original data    
    n = []
    for p in pat:
        if p in dat.p_id.values:
            n.append(p)
    n = len(n)

    # Match image and tabular data
    # For patients that are in both sets we fill the neccessary datastructures
    # TODO: Can be optimized? Preinitialization neccessary?
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
    
    
    X = X.squeeze()
    X = np.float32(X)

    logging.info(f"X img out shape: {X.shape}")
    logging.info(f"X tab out shape: {X_tab.shape}")
    logging.info(f"Y mrs out shape: {Y_mrs.shape}")
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

    # TODO: Move this to config.. Only neccessary for tracking by papers.
    # TODO: Move this out of method -> this info/logic should not be kept here.
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
        logging.error("Split version is not set. We quit this now.")
        raise ValueError("Split version is not set.")
    elif split_version not in split_dict:
        logging.error(f"Split version '{split_version}' is not in {split_dict}. We quit this now.")
        raise ValueError(f"Split version '{split_version}' is not in {split_dict}.")
    else:
        logging.info(f"Using random seed for split '{split_version}: {split_dict.get(split_version)}'")
    
    
    skf = StratifiedKFold(n_splits=n_splits, 
                          shuffle=True, 
                          random_state=split_dict[split_version])
    folds = []
    
    # TODO: Here we only split out info for binary... Ordinal output (Y_mrs) not handled?
    logging.info("Splitting by binned target variable Y_new:") 
    for train_index, test_index in skf.split(p_id, Y_new): # 10 Fold sigmoid stratified with Outcome Good/Bad (V0, V2, V3)
    # for train_index, test_index in skf.split(p_id, Y_mrs): # 10 Fold sigmoid stratified with Outcome MRS (V1)
        folds.append(p_id[test_index])
        logging.info(f"Current split has {sum(Y_new[train_index])} bad outcomes in the train samples and {sum(Y_new[test_index])} bad outcomes in the test samples.")


    for i, fold in enumerate(folds):
        logging.info(f"{len(fold)} samples in fold {i}")
        id_tab["fold" + str(i)] = "train" 
        
        # increment for val (+5 so that no fold has only 40 in train & test) # TODO: what does this mean???
        j = i+5
        if j >= len(folds):
            j = j-n_splits

        id_tab.loc[id_tab["p_id"].isin(fold), "fold"+str(i)] = "test"
        id_tab.loc[id_tab["p_id"].isin(folds[j]), "fold"+str(i)] = "val"


    # Finally we save
    id_tab.to_csv(tabular_output_path, index=False)
    np.save(image_output_path, X)

    # TODO: Return paths saved? decide later
    return None


def validate_transformed_data(
        patient_id:int, 
        image_input_path:str,
        tabular_output_path:str, 
        image_output_path:str,
        ) -> Tuple[bool, int, int]:
    """
    Validate that the patient with id patient_id actually has the correct data linked 
    from before and after the transformation
    """
    
    # load the transformed data
    id_tab = pd.read_csv(tabular_output_path, sep=",")
    X = np.load(image_output_path)

    # load the raw data
    with h5py.File(image_input_path, "r") as h5:
        X_in = h5["X"][:]
        _ = h5["Y_img"][:]
        _ = h5["Y_pat"][:]
        pat = h5["pat"][:]
    

    index1 = id_tab[id_tab.p_id == patient_id].p_idx.values[0] -1
    logging.info(f"DataFrame index of patient {patient_id} in the transformed data: {index1}.")
    index2 = np.argwhere(pat == patient_id).squeeze()
    logging.info(f"DataFrame index of patient {patient_id} in the raw data: {index2}.")


    im1 = X[index1].astype("float64")
    im2 = X_in.squeeze()[index2].astype("float64")
    data_is_equals = np.allclose(im1, im2)
    assert data_is_equals, f"The patients data, for id {patient_id}, before and after transformation is not equals."
    logging.info(f"The data for patient {patient_id} is equals")

    return data_is_equals, index1, index2


def plot_info_split_data(tabular_output_path:str, n_splits:int = 10) -> None:
    id_tab = pd.read_csv(tabular_output_path, sep=",")


    id_tab["unfavorable"].value_counts()

    for i in range(n_splits):
        fig, (ax1, ax2) = plt.subplots(1,2)
        sns.countplot(x = id_tab[id_tab["fold"+str(i)]=="test"].mrs, ax = ax1)
        sns.countplot(x = id_tab[id_tab["fold"+str(i)]=="test"].unfavorable, ax = ax2)

    return None


def plot_count_mrs_binned_mrs(Y_mrs, Y_new):
    fig, (ax1, ax2) = plt.subplots(1,2, figsize=(15, 8))
    sns.countplot(x = Y_mrs, ax = ax1)
    sns.countplot(x = Y_new, ax = ax2)

    print("Left: Distribution of the MRS score")
    print("Right: Distribution of the binary outcome (MRS>=2)")
    print(f"Binary count: Good outcome: {sum(Y_new == 0)}, Bad outcome{sum(Y_new == 1)}")

def main():
    SPLIT_VERSION = "V3"
    N_SPLITS = 10

    # should only non TIA (transient ischemic attack) patients be included?
    only_non_tia = True

    IMG_DIR = "/host-homes/hezo/stroke_perfusion/data/"
    RAW_TABULAR = IMG_DIR + 'baseline_data_zurich_prepared.csv'
    RAW_IMAGES = IMG_DIR + 'dicom_2d_192x192x3_clean_interpolated_18_02_2021_preprocessed2.h5'

    OUTPUT_DIR = "/home/dari/explainable_AI/data/"
    TRANSFORMED_TABULAR = f"{OUTPUT_DIR}{N_SPLITS}Fold_ids_{SPLIT_VERSION}-TEST.csv"
    TRANSFORMED_IMAGES = f"{OUTPUT_DIR}prepocessed_dicom_3d-TEST.npy"

    relevant_features = ["age", "sexm", "nihss_baseline", "mrs_before",
                        "stroke_beforey", "tia_beforey", "ich_beforey", 
                        "rf_hypertoniay", "rf_diabetesy", "rf_hypercholesterolemiay", 
                        "rf_smokery", "rf_atrial_fibrillationy", "rf_chdy"]

    load_transform_dataset(
        tabular_input_path=RAW_TABULAR, 
        image_input_path=RAW_IMAGES, 
        tabular_output_path=TRANSFORMED_TABULAR,
        image_output_path=TRANSFORMED_IMAGES,
        only_non_tia=only_non_tia,
        relevant_features=relevant_features,
        favorable_mrs=[0,1,2],
        n_splits=N_SPLITS,
        split_version=SPLIT_VERSION,
        )
    
    data_is_valid, id_before_transform, id_after_transform = validate_transformed_data(
        patient_id=460, 
        image_input_path=RAW_IMAGES, 
        tabular_output_path=TRANSFORMED_TABULAR,
        image_output_path=TRANSFORMED_IMAGES
        )


    

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    main()