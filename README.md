# Explainable Deep Neural Networks for MRI Based Stroke Analysis

This repository comprises code for generating, evaluating and visually explaining models for binary stroke outcome predictions which are described here [arXiv:2206.13302 "Deep transformation models for predicting functional outcome after acute ischemic stroke"](https://arxiv.org/abs/2206.13302) (along with the code [github code](https://github.com/LucasKook/dtm-usz-stroke)).  The model is trained on 3D MRI data (DWI-Images) of stroke patients and predicts the binary outcome of the modified Rankin Scale (mRS) at 90 days after stroke. The mRS is an ordinal scale from 0 to 6, where 0 means no symptoms and 6 means death. The model predicts whether the patient will have a good outcome (mRS <= 2) or a bad outcome (mRS > 2). 

The goal of this project is to explain the predictions of a binary stroke outcome model based on image data only. We aim to du so by highlighting regions in the images that were crucial for the made prediction.  Heatmaps for the model predictions are generated using the [Grad-CAM](https://arxiv.org/abs/1610.02391) and 3D-Occlusion methods.

The models and heatmaps are trained and evaluatien with different data splits. First the results of the above mentioned papers were reproduced based on the 6th split of the above mentioned paper. In order to generate heatmaps for each patient new 10 folds cross validation (CV) splits are generated. The splits from the paper are randomly choosen and therefore not all patients are in a test set. Different model architectures are tested and compared:

Different data folds and architectures are tested and compared. Fhe following table gives some information about the different implemented versions.
**Number of Folds** and **Stratification** refer to the number of folds generated their stratification. **Seed** is the random seed used for splitting.
**Layer Connection** refers to the connection between the convolutional and fully connected part of the neural network. **Activation Function** is the activation function used in the last layer.
**Number of Ensembles** refers to the number of models trained for each split (*total numbers of models = number of folds x number of ensembles*). **Additional Information** gives some additional information about the model. 

Version Name | Number of Folds | Stratification | Seed | Layer Connection | Activation Function | Number of Ensembles | Additional Information
--- | --- | --- | --- | --- | --- | --- | ---
andrea_split       | 6  | Random                 | ?   | Average Pooling Layer | sigmoid | 5  | same splits and training as in paper, only trained for split 6
10Fold_sigmoid_V0  | 10 | binary (mrs > or <= 2) | 100 | Average Pooling Layer | sigmoid | 5  | twice trained with different seeds
10Fold_softmax_V0  | 10 | binary (mrs > or <= 2) | 100 | Average Pooling Layer | softmax | 5  | same Folds as 10Fold_sigmoid_V0 
10Fold_sigmoid_V1  | 10 | mrs                    | 999 | Average Pooling Layer | sigmoid | 10 |
10Fold_softmax_V1  | 10 | mrs                    | 999 | Average Pooling Layer | softmax | 10 | same Folds as 10Fold_sigmoid_V1 
10Fold_sigmoid_V2  | 10 | binary (mrs > or <= 2) | 500 | Average Pooling Layer | sigmoid | 5  |
10Fold_sigmoid_V2f | 10 | binary (mrs > or <= 2) | 500 | Flatten Layer         | sigmoid | 5  | same Folds as 10Fold_sigmoid_V2 
10Fold_sigmoid_V3  | 10 | binary (mrs > or <= 2) | 200 | Average Pooling Layer | sigmoid | 5  | without TIA patients

All data, intermediate and final results are saved locally on a ZHAW server.

## Files

The end product of this repository is the visualization of the heatmaps. However, to generate the heatmaps, the data must be splitted in folds, the model must be trained and evaluated.  
After these steps the heatmaps can be generated and visualized. This is possible by generating all heatmaps and save the genereated heatmaps into a pdf. Alternatively, the heatmaps can be generated and visualized at the same time with an interactive slider. Or the heatmaps can be generated on the fly and visualized with an interactive slider.

### 10 fold CV Preparation

- `split_data.ipynb`: Split data into 10 Fold CV splits (8 as trainings, 1 as validation and 1 as test set). Takes preprocessed images (e.g. *dicom_2d_192x192x3_clean_interpolated_18_02_2021_preprocessed2.h5*) 
and preporcessed tabular data (e.g. *baseline_data_zurich_prepared.csv*) as input. Returns a numpy array of the images which are eligible for training and a pandas dataframe with the tabular data (image index, patient id, mrs, binary mrs, fold assignments).

### Model Training

- `usz_binary_model_3d_paper_splits.ipynb`: Train one model of the paper splits with the same architecture as in the paper and compare the calibration plots from the paper with the newly trained model. As input the preprocessed images from the paper (*dicom_2d_192x192x3_clean_interpolated_18_02_2021_preprocessed2.h5*), the preprocessed tabular data (*baseline_data_zurich_prepared.csv*) and splits from the paper (*andrea_splits.csv*) are needed. Saves the trained weights. Alternatively the model is not trained an the weights are loaded.
- `usz_binary_model_3d_10fold_cv.ipynb`: Train a 10 Fold CV model with the same architecture as in the paper. The preprocessed images and tabular data from *split_data.ipynb* are needed as input. Saves the trained weights. Alternatively weights are loaded.  

### Model Evaluation

- `result_comparison.ipynb`: Compare the results achieved in the paper with the results of the one in *usz_binary_model_3d_paper_splits.ipynb*. Needs the all weights trained in  *usz_binary_model_3d_paper_splits.ipynb*, the preprocessed images from the paper (*dicom_2d_192x192x3_clean_interpolated_18_02_2021_preprocessed2.h5*), the preprocessed tabular data (*baseline_data_zurich_prepared.csv*) and splits from the paper (*andrea_splits.csv*). Furthermore, the results from the paper (*stroke_cimrsbinary_lossnll_wsno_augyes_cdftest_spl6_ensX.csv*) and the calibration plot data (*bincal_splnll.csv*, *bincal_avgnll.csv*) are needed.
- `result_assembly.ipynb`: Assemble the results of the 10 Fold CV (and all ensembles) into one dataframe. As input the preprocessed images and tabular data from *split_data.ipynb* and the weights from *usz_binary_model_3d_10fold_cv.ipynb* are needed. Moreover, when the model should be assembled for the splits from andrea the splits from *andrea_splits.csv* and weights from *usz_binary_model_3d_paper_splits.ipynb* are needed. Saves a dataframe with the results of all ensembles for each split (also trafo averaged predictions). 

### Model Visualization

- `gradcam_all_models.ipynb`: Explore the Grad-CAM heatmaps. The trained weights, the assembled results and the preprocessed images are needed as input. The notebook returns nothing and only serves to explore the Grad-CAM heatmaps.
- `gradcam_allmodels_slider.ipynb`: Generate Grad-CAM heatmaps and visualize them with an interactive slider. The trained weights, the assembled results, the preprocessed images and the unnormalized tabular data (*baseline_data_zurich_prepared0.csv*) are needed as input. The notebook returns nothing and only serves to generate and visualize the Grad-CAM heatmaps.
- `occlusion_allModels_slider.ipynb`: Generate 3D-Occlusion heatmaps and visualize them with an interactive slider. The trained weights, the assembled results, the preprocessed images and the unnormalized tabular data are needed as input. The notebook returns nothing and only serves to generate and visualize the 3D-Occlusion heatmaps.
- `generate_all_heatmaps.ipynb`: Generates heatmaps (occlusion or gradcam) for all patients. As input the trained weights, the assembled results, the preprocessed images and the unnormalized tabular data are needed. The heatmaps are saved as numpy arrays. The standard visualization is then saved as pngs and all (or only wrongly predicted patients) png are saved into one pdf.
- `heatmap_slider.ipynb`: Visualize already generated heatmaps with an interactive slider. As input the generated heatmaps from *generate_all_heatmaps.ipynb*, the assembled results, the preprocessed images and the unnormalized tabular data are needed. The notebook returns nothing and only serves to visualize the heatmaps with an interactive slider to compare occlusion and Grad-CAM.

### Functions

- `functions_read_data.py`: Helper functions for reading the data.
- `functions_model_definition.py`: Helper functions for defining the model.
- `functions_metrics.py`: Helper functions for calculating metrics and calibration plots.
- `functions_gradcam.py`: Helper functions for generating Grad-CAM heatmaps.
- `functions_occlusion.py`: Helper functions for generating 3D-Occlusion heatmaps.
- `functions_plot_heatmap.py`: Helper functions for plotting heatmaps.
- `functions_slider.py`: Helper functions for visualizing heatmaps with an interactive slider.
