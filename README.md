# Explainable Deep Neural Networks for MRI Based Stroke Analysis

This repository comprises code for generating, evaluating and visualizing models for binary stroke outcome predictions and is based on the paper [arXiv:2206.13302 "Deep transformation models for predicting functional outcome after acute ischemic stroke"](https://arxiv.org/abs/2206.13302) ([github code](https://github.com/LucasKook/dtm-usz-stroke)).

The goal is to explain the predictions of a binary stroke outcome model based on image data only. The model is trained on 3D MRI data (DWI-Images) of stroke patients and predicts the binary outcome of the modified Rankin Scale (mRS) at 90 days after stroke. The mRS is a scale from 0 to 6, where 0 means no symptoms and 6 means death. The model predicts whether the patient will have a good outcome (mRS <= 2) or a bad outcome (mRS > 2).  
Heatmaps for the model predictions are generated using the [Grad-CAM](https://arxiv.org/abs/1610.02391) and 3D-Occlusion methods. 

The models and heatmaps are generated based on the splits of the above mentioned paper and new splits. The new models on the paper splits are compared with the original ones to reproduce the achieved results. In order to generate heatmaps for each patient new 10 Fold splits are generated. Different model architectures are tested and compared:

- andrea_split: splits and training as in paper 
- 10Fold_sigmoid_V0: 10 stratifed (with outcome mrs > 2 or mrs <= 2) Folds trained with the last layer beeing activated with sigmoid (5 ensembles per split)
- 10Fold_softmax_V0: same Folds as 10Fold_sigmoid_V0 but last layer activated with softmax (5 ensembles per split)
- 10Fold_softmax_V1: new 10 Fold stratified (with mrs) and last layer activated with softmax (10 ensembles per split)
- 10Fold_sigmoid_V1: same Folds as 10Fold_softmax_V1 and last layer activated with sigmoid (10 ensembles per split)
- 10Fold_sigmoid_V2: 10 Fold binary stratified (mrs > or <= 2) other seed than V0, and last layer activated with sigmoid (5 ensembles per split)
- 10Fold_sigmoid_V2f: same as 10Fold_sigmoid_V2 but with flatten Layer
- 10Fold_signoid_V3: 10 Fold binary stratified (mrs > or <= 2) without TIA patients, other seed than V0 and V2 and last layer activated wih sigmoid (5 ensembles per split)

The 10Fold_sigmoid_V0 version was trained twice to compare different initiallitations.  

All data, intermediate and final results are saved locally on a ZHAW server.

## Files

### Data Preparation

- `split_data.ipynb`: Split data into 10 Fold CV splits (8 as trainings, 1 as validation and 1 as test set).

### Model Training

- `usz_binary_model_3d_paper_splits.ipynb`: Train models on paper splits and the same architecture as in the paper.
- `usz_binary_model_3d_10fold_cv.ipynb`: Train a 10 Fold CV model with the same architecture as in the paper.

### Model Evaluation

- `result_comparison.ipynb`: Compare the results achieved in the paper with the results of the one in usz_binary_model_3d_paper_splits.ipynb.
- `result_assembly.ipynb`: Assemble the results of the 10 Fold CV (and all ensembles) into one dataframe.

### Model Visualization

- `gradcam_all_models.ipynb`: Explore the Grad-CAM heatmaps.
- `gradcam_allmodels_slider.ipynb`: Generate Grad-CAM heatmaps and visualize them with an interactive slider.
- `occlusion_allModels_slider.ipynb`: Generate 3D-Occlusion heatmaps and visualize them with an interactive slider.
- `generate_all_heatmaps.ipynb`: Generates heatmaps (occlusion or gradcam) for all patients.
- `heatmap_slider.ipynb`: Visualize already generated heatmaps with an interactive slider.

### Functions

- `functions_read_data.py`: Helper functions for reading the data.
- `functions_model_definition.py`: Helper functions for defining the model.
- `functions_metrics.py`: Helper functions for calculating metrics and calibration plots.
- `functions_gradcam.py`: Helper functions for generating Grad-CAM heatmaps.
- `functions_occlusion.py`: Helper functions for generating 3D-Occlusion heatmaps.
- `functions_plot_heatmap.py`: Helper functions for plotting heatmaps.
- `functions_slider.py`: Helper functions for visualizing heatmaps with an interactive slider.
