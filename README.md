# Explainable Deep Neural Networks for MRI Based Stroke Analysis

This repository comprises code for generating, evaluating and visualizing models for binary stroke outcome predictions and is based on the paper [arXiv:2206.13302 "Deep transformation models for predicting functional outcome after acute ischemic stroke"](https://arxiv.org/abs/2206.13302) ([github code](https://github.com/LucasKook/dtm-usz-stroke)).

All data, intermediate and final results are saved locally on a ZHAW server.

## Files

### Data Preparation

- `Split_Data_into_10Fold.ipynb`:

### Model Training

- `USZ_binary_model_3d_brdd_10Fold_CV.ipynb`: 
- `USZ_binary_model_3d_brdd_andrea_splits.ipynb`:

### Model Evaluation

- `Result_Comparison.ipynb`:
- `Result_Assembly.ipynb`:

### Model Visualization

- `GradCam.ipynb`:
- `GradCam_1Model.ipynb`:
- `GradCam_allModels.ipynb`:
- `GradCam_allModels_slider.ipynb`:
- `GradCam_generate_allPics.ipynb`:
- `Occlusion_allModels_slider.ipynb`:

### Functions

- `function_model_definition.py`:
- `function_read_data.py`:
- `functions_gradcam.py`:
- `helper_functions.py`:
- `plot_functino_gradcam.py`:

### GradCam Understanding

- `Understand_GradCam_imagenet.ipynb`:

## Modelversions

- andrea_split: splits and training as in paper 
- 10Fold_sigmoid_V0 (old name: 10Fold_sigmoid): 10 stratifed (with outcome mrs > 2 or mrs <= 2) Folds trained with the last layer beeing activated with sigmoid (5 ensembles per split)
- 10Fold_softmax_V0: same Folds as 10Fold_sigmoid but last layer activated with softmax (5 ensembles per split)
- 10Fold_softmax_V1: new 10 Fold stratified (with mrs) and last layer activated with softmax (10 ensembles per split)
- 10Fold_sigmoid_V1: same Folds as 10Fold_softmax_V1 and last layer activated with sigmoid (10 ensembles per split)
- 10Fold_sigmoid_V2: 10 Fold binary stratified (mrs > or <= 2) other seed than V0, and last layer activated with sigmoid (5 ensembles per split)
- 10Fold_sigmoid_V2f: same as 10Fold_sigmoid_V2 but with flatten Layer
- 10Fold_signoid_V3: 10 Fold binary stratified (mrs > or <= 2) without TIA patients, other seed than V0 and V2 and last layer activated wih sigmoid (5 ensembles per split)
