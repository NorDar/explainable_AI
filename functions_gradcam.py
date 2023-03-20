import tensorflow as tf
import numpy as np

from skimage.transform import resize
from matplotlib import pyplot as plt

### GradCam for specific layer
def grad_cam_3d(img, model_3d, layer, pred_index=None, inv_hm=False):
    # pred_index: output channel when sigmoid should always be 0, when softmax 0 favorable, 1 unfavorable
    
    # First, we create a MODEL that maps the input image to the activations
    # of the last conv layer as well as the output predictions
    grad_model = tf.keras.models.Model([model_3d.inputs], [model_3d.get_layer(layer).output, model_3d.output])
    
    # Then, we compute the GRADIENT for our input image
    # with respect to the activations of the last conv layer
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img)
        if pred_index is None:
            pred_index = tf.argmax(predictions[0])
        class_channel = predictions[:, pred_index] # when sigmoid, pred_index must be None or 0

    # This is the gradient of the output neuron
    # with regard to the output feature map of the last conv layer
    grads = tape.gradient(class_channel, conv_outputs)[0]
    
    # Average gradients spatially
    weights = tf.reduce_mean(grads, axis=(0, 1, 2)) # pooled grads
    
    # Build a ponderated map of filters according to gradients importance
    # We multiply each channel in the feature map array
    # by "how important this channel is" with regard to the top predicted class
    # then sum all the channels to obtain the heatmap class activation
    output = conv_outputs[0]    
    
    # slower implementation:
#     cam = np.zeros(output.shape[0:3], dtype=np.float32)
#     for index, w in enumerate(weights):
#         cam += w * output[:, :, :, index]

#     capi=resize(cam,(img.shape[1:]))
#     capi = np.maximum(capi,0)
#     heatmap = (capi - capi.min()) / (capi.max() - capi.min())
#     resized_img = img.reshape(img.shape[1:])

    # faster implementation:
    heatmap = output @ weights[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = resize(heatmap, img.shape[1:])
    
    # For visualization purpose, we will also normalize the heatmap between 0 & 1
    if inv_hm:
        heatmap = heatmap * (-1)
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    heatmap = heatmap.numpy()
    resized_img = img.reshape(img.shape[1:])
    
    return heatmap, resized_img



### GradCam for multiple layers (average, median and max activations are possible)
def multi_layers_grad_cam_3d(img, model_3d, layers, mode = "mean", normalize = True, pred_index=None, invert_hm="none"):
    valid_modes = ["mean", "median", "max"]
    if mode not in valid_modes:
        raise ValueError("multi_layers_grad_cam_3d: mode must be one of %r." % valid_modes)
    valid_hm_inverts = ["none", "all", "last"]
    if invert_hm not in valid_hm_inverts:
        raise ValueError("multi_layers_grad_cam_3d: invert_hm must be one of %r." % valid_hm_inverts)
        
    if not isinstance(layers, list):
        layers = [layers]
    
    h_l = []
    for i, layer in enumerate(layers):
        if (i == len(layers)-1 and invert_hm == "last") or invert_hm == "all":
            heatmap, resized_img = grad_cam_3d(img = img, model_3d = model_3d , layer = layer, 
                                               pred_index=pred_index, inv_hm=True)
        else:
            heatmap, resized_img = grad_cam_3d(img = img, model_3d = model_3d , layer = layer, 
                                               pred_index=pred_index, inv_hm=False)
        h_l.append(heatmap)
    
    h_l = np.array(h_l)
    if mode == "mean":
        heatmap = np.mean(h_l, axis = 0)
    elif mode == "median":
        heatmap = np.median(h_l, axis = 0)
    elif mode == "max":
        heatmap = np.max(h_l, axis = 0) 
        
    if normalize and heatmap.max() != 0:
        heatmap = ((heatmap - heatmap.min())/heatmap.max())
    
    return (heatmap, resized_img)

# applies grad cams to multiple models (and layers)
def multi_models_grad_cam_3d(img, cnn, model_names, layers, model_mode = "mean", layer_mode = "mean", normalize = True, pred_index=None,
                             invert_hm="none"):
    valid_modes = ["mean", "median", "max"]
    if model_mode not in valid_modes:
        raise ValueError("multi_models_grad_cam_3d: model_mode must be one of %r." % valid_modes)
        
    if not isinstance(layers, list):
        layers = [layers]
    
    h_l = []
    for model_name in model_names:
        cnn.load_weights(model_name)
        heatmap, resized_img = multi_layers_grad_cam_3d(
            img = img, model_3d = cnn , layers = layers, mode = layer_mode, normalize = normalize, 
            pred_index=pred_index, invert_hm=invert_hm)
        h_l.append(heatmap)
    
    h_l = np.array(h_l)
    if model_mode == "mean":
        heatmap = np.mean(h_l, axis = 0)
    elif model_mode == "median":
        heatmap = np.median(h_l, axis = 0)
    elif model_mode == "max":
        heatmap = np.max(h_l, axis = 0)
        
    if normalize and heatmap.max() != 0:
        heatmap = ((heatmap - heatmap.min())/heatmap.max())
        
    target_shape = h_l.shape[:-1]
    max_hm_slice = np.array(np.unravel_index(h_l.reshape(target_shape).reshape(len(h_l), -1).argmax(axis = 1), 
                                             h_l.reshape(target_shape).shape[1:])).transpose()
    hm_mean_std = np.sqrt(np.mean(np.var(h_l, axis = 0)))
        
    return heatmap, resized_img, max_hm_slice, hm_mean_std


# Prepares data in order to use multi_models_grad_cam_3d and plot_gradcams_last_avg_org
def get_img_and_models(p_ids, results, pats, imgs, gen_model_name, num_models = 5):
    imgs = np.expand_dims(imgs, axis = -1)
    
    # extract table with all matches for p_ids
    res_tab = results[results.p_id.isin(p_ids)].sort_values(["p_id", "test_split"]).reset_index()
    
    res_imgs = []
    res_mod_names = []
    
    res_test_splits = list(res_tab.test_split)
    for i, p_id in enumerate(list(res_tab.p_id)):
        index = np.argwhere(pats == p_id).squeeze()
        res_imgs.append(imgs[index])
        
        # generate model names
        res_mod_names.append(
            [gen_model_name(res_test_splits[i], j) for j in range(num_models)]
        )
            
    return (res_tab, np.array(res_imgs), res_mod_names)

  
    