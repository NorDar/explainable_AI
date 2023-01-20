import tensorflow as tf
import numpy as np

from skimage.transform import resize
from matplotlib import pyplot as plt

### GradCam for specific layer
def grad_cam_3d(img, model_3d, layer):
    
    # First, we create a MODEL that maps the input image to the activations
    # of the last conv layer as well as the output predictions
    grad_model = tf.keras.models.Model([model_3d.inputs], [model_3d.get_layer(layer).output, model_3d.output])
    
    # Then, we compute the GRADIENT for our input image
    # with respect to the activations of the last conv layer
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img)    
        loss = predictions[0][0]

    # This is the gradient of the output neuron
    # with regard to the output feature map of the last conv layer
    grads = tape.gradient(loss, conv_outputs)[0]
    
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
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    heatmap = heatmap.numpy()
    resized_img = img.reshape(img.shape[1:])
    
    return heatmap, resized_img



### GradCam for multiple layers (average, median and max activations are possible)
def multi_layers_grad_cam_3d(img, model_3d, layers, mode = "mean"):
    valid_modes = ["mean", "median", "max"]
    if mode not in valid_modes:
        raise ValueError("multi_layers_grad_cam_3d: mode must be one of %r." % valid_modes)
    
    h_l = []
    for layer in layers:
        heatmap, resized_img = grad_cam_3d(img = img, model_3d = model_3d , layer = layer)
        h_l.append(heatmap)
    
    h_l = np.array(h_l)
    if mode == "mean":
        heatmap = np.mean(h_l, axis = 0)
    elif mode == "median":
        heatmap = np.median(h_l, axis = 0)
    elif mode == "max":
        heatmap = np.max(h_l, axis = 0)
        
    
    return (heatmap, resized_img)
    