import warnings
import skimage

import numpy as np

from matplotlib import pyplot as plt

def plot_gradcam(resized_img, heatmap, 
                 version = "overlay",
                 mode = "avg",
                 slices = None,
                 heatmap_threshold = None,
                 negative = False,
                 pic_size = (128,128),
                 show = True,
                 add_plot = None):
    # mode: avg => averages (mean) of heatmaps and image. avg_heatmap_correction can be applied for heatmap average
    #       max => extracts slice with highest activation for heatmap and image
    #       def => extracts defined slices. if not defined by "slices", than extract middle slice of each view
    # slices: should be none or a tuple of shape 3. defines which slice to take when mode == "def"
    # heatmap_threshold: if not None than should be between 0 and 1. At which proportion of the heatmap values, 
    #                     the heatmap should be set to 0. This can reduce noise of GradCam. "gradcam threshold"
    # add_plot: if not NULL, a tuple of (current_row, total_rows) must be given (current_row starts counting with 0)
    
    valid_versions = ["overlay", "original", "activation"]
    valid_modes = ["avg", "max", "def"]
    if version not in valid_versions:
        raise ValueError("plot_gradcam: version must be one of %r." % valid_versions)
    if mode not in valid_modes:
        raise ValueError("plot_gradcam: mode must be one of %r." % valid_modes)
    if heatmap_threshold is not None:
        if not (heatmap_threshold < 1 and heatmap_threshold > 0):
            raise ValueError("plot_gradcam: if heatmap_threshold is not None than must be between 0 and 1")
    
    if slices is None and mode == "def":
        warnings.warn("plot_gradcam: slices are not defined but mode is set to def. Plot slices (64,64,14)")
        slices = (64,64,14)
    elif slices is not None and mode in ["avg", "max"]:
        warnings.warn("plot_gradcam: slices are defined but mode is not set to def. Ignore value of slice!")
        
    if mode == "max":
        slices = np.unravel_index(heatmap.argmax(), heatmap.shape)
    
    
    if negative:
        resized_img = np.negative(resized_img)

    if add_plot is None:
        fig = plt.figure(figsize = (15,15))
    
    # Define all plots (image and heatmap even if not used)
    # Also define captions
    if mode == "avg":
        img_ax = np.mean(resized_img, axis = 2)
        map_ax = np.mean(heatmap, axis = 2)
        img_cor = skimage.transform.resize(np.rot90(np.mean(resized_img, axis = 0)), output_shape=pic_size)
        map_cor = skimage.transform.resize(np.rot90(np.mean(heatmap, axis = 0)), output_shape=pic_size)
        img_sag = skimage.transform.resize(np.fliplr(np.rot90(np.mean(resized_img, axis = 1))), output_shape=pic_size)
        map_sag = skimage.transform.resize(np.fliplr(np.rot90(np.mean(heatmap, axis = 1))), output_shape=pic_size)
        captions = ["Axial Avg", "Coronal Avg", "Sagital Avg"]            
    else:
        img_ax = resized_img[:,:,slices[2]]
        map_ax = heatmap[:,:,slices[2]]
        img_cor = skimage.transform.resize(np.rot90(resized_img[slices[0],:,: ]), output_shape=pic_size)
        map_cor = skimage.transform.resize(np.rot90(heatmap[slices[0],:,:]), output_shape=pic_size)
        img_sag = skimage.transform.resize(np.fliplr(np.rot90(resized_img[:,slices[1],: ])), output_shape=pic_size)
        map_sag = skimage.transform.resize(np.fliplr(np.rot90(heatmap[:,slices[1],:])), output_shape=pic_size)
        captions = ["Ax. Slice: " + str(slices[2]), "Cor. Slice: " + str(slices[0]), "Sag. Slice: " + str(slices[1])]
        
    # apply heatmap_threshold if it is defined (GradCam threshold)
    if heatmap_threshold is not None:
        ax_th = np.max(map_ax)*heatmap_threshold
        map_ax[map_ax < ax_th] = 0
        cor_th = np.max(map_cor)*heatmap_threshold
        map_cor[map_cor < cor_th] = 0
        sag_th = np.max(map_sag)*heatmap_threshold
        map_sag[map_sag < sag_th] = 0

    images = [img_ax, img_cor, img_sag]
    heatmaps = [map_ax, map_cor, map_sag]

    # For all three views plot desired version and add caption
    for i in range(3):
        if add_plot is None:
            ax = plt.subplot(1,3, i+1)
        elif len(add_plot) == 2:
            ax = plt.subplot(add_plot[1], 3, (add_plot[0]*3)+(i+1))
            
        if version in ["overlay", "original"]:
            ax.imshow(images[i], cmap='gray')
        if version in ["overlay", "activation"]:
            ax.imshow(heatmaps[i], alpha=0.4, cmap="jet")
        ax.set_title(captions[i])
        plt.axis('off')
        
    if show:
        plt.show()