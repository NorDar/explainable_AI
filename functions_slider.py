from __future__ import print_function

import numpy as np
import ipywidgets as widgets

from ipywidgets import interact, interactive, fixed, interact_manual, HBox, VBox, Layout, AppLayout
from IPython.display import display
from termcolor import colored

import functions_gradcam as gc
import functions_occlusion as oc
import functions_plot_heatmap as phm

def gradcam_interactive_plot(p_id, vis_layers,
                             cnn, all_results, pat, X_in,
                             generate_model_name, num_models,
                             pat_dat,
                             pred_hm_only=True, 
                             heatmaps = None):
    p_ids = [p_id]
    (res_table, res_images, res_model_names) = gc.get_img_and_models(
        p_ids, results = all_results, pats = pat, imgs = X_in, 
        gen_model_name = generate_model_name,
        num_models = num_models)
    
    print("patient id: ", res_table.p_id[0])
    print("age: ", pat_dat[pat_dat["p_id"] == res_table.p_id[0]]["age"].values[0])
    print("true mrs: ", res_table.mrs[0])
    print("true class: ", res_table.unfavorable[0])
    print(colored("pred class: "+str(res_table.y_pred_class[0]), 
                'green' if res_table["pred_correct"][0] == True else 'red'))
    print("pred prob (class 1): ", res_table.y_pred_trafo_avg[0])
    print("pred uncertainty: ", res_table.y_pred_unc[0])
    # print("heatmap unc. last layer: ", res_table.y_pred_unc[0])
    
    ## Generate heatmap
    if pred_hm_only:
        invert_hm = "all" if res_table.y_pred_class[0] == 0 else "none"
        gcpp_hm = "last"
        cmap = "jet"
        hm_positive=True
    else:
        invert_hm = "none"
        gcpp_hm = "none"
        cmap = "bwr"
        hm_positive=False
    
    if heatmaps is None:        
        heatmap, resized_img, max_hm_slice, hm_mean_std = gc.multi_models_grad_cam_3d(
            img = np.expand_dims(res_images[0], axis = 0), 
            cnn = cnn,
            model_names = res_model_names[0],
            layers = vis_layers[3],
            model_mode = "mean",
            pred_index = 0,
            invert_hm = invert_hm,
            gcpp_hm = gcpp_hm)
    else:
        heatmap = heatmaps[np.argwhere(pat == p_id).squeeze()]
        resized_img = res_images[0]

    slices = np.unravel_index(heatmap.argmax(), heatmap.shape)
    print("max slices:", (slices[2], slices[0], slices[1]))
    
    ## Plot Heatmap Average
    phm.plot_heatmap(resized_img, heatmap,
                version = "overlay",
                mode = "avg",
                hm_colormap=cmap,
                hm_positive=hm_positive,
                colorbar=True)

    ## Plot Heatmap Slider
    def slicer(axi_slider, cor_slider, sag_slider):
        phm.plot_heatmap(resized_img, heatmap,
                version = "overlay",
                mode = "def",
                slices = (cor_slider,sag_slider,axi_slider),
                hm_colormap=cmap,
                hm_positive=hm_positive,
                colorbar=True)
        phm.plot_heatmap(resized_img, heatmap,
                version = "original",
                mode = "def",
                slices=(cor_slider,sag_slider,axi_slider),
                hm_colormap=cmap,
                hm_positive=hm_positive,
                slice_line=True)

    w=interactive(
        slicer, 
        axi_slider=widgets.IntSlider(value=slices[2],min=0,max=27,step=1), 
        cor_slider=widgets.IntSlider(value=slices[0],min=0,max=127,step=1), 
        sag_slider=widgets.IntSlider(value=slices[1],min=0,max=127,step=1))

    slider_layout = Layout(display='flex', flex_flow='row', 
                        justify_content='space-between', align_items='center',
                        width='9.2in')
    images_layout = Layout(display='flex', flex_flow='row', 
                        justify_content='space-between', align_items='center',
                        width='15', height='15')

    display(VBox([
        HBox([w.children[0],w.children[1], w.children[2]], layout=slider_layout),
        HBox([w.children[3]], layout=images_layout)
    ]))      
    w.update()
   
   
 
def occlusion_interactive_plot(p_id, occ_size, occ_stride,
                               cnn, all_results, pat, X_in,
                               generate_model_name, num_models,
                               pat_dat,
                               pred_hm_only=True,
                               heatmaps = None):
    p_ids = [p_id]
    (res_table, res_images, res_model_names) = gc.get_img_and_models(
        p_ids, results = all_results, pats = pat, imgs = X_in, 
        gen_model_name = generate_model_name,
        num_models = num_models)
    
    print("patient id: ", res_table.p_id[0])
    print("age: ", pat_dat[pat_dat["p_id"] == res_table.p_id[0]]["age"].values[0])
    print("true mrs: ", res_table.mrs[0])
    print("true class: ", res_table.unfavorable[0])
    print(colored("pred class: "+str(res_table.y_pred_class[0]), 
                'green' if res_table["pred_correct"][0] == True else 'red'))
    print("pred prob (class 1): ", res_table.y_pred_trafo_avg[0])
    print("pred uncertainty: ", res_table.y_pred_unc[0])
    # print("heatmap unc. last layer: ", res_table.y_pred_unc[0])
      
    ## Generate heatmap
    if pred_hm_only:
        invert_hm = "pred_class"
        both_directions = False
        cmap = "jet"
        hm_positive=True
    else:
        invert_hm = "never"
        both_directions = True
        cmap = "bwr"
        hm_positive=False   
    
    if heatmaps is None:
        (heatmap, resized_img, max_hm_slice, hm_mean_std) =  oc.volume_occlusion(
            volume = res_images, 
            res_tab = res_table, 
            occlusion_size = np.array(occ_size), 
            cnn = cnn,
            invert_hm=invert_hm,
            both_directions=both_directions,
            model_names = res_model_names[0],
            occlusion_stride = occ_stride)
    else:
        heatmap = heatmaps[np.argwhere(pat == p_id).squeeze()]
        resized_img = res_images[0]
    
    slices = np.unravel_index(heatmap.argmax(), heatmap.shape)
    print("max slices:", (slices[2], slices[0], slices[1]))
    
    ## Plot Heatmap Average
    phm.plot_heatmap(resized_img, heatmap,
                version = "overlay",
                mode = "avg",
                hm_colormap=cmap,
                hm_positive=hm_positive,
                colorbar=True)

    ## Plot Heatmap Slider
    def slicer(axi_slider, cor_slider, sag_slider):
        phm.plot_heatmap(resized_img, heatmap,
                version = "overlay",
                mode = "def",
                slices = (cor_slider,sag_slider,axi_slider),
                hm_colormap=cmap,
                hm_positive=hm_positive,
                colorbar=True)
        phm.plot_heatmap(resized_img, heatmap,
                version = "original",
                mode = "def",
                slices=(cor_slider,sag_slider,axi_slider),
                hm_colormap=cmap,
                hm_positive=hm_positive,
                slice_line=True)

    w=interactive(
        slicer, 
        axi_slider=widgets.IntSlider(value=slices[2],min=0,max=27,step=1), 
        cor_slider=widgets.IntSlider(value=slices[0],min=0,max=127,step=1), 
        sag_slider=widgets.IntSlider(value=slices[1],min=0,max=127,step=1))

    slider_layout = Layout(display='flex', flex_flow='row', 
                        justify_content='space-between', align_items='center',
                        width='9.2in')
    images_layout = Layout(display='flex', flex_flow='row', 
                        justify_content='space-between', align_items='center',
                        width='15', height='15')

    display(VBox([
        HBox([w.children[0],w.children[1], w.children[2]], layout=slider_layout),
        HBox([w.children[3]], layout=images_layout)
    ]))      
    w.update()