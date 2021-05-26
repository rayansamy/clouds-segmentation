"""
from tools.dogs import *
from tools.voc import (
        VOC_CLASSES, VOC_ocv, enforce_all_seeds,
        transforms_voc_ocv_eval, sequence_batch_collate_v2)
"""
import sklearn.metrics
import numpy as np
import pandas as pd
from tqdm import tqdm
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from torchvision import transforms
import matplotlib.patches as patches
import torch
import torch.nn as nn
import torchvision.models as models
import PIL
from skimage.filters import threshold_otsu

import torch.nn.functional as F
"""
from pytorch_grad_cam import GradCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
"""

import cv2
import scipy.ndimage as ndimage
import scipy.spatial as spatial

import logging
#from tools.snippets import (quick_log_setup, mkdir)
import pickle
import pickletools


class ExtractActiv:
    """
    Allow extraction of each output of layer 
    And Attach it to a function for saving gradioents
    """
    def __init__(self, model, target):
        self.gradients = [] 
        self.model = model
        self.target = target

    def save_gradient(self, grad):
        self.gradients.append(grad)

    def __call__(self, x):
        out = []
        self.gradients = []
        for name, layer in self.model._modules.items():
            # Doing a forward pass
            x = layer(x)
            if name in self.target:
                #  Register hook on last conv layer so we can get its gradient later
                x.register_hook(self.save_gradient) 
                out += [x]
        return out, x

class ModelOutputs:
    """
    Process forward pass and return :
    - Network output
    - Activation from wanted last conv layer
    - Gradients from wanted target layer
    """
    def __init__(self, model, feature_module, target):
        self.model = model
        self.feature = feature_module
        self.feature_extractor = ExtractActiv(self.feature, target)

    def get_gradients(self):
        return self.feature_extractor.gradients
    
    def __call__(self, inpu):
        activations = []
        for name, layer in self.model._modules.items():
            if layer == self.feature:
                activations, inpu = self.feature_extractor(inpu)
            elif "avgpool" in name.lower():
                inpu = layer(inpu)
                inpu = inpu.view(inpu.size(0), -1)
            else:
                inpu = layer(inpu)
        
        return activations, inpu



class GradCam:
    """
        Class for computing forward pass on a given model and extract Heatmap.
    """
    def __init__(self, model, feature_layer=None, target_layer=2, use_cuda=False):
        self.model = model
        # GradCam have to be run with fixed weights
        self.model.eval() 
        self.convlayer = feature_layer
        self.workflow = ModelOutputs(self.model, self.convlayer, target_layer)  
        self.use_cuda = use_cuda

    def forward(self, image):
        return self.model(image)

    def __call__(self, input_img, target_category=None):
        # If we're using a gpu take the image on gpu
        if self.use_cuda:
            input_img = input_img.cuda()

        # Doing a forward pass with the image
        features, output = self.workflow(input_img)
        if target_category == None:
            target_category = np.argmax(output.cpu().data.numpy())
            
        # We consider only the gradient of the targeted class so we set the output of others to zero
        one_hot = np.zeros((1, output.size()[-1]), dtype=np.float32)
        one_hot[0][target_category] = 1
        one_hot = torch.Tensor(torch.from_numpy(one_hot))
        one_hot.requires_grad = True
        if self.use_cuda:
            one_hot = one_hot.cuda()
        
        one_hot = torch.sum(one_hot * output)

        # We calculate gradient then we extract it with the appropriate class
        #self.convlayer.zero_grad()
        self.model.zero_grad()
        one_hot.backward(retain_graph=True)
        #print("RATATATA : "+str(self.workflow.get_gradients()))
        grads_val = self.workflow.get_gradients()[-1].cpu().data.numpy()

        target = features[-1]
        target = target.cpu().data.numpy()[0, :]

        # We do a pooling over width and height of last feature layer
        weights = np.mean(grads_val, axis=(2, 3))[0, :]
        cam = np.zeros(target.shape[1:], dtype=np.float32)

        for i, w in enumerate(weights):
            cam += w * target[i, :, :]

        # We upscale the resulting heatmap
        cam = np.maximum(cam, 0)
        cam = cv2.resize(cam, input_img.shape[2:][::-1])
        cam = cam - np.min(cam)
        cam = cam / np.max(cam)
        return cam




def preprocess_image(img):
    """
    Normalizing an image using mean and std of VOC dataset.
    """

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
    preprocessing = transforms.Compose([
        transforms.ToTensor(),
        normalize,
    ])
    return preprocessing(img.copy()).unsqueeze(0)


def get_bdbox_from_heatmap(heatmap, threshold=0.2, smooth_radius=20):
    """
    Function to extract bounding boxes of objects in heatmap
    Input :
        Heatmap : matrix extracted with GradCAM. 
        threshold : value defining the values we consider , increasing it increases the size of bounding boxes.
        smooth_radius : radius on which each pixel is blurred. 
    Output :
        returned_objects : List of bounding boxes, N_objects * [ xmin, xmax, ymin, ymax, width, height ]
    """

    # If heatmap is all zeros i initialize a default bounding box which wraps entire image
    xmin = 0
    xmax = heatmap.shape[1]
    ymin = 0
    ymax = heatmap.shape[0]
    width = xmax-xmin
    height = ymax-ymin
    
    returned_objects = []

    # Count if there is any "hot" value on the heatmap
    count = (heatmap > threshold).sum() 
    
    # Blur the image to have continuous regions
    heatmap = ndimage.uniform_filter(heatmap, smooth_radius)

    # Threshold the heatmap with 1 for values > threshold and 0 else
    thresholded = np.where(heatmap > threshold, 1, 0)

    # Apply morphological filter to fill potential holes in the heatmap
    thresholded =  ndimage.morphology.binary_fill_holes(thresholded)

    # Detect all independant objects in the image
    labeled_image, num_features = ndimage.label(thresholded)
    objects = ndimage.measurements.find_objects(labeled_image)
    
    # We loop in each object ( if any is detected ) and append it to a global list
    if count > 0:
        for obj in objects:
            x = obj[1]
            y = obj[0]
            xmin = x.start
            xmax = x.stop
            ymin = y.start
            ymax = y.stop

            width = xmax-xmin
            height = ymax-ymin
            
            returned_objects.append([xmin, xmax, ymin, ymax, width, height])
    else:
        returned_objects.append([xmin, xmax, ymin, ymax, width, height])
    return returned_objects


def eval_image(model, gradcam, path, target_category=4):
    """
    Evaluate an Image with GradCAM algorithm
    Input :
        model : Resnet50 model
        path : path for the image to predict heatmap from
        target_category : which category prediction we're interested in
    Output :
        input_img : the image after preprocessing
        grayscale_cam : heatmap of relevant pixels in picture
        cam : Image + heatmap
        img : original image
    """

    img = cv2.imread(path, 1)
    img = cv2.resize(img, (768, 512))
    img = np.float32(img) / 255
    # Opencv loads as BGR:
    img = img[:, :, ::-1]
    input_img = preprocess_image(img)

    # If None, returns the map for the highest scoring category.
    # Otherwise, targets the requested category.
    grayscale_cam = gradcam(input_img, target_category)
    
    cam = show_cam_on_image(img, grayscale_cam)
    
    return input_img, grayscale_cam, cam, img

def show_cam_on_image(img, mask):
    """
    Input :
        mask : Heatmap from GradCAM
        img : Original Image
    Output :
        cam : The image with a heatmap mask on it
    """

    heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255
    cam = heatmap + np.float32(img)
    cam = cam / np.max(cam)
    return np.uint8(255 * cam)




def eval_voc_grad(visualise=False, threshold=0.65):
    """
    Function that applies GradCAM algorithm with modified ResNet50 model and then evaluates ap performance.
    if visualize = True, images and bounding boxes will show while processing.
    """
    

    device = 'cpu'

    # Load the finetuned model
    """
    model = torch.load("../zooniverse-resized-image_net_model", map_location=device)
    print(model)
    return 
    """   
    model = models.resnet18(pretrained=True)
    
    
    model.fc = nn.Sequential(
    nn.Linear(512, 100),
    nn.ReLU(),
    nn.Dropout(0.5),

    nn.Linear(100, 20),
    nn.ReLU(),
    nn.Dropout(0.5),

    nn.Linear(20, 4),
    nn.LogSoftmax(dim=1)
)
    model.load_state_dict(torch.load("../zooniverse-resized-image_state_dict", map_location=device))
    #target_layer = model.layer4[-1]
    #model.eval()
    images = ["../data/3.jpg"]
    # (self, model, target_layer_names, use_cuda):
    grad_cam = GradCam(model, feature_layer=model.layer4, target_layer=['1'], use_cuda=False)
    
    for image in images:
        for category in [0, 1, 2, 3]:
            input_img, grayscale_cam, cam, img = eval_image(model, grad_cam, image, target_category=category)
            
            # Extract all objects from heatmap
            objects = get_bdbox_from_heatmap(grayscale_cam, threshold=threshold)
            #  Loop in each one of the objects and adding it in the detection output
            for obj in objects[::-1]:
                xmin, xmax, ymin, ymax, width, height = obj
                bbox = np.r_[int(xmin), int(ymin), int(xmax), int(ymax)]
            

            # Visualizing image with Ground truth bounding boxes and predicted ones 
            if visualise:
                fig,ax = plt.subplots(1)
                ax.imshow(cam)
                plt.text(0, 0, "Category "+str(category))
                plt.show()

    # Evaluation step


def predict_image(model, image, prediction):
    threshold = 0.5
    device = 'cpu'
    # (self, model, target_layer_names, use_cuda):
    grad_cam = GradCam(model, feature_layer=model.layer4, target_layer=['1'], use_cuda=False)
    fig,axs = plt.subplots(2, 2)
    fig.set_figheight(20)
    fig.set_figwidth(20)
    fig.suptitle("PREDICTED BY THE MODEL AS "+str(prediction), fontsize=14)
    for category in [0, 1, 2, 3]:
        #print("category : "+str(category))
        input_img, grayscale_cam, cam, img = eval_image(model, grad_cam, image, target_category=category)
        if category==0:
            axs[0, 0].imshow(cam)
            axs[0, 0].set_title("Activated Pixels for prediction FISH")
        if category==1:
            axs[0, 1].imshow(cam)
            axs[0, 1].set_title("Activated Pixels for prediction FLOWER")
        if category==2:
            axs[1, 0].imshow(cam)
            axs[1, 0].set_title("Activated Pixels for prediction GRAVEL")
        if category==3:
            axs[1, 1].imshow(cam)
            axs[1, 1].set_title("Activated Pixels for prediction SUGAR")
        # Visualizing image with Ground truth bounding boxes and predicted ones 
        
    plt.show()

    # Evaluation step


if __name__ == '__main__':
    eval_voc_grad(visualise=True)
