import torch
from torch.autograd import Variable

from PIL import Image
import numpy as np

from functools import partial
import sys

import cv2

from models import *
from utils import *

def load_image(filename = './lena.jpg'):
    img = Image.open(filename)

    return img

def preprocess(img):
    img = np.asarray(img.resize((224, 224))) # resize to 224 * 224 (W * H), np.asarray returns (H, W, C)
    img = img.transpose(2, 0, 1)    # reshape to (C, H, W)
    img = img[np.newaxis, :, :, :]  # add one dim to (1, C, H, W)

    return Variable(torch.FloatTensor(img.astype(float)))

# store all feature maps and max pooling locations during forward pass
def store_feat_maps(model):

    def hook(module, input, output, key):
       if isinstance(module, nn.MaxPool2d):
           model.feat_maps[key] = output[0]
           model.pool_locs[key] = output[1]
       else:
           model.feat_maps[key] = output

    for idx, layer in enumerate(model._modules.get('features')):    # _modules returns an OrderedDict
        layer.register_forward_hook(partial(hook, key = idx))


if __name__ == '__main__':

    img_file = './lena.jpg'

    # load an image
    img = load_image(img_file)

    # preprocess an image, return a pytorch Variable
    input_img = preprocess(img)

    vgg16_conv = VGG16_Conv(1000)   # ImageNet class categories, build vgg16 forward network
    vgg16_conv.eval()   # evaluation mode

    store_feat_maps(vgg16_conv) # store all feature maps and max pooling locations during forward pass

    conv_output = vgg16_conv(input_img)

    vgg16_deconv = VGG16_Deconv()   # build vgg16 backward network
    vgg16_deconv.eval()

    plt.ion()  # remove blocking
    plt.figure(figsize = (10, 5))

    while True:
        layer = input('Which layer to view (0-30, -1 to exit): ')
        try:
            layer = int(layer)
        except ValueError:
            continue

        if layer < 0:
            sys.exit(0)

        feat_map = vgg16_conv.feat_maps[layer].data.numpy().transpose(1, 2, 3, 0) # (1, C, H, W) -> (C, H, W, 1)
        feat_map_grid = vis_grid(feat_map)  # represent a feature map in a grid
        vis_layer(feat_map_grid)    # visualize a feature map

        # only transpose convolve from Conv2d or ReLU layers
        if (layer not in vgg16_conv.conv_layer_indices) and (layer - 1 not in vgg16_conv.conv_layer_indices):
            print('Not Conv2d or ReLu layer selected!')
            continue    # RelU layers are always follow Conv2d layers directly

        n_activation = feat_map.shape[0]    # n_activation = n_chs

        marker = None
        while True:
            select = input('Select an activation? (y/[n]): ') == 'y'    # choose one activation from a feature map
            if marker != None:
                marker.pop(0).remove()

            if not select:
                break

            _, H, W, _ = feat_map.shape
            # gridH = ceil(sqrt(n_chs)) * H + ceil(sqrt(n_chs)), gridW = ceil(sqrt(n_chs)) * W + ceil(sqrt(n_chs))
            gridH, gridW, _ = feat_map_grid.shape

            col_steps = gridW // (W + 1)    # ceil(sqrt(n_chs))

            print('Click on an activation to continue:')
            xPos, yPos = plt.ginput(1)[0]
            xIdx = xPos // (W + 1)  # additional "1" represents black cutting-line
            yIdx = yPos // (H + 1)
            activation_idx = int(col_steps * yIdx + xIdx)   # index starts from 0

            if activation_idx >= n_activation:
                print('Invalid activation selected!')
                continue

            new_feat_map = vgg16_conv.feat_maps[layer].clone()
            # set all activations to zero, except for the selected one
            if activation_idx == 0:
                new_feat_map[:, 1:, :, :] = 0
            else:
                new_feat_map[:, :activation_idx, :, :] = 0
                if activation_idx != vgg16_conv.feat_maps[layer].shape[1] - 1:
                    new_feat_map[:, activation_idx + 1:, :, :] = 0

            deconv_output = vgg16_deconv(new_feat_map, layer, activation_idx, vgg16_conv.pool_locs)

            # transform and normalize a deconvolutional image
            img = tn_deconv_img(deconv_output)

            heatmap = cv2.applyColorMap(cv2.resize(img, (img.shape[1], img.shape[0])), cv2.COLORMAP_JET)

            plt.subplot(121)
            marker = plt.plot(xPos, yPos, marker = '+', color = 'red')
            plt.subplot(122)
            plt.imshow(heatmap) # img