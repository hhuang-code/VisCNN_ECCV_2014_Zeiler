import torch.nn as nn
import torchvision.models as models

from collections import OrderedDict

import pdb

class VGG16_Conv(nn.Module):

    def __init__(self, n_classes = 1000):   # ImageNet class categories
        super(VGG16_Conv, self).__init__()

        self.features = nn.Sequential(
            # conv1
            nn.Conv2d(3, 64, 3, padding = 1),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding = 1),
            nn.ReLU(),
            nn.MaxPool2d(2, stride = 2, return_indices = True),
            # conv2
            nn.Conv2d(64, 128, 3, padding = 1),
            nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding = 1),
            nn.ReLU(),
            nn.MaxPool2d(2, stride = 2, return_indices = True),
            # conv3
            nn.Conv2d(128, 256, 3, padding = 1),
            nn.ReLU(),
            nn.Conv2d(256, 256, 3, padding = 1),
            nn.ReLU(),
            nn.Conv2d(256, 256, 3, padding = 1),
            nn.ReLU(),
            nn.MaxPool2d(2, stride = 2, return_indices = True),
            # conv4
            nn.Conv2d(256, 512, 3, padding = 1),
            nn.ReLU(),
            nn.Conv2d(512, 512, 3, padding = 1),
            nn.ReLU(),
            nn.Conv2d(512, 512, 3, padding = 1),
            nn.ReLU(),
            nn.MaxPool2d(2, stride = 2, return_indices = True),
            # conv5
            nn.Conv2d(512, 512, 3, padding = 1),
            nn.ReLU(),
            nn.Conv2d(512, 512, 3, padding = 1),
            nn.ReLU(),
            nn.Conv2d(512, 512, 3, padding = 1),
            nn.ReLU(),
            nn.MaxPool2d(2, stride = 2, return_indices = True)
        )

        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),  # 224x244 image pooled down to 7x7 from features
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(4096, n_classes)
        )

        self.feat_maps = OrderedDict()  # store all (conv) feature maps

        self.pool_locs = OrderedDict() # store all max locations for pooling layers

        # index of convolutional layers
        self.conv_layer_indices = [0, 2, 5, 7, 10, 12, 14, 17, 19, 21, 24, 26, 28]

        self.init_weights() # initialize weights

    # initialize weights using pre-trained vgg16 on ImageNet
    def init_weights(self):
        vgg16_pretrained = models.vgg16(pretrained = True)
        for idx, layer in enumerate(vgg16_pretrained.features): # feature component
            if isinstance(layer, nn.Conv2d):
                self.features[idx].weight.data = layer.weight.data
                self.features[idx].bias.data = layer.bias.data

        for idx, layer in enumerate(vgg16_pretrained.classifier):   # classifier component
            if isinstance(layer, nn.Linear):
                self.classifier[idx].weight.data = layer.weight.data
                self.classifier[idx].bias.data = layer.bias.data

    def forward(self, x):
        for idx, layer in enumerate(self.features): # pass self.features
            if isinstance(layer, nn.MaxPool2d):
                x, locs = layer(x)
            else:
                x = layer(x)

        x = x.view(x.size()[0], -1) # reshape to (1, 512 * 7 * 7)

        output = self.classifier(x) # pass self.classifier

        return output