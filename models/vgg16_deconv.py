import torch.nn as nn
import torchvision.models as models

import sys

class VGG16_Deconv(nn.Module):

    def __init__(self):
        super(VGG16_Deconv, self).__init__()

        self.features = nn.Sequential(
            # deconv1
            nn.MaxUnpool2d(2, stride = 2),
            nn.ReLU(),
            nn.Conv2d(512, 512, 3, padding = 1),
            nn.ReLU(),
            nn.Conv2d(512, 512, 3, padding = 1),
            nn.ReLU(),
            nn.Conv2d(512, 512, 3, padding = 1),
            # deconv2
            nn.MaxUnpool2d(2, stride = 2),
            nn.ReLU(),
            nn.ConvTranspose2d(512, 512, 3, padding = 1),
            nn.ReLU(),
            nn.ConvTranspose2d(512, 512, 3, padding = 1),
            nn.ReLU(),
            nn.ConvTranspose2d(512, 256, 3, padding = 1),
            # deconv3
            nn.MaxUnpool2d(2, stride = 2),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 256, 3, padding = 1),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 256, 3, padding = 1),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, 3, padding = 1),
            # deconv4
            nn.MaxUnpool2d(2, stride = 2),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 128, 3, padding = 1),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 3, padding = 1),
            # deconv5
            nn.MaxUnpool2d(2, stride = 2),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 64, 3, padding = 1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 3, 3, padding = 1)
        )

        # forward idx : backward idx
        self.conv2deconv_indices = {0:30, 2:28, 5:25, 7:23, 10:20, 12:18, 14:16, 17:13, 19:11, 21:9, 24:6, 26:4, 28:2}
        # forward idx : backward idx; not align
        self.conv2deconv_bias_indices = {0:28, 2:25, 5:23, 7:20, 10:18, 12:16, 14:13, 17:11, 19:9, 21:6, 24:4, 26:2}
        # forwardidx : backward idx
        self.relu2relu_indices = {1:29, 3:27, 6:24, 8:22, 11:19, 13:17, 15:15, 18:12, 20:10, 22:8, 25:5, 27:3, 29:1}
        # backward idx : forward idx
        self.unpool2pool_indices = {26:4, 21:9, 14:16, 7:23, 0:30}

        self.init_weights()  # initialize weights

    # initialize weights using pre-trained vgg16 on ImageNet
    def init_weights(self):
        vgg16_pretrained = models.vgg16(pretrained = True)
        for idx, layer in enumerate(vgg16_pretrained.features): # feature component
            if isinstance(layer, nn.Conv2d):
                self.features[self.conv2deconv_indices[idx]].weight.data = layer.weight.data
                if idx in self.conv2deconv_bias_indices:    # bias in first backward layer is randomly set
                    self.features[self.conv2deconv_bias_indices[idx]].bias.data = layer.bias.data

    def forward(self, x, layer, activation_idx, pool_locs):
        if layer in self.conv2deconv_indices:
            start_idx = self.conv2deconv_indices[layer]
        elif layer in self.relu2relu_indices:
            start_idx = self.relu2relu_indices[layer]
        else:
            print('No such Conv2d or RelU layer!')
            sys.exit(0)

        for idx in range(start_idx, len(self.features)):
            if isinstance(self.features[idx], nn.MaxUnpool2d):
                x = self.features[idx](x, pool_locs[self.unpool2pool_indices[idx]])
            else:
                x = self.features[idx](x)

        return x