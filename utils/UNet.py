# create U-Net class inheriting from torch.nn.Module

import os, sys, glob
import math
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import torch
from tqdm.auto import tqdm
from PIL import Image

class UNet(torch.nn.Module):

    def __init__(self,
                enc_layer_sizes = [16, 32, 64, 128],
                dec_layer_sizes = [128, 64, 32, 16],
                in_channels=3,
                out_channels=4,
                dropout_rate=0.1,
                conv_per_block=1):

        super().__init__() #inherit attrs. from Module

        self.enc_layer_sizes = enc_layer_sizes
        self.dec_layer_sizes = dec_layer_sizes
        self.num_enc_layers = len(enc_layer_sizes)
        self.num_dec_layers = len(dec_layer_sizes)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = 3
        self.padding = 1
        self.dropout_rate=dropout_rate
        self.conv_per_block=conv_per_block

        # create lists to hold each layer group
        self.enc_layers = torch.nn.ModuleList()
        self.bottleneck_layers = torch.nn.ModuleList()
        self.dec_layers = torch.nn.ModuleList()

        # create encoder blocks
        for i in range(self.num_enc_layers-1):
            if i == 0:
                self.enc_layers += self.enc_layer(self.in_channels, self.enc_layer_sizes[i], self.kernel_size, self.padding)
            else:
                self.enc_layers += self.enc_layer(self.enc_layer_sizes[i-1], self.enc_layer_sizes[i], self.kernel_size, self.padding)

        # create bottleneck block        
        self.bottleneck_layers += self.enc_layer(self.enc_layer_sizes[-2], self.enc_layer_sizes[-1], self.kernel_size, self.padding, pool=False)

        # create decoder blocks
        for i in range((self.num_dec_layers-1)):
            self.dec_layers += self.dec_layer(self.dec_layer_sizes[i], self.dec_layer_sizes[i+1], self.kernel_size, self.padding)

        # add final layer
        self.final_layer = torch.nn.Conv2d(self.dec_layer_sizes[-1], out_channels, kernel_size=1, padding=0)
        self.final_activation = torch.nn.Softmax(dim=1)


    def conv_block(self, in_channels, out_channels, kernel_size, padding):
        
        conv_block = torch.nn.Sequential()
        for i in range(self.conv_per_block):

            if i == 0:
                conv_block += torch.nn.Sequential(torch.nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=self.kernel_size,
                    padding=self.padding),
                torch.nn.BatchNorm2d(out_channels),
                torch.nn.LeakyReLU(),
                torch.nn.Dropout2d(self.dropout_rate)
                )
            else:
                # define base conv_block structure
                conv_block += torch.nn.Sequential(
                    torch.nn.Conv2d(
                        in_channels=out_channels,
                        out_channels=out_channels,
                        kernel_size=self.kernel_size,
                        padding=self.padding),
                    torch.nn.BatchNorm2d(out_channels),
                    torch.nn.LeakyReLU(),
                    torch.nn.Dropout2d(self.dropout_rate)
                )
        return conv_block
    
    def enc_layer(self, in_channels, out_channels, kernel_size, padding, pool=True):

        # define encoder layer structure
        enc_layer = self.conv_block(in_channels,
                                    out_channels,
                                    kernel_size=self.kernel_size, 
                                    padding=self.padding)
        if pool == True:
            enc_layer.append(torch.nn.MaxPool2d(2))

        return enc_layer

    def dec_layer(self, in_channels, out_channels, kernel_size, padding, upsample=True):

        # define decoder layer structure
        if upsample == True:
            upsample = torch.nn.Sequential(
                torch.nn.Upsample(scale_factor=2, mode='nearest'),
                torch.nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=self.kernel_size,
                    padding=self.padding)
            )
            #prepend upsampling layer to conv_block to make decoder
            dec_layer = upsample + self.conv_block(in_channels, 
                                                    out_channels, 
                                                    self.kernel_size, 
                                                    self.padding)

        else:
            dec_layer = self.conv_block(in_channels, out_channels, self.kernel_size, self.padding)

        return dec_layer


    # define forward pass
    def forward(self, x):
        
        cache = []
        #propagate through encoder layers
        for i, enc_layer in enumerate(self.enc_layers):
            x = enc_layer(x)
            if (i+1) % 5 == 4: #5 layers per encoder block; on the 4th layer, save output in cache
                cache.append(x) #save layer outputs in cache for skip connections

        #propagate through bottleneck layer
        for bottleneck_layer in self.bottleneck_layers:
            x = bottleneck_layer(x)

        #propagate through decoder layers
        j = 0 # set index control var for cache
        for i, dec_layer in enumerate(self.dec_layers):
            x = dec_layer(x)
            if (i+1) % 6 == 2: # 6 layers per decoder block; on the 2nd layer, concatenate with cache
                x = torch.cat([x, cache[-(j+1)]], dim=1)
                j += 1

        #apply final conv layer
        x = self.final_layer(x)
        x = self.final_activation(x)
        return x