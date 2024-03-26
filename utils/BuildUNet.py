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

    '''
    Builds u-net, an encoder-decoder architecture for segmentation.
    
    Args:
        layer_sizes (list): list of integers representing the number of channels in each layer, where the length of the list is the number of layers
        in_channels (int): number of channels in input images
        out_channels (int): number of channels in output segmentations
        dropout_rate (float): dropout rate for dropout layers
        conv_per_block (int): number of convolutional layers per block
        hidden_activation (torch.nn.Module): activation function for hidden layers
        output_activation (torch.nn.Module): activation function for output layer

    Inputs:
        batch (tensor of size (B, in_channels, H, W)): batch of input images
    
    Returns:
        batch (tensor of size (B, out_channels, H, W)): batch of output segmentations
    '''

    def __init__(self,
                layer_sizes = [16, 32, 64, 128],
                in_channels=3,
                out_channels=4,
                dropout_rate=0.1,
                conv_per_block=1,
                hidden_activation=torch.nn.SELU(),
                output_activation=None):

        super().__init__() #inherit attrs. from Module

        self.enc_layer_sizes = layer_sizes
        self.dec_layer_sizes = layer_sizes[::-1]
        self.num_enc_layers = len(layer_sizes)
        self.num_dec_layers = len(layer_sizes)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = 3
        self.padding = 1
        self.dropout_rate=dropout_rate
        self.conv_per_block=conv_per_block
        self.hidden_activation=hidden_activation

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
        if output_activation is not None:
            self.final_activation = output_activation
        else:
            self.final_activation = torch.nn.Identity()


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
                self.hidden_activation,
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
                    self.hidden_activation,
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

            if (i+1) % ((self.conv_per_block * 4) + 1) == (self.conv_per_block * 4): #(self.conv_per_block * 4) layers per encoder block; on the penultimate layer, save output in cache
                cache.append(x) #save layer outputs in cache for skip connections

        #propagate through bottleneck layer
        for bottleneck_layer in self.bottleneck_layers:
            x = bottleneck_layer(x)

        #propagate through decoder layers
        j = 0 # set index control var for cache
        for i, dec_layer in enumerate(self.dec_layers):
            
            x = dec_layer(x)

            if (i+1) % ((self.conv_per_block * 4) + 2) == 2: # (self.conv_per_block * 4) + 2 layers per decoder block; on the 2nd layer, concatenate with cache
                x = torch.cat([x, cache[-(j+1)]], dim=1)
                j += 1

        #apply final conv layer
        x = self.final_layer(x)
        x = self.final_activation(x)
        return x