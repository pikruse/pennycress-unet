import torch, pdb
import torch.nn as nn

class BuildUNet(nn.Module):
    
    '''
    Builds u-net, an encoder-decoder architecture for segmentation.
    
    Args:
        input_features (int): number of input features/channels
        output_features (int): number of output channels
        layers (list): contains integer layer sizes
        hidden_activation (callable): instantiated activation function
        output_activation (callable): instantiated activation function
        n_convs (int): number of convolutions per layer
        use_batchnorm (bool): indicates whether to use batchnorm
        drouput_rate (float): dropout rate
        dimension (int): data dimensionality (1D, 2D, or 3D)
    
    Inputs:
        batch (tensor): batch of input images
    
    Returns:
        batch (tensor): batch of output segmentations
    '''

    def __init__(
        self, 
        input_features,
        output_features,
        layers, 
        hidden_activation=nn.LeakyReLU(),
        output_activation=None,
        n_convs=1,
        use_batchnorm=True,
        dropout_rate=0.0,
        dimension=2):
        
        super().__init__()
        self.input_features = input_features
        self.output_features = output_features
        self.layers = layers
        self.hidden_activation = hidden_activation
        self.output_activation = output_activation
        self.n_convs = n_convs
        self.use_batchnorm = use_batchnorm
        self.dropout_rate = dropout_rate
        self.dimension = dimension

        # select between 1D, 2D, and 3D convolutions
        if self.dimension == 1:
            self.conv_fun = nn.Conv1d
            self.drop_fun = nn.Dropout1d
            self.batch_fun = nn.BatchNorm1d
        elif self.dimension == 3:
            self.conv_fun = nn.Conv3d
            self.drop_fun = nn.Dropout3d
            self.batch_fun = nn.BatchNorm3d
        else:
            self.conv_fun = nn.Conv2d
            self.drop_fun = nn.Dropout2d
            self.batch_fun = nn.BatchNorm2d

        # input convolutions
        current_features = self.layers[0]
        self.input_conv = self.convolution(input_features, current_features)

        # encoder convolutions
        self.encoder_ops = nn.ModuleList()
        for layer in self.layers[1:]:
            self.encoder_ops.append(nn.Sequential(
                self.downsample(current_features, current_features),
                self.convolution(current_features, layer)))
            current_features = layer
        
        # decoder operations
        self.decoder_ups = nn.ModuleList()
        self.decoder_convs = nn.ModuleList()
        for layer in self.layers[::-1][1:]:
            self.decoder_ups.append(self.upsample(current_features, current_features))
            self.decoder_convs.append(self.convolution(current_features+layer, layer))
            current_features = layer

        # output convolution
        self.output_conv = self.conv_fun(
                in_channels=current_features,
                out_channels=output_features,
                kernel_size=3,
                stride=1,
                padding=1)

    def convolution(self, n_inputs, n_outputs):

        # convolutional layers (conv, batchnorm, activation, dropout)
        operations = []
        for i in range(self.n_convs):
            operations.append(self.conv_fun(
                in_channels=n_inputs,
                out_channels=n_outputs,
                kernel_size=3,
                stride=1,
                padding=1))
            n_inputs = n_outputs
            if self.use_batchnorm:
                operations.append(self.batch_fun(n_outputs))
            operations.append(self.hidden_activation)
            operations.append(self.drop_fun(p=self.dropout_rate))

        return nn.Sequential(*operations)

    def downsample(self, n_inputs, n_outputs):

        # pooling layer (conv, batchnorm, activation)
        operations = []
        operations.append(self.conv_fun(
            in_channels=n_inputs,
            out_channels=n_outputs,
            kernel_size=3,
            stride=2,
            padding=1))
        if self.use_batchnorm:
            operations.append(self.batch_fun(n_outputs))
        operations.append(self.hidden_activation)

        return nn.Sequential(*operations)

    def upsample(self, n_inputs, n_outputs):

        # upsampling layer (upsample, conv, batchnorm, activation)
        operations = []
        operations.append(nn.Upsample(
            scale_factor=2, 
            mode='nearest'))
        operations.append(self.conv_fun(
            in_channels=n_inputs,
            out_channels=n_outputs,
            kernel_size=3,
            stride=1,
            padding=1))
        if self.use_batchnorm:
            operations.append(self.batch_fun(n_outputs))
        operations.append(self.hidden_activation)

        return nn.Sequential(*operations)

    def forward(self, batch):

        # encode 
        context = [self.input_conv(batch)]
        for i in range(len(self.encoder_ops)):
            context.append(self.encoder_ops[i](context[i]))

        # decode
        batch = context[-1]
        for i in range(len(self.decoder_ups)):
            batch = torch.cat([
                self.decoder_ups[i](batch),
                context[::-1][i+1]], dim=1)
            batch = self.decoder_convs[i](batch)

        # output convolution
        batch = self.output_conv(batch)
        if self.output_activation is not None:
            batch = self.output_activation(batch)
        
        return batch