# ============================================================= #
# U-Net-Id                                                      #
#                                                               #
# This file contains building block classes for U-Net, and the  #
# U-Net-Id class itself. All subclass PyTorch's nn.Module.      #
# ============================================================= #

import torch
import torch.nn as nn
import numpy as np

##########
## UNET ##
##########

# NETWORK COMPONENTS #
# ------------------ #

class DoubleConv(nn.Module):
    '''
    Two consecutive 3x3 2D convolution operations and an ELU activation function.

    The first convolution increases (or decreases) the number of channels. 
    The second convolution has the same number of in and out channels. 
    The number of in and out channels are set by arguments to the __init__ function.

    Attributes:
        conv1 (torch.nn.Conv2d): A 2D convolution with kernel size of 3, 'same' padding in reflect mode
        conv2 (torch.nn.Conv2d): A 2D convolution with kernel size of 3, 'same' padding in reflect mode
        elu (torch.nn.ELU): ELU function
    '''

    def __init__(self, c_in, c_out):
        '''
        Initializes class attributes.

        Args:
            c_in (int): Number of channels at the input of the convolution block.
            c_out (int): Number of channels at the output of the convolution block.
        '''
        super().__init__()
        self.conv1 = nn.Conv2d(c_in, c_out, kernel_size=3, padding='same', padding_mode='reflect')
        self.conv2 = nn.Conv2d(c_out, c_out, kernel_size=3, padding='same', padding_mode='reflect')
        self.elu = nn.ELU()
        
    def forward(self, inputs):
        '''
        Defines forward pass of DoubleConv.

        Args:
            inputs (torch.Tensor): Input to the DoubleConv
        '''
        x = self.conv1(inputs)
        x = self.elu(x)
        x = self.conv2(x)
        x = self.elu(x)
        return x

class EncoderBlock(nn.Module):
    '''
    Encoder block for a U-Net.

    The encoder block consists of a double convolution and a 2x2 max pooling operation. 
    It reduces image size and increases channel depth, capturing context.

    Attributes:
        c_in (int): Number of channels at the input of the EncoderBlock.
        c_out (int): Number of channels at the output of the EncoderBlock.
    '''

    def __init__(self, c_in, c_out):
        '''
        Initializes class attributes.

        Args:
            c_in (int): Number of channels at the input of the EncoderBlock.
            c_out (int): Number of channels at the output of the EncoderBlock.
        '''
        super().__init__()
        self.conv = DoubleConv(c_in, c_out)
        self.pool = nn.MaxPool2d((2, 2))
     
    def forward(self, inputs):
        '''
        Defines forward pass of EncoderBlock.

        Args:
            inputs (torch.Tensor): Input to the EncoderBlock
        '''
        x = self.conv(inputs)
        p = self.pool(x)
        return x, p

class DecoderBlock(nn.Module):
    '''
    Decoder block for a U-Net.

    The decoder block consists of a 2x2 transposed convolution operation followed 
    by a double convolution. 
    Between the two operations, the image is concatenated with another image. 
    It increases image size and decreases channel depth, localizing details.

    Attributes:
        c_in (int): Number of channels at the input of the DecoderBlock.
        c_out (int): Number of channels at the output of the DecoderBlock.
    '''

    def __init__(self, c_in, c_out):
        '''
        Initializes class attributes.

        Args:
            c_in (int): Number of channels at the input of the DecoderBlock.
            c_out (int): Number of channels at the output of the DecoderBlock.
        '''
        super().__init__()
        self.up = nn.ConvTranspose2d(c_in, c_out, kernel_size=2, stride=2, padding=0)
        self.conv = DoubleConv(c_in, c_out)
    
    def forward(self, inputs, skip):
        '''
        Defines forward pass of DecoderBlock.

        Args:
            inputs (torch.Tensor): Input to the DecoderBlock
            skip (torch.Tensor): Tensor (image) to be concatenated to the input.
        '''
        x = self.up(inputs)
        x = torch.cat([x, skip], axis=1)
        x = self.conv(x)
        return x

class Bottleneck(nn.Module):
    '''
    Wrapper for DoubleConv class. (Bottleneck for a U-Net)

    Attributes:
        conv (DoubleConv): Convolutions and activation functions as defined in DoubleConv
    '''

    def __init__(self, c_in, c_out):
        '''
        Initializes class attributes.

        Args:
            c_in (int): Number of channels at the input of the DoubleConv block.
            c_out (int): Number of channels at the output of the DoubleConv block.
        '''
        super().__init__()
        self.conv = DoubleConv(c_in, c_out)
    
    def forward(self, inputs):
        '''
        Defines forward pass of Bottleneck.

        Args:
            inputs (torch.Tensor): Input to the Bottleneck
        '''
        x = self.conv(inputs)
        return x

class OutConv(nn.Module):
    '''
    A 1x1 2D convolution operation with a sigmoid activation function.

    This is the output layer of a U-Net in the U-Net-Id architecture.

    Attributes:
        conv (torch.nn.Conv2d): A 2D convolution with kernel size of 1, 'same' padding in reflect mode
        sigmoid (torch.nn.Sigmoid): Sigmoid function
    '''
    
    def __init__(self, c_in, c_out):
        '''
        Initializes class attributes.

        Args:
            c_in (int): Number of channels at the input of the OutConv block.
            c_out (int): Number of channels at the output of the OutConv block.
        '''
        super().__init__()
        self.conv = nn.Conv2d(c_in, c_out, kernel_size=1, padding='same', padding_mode='reflect')
        self.sigmoid = nn.Sigmoid()

    def forward(self, inputs):
        '''
        Defines forward pass of OutConv.

        Args:
            inputs (torch.Tensor): Input to the OutConv
        '''
        x = self.conv(inputs)
        x = self.sigmoid(x)
        return x



# U-NET #
# ----- #

class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()
        
        # Encoder
        self.e1 = EncoderBlock(1, 64)
        self.e2 = EncoderBlock(64, 128)
        self.e3 = EncoderBlock(128, 256)
        self.e4 = EncoderBlock(256, 512)
        
        # Bottleneck
        self.b = Bottleneck(512, 1024)
        
        # Decoder
        self.d1 = DecoderBlock(1024, 512)
        self.d2 = DecoderBlock(512, 256)
        self.d3 = DecoderBlock(256, 128)
        self.d4 = DecoderBlock(128, 64)
        
        # Output layer
        self.out = OutConv(64, 1)
    
    def forward(self, inputs):
        
        # Encoder
        s1, p1 = self.e1(inputs)
        s2, p2 = self.e2(p1)
        s3, p3 = self.e3(p2)
        s4, p4 = self.e4(p3)
        
        # Bottleneck
        b = self.b(p4)
        
        # Decoder
        d1 = self.d1(b, s4)
        d2 = self.d2(d1, s3)
        d3 = self.d3(d2, s2)
        d4 = self.d4(d3, s1)
        
        # Classifier
        outputs = self.out(d4)
        return outputs


#####################
## FULL UNETID NET ##
#####################

# Clasees for the final part of the U-Net-Id.
# TODO: Give them better names and docs (integrate into rest of file structure)

class BlueConv(nn.Module):
    # 3x3 kernel 2D convolution with ELU activation fcn
    def __init__(self, c_in, c_out):
        super().__init__()
        self.conv = nn.Conv2d(c_in, c_out, kernel_size=3, padding='same', padding_mode='reflect')
        self.elu = nn.ELU()
    
    def forward(self, inputs):
        x = self.conv(inputs)
        x = self.elu(x)
        return x

class RedConv(nn.Module):
    # 1x1 kernel 2D convolution with sigmoid activation fcn
    def __init__(self, c_in, c_out):
        super().__init__()
        self.conv = nn.Conv2d(c_in, c_out, kernel_size=1, padding='same', padding_mode='reflect')
        self.sigmoid = nn.Sigmoid()

    def forward(self, inputs):
        x = self.conv(inputs)
        x = self.sigmoid(x)
        return x


# U-Net-Id #
# -------- #

class Unetid(nn.Module):
    def __init__(self):
        super().__init__()

        # U-Nets (same architecture, different params)
        self.unet_a = UNet()
        self.unet_b = UNet()
        self.unet_c = UNet()

        # Second section of network
        self.conv1_a = BlueConv(3, 64)
        self.conv2_a = BlueConv(64, 32)
        self.conv3_a = RedConv(32, 1)

        self.conv1_b = BlueConv(3, 64)
        self.conv2_b = BlueConv(64, 32)
        self.conv3_b = RedConv(32, 1)
        
        self.conv1_c = BlueConv(3, 64)
        self.conv2_c = BlueConv(64, 32)
        self.conv3_c = RedConv(32, 1)

    def forward(self, inputs):

        xa = self.unet_a(inputs)
        xb = self.unet_b(inputs)
        xc = self.unet_c(inputs)

        inputsnext = torch.cat([xa, xb, xc], axis=1)

        ya = self.conv3_a(self.conv2_a(self.conv1_a(inputsnext)))
        yb = self.conv3_b(self.conv2_b(self.conv1_b(inputsnext)))
        yc = self.conv3_c(self.conv2_c(self.conv1_c(inputsnext)))

        output = torch.cat([ya, yb, yc], axis=1)
        return output
