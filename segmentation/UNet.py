import torch
import torch.nn as nn

########################
## NETWORK COMPONENTS ##
########################

class DoubleConv(nn.Module):
    '''
    Two 2D convolution operations and a ReLU activation function.

    The first convolution increases (or decreases) the number of channels. 
    The second convolution has the same number of in and out channels. 
    The number of in and out channels are set by arguments to the __init__ function.

    Attributes:
        conv1 (torch.nn.Conv2d): A 2D convolution with kernel size of 3, 'same' padding in reflect mode
        conv2 (torch.nn.Conv2d): A 2D convolution with kernel size of 3, 'same' padding in reflect mode
        relu (torch.nn.ReLU): ReLU function
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
        self.relu = nn.ReLU()
        
    def forward(self, inputs):
        '''
        Defines forward pass of DoubleConv.

        Args:
            inputs (torch.Tensor): Input to the DoubleConv
        '''
        x = self.conv1(inputs)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.relu(x)
        return x

class EncoderBlock(nn.Module):
    '''
    Encoder block for a U-Net.

    The encoder block consists of a double convolution and a max pooling operation. 
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

    The decoder block consists of a transposed convolution operation followed by a double convolution. 
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

