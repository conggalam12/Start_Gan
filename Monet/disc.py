import torch.nn as nn

class CNNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=2, device="cuda"):
        """
        Initialize the CNNBlock with convolution, batch normalization, and activation layers.
        
        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            stride (int, optional): Stride for the convolution. Defaults to 2.
            device (str, optional): Device to run the operations on ('cuda' or 'cpu'). Defaults to "cuda".
        """
        super().__init__()  # Initialize the parent class
        self.device = device  # Set the device
        self.conv = nn.Sequential(
            nn.Conv2d(
                in_channels, 
                out_channels, 
                kernel_size=4, 
                stride=stride, 
                bias=False, 
                padding_mode='reflect', 
                device=self.device
            ),  # Convolutional layer with specified parameters
            nn.BatchNorm2d(out_channels, device=self.device),  # Batch normalization to stabilize and accelerate training
            nn.LeakyReLU(0.2),  # Leaky ReLU activation with negative slope of 0.2
        )
        
    def forward(self, x):
        """
        Forward pass through the CNNBlock.
        
        Args:
            x (torch.Tensor): Input tensor.
        
        Returns:
            torch.Tensor: Output tensor after convolution, batch normalization, and activation.
        """
        return self.conv(x)  # Apply the sequential layers to the input


class Discriminator(nn.Module):
    def __init__(self, in_channels=3, features=[64, 128, 256, 512], device="cuda"):
        """
        Initialize the Discriminator model using a series of CNNBlocks.
        
        Args:
            in_channels (int, optional): Number of input channels. Defaults to 3 (e.g., RGB images).
            features (list, optional): List of feature sizes for each CNNBlock. Defaults to [64, 128, 256, 512].
            device (str, optional): Device to run the operations on ('cuda' or 'cpu'). Defaults to "cuda".
        """
        super().__init__()  # Initialize the parent class
        self.device = device  # Set the device
        self.initial = nn.Sequential(
            nn.Conv2d(
                in_channels, 
                features[0], 
                kernel_size=4, 
                stride=2, 
                padding=1, 
                padding_mode='reflect', 
                device=self.device
            ),  # Initial convolutional layer
            nn.LeakyReLU(0.2)  # Leaky ReLU activation
        )
        
        layers = []  # Initialize a list to hold subsequent CNNBlocks
        in_channels = features[0]  # Set the initial number of input channels
        for feature in features[1:]:
            layers.append(
                CNNBlock(in_channels, feature, stride=1 if feature == features[-1] else 2, device=self.device),
            )  # Append CNNBlock with appropriate stride
            in_channels = feature  # Update the number of input channels for the next block
        
        layers.append(
            nn.Conv2d(
                in_channels, 
                1, 
                kernel_size=4, 
                stride=1, 
                padding=1, 
                padding_mode='reflect', 
                device=self.device
            )
        )  # Final convolutional layer to produce a single output channel (real/fake classification)
            
        self.model = nn.Sequential(*layers)  # Combine all layers into a sequential model
        
    def forward(self, x):
        """
        Forward pass through the Discriminator.
        
        Args:
            x (torch.Tensor): Input tensor (e.g., an image).
        
        Returns:
            torch.Tensor: Output tensor representing the discriminator's prediction.
        """
        x = self.initial(x)  # Apply the initial convolution and activation
        return self.model(x)  # Apply the subsequent CNNBlocks and final convolution