import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
class Block(nn.Module):
    def __init__(self, in_channels, out_channels, down=True, act='relu', use_dropout=False, device="cpu"):
        """
        Initialize a Block, which can act as either a downsampling or upsampling layer.
        
        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            down (bool, optional): If True, performs downsampling using Conv2d; else, upsampling using ConvTranspose2d. Defaults to True.
            act (str, optional): Activation function to use ('relu' or 'leaky'). Defaults to 'relu'.
            use_dropout (bool, optional): Whether to include a dropout layer. Defaults to False.
            device (str, optional): Device to run the operations on ('cuda' or 'cpu'). Defaults to "cuda".
        """
        super().__init__()  # Initialize the parent class
        self.device = device  # Set the device
        self.conv = nn.Sequential(
            nn.Conv2d(
                in_channels, 
                out_channels, 
                kernel_size=4, 
                stride=2, 
                padding=1, 
                bias=False, 
                padding_mode='reflect', 
                device=self.device
            ) if down else nn.ConvTranspose2d(
                in_channels, 
                out_channels, 
                kernel_size=4, 
                stride=2, 
                padding=1, 
                bias=False, 
                device=self.device
            ),  # Choose Conv2d for downsampling or ConvTranspose2d for upsampling
            nn.BatchNorm2d(out_channels, device=self.device),  # Batch normalization
            nn.ReLU() if act == 'relu' else nn.LeakyReLU(0.2),  # Activation function
        )
        self.use_dropout = use_dropout  # Set whether to use dropout
        self.dropout = nn.Dropout(0.5)  # Define a dropout layer with 50% dropout rate
    
    def forward(self, x):
        """
        Forward pass through the Block.
        
        Args:
            x (torch.Tensor): Input tensor.
        
        Returns:
            torch.Tensor: Output tensor after convolution, normalization, activation, and optional dropout.
        """
        x = self.conv(x)  # Apply convolution, normalization, and activation
        return self.dropout(x) if self.use_dropout else x  # Apply dropout if enabled, else return the tensor


class Generator(nn.Module):
    def __init__(self, in_channels=3, features=64, device="cuda"):
        """
        Initialize the Generator model using a series of downsampling and upsampling Blocks.
        
        Args:
            in_channels (int, optional): Number of input channels. Defaults to 3 (e.g., RGB images).
            features (int, optional): Base number of feature channels. Defaults to 64.
            device (str, optional): Device to run the operations on ('cuda' or 'cpu'). Defaults to "cuda".
        """
        super().__init__()  # Initialize the parent class
        self.device = device  # Set the device
        self.initial_down = nn.Sequential(
            nn.Conv2d(
                in_channels, 
                features, 
                kernel_size=4, 
                stride=2, 
                padding=1, 
                padding_mode='reflect', 
                device=self.device
            ),  # Initial convolutional layer
            nn.LeakyReLU(0.2),  # Leaky ReLU activation
        )  # Output size: 128
        
        # Define downsampling Blocks
        self.down1 = Block(features, features*2, down=True, act='leaky', use_dropout=False)   # Output size: 64
        self.down2 = Block(features*2, features*4, down=True, act='leaky', use_dropout=False) # Output size: 32
        self.down3 = Block(features*4, features*8, down=True, act='leaky', use_dropout=False) # Output size: 16
        self.down4 = Block(features*8, features*8, down=True, act='leaky', use_dropout=False) # Output size: 8
        self.down5 = Block(features*8, features*8, down=True, act='leaky', use_dropout=False) # Output size: 4
        self.down6 = Block(features*8, features*8, down=True, act='leaky', use_dropout=False) # Output size: 2
        
        # Bottleneck layer
        self.bottleneck = nn.Sequential(
            nn.Conv2d(
                features*8, 
                features*8, 
                kernel_size=4, 
                stride=2, 
                padding=1, 
                padding_mode='reflect', 
                device=self.device
            ),  # Convolutional layer at the bottleneck
            nn.ReLU(),  # ReLU activation
        )  # Output size: 1x1
        
        # Define upsampling Blocks with skip connections
        self.up1 = Block(features*8, features*8, down=False, act='relu', use_dropout=True)  # Output size: 2
        self.up2 = Block(features*8*2, features*8, down=False, act='relu', use_dropout=True)  # Output size: 4
        self.up3 = Block(features*8*2, features*8, down=False, act='relu', use_dropout=True)  # Output size: 8
        self.up4 = Block(features*8*2, features*8, down=False, act='relu', use_dropout=False) # Output size: 16
        self.up5 = Block(features*8*2, features*4, down=False, act='relu', use_dropout=False) # Output size: 32
        self.up6 = Block(features*4*2, features*2, down=False, act='relu', use_dropout=False) # Output size: 64
        self.up7 = Block(features*2*2, features, down=False, act='relu', use_dropout=False)   # Output size: 128
        
        # Final upsampling layer to reconstruct the image
        self.finil_up = nn.Sequential(
            nn.ConvTranspose2d(
                features*2, 
                in_channels, 
                kernel_size=4, 
                stride=2, 
                padding=1, 
                device=self.device
            ),  # Transposed convolution to upsample to original image size
            nn.Tanh(),  # Tanh activation to scale the output between -1 and 1
        )
        
    def forward(self, x):
        """
        Forward pass through the Generator, implementing the U-Net architecture with skip connections.
        
        Args:
            x (torch.Tensor): Input tensor (e.g., an image).
        
        Returns:
            torch.Tensor: Output tensor representing the generated image.
        """
        d1 = self.initial_down(x)  # Apply initial downsampling
        d2 = self.down1(d1)  # Apply first downsampling Block
        d3 = self.down2(d2)  # Apply second downsampling Block
        d4 = self.down3(d3)  # Apply third downsampling Block
        d5 = self.down4(d4)  # Apply fourth downsampling Block
        d6 = self.down5(d5)  # Apply fifth downsampling Block
        d7 = self.down6(d6)  # Apply sixth downsampling Block
        
        bottleneck = self.bottleneck(d7)  # Apply bottleneck layer
        
        up1 = self.up1(bottleneck)  # Apply first upsampling Block
        up2 = self.up2(torch.cat([up1, d7], 1))  # Concatenate with corresponding downsampled feature map and apply second upsampling Block
        up3 = self.up3(torch.cat([up2, d6], 1))  # Concatenate with corresponding downsampled feature map and apply third upsampling Block
        up4 = self.up4(torch.cat([up3, d5], 1))  # Concatenate with corresponding downsampled feature map and apply fourth upsampling Block
        up5 = self.up5(torch.cat([up4, d4], 1))  # Concatenate with corresponding downsampled feature map and apply fifth upsampling Block
        up6 = self.up6(torch.cat([up5, d3], 1))  # Concatenate with corresponding downsampled feature map and apply sixth upsampling Block
        up7 = self.up7(torch.cat([up6, d2], 1))  # Concatenate with corresponding downsampled feature map and apply seventh upsampling Block

        # Concatenate with initial downsampled feature map and apply final upsampling to generate the output image
        return self.finil_up(torch.cat([up7, d1], 1))
    

if __name__ == "__main__":
    image = Image.open(r"C:\Users\cong_nguyen\Documents\Python\Start_Gan\Monet\img\047da870f6.jpg").convert('RGB')
    transform_list = []
    transform_list.append(transforms.ToTensor())
    transform_list.append(
            transforms.Normalize(
                mean=[0.5,0.5,0.5],
                std=[0.5,0.5,0.5]
            )
        )
    transform = transforms.Compose(transform_list)
    sample_input = transform(image)
    generator = Generator(3,64,"cpu")
    generator.load_state_dict(torch.load(r'C:\Users\cong_nguyen\Documents\Python\Start_Gan\Monet\weight\generator_weights.pth',weights_only=True,map_location="cpu"))
    generator.eval()
    output = generator(sample_input.unsqueeze(0))
    output = output.squeeze(0)
    output = output*0.5 + 0.5
    transform = transforms.ToPILImage()
    image = transform(output)
    image.save(r"C:\Users\cong_nguyen\Documents\Python\Start_Gan\Monet\result\demo3.jpg")