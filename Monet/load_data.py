from torch.utils.data import Dataset
import torchvision
import os
from PIL import Image
class MonetPhoto(Dataset):
    def __init__(self, data_root, monet_path, photo_path):
        self.data_root = data_root  # Set the root directory for data
        self.monet_path = monet_path  # Set the subdirectory for Monet images
        self.monet_images = os.listdir(os.path.join(data_root, monet_path))  # List all Monet image filenames
        self.photo_path = photo_path  # Set the subdirectory for photo images
        self.photo_images = os.listdir(os.path.join(data_root, photo_path))  # List all photo image filenames
        self.transform =  torchvision.transforms.Compose([
                            torchvision.transforms.Resize((256, 256)),  # Resize images to 256x256 pixels
                            torchvision.transforms.ToTensor(),  # Convert PIL images to PyTorch tensors
                            torchvision.transforms.Normalize(0.5, 0.5) 
                        ])

    def __len__(self):
        return max(len(self.monet_images), len(self.photo_images)) 

    def __getitem__(self, idx):
        monet_image = Image.open(os.path.join(self.data_root, self.monet_path, self.monet_images[idx % len(self.monet_images)]))
        photo_image = Image.open(os.path.join(self.data_root, self.photo_path, self.photo_images[idx % len(self.photo_images)]))
        
        monet_image = self.transform(monet_image)  
        photo_image = self.transform(photo_image)  
        
        return monet_image, photo_image  
