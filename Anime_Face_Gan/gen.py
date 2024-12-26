import torch.nn as nn
import torch
from torchvision.utils import save_image
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            # input layer
            nn.ConvTranspose2d(100, 64 * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(64 * 8),
            nn.ReLU(True),
            # 1st hidden layer
            nn.ConvTranspose2d(64 * 8, 64 * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64 * 4),
            nn.ReLU(True),
            # 2nd hidden layer
            nn.ConvTranspose2d(64 * 4, 64 * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64 * 2),
            nn.ReLU(True),
            # 3rd hidden layer
            nn.ConvTranspose2d(64 * 2, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            # output layer
            nn.ConvTranspose2d(64, 3, 4, 2, 1, bias=False),
            nn.Tanh()
        )

    def forward(self, input):
        return self.main(input)
    

if __name__ == "__main__":
    model = Generator()
    model.load_state_dict(torch.load(r'C:\Users\cong_nguyen\Documents\Python\Start_Gan\Anime_Face_Gan\weights\anime_gen.pt',weights_only=True,map_location="cpu"))
    model.eval()
    sample_input = torch.rand(1,100,1,1)
    output = model(sample_input)
    output = output*0.5 + 0.5
    output = output.squeeze(0)
    transform = transforms.ToPILImage()
    image = transform(output)
    image.save(r"C:\Users\cong_nguyen\Documents\Python\Start_Gan\Anime_Face_Gan\result\demo.jpg")