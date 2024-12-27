
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
from gen import Generator
if __name__ == "__main__":
    image = Image.open(r"C:\Users\cong_nguyen\Documents\Python\Start_Gan\Monet\img\0162322d2d.jpg").convert('RGB')
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
    image.save(r"C:\Users\cong_nguyen\Documents\Python\Start_Gan\Monet\result\0162322d2d.jpg")