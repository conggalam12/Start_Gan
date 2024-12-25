import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as dset
import torchvision.utils as vutils
from torchvision.utils import make_grid
import matplotlib.pyplot as plt
from discriminator import Discriminator
from generator import Generator
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

CUDA = torch.cuda.is_available()
print("PyTorch version: {}".format(torch.__version__))
if CUDA:
    print("CUDA version: {}\n".format(torch.version.cuda))

if CUDA:
    torch.cuda.manual_seed(1)
device = torch.device("cuda:0" if CUDA else "cpu")
cudnn.benchmark = True

dataset = dset.MNIST(root='data', download=True,
                     transform=transforms.Compose([
                     transforms.Resize(64),
                     transforms.ToTensor(),
                     transforms.Normalize((0.5,), (0.5,))
                     ]))

# Dataloader
dataloader = torch.utils.data.DataLoader(dataset, batch_size=128,
                                         shuffle=True, num_workers=2)

# Plot training images
# real_batch = next(iter(dataloader))
# plt.figure(figsize=(8,8))
# plt.axis("off")
# plt.title("Training Images")
# plt.imshow(np.transpose(vutils.make_grid(real_batch[0].to(device)[:64], padding=2, normalize=True).cpu(),(1,2,0)))


generator = Generator().to(device)
generator.apply(weights_init)

discriminator = Discriminator().to(device)
discriminator.apply(weights_init)

criterion = nn.BCELoss()

# Create batch of latent vectors that I will use to visualize the progression of the generator
viz_noise = torch.randn(128, 100, 1, 1, device=device)

# Setup Adam optimizers for both G and D
optimizerD = optim.Adam(discriminator.parameters(), lr=1e-4, betas=(0.5, 0.999))
optimizerG = optim.Adam(generator.parameters(), lr=1e-4, betas=(0.5, 0.999))

g_losses = []
d_losses = []

for epoch in range(100):
    g_loss_epoch = 0
    d_loss_epoch = 0
    
    for i, data in enumerate(dataloader, 0):
        batch_size = data[0].size(0)
        
        # ---------------------
        #  Train Discriminator
        # ---------------------
        discriminator.zero_grad()
        
        # Train with real images
        real_images = data[0].to(device)
        real_labels = torch.ones(batch_size, 1).to(device) * 0.9  # Label smoothing
        fake_labels = torch.zeros(batch_size, 1).to(device)
        
        d_real_output = discriminator(real_images)
        d_real_loss = criterion(d_real_output, real_labels)
        d_real_loss.backward()
        
        # Train with fake images
        noise = torch.randn(batch_size, 100, 1, 1, device=device)
        fake_images = generator(noise)
        d_fake_output = discriminator(fake_images.detach())  # Detach to avoid training G
        d_fake_loss = criterion(d_fake_output, fake_labels)
        d_fake_loss.backward()
        
        # Total discriminator loss
        d_loss = (d_real_loss + d_fake_loss) * 0.5  # Average the losses
        optimizerD.step()
        
        # -----------------
        #  Train Generator
        # -----------------
        generator.zero_grad()
        
        # Generate new fake images
        noise = torch.randn(batch_size, 100, 1, 1, device=device)  # New noise
        fake_images = generator(noise)
        g_output = discriminator(fake_images)
        g_loss = criterion(g_output, real_labels)
        g_loss.backward()
        optimizerG.step()
        
        # Save losses
        g_loss_epoch += g_loss.item()
        d_loss_epoch += d_loss.item()
        
        # Print batch progress
        if i % 100 == 0:
            print(f'[{epoch}/{100}][{i}/{len(dataloader)}] '
                  f'D_loss: {d_loss.item():.4f} G_loss: {g_loss.item():.4f}')
    
    # Calculate average epoch losses
    g_losses.append(g_loss_epoch/len(dataloader))
    d_losses.append(d_loss_epoch/len(dataloader))
    
    # Print epoch stats
    print(f'Epoch [{epoch}/{100}] '
          f'D_loss: {d_loss_epoch/len(dataloader):.4f} '
          f'G_loss: {g_loss_epoch/len(dataloader):.4f}')
    
    # Generate and save sample images
    if epoch % 5 == 0:
        with torch.no_grad():
            generator.eval()
            n = torch.randn(16, 100, 1, 1).cuda()
            random_samples = generator(n).view(16, 1, 64, 64)
            grid = make_grid(random_samples.cpu(), nrow=4, normalize=True)
            plt.figure(figsize=(10, 10))
            plt.imshow(grid.permute(1, 2, 0))
            plt.axis('off')
            plt.title(f'Generated Images at Epoch {epoch}', fontsize=15)
            plt.savefig(f'generated_images_epoch_{epoch}.png', 
                       bbox_inches='tight', pad_inches=0.1)
            plt.close()
            generator.train()

