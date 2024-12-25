import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision import transforms , datasets
from disc import Discriminator
from gen import Generator
import torch.nn.functional as F
import tqdm
from torchvision.utils import save_image
import os
def train_discriminator(real_images, opt_d):
    # Clear discriminator gradients
    opt_d.zero_grad()

    # Pass real images through discriminator
    real_preds = discriminator(real_images)
    real_targets = torch.ones(real_images.size(0), 1, device=device)
    real_loss = F.binary_cross_entropy(real_preds, real_targets)
    real_score = torch.mean(real_preds).item()
    
    # Generate fake images
    latent = torch.randn(batch_size, 100, 1, 1, device=device)
    fake_images = generator(latent)

    # Pass fake images through discriminator
    fake_targets = torch.zeros(fake_images.size(0), 1, device=device)
    fake_preds = discriminator(fake_images)
    fake_loss = F.binary_cross_entropy(fake_preds, fake_targets)
    fake_score = torch.mean(fake_preds).item()

    # Update discriminator weights
    loss = real_loss + fake_loss
    loss.backward()
    opt_d.step()
    return loss.item(), real_score, fake_score
def train_generator(opt_g):
    # Clear generator gradients
    opt_g.zero_grad()
    
    # Generate fake images
    latent = torch.randn(batch_size, 100, 1, 1, device=device)
    fake_images = generator(latent)
    
    # Try to fool the discriminator
    preds = discriminator(fake_images)
    targets = torch.ones(batch_size, 1, device=device)
    loss = F.binary_cross_entropy(preds, targets)
    
    # Update generator weights
    loss.backward()
    opt_g.step()
    
    return loss.item()
def denorm(img_tensors):
    return img_tensors * 0.5 + 0.5
def save_samples(index, latent_tensors,generator,sample_dir = "result"):
    fake_images = generator(latent_tensors)
    fake_fname = 'generated-images-{0:0=4d}.png'.format(index)
    save_image(denorm(fake_images), os.path.join(sample_dir, fake_fname), nrow=8)
    print('Saving', fake_fname)

    
trans = transforms.Compose([
                     transforms.Resize(64),
                     transforms.CenterCrop(64) ,
                     transforms.ToTensor(),
                     transforms.Normalize((0.5,), (0.5,))
                    ])

dataset = datasets.ImageFolder(root='data\3\images', transform=trans)

batch_size = 64  # Adjust as needed
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

discriminator = Discriminator()
discriminator.to(device)

generator = Generator()
generator.to(device)

torch.cuda.empty_cache()
    
losses_g = []
losses_d = []
real_scores = []
fake_scores = []

# Create optimizers
opt_d = torch.optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))
opt_g = torch.optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
fixed_latent = torch.randn(64, 100, 1, 1, device=device)

for epoch in range(100):
    for real_images, _ in tqdm(dataloader):
        # Train discriminator
        loss_d, real_score, fake_score = train_discriminator(real_images, opt_d)
        # Train generator
        loss_g = train_generator(opt_g)
        
    # Record losses & scores
    losses_g.append(loss_g)
    losses_d.append(loss_d)
    real_scores.append(real_score)
    fake_scores.append(fake_score)
    
    # Log losses & scores (last batch)
    print("Epoch [{}/{}], loss_g: {:.4f}, loss_d: {:.4f}, real_score: {:.4f}, fake_score: {:.4f}".format(
        epoch+1, 100, loss_g, loss_d, real_score, fake_score))

    # Save generated images
    save_samples(epoch+1, fixed_latent, show=False)