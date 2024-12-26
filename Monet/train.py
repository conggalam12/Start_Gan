import torch
from torch.utils.data import  DataLoader, random_split
import torchvision.transforms as transforms
from pathlib import Path
from tqdm import tqdm
from load_data import MonetPhoto
from disc import Discriminator
from gen import Generator

transform = transforms.ToPILImage()

class Trainer:
    def __init__(
        self,
        train_data: DataLoader,
        val_data: DataLoader,
        generator: torch.nn.Module,
        discriminator: torch.nn.Module,
        nb_epochs: int = 5,
        device: str = "cuda",
        save_path: str = None,
    ):
        """
        Initialize the Trainer with data loaders, models, training parameters, and device configuration.
        
        Args:
            train_data (DataLoader): DataLoader for training data.
            val_data (DataLoader): DataLoader for validation data.
            generator (torch.nn.Module): Generator model.
            discriminator (torch.nn.Module): Discriminator model.
            nb_epochs (int, optional): Number of training epochs. Defaults to 5.
            device (str, optional): Device to run the training on ('cuda' or 'cpu'). Defaults to "cuda".
            save_path (str, optional): Path to save the trained models. Defaults to None.
        """
        self.train_data = train_data        # Assign training data loader
        self.val_data = val_data            # Assign validation data loader
        self.generator = generator          # Assign generator model
        self.discriminator = discriminator  # Assign discriminator model
        self.nb_epochs = nb_epochs          # Set number of epochs
        self.device = device                # Set device for computation
        self.save_path = Path(save_path) if save_path else save_path  # Set save path as Path object if provided

        # Initialize a fixed set of validation samples for monitoring generator progress
        self.z = next(iter(self.val_data))[0][:32].to(self.device)
        
        # Initialize a dictionary to store training logs
        self.logs = {
            "Step": [],
            "Train_g_loss": [],
            "Train_d_loss": [],
            "Val_g_loss": [],
            "Val_d_loss": [],
            "Samples": [],
        }

    def init_optimizers(self, lr: float=3e-4, betas: tuple=(0.5, 0.999)):
        """
        Initialize the optimizers for the generator and discriminator.
        
        Args:
            lr (float, optional): Learning rate for the optimizers. Defaults to 3e-4.
            betas (tuple, optional): Beta parameters for the Adam optimizer. Defaults to (0.5, 0.999).
        """
        # Initialize Adam optimizer for the generator
        self.g_optimizer = torch.optim.Adam(
            self.generator.parameters(), lr=lr, betas=betas
        )
        # Initialize Adam optimizer for the discriminator
        self.d_optimizer = torch.optim.Adam(
            self.discriminator.parameters(), lr=lr, betas=betas
        )

    def train(self):
        """
        Execute the training loop for the GAN, including training and validation phases,
        logging, and model checkpointing.
        
        Returns:
            dict: Dictionary containing training logs.
        """
        # Ensure that both optimizers are initialized before training
        assert (self.g_optimizer is not None) and (
            self.d_optimizer is not None
        ), "Please run Trainer().init_optimizer()"
      
        best_score = torch.inf  # Initialize the best validation score to infinity

        # Iterate over each epoch
        for i in range(self.nb_epochs):
            # Initialize cumulative losses for the epoch
            train_d_loss, train_g_loss, val_d_loss, val_g_loss = 0, 0, 0, 0
            self.generator.train()      # Set generator to training mode
            self.discriminator.train()  # Set discriminator to training mode
            
            # Training loop with progress bar
            loop = tqdm(
                enumerate(self.train_data),
                desc=f"Epoch {i + 1}/{self.nb_epochs} train",
                leave=False,
                total=len(self.train_data),
            )
            for step, (x, y) in loop:
                x = x.to(self.device)  # Move input data to the specified device
                y = y.to(self.device)  # Move target data to the specified device

                # Perform a training step and retrieve generator and discriminator losses
                g_loss, d_loss = self.train_step(x, y)

                # Accumulate the losses
                train_g_loss += g_loss
                train_d_loss += d_loss

                # Update the progress bar with average losses
                loop.set_postfix_str(
                    f"g_loss: {train_g_loss / (step + 1) :.2f}, d_loss: {train_d_loss / (step + 1) :.2f}"
                )

            # Validation phase
            self.generator.eval()  # Set generator to evaluation mode
            self.discriminator.eval()  # Set discriminator to evaluation mode

            # Validation loop with progress bar
            loop = tqdm(
                enumerate(self.val_data),
                desc=f"Epoch {i + 1}/{self.nb_epochs} validation",
                leave=True,
                total=len(self.val_data),
            )
            for step, (x, y) in loop:
                x = x.to(self.device)  # Move input data to the specified device
                y = y.to(self.device)  # Move target data to the specified device

                # Perform a validation step and retrieve generator and discriminator losses
                g_loss, d_loss = self.val_step(x, y)

                # Accumulate the validation losses
                val_g_loss += g_loss
                val_d_loss += d_loss

                # Update the progress bar with average validation losses
                loop.set_postfix_str(
                    f"g_loss: {val_g_loss / (step + 1) :.2f} d_loss: {val_d_loss / (step + 1) :.2f}"
                )
            
            # Check if the current validation loss is the best so far
            if self.save_path and best_score > val_g_loss:
                best_score = val_g_loss  # Update the best score
                self.save_model()  # Save the current model as the best model

            # Log the metrics for the current epoch
            self.log_metrics(
                step=i,
                train_g_loss=train_g_loss,
                train_d_loss=train_d_loss,
                val_g_loss=val_g_loss,
                val_d_loss=val_d_loss,
            )
            # Generate and display sample images from the generator
            fake_img = self.generator((photos[0].unsqueeze(0)).to('cuda'))
            fake_img = fake_img[0].cpu().detach()
            fake_img = fake_img*0.5 + 0.5
            fake_img = transform(fake_img)
            fake_img.save(f'result/epoch_{i}.png')
        return self.logs  # Return the training logs

    def train_step(
        self,
        x: torch.Tensor,
        y: torch.Tensor
    ) -> tuple:
        """
        Perform a single training step for both generator and discriminator.
        
        Args:
            x (torch.Tensor): Input tensor (e.g., photos).
            y (torch.Tensor): Target tensor (e.g., Monet paintings).
            
        Returns:
            tuple: Generator loss and discriminator loss.
        """
        self.g_optimizer.zero_grad(set_to_none=True)  # Reset generator gradients
        self.d_optimizer.zero_grad(set_to_none=True)  # Reset discriminator gradients
        # Generate fake images and classify them with the discriminator
        fake_ = self.discriminator(
            self.generator(x)
        )

        # Classify real images with the discriminator
        real = self.discriminator(y)
        # Calculate discriminator loss: real images should be classified as ones and fake as zeros
        d_loss = (
            torch.nn.functional.mse_loss(real, torch.ones_like(real, device=self.device)) + 
            torch.nn.functional.mse_loss(fake_, torch.zeros_like(fake_, device=self.device))
        )

        d_loss.backward()  # Backpropagate discriminator loss
        self.d_optimizer.step()  # Update discriminator weights
        
        # Re-generate fake images for generator training
        fake = self.discriminator(
            self.generator(x)
        )
        
        # Calculate generator loss: fake images should be classified as ones
        g_loss = torch.nn.functional.mse_loss(fake, torch.ones_like(fake, device=self.device))
        g_loss.backward()  # Backpropagate generator loss
        self.g_optimizer.step()  # Update generator weights
        
        return g_loss.item(), d_loss.item()  # Return scalar losses

    @torch.no_grad()
    def val_step(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
    ) -> tuple:
        """
        Perform a single validation step without updating the model.
        
        Args:
            x (torch.Tensor): Input tensor (e.g., photos).
            y (torch.Tensor): Target tensor (e.g., Monet paintings).
            
        Returns:
            tuple: Generator loss and discriminator loss.
        """
        # Generate fake images and classify them with the discriminator
        fake = self.discriminator(
            self.generator(x)
        )
        # Classify real images with the discriminator
        real = self.discriminator(y)
        # Calculate generator loss: fake images should be classified as ones
        g_loss = torch.nn.functional.mse_loss(fake, torch.ones_like(fake, device=self.device))
        # Calculate discriminator loss: real images as ones and fake as zeros
        d_loss = (
            torch.nn.functional.mse_loss(real, torch.ones_like(real, device=self.device)) + 
            torch.nn.functional.mse_loss(fake, torch.zeros_like(fake, device=self.device))
        )
        return g_loss.item(), d_loss.item()  # Return scalar losses

    @torch.no_grad()
    def log_metrics(
        self,
        step: int,
        train_g_loss: torch.Tensor,
        train_d_loss: torch.Tensor,
        val_g_loss: torch.Tensor,
        val_d_loss: torch.Tensor,
    ):
        """
        Log the training and validation metrics for the current epoch.
        
        Args:
            step (int): Current epoch number.
            train_g_loss (torch.Tensor): Cumulative generator loss during training.
            train_d_loss (torch.Tensor): Cumulative discriminator loss during training.
            val_g_loss (torch.Tensor): Cumulative generator loss during validation.
            val_d_loss (torch.Tensor): Cumulative discriminator loss during validation.
        """
        self.logs["Step"].append(step)  # Log the current epoch
        self.logs["Train_g_loss"].append(train_g_loss / len(self.train_data))  # Log average training generator loss
        self.logs["Train_d_loss"].append(train_d_loss / len(self.train_data))  # Log average training discriminator loss
        self.logs["Val_g_loss"].append(val_g_loss / len(self.val_data))  # Log average validation generator loss
        self.logs["Val_d_loss"].append(val_d_loss / len(self.val_data))  # Log average validation discriminator loss # Log generated sample images

    def save_model(self, full: bool = False):
        """
        Save the generator and discriminator models.
        
        Args:
            full (bool, optional): Whether to save the full model or just the state dictionaries. Defaults to False.
        """
        if full:
            # Save the entire generator and discriminator models
            torch.save(self.generator, Path(self.save_path) / "generator.pth")
            torch.save(self.discriminator, Path(self.save_path) / "discriminator.pth")
        else:
            # Save only the state dictionaries (weights) of the models
            torch.save(
                self.generator.state_dict(),
                Path(self.save_path) / "generator_weights.pth",
            )
            torch.save(
                self.discriminator.state_dict(),
                Path(self.save_path) / "discriminator_weights.pth",
            )

generator = Generator(3,64)
discriminator = Discriminator()

device = "cuda"

dataset = MonetPhoto(data_root='data',monet_path='monet_jpg',photo_path='photo_jpg')

train_size, test_size = int(len(dataset) * 0.8), len(dataset) - int(len(dataset) * 0.8)

training_data, testing_data = random_split(dataset, [train_size, test_size])
train_dataloader = DataLoader(training_data.dataset, batch_size=32, shuffle=True, num_workers=2)
test_dataloader = DataLoader(testing_data.dataset, batch_size=32, shuffle=True, num_workers=2)
mones, photos = next(iter(train_dataloader))
trainer = Trainer(
    train_data=train_dataloader,
    val_data= test_dataloader,
    generator=generator,
    discriminator= discriminator,
    nb_epochs= 50,
    device = device,
    save_path = 'result/',
)


trainer.init_optimizers()
logs = trainer.train()
trainer.save_model()