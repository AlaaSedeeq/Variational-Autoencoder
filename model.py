import os
import numpy as np 
import pandas as pd
from collections import namedtuple
from tqdm import tqdm 
import matplotlib.pyplot as plt
from matplotlib import cm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.sampler import SubsetRandomSampler

from torchvision.datasets import MNIST 
from torchvision.transforms import transforms as T
from torchvision.utils import save_image, make_grid
from torchsummary import summary

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# MNIST dataset
dataset = torchvision.datasets.MNIST(root='../../data',
                                     train=True,
                                     transform=transforms.ToTensor(),
                                     download=True)

# Data loader
data_loader = torch.utils.data.DataLoader(dataset=dataset,
                                          batch_size=batch_size, 
                                          shuffle=True)



class Parameters:
    def __init__(self):
        self.input_shape = 28 * 28
        self.encoder_dim = 512 
        self.latent_dim = 32
        self.n_epochs = 20
        self.lr = 0.001
        self.batch_size = 128
        
MNIST_P = Parameters()



class MNIST_VAE(nn.Module):
    def __init__(self, MNIST_P):
        super(MNIST_VAE, self).__init__()
        # input layer
        self.input_net = nn.Sequential(
            nn.Linear(MNIST_P.input_shape, MNIST_P.encoder_dim),
            nn.ReLU()
        )
        # encoder layers (mu and std)
        self.mu_net = nn.Linear(MNIST_P.encoder_dim, MNIST_P.latent_dim)
        self.var_net = nn.Linear(MNIST_P.encoder_dim, MNIST_P.latent_dim)
        
        self.decoder_net = nn.Sequential(
            nn.Linear(MNIST_P.latent_dim, MNIST_P.encoder_dim),
            nn.ReLU(),
            nn.Linear(MNIST_P.encoder_dim, MNIST_P.input_shape),
            nn.Sigmoid()
        )
        
    def encoder(self, x):
        input_ = self.input_net(x)
        mu, var = self.mu_net(input_), self.var_net(input_)
        return mu, var
    
    def reparameterize(self, mu, var):
        # normal distribution with mean 0 and variance 1
        std = torch.exp(var/2)
        rand_normal = torch.randn_like(std)
        return mu + rand_normal * std

    def decoder(self, z):
        decoded = self.decoder_net(z)
        return decoded
    
    def forward(self, x):
        mu, var = self.encoder(x)
        latent_space = self.reparameterize(mu, var)
        decoded = self.decoder_net(latent_space)
        return decoded, mu, var
    
print(summary(MNIST_VAE(MNIST_P), (1, 28*28)))



# Compute reconstruction loss and kl divergence
def Criterion(x_reconst, x, var, mu):
    decoder_loss = F.binary_cross_entropy(x_reconst, x, reduction='sum')
    kl_div = - 0.5 * torch.sum(1 + var - mu.pow(2) - var.exp())
    return decoder_loss , kl_div





#########################################
#            Trainer Class              #
#########################################

class Trainer:
    
    def __init__(self, model, optimizer, criterion, scheduler, load_path=None):
        self.__class__.__name__ = "PyTorch Trainer"
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.scheduler = scheduler

        # Create a directory to save images if not exists
        sample_dir = 'model ouput'
        try: os.makedirs(sample_dir)
        except: pass
        
        # if model exist
        if load_path: self.model = torch.load(load_path)
        
    def save_model(self, path):
        torch.save(self.model.state_dict(), path)
        
        
    def display_volumes(
        self,
        x,
        path
    ):
        def show(
            img, 
            label=None, 
            alpha=0.5
        ):
            plt.figure(figsize=(20,8))
            npimg = img.numpy()
            img = plt.imshow(
                np.transpose(
                    npimg, 
                    (1, 2, 0)), 
                interpolation="none"
            )
            plt.savefig(path)

        cmap_vol = np.apply_along_axis(cm.viridis, 0, x.numpy())
        cmap_vol = torch.from_numpy(np.squeeze(cmap_vol))

        show(make_grid(cmap_vol))
    
    def run(self, train_loader):                
        
        for epoch in range(MNIST_P.n_epochs):
            lr = self.optimizer.param_groups[0]['lr'] if not self.scheduler else self.scheduler.get_last_lr()[0]
            data_iter = iter(train_loader)
            prog_bar = tqdm(range(len(train_loader)))
            for step in prog_bar: # iter over batches
                ######################
                # Get the data ready #
                ######################
                # get the input images and their corresponding labels
                images, _ = data_iter.next() # no need for labels

                # wrap them in a torch Variable and move tnsors to the configured device
                images = Variable(images).to(device).view(-1, MNIST_P.input_shape)                                  
                
                ################
                # Forward Pass #
                ################
                # Feed input images
                x_decoded, mu, var = model(images)
                
                # Find the Loss
                # Backprop and optimize
                decoder_loss , kl_div = self.criterion(x_decoded, images, mu, var)
                loss = decoder_loss + kl_div
                
                #################
                # Backward Pass #
                #################
                # Calculate gradients
                loss.backward()
                # Update Weights
                self.optimizer.step()
                # clear the gradient
                self.optimizer.zero_grad()

                prog_bar.set_description('Epoch {}/{}, Decoder Loss: {:.4f}, KL Div: {:.4f}, lr={:.4f}'\
                                         .format(epoch+1, MNIST_P.n_epochs, decoder_loss, kl_div ,lr))

            # see model's performace each epoch
            with torch.no_grad():
                # Save the sampled images
                z = torch.randn(MNIST_P.batch_size, MNIST_P.latent_dim).to(device)
                out = model.decoder(z).view(-1, 1, 28, 28)
                self.display_volumes(out, os.path.join('model ouput', 'Epoch-{} Sampled.png'.format(epoch+1)))

                # Save the decoded images
                out, _, _ = model(images)
                image_cat = torch.cat([images.view(-1, 1, 28, 28), out.view(-1, 1, 28, 28)], dim=3)
                self.display_volumes(image_cat, os.path.join('model ouput', 'Epoch-{} Decoded.png'.format(epoch+1)))

                
            # Decrease the lr
            if self.scheduler:
                scheduler.step()
                
                
                
# Defining Parameters
model = MNIST_VAE(MNIST_P)
criterion = Criterion
optimizer = torch.optim.Adam(model.parameters())

# Refresh tqdm bar
tqdm.refresh

# Define model trainer and start training 
model_trainer = Trainer(model, optimizer, criterion, None, '')
model_trainer.run(mnist_dataloder)