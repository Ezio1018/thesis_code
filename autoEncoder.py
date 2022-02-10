from json import load
from this import d
import torch
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader,random_split
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
from const import *
import numpy as np
from queue import Queue
from preprocess import getDL

class Encoder(nn.Module):
    
    def __init__(self, encoded_space_dim):
        super().__init__()
        
        ### Convolutional section
        self.encoder_cnn = nn.Sequential(
            # First convolutional layer
            nn.Conv2d(1, 8, 3, stride=2, padding=1),
            #nn.BatchNorm2d(8),
            nn.ReLU(True),
            # Second convolutional layer
            nn.Conv2d(8, 16, 3, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(True),
            # Third convolutional layer
            nn.Conv2d(16, 32, 3, stride=2, padding=0),
            #nn.BatchNorm2d(32),
            nn.ReLU(True)
        )
        
        ### Flatten layer
        self.flatten = nn.Flatten(start_dim=1)

        ### Linear section
        self.encoder_lin = nn.Sequential(
            # First linear layer
            nn.Linear(3 * 3 * 32, 128),
            nn.ReLU(True),
            # Second linear layer
            nn.Linear(128, encoded_space_dim)
        )
        
    def forward(self, x):
        # Apply convolutions
        x = self.encoder_cnn(x)
        # Flatten
        x = self.flatten(x)
        # # Apply linear layers
        x = self.encoder_lin(x)
        return x

class Decoder(nn.Module):
    
    def __init__(self, encoded_space_dim):
        super().__init__()

        ### Linear section
        self.decoder_lin = nn.Sequential(
            # First linear layer
            nn.Linear(encoded_space_dim, 128),
            nn.ReLU(True),
            # Second linear layer
            nn.Linear(128, 3 * 3 * 32),
            nn.ReLU(True)
        )

        ### Unflatten
        self.unflatten = nn.Unflatten(dim=1, unflattened_size=(32, 3, 3))

        ### Convolutional section
        self.decoder_conv = nn.Sequential(
            # First transposed convolution
            nn.ConvTranspose2d(32, 16, 3, stride=2, output_padding=0),
            nn.BatchNorm2d(16),
            nn.ReLU(True),
            # Second transposed convolution
            nn.ConvTranspose2d(16, 8, 3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(8),
            nn.ReLU(True),
            # Third transposed convolution
            nn.ConvTranspose2d(8, 1, 3, stride=2, padding=1, output_padding=1)
        )
        
    def forward(self, x):
        # Apply linear layers
        x = self.decoder_lin(x)
        # Unflatten
        x = self.unflatten(x)
        # Apply transposed convolutions
        x = self.decoder_conv(x)
        # Apply a sigmoid to force the output to be between 0 and 1 (valid pixel values)
        x = torch.sigmoid(x)
        return x

class model():
    def __init__(self):
        super().__init__()
        self.encoder = Encoder(encoded_space_dim=SPACE_DIM)
        self.decoder = Decoder(encoded_space_dim=SPACE_DIM)
        self.env,_=getDL()
        self.lossBuffer=[Queue(BUFFER_SIZE) for i in range(len(list(self.env)))]
        self.buffer_counter=0
        ### Define the loss function
        self.loss_fn = torch.nn.MSELoss()

        params_to_optimize = [
            {'params': self.encoder.parameters()},
            {'params': self.decoder.parameters()}
        ]

        self.optim = torch.optim.Adam(params_to_optimize, lr=AUTOENCODER_LR, weight_decay=1e-05)
        #optim = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=6e-05)
        # Move both the encoder and the decoder to the selected device
        self.encoder=self.encoder.to(DEVICE)
        self.decoder=self.decoder.to(DEVICE)


    def compute_total_loss(self,data):
        for d in data:
            d = d.to(DEVICE)
             # Encode data
            encoded_data = self.encoder(d[0])
            # Decode data
            decoded_data = self.decoder(encoded_data)
            loss=(decoded_data-d[0])**2
            total_loss.append(loss,d[1])
        return total_loss
            



    ### Training function
    def train_epoch(self,lossbuffer,data,):
        # Set train mode for both the encoder and the decoder
        self.encoder.train()
        self.train_loss = []
        dataloader=DataLoader(lossbuffer,BATCH_SIZE,shuffle=True)
        # Iterate the dataloader (we do not need the label values, this is unsupervised learning)
        for image_batch, _ in dataloader: # with "_" we just ignore the labels (the second element of the dataloader tuple)
            # Move tensor to the proper device
            image_batch = image_batch.to(DEVICE)
            # Encode data
            encoded_data = self.encoder(image_batch)
            # Decode data
            decoded_data = self.decoder(encoded_data)
            # Evaluate loss
            loss = self.loss_fn(decoded_data, image_batch)
            # Backward pass
            self.optim.zero_grad()
            loss.backward()
            self.optim.step()
            # Print batch loss
            self.compute_total_loss(data)
            print('\t partial train loss (single batch): %f' % (loss.data))
            self.train_loss.append(loss.detach().cpu().numpy())
            
        return np.mean(self.train_loss)

### Prepare patches

