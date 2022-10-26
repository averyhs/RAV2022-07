import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
from tqdm import tqdm
from datetime import date
from UNet import UNet

from params import epochs, learning_rate


class Model:
    def __init__(self):
        self.net = UNet()
        self.loss_fcn = nn.MSELoss()
        self.optimizer = optim.Adam(self.net.parameters(), lr=learning_rate)
        self.loss_hist = []
    
    def train(self, data, savefile=None):
        # Load checkpoint file if provided
        if savefile is not None:
            checkpoint = torch.load(savefile)
            self.net.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            e = checkpoint['epoch']        
        else:
            e = -1 # e is used in the epoch loop so needs to be initialized
        
        for epoch in range(e+1, epochs+e+1):
            print('Epoch {}'.format(epoch+1))
            avg_loss = 0 # Reset average loss
            
            # Set the model to TRAINING MODE
            # (Some network components behave differently when training vs testing.
            # .eval() for testing)
            self.net.train()
            
            b=0 # number of batches
            for X_batch, Y_batch in tqdm(data):
                # Make everything compatible by casting to float (it keeps expectong Bytes idk why)
                X_batch = X_batch.float()
                Y_batch = Y_batch.float()
                
                # Unsqueeze to include 2nd dimension (arg of 1) representing CHANNEL (only have 1 channel)
                X_batch = X_batch.unsqueeze(1)
                Y_batch = Y_batch.unsqueeze(1)
                
                # TODO: Setup and enable multiprocessing (and GPU if available)
                # ...multiprocessing...
                
                # Set parameter gradients to zero
                # (TODO: Set gradients to ones from a previous model)
                self.optimizer.zero_grad()
                
                # Forward pass
                Y_pred = self.net(X_batch)
                loss = self.loss_fcn(Y_pred, Y_batch)  # compute loss
                
                # Backward pass to compute gradients,
                # then update weights
                loss.backward()
                self.optimizer.step()
                
                # Compute average loss for this epoch
                avg_loss += loss / len(data)
                
                b=b+1 # Increment num batches
            
            # Update loss records
            self.loss_hist.append(avg_loss)
            
            # Save checkpoint (if loss is new min)
            print('avg_loss:',avg_loss.item())
            if avg_loss==min(self.loss_hist):
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.net.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'loss': avg_loss
                }, 'model.pt') # overwrites

    def test():
        pass

    def run():
        pass

