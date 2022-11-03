import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
import numpy as np
from tqdm import tqdm
from datetime import date
import torchvision.transforms as transforms
from NN import Unetid

from params import epochs, learning_rate


class Model:
    def __init__(self):
        self.net = Unetid()
        self.loss_fcn = nn.MSELoss()
        self.optimizer = optim.Adam(self.net.parameters(), lr=learning_rate)
    
    def train(self, data, savefile=None, recordfile=None):
        # TODO: train/test holdout

        # Load checkpoint file if provided
        if savefile is not None:
            checkpoint = torch.load(savefile)
            self.net.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        else:
            checkpoint = None
        
        # Load records file if provided
        if recordfile is not None:
            records = np.load(recordfile) # load .npz file
            epoch_hist = records['epoch_hist'].astype(np.int64)
            loss_hist = records['loss_hist']
        else:
            epoch_hist = np.array([])
            loss_hist = np.array([])

        # Set current epoch (last completed epoch)
        if len(epoch_hist) > 0:
            e = epoch_hist[-1]
        elif savefile is not None:
            e = checkpoint['epoch']
        else:
            e = -1
        
        # BUG: EPOCH NUMBER - updating loss history and checkpointing used to happen once at the end of every epoch.
        #      Now because of large batches and long epoch times, it happens multiple times in the middle of an epoch.
        #      The problem is, when picking up where we left off, we add 1 to the last logged epoch. This now gives 
        #      inaccurate number of epochs (records say there have been more epochs than have actually been completed).
        
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
                
                # Correct for inconsistencies in how i've shaped things vs how Torch wants it
                X_batch = X_batch.unsqueeze(1) # Unsqueeze single channel to include 2nd dimension (channel)
                Y_batch = torch.permute(Y_batch, (0, 3, 1, 2)) # Put channel as second dim instead of last
                
                # TODO: Setup and enable multiprocessing (and GPU if available)
                # ...multiprocessing...
                
                # Set parameter gradients to zero
                self.optimizer.zero_grad()
                
                # Forward pass
                Y_pred = self.net(X_batch)
                loss = self.loss_fcn(Y_pred, Y_batch) # compute loss
                
                # Backward pass to compute gradients,
                # then update weights
                loss.backward()
                self.optimizer.step()
                
                # Compute average loss for this epoch
                avg_loss += loss / len(data)
                
                # Update loss records and save checkpoint* every 5 batches
                # BUG: EPOCH NUMBER - updating loss history and checkpointing used to happen once at the end of every epoch.
                #      Now because of large batches and long epoch times, it happens multiple times in the middle of an epoch.
                #      The problem is, when picking up where we left off, we add 1 to the last logged epoch. This now gives 
                #      inaccurate number of epochs (records say there have been more epochs than have actually been completed).
                if b%5==0:
                    # Update loss records
                    epoch_hist = np.append(epoch_hist, epoch)
                    loss_hist = np.append(loss_hist, avg_loss.item())
                    np.savez('records.npz', epoch_hist=epoch_hist, loss_hist=loss_hist)

                    # Save checkpoint if loss is new min
                    if avg_loss.item() == min(loss_hist):
                        torch.save({
                            'epoch': epoch,
                            'model_state_dict': self.net.state_dict(),
                            'optimizer_state_dict': self.optimizer.state_dict(),
                            'loss': avg_loss
                        }, 'model.pt') # overwrites

                b=b+1 # Increment num batches
            
            # Display average loss of epoch
            print('avg_loss:',avg_loss.item())


