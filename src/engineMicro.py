import time
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data import random_split
from torch.utils.data import TensorDataset


def train(model, loss_func, optim, scheduler, dataloaders, epochs, device, src_mask, tgt_mask, verbose=True):
    '''
    - model: model to train
    - loss_func: loss function
    - optim: optimizer
    - scheduler: lr scheduler
    - dataloaders: tuple of train and val dataloaders
    - epochs: number of training iterations
    - device: for gpu
    - src_mask: mask for encoder output
    - tgt_mask: mask for target shifted right
    '''
    train_dl, val_dl = dataloaders
    
    train_losses, val_losses= [],[]

    start_time = time.time()
    last_epoch_time = start_time
    
    print("################")
    print("#Begin training#")
    print("################")
    for epoch in range(epochs):
        model.train()
        train_loss, running_loss, total_count,running_total_count = 0,0,0,0
        for  i, (X,y) in enumerate(train_dl):     
            X = X.to(device)
            y = y.to(device)
            
            total_count += y.shape[0]
            running_total_count += y.shape[0]
            #zero out the previous grads
            optim.zero_grad()

            #compute output
            tgt = torch.concat((X[:,-1,0].unsqueeze(1), y[:,:-1]), axis=1)
            out = model(X,tgt,src_mask, tgt_mask)

            #backward
            loss = torch.sqrt(loss_func(out,y)) #use RMSE so units are not squared
            
            train_loss += loss.item()
            if verbose:
                running_loss += loss.item()

            loss.backward()
            optim.step()
          
            #print stats    
            if verbose and i % 10 == 9:    # print every 10 mini-batches
                print(f'[Epoch: {epoch}, Batch:{i + 1:5d}] loss: {running_loss / running_total_count:.3f}')
                running_loss = 0.0
                running_total_count = 0

        train_loss /= total_count
      
        train_losses.append(train_loss)

        #validation
        val_loss = test(model, loss_func, val_dl, device, src_mask, tgt_mask)
   
        val_losses.append(val_loss)
        curr_time = time.time()
        print(f"Epoch:{epoch}  Train Loss:{train_loss:.05f}  Val Loss:{val_loss:.05f}, Total Time:{curr_time - start_time}, Epoch Time:{curr_time - last_epoch_time}")
        last_epoch_time = curr_time

        if scheduler:
             scheduler.step(val_loss)

    print("###############")
    print("#Done training#")
    print("###############") 

    makePrettyGraphs(train_losses, val_losses)

    return model, train_losses,  val_losses


def test(model, loss_func, dataloader, device, src_mask, tgt_mask):
    '''
    - model: model to infer with
    - loss_func: loss function
    - dataloader: dataloader to evaluate on
    - device: for gpu
    - src_mask: mask for encoder output
    - tgt_mask: mask for target shifted right
    '''
    model.eval()
   
    loss_total,total_count = 0,0
  
    with torch.no_grad():
            for X,y in dataloader:
                X = X.to(device)
                y = y.to(device)
                total_count += y.shape[0]

                tgt = torch.concat((X[:,-1,0].unsqueeze(1), y[:,:-1]), axis=1)
                out = model(X,tgt,src_mask,tgt_mask)  
                
                loss = torch.sqrt(loss_func(out,y)) #use RMSE so units are not squared
                loss_total += loss.item()
                
             
    return loss_total/total_count



def makePrettyGraphs(train_losses,  val_losses):
    '''
    - list of train and val accuracies and losses
    '''
    #pretty graphs
    ind = [x for x in range(1,len(train_losses)+1)]
    plt.subplot(1,2,1)
    plt.plot(ind, train_losses)
    plt.title("Average Train Loss per sample VS Epoch")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")


    plt.subplot(1,2,2)
    plt.plot(ind, val_losses)
    plt.title("Average Validation Loss per sample VS Epoch")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")


    plt.tight_layout()
    


    