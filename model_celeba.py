import numpy as np
import pandas as pd 

import torch
import torch.nn as nn
import torch.nn.functional as F

from lib.datasets import CelebA
from lib.models import CelebAModel
from lib.utils import *

def main():
    print('Pytorch is running on version: ' + torch.__version__)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # args
    batch_size_per_gpu = 128
    seed = 33907
    epochs = 125
    # set seeds
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # dataloader
    celeba = CelebA(batch_size=batch_size_per_gpu)
    train_dataloader = celeba.train_dataloader
    val_dataloader = celeba.val_dataloader
    # model, optimizer, and scheduler
    model = CelebAModel(weights=None).to(device)
    optimizer = torch.optim.RMSprop(
        model.parameters(), lr=0.064, alpha=0.9, eps=0.0316, 
        weight_decay=1e-5, momentum=0.9,
    )
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.921)

    def train():
        train_stat = np.array([])
        model.train()
        # training loop
        for batch_idx, (data, label) in enumerate(train_dataloader):
            label, sens = celeba.process_label(label)
            data, label, sens = data.to(device), label.to(torch.float32).to(device), sens.to(device)
            instance = normalize(data)
            optimizer.zero_grad()
            logit = model(instance)
            loss = F.binary_cross_entropy(logit, label)
            loss.backward()
            optimizer.step()
            # collecting performance information
            pred = torch.where(logit> 0.5, 1, 0).to(device)
            stat = celeba.process_pred(pred, label, sens)
            stat = stat[np.newaxis, :]
            train_stat = train_stat+stat if len(train_stat) else stat
        return train_stat # in shape (1, attribute, 8)
        
    def val():
        val_stat = np.array([])
        model.eval()
        with torch.no_grad():
            # validaton loop
            for batch_idx, (data, label) in enumerate(val_dataloader):
                label, sens = celeba.process_label(label)
                data, label, sens = data.to(device), label.to(torch.float32).to(device), sens.to(device)
                instance = normalize(data)            
                logit = model(instance)
                # collecting performance information
                pred = torch.where(logit> 0.5, 1, 0).to(device)
                stat = celeba.process_pred(pred, label, sens)
                stat = stat[np.newaxis, :]
                val_stat = val_stat+stat if len(val_stat) else stat
            return val_stat # in shape (1, attribute, 8)
    
    def get_basic_stat(idx, stat):
        # Inupt: stat in shape (1, attribute, 8)
        mtp, mfp, mfn, mtn = stat[0,idx,0:4]
        ftp, ffp, ffn, ftn = stat[0,idx,4:8]
        # accuracy
        macc = (mtp+mtn)/(mtp+mfp+mfn+mtn)
        facc = (ftp+ftn)/(ftp+ffp+ffn+ftn)
        tacc = (mtp+mtn+ftp+ftn)/(mtp+mfp+mfn+mtn+ftp+ffp+ffn+ftn)
        # fairness
        mtpr, mtnr = mtp/(mtp+mfn), mtn/(mtn+mfp)
        ftpr, ftnr = ftp/(ftp+ffn), ftn/(ftn+ffp)
        equality_of_opportunity = abs(mtpr-ftpr)
        equalized_odds = abs(mtpr-ftpr) + abs(mtnr-ftnr)
        return macc, facc, tacc, equality_of_opportunity, equalized_odds

    # performance recording
    train_stat = np.array([])
    val_stat = np.array([])
    # Run the code
    print(f'Start training')
    for i in range(epochs):
        train_stat_per_epoch = train()
        val_stat_per_epoch = val()
        scheduler.step()

        train_stat = np.concatenate((train_stat, train_stat_per_epoch), axis=0) if len(train_stat) else train_stat_per_epoch
        val_stat = np.concatenate((val_stat, val_stat_per_epoch), axis=0) if len(val_stat) else val_stat_per_epoch
        # print some basic statistic
        print(f'Epoch: {i:02d}')
        for a in range(4): # all attributes
            macc, facc, tacc, equality_of_opportunity, equalized_odds = get_basic_stat(a, val_stat_per_epoch)
            print(f'    {macc:.4f} - {facc:.4f} - {tacc:.4f} -- {equality_of_opportunity:.4f} - {equalized_odds:.4f}')
        # save model checkpoint
        save_model(model, optimizer, name=f'{seed}_CelebA_{i:04d}_')
    # save basic statistic
    save_stats(train_stat, f'{seed}_CelebA_train_')
    save_stats(val_stat, f'{seed}_CelebA_val_')


if __name__ == '__main__':
    main()