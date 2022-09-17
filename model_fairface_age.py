import time
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from lib.datasets import FairFace
from lib.models import AgeModel
from lib.fairness import *
from lib.utils import *

def main(args):
    print('Pytorch is running on version: ' + torch.__version__)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # args
    seed = 33907
    # set seeds
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # dataloader
    fairface = FairFace(batch_size=args.batch_size)
    train_dataloader = fairface.train_dataloader
    val_dataloader = fairface.val_dataloader
    # model, optimizer, and scheduler
    model = AgeModel(weights=None).to(device)
    optimizer = torch.optim.RMSprop(
        model.parameters(), lr=0.064, alpha=0.9, eps=0.0316, 
        weight_decay=1e-5, momentum=0.9,
    )
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.921)
    if args.resume:
        model, optimizer, scheduler = load_model(model, optimizer, scheduler, args.resume)


    def train():
        train_stat = np.array([])
        model.train()
        # training loop
        for batch_idx, (data, label) in enumerate(train_dataloader):
            label, sens = filter_agemodel_label(label)
            data, label, sens = data.to(device), label.to(torch.float32).to(device), sens.to(device)
            instance = normalize(data)
            optimizer.zero_grad()
            logit = model(instance)
            loss = F.binary_cross_entropy(logit, label)
            loss.backward()
            optimizer.step()
            # collecting performance information
            pred =  to_prediction(logit, model_name='FairFace')
            stat = calc_groupcm(pred, label, sens)
            stat = stat[np.newaxis, :]
            train_stat = train_stat+stat if len(train_stat) else stat
        return train_stat # in shape (1, 1, 8)
        
    def val():
        val_stat = np.array([])
        model.eval()
        with torch.no_grad():
            # validaton loop
            for batch_idx, (data, label) in enumerate(val_dataloader):
                label, sens = filter_agemodel_label(label)
                data, label, sens = data.to(device), label.to(torch.float32).to(device), sens.to(device)
                instance = normalize(data)
                logit = model(instance)
                # collecting performance information
                pred =  to_prediction(logit, model_name='FairFace')
                stat = calc_groupcm(pred, label, sens)
                stat = stat[np.newaxis, :]
                val_stat = val_stat+stat if len(val_stat) else stat
            return val_stat # in shape (1, 1, 8)

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

    train_stat_per_epoch = train()
    val_stat_per_epoch = val()
    # Run the code
    print(f'Start training')
    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        train_stat_per_epoch = train()
        scheduler.step()
        val_stat_per_epoch = val()

        train_stat = np.concatenate((train_stat, train_stat_per_epoch), axis=0) if len(train_stat) else train_stat_per_epoch
        val_stat = np.concatenate((val_stat, val_stat_per_epoch), axis=0) if len(val_stat) else val_stat_per_epoch
        # print some basic statistic
        print(f'Epoch: {epoch:02d}')
        # only have 1 attribute
        macc, facc, tacc, equality_of_opportunity, equalized_odds = get_basic_stat(0, val_stat_per_epoch)
        print(f'    {macc:.4f} - {facc:.4f} - {tacc:.4f} -- {equality_of_opportunity:.4f} - {equalized_odds:.4f}')
        # save model checkpoint
        save_model(model, optimizer, scheduler, name=f'{seed}_FairFaceAge_{epoch:04d}')
    # save basic statistic
    save_stats(train_stat, f'{seed}_FairFaceAge_train')
    save_stats(val_stat, f'{seed}_FairFaceAge_val')
    total_time = time.time() - start_time
    print(f'Training time: {total_time/60:.4f} mins')

def get_args():
    import argparse

    parser = argparse.ArgumentParser(description="Model training")
    parser.add_argument("-b", "--batch-size", default=128, type=int, help="batch size for model inputs")
    parser.add_argument("--epochs", default=125, type=int, help="number of epochs to run")
    parser.add_argument("--resume", default="", help="name of a checkpoint, without .pth")
    parser.add_argument("--start-epoch", default=0, type=int, help="start epoch, it won't do any check with the weight loaded")

    return parser

if __name__ == '__main__':
    args = get_args().parse_args()
    main(args)