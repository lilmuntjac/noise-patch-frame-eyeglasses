import time
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from lib.datasets import FairFace
from lib.models import FairFaceModel
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
    model = FairFaceModel(weights=None).to(device)
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
            data, label = data.to(device), label.to(device)
            instance = normalize(data)
            optimizer.zero_grad()
            logit = model(instance)
            loss = F.cross_entropy(logit[:,0:7], label[:,0]) + \
                   F.cross_entropy(logit[:,7:9], label[:,1]) + \
                   F.cross_entropy(logit[:,9:18], label[:,2])
            loss.backward()
            optimizer.step()
            # collecting performance information
            result = fairface.process_logit(logit, label, 'race')
            result = result[np.newaxis, :]
            train_stat = train_stat+result if len(train_stat) else result
        return train_stat # in shape (1, attribute, 2)
        
    def val():
        val_stat = np.array([])
        model.eval()
        with torch.no_grad():
            # validaton loop
            for batch_idx, (data, label) in enumerate(val_dataloader):
                data, label = data.to(device), label.to(device)
                instance = normalize(data)
                optimizer.zero_grad()
                logit = model(instance)
                # collecting performance information
                result = fairface.process_logit(logit, label, 'race')
                result = result[np.newaxis, :]
                val_stat = val_stat+result if len(val_stat) else result
            return val_stat # in shape (1, attribute, 2)

    # performance recording
    train_stat = np.array([])
    val_stat = np.array([])
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
        g1_race_acc, g1_gender_acc, g1_age_acc = val_stat_per_epoch[0,:,0] / (val_stat_per_epoch[0,:,0]+val_stat_per_epoch[0,:,1])
        g2_race_acc, g2_gender_acc, g2_age_acc = val_stat_per_epoch[0,:,2] / (val_stat_per_epoch[0,:,2]+val_stat_per_epoch[0,:,3])
        print(f'    {g1_race_acc:.4f} {g1_gender_acc:.4f} {g1_age_acc:.4f}')
        print(f'    {g2_race_acc:.4f} {g2_gender_acc:.4f} {g2_age_acc:.4f}')
        # save model checkpoint
        save_model(model, optimizer, scheduler, name=f'{seed}_FairFace_{epoch:04d}')
    # save basic statistic
    save_stats(train_stat, f'{seed}_FairFace_train')
    save_stats(val_stat, f'{seed}_FairFace_val')
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