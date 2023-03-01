import time
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from lib.datasets import CelebA
from lib.models import CelebAModel
from lib.fairnessBinary import *
from lib.utils import *

def main(args):
    print('Pytorch is running on version: ' + torch.__version__)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    start_time = time.time()
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    # dataset, dataloader (CelebA)
    celeba = CelebA(batch_size=args.batch_size)
    train_dataloader = celeba.train_dataloader
    val_dataloader = celeba.val_dataloader
    # model, optimizer, and scheduler
    model = CelebAModel(weights=None).to(device)
    optimizer = torch.optim.RMSprop(
        model.parameters(), lr=args.lr, alpha=0.9, eps=0.0316, 
        weight_decay=1e-5, momentum=0.9,
    )
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.921)
    model_ckpt_path = Path(args.model_ckpt_root)/args.model_name
    model_stat_path = Path(args.model_stat_root)/args.model_name
    if args.resume:
        model, optimizer, scheduler = load_model(model, optimizer, scheduler, name=args.resume, root_folder=model_ckpt_path)
        train_stat = load_stats(name=args.model_name+'_train', root_folder=model_stat_path)
        val_stat = load_stats(name=args.model_name+'_val', root_folder=model_stat_path)
    else:
        train_stat, val_stat = np.array([]), np.array([])
    total_time = time.time() - start_time
    print(f'Preparation done in {total_time:.4f} secs')

    def train():
        train_stat = np.array([])
        model.train()
        # training loop
        for batch_idx, (data, label) in enumerate(train_dataloader):
            label, sens = filter_celeba_label(label)
            data, label, sens = data.to(device), label.to(torch.float32).to(device), sens.to(device)
            instance = normalize(data)
            optimizer.zero_grad()
            logit = model(instance)
            loss = F.binary_cross_entropy(logit, label)
            loss.backward()
            optimizer.step()
            # collecting performance information
            pred = to_prediction(logit, model_name='CelebA')
            stat = calc_groupcm(pred, label, sens)
            stat = stat[np.newaxis, :]
            train_stat = train_stat+stat if len(train_stat) else stat
        return train_stat # in shape (1, attribute, 8)
        
    def val():
        val_stat = np.array([])
        model.eval()
        with torch.no_grad():
            # validaton loop
            for batch_idx, (data, label) in enumerate(val_dataloader):
                label, sens = filter_celeba_label(label)
                data, label, sens = data.to(device), label.to(torch.float32).to(device), sens.to(device)
                instance = normalize(data)            
                logit = model(instance)
                # collecting performance information
                pred = to_prediction(logit, model_name='CelebA')
                stat = calc_groupcm(pred, label, sens)
                stat = stat[np.newaxis, :]
                val_stat = val_stat+stat if len(val_stat) else stat
            return val_stat # in shape (1, attribute, 8)
    # summarize the status in validation set for some adjustment
    def get_stats_per_epoch(stat):
        # Input: statistics for a single epochs, shape (1, attributes, 8)
        mtp, mfp, mfn, mtn = [stat[0,:,i] for i in range(0, 4)]
        ftp, ffp, ffn, ftn = [stat[0,:,i] for i in range(4, 8)]
        # Accuracy
        macc = (mtp+mtn)/(mtp+mfp+mfn+mtn)
        facc = (ftp+ftn)/(ftp+ffp+ffn+ftn)
        tacc = (mtp+mtn+ftp+ftn)/(mtp+mfp+mfn+mtn+ftp+ffp+ffn+ftn)
        # Fairness
        mtpr, mtnr = mtp/(mtp+mfn), mtn/(mtn+mfp)
        ftpr, ftnr = ftp/(ftp+ffn), ftn/(ftn+ffp)
        tpr_diff, tnr_diff = abs(mtpr-ftpr), abs(mtnr-ftnr)
        equality_of_opportunity = tpr_diff
        equalized_odds = tpr_diff+tnr_diff
        stat_dict = {"male_acc": macc, "female_acc": facc, "total_acc": tacc,
                     "tpr_diff": tpr_diff, "tnr_diff": tnr_diff, 
                     "equality_of_opportunity": equality_of_opportunity, "equalized_odds": equalized_odds}
        return stat_dict

    # Run the code
    print(f'Start training model')
    start_time = time.time()

    for epoch in range(args.start_epoch, args.epochs):
        train_stat_per_epoch = train()
        # scheduler.step()
        val_stat_per_epoch = val()
        train_stat = np.concatenate((train_stat, train_stat_per_epoch), axis=0) if len(train_stat) else train_stat_per_epoch
        val_stat = np.concatenate((val_stat, val_stat_per_epoch), axis=0) if len(val_stat) else val_stat_per_epoch
        # print some basic statistic
        print(f'Epoch: {epoch:02d}')
        for a in range(4): # all attributes
            print(f'    attribute: {a}')
            stat_dict = get_stats_per_epoch(train_stat_per_epoch)
            macc, facc, tacc = stat_dict["male_acc"][a], stat_dict["female_acc"][a], stat_dict["total_acc"][a]
            equality_of_opportunity, equalized_odds = stat_dict["equality_of_opportunity"][a], stat_dict["equalized_odds"][a]
            print(f'    train    {macc:.4f} - {facc:.4f} - {tacc:.4f} -- {equality_of_opportunity:.4f} - {equalized_odds:.4f}')
            stat_dict = get_stats_per_epoch(val_stat_per_epoch)
            macc, facc, tacc = stat_dict["male_acc"][a], stat_dict["female_acc"][a], stat_dict["total_acc"][a]
            equality_of_opportunity, equalized_odds = stat_dict["equality_of_opportunity"][a], stat_dict["equalized_odds"][a]
            print(f'    val      {macc:.4f} - {facc:.4f} - {tacc:.4f} -- {equality_of_opportunity:.4f} - {equalized_odds:.4f}')
        # save model checkpoint
        save_model(model, optimizer, scheduler, name=f'{args.model_name}_{epoch:04d}', root_folder=model_ckpt_path)
    # save basic statistic
    save_stats(train_stat, f'{args.model_name}_train', root_folder=model_stat_path)
    save_stats(val_stat, f'{args.model_name}_val', root_folder=model_stat_path)
    total_time = time.time() - start_time
    print(f'Training time: {total_time/60:.4f} mins')

def get_args():
    import argparse

    parser = argparse.ArgumentParser(description="Model training")
    # Training related arguments
    parser.add_argument("--seed", default=32138, type=int, help="seed for the adversarial instance training process")
    parser.add_argument("-b", "--batch-size", default=128, type=int, help="batch size for model inputs")
    parser.add_argument("--epochs", default=125, type=int, help="number of epochs to run")
    parser.add_argument("--lr", default=1e-4, type=float, help="step size for model training")
    # For model trained
    parser.add_argument("--model-ckpt-root", default='/tmp2/aislab/makila/model_checkpoint', type=str, help='root path for model checkpoint')
    parser.add_argument("--model-stat-root", default='/tmp2/aislab/makila/model_stats', type=str, help='root path for model statistic')
    parser.add_argument("--model-name", default='default_model', type=str, help='name for this model trained')
    parser.add_argument("--resume", default="", help="name of a checkpoint, without .pth")
    parser.add_argument("--start-epoch", default=0, type=int, help="start epoch, it won't do any check with the weight loaded")

    return parser

if __name__ == '__main__':
    args = get_args().parse_args()
    main(args)