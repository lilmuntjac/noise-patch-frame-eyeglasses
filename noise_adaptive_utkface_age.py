# focus on cutoff 
import time
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from lib.datasets import UTKface
from lib.models import AgeModel
from lib.fabricator import NoiseOverlay
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
    utkface = UTKface(batch_size=args.batch_size)
    train_dataloader = utkface.train_dataloader
    val_dataloader = utkface.val_dataloader

    # base model, base model optimizer, and base model scheduler
    model = AgeModel(weights=None).to(device)
    optimizer = torch.optim.RMSprop(
        model.parameters(), lr=0.064, alpha=0.9, eps=0.0316, 
        weight_decay=1e-5, momentum=0.9,
    )
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.921)
    model, optimizer, scheduler = load_model(model, optimizer, scheduler, args.model)

    # adversarial element, adversarial optimizer, adversarial scheduler
    if args.resume:
        master = load_stats(args.resume, root_folder='./noise')
        master = torch.from_numpy(master).to(device)
    else:
        master = torch.zeros((1, 3, 224, 224)).to(device)
    master = nn.Parameter(master)
    adversary_optimizer = torch.optim.SGD([master], lr=args.lr, )
    # adversary_scheduler = torch.optim.lr_scheduler.StepLR(adversary_optimizer, step_size=1, gamma=0.921)

    # NoiseOverlay
    noise_overlay = NoiseOverlay()

    # coefficient of the utility loss
    coef = torch.tensor(args.coef).to(device)

    def train():
        train_stat = np.array([])
        model.eval()
        # training loop
        for batch_idx, (data, label) in enumerate(train_dataloader):
            label, sens = filter_agemodel_label(label)
            data, label, sens = data.to(device), label.to(torch.float32).to(device), sens.to(device)
            adv_image, label = noise_overlay.apply(data, label, master)
            adv_image = normalize(adv_image)
            adversary_optimizer.zero_grad()
            logit = model(adv_image)
            if args.fairness_matrix == 'equality of opportunity':
                loss = loss_EOpp_perturbed_adaptive(logit, label, sens, coef)
            elif args.fairness_matrix == 'equalized odds':
                loss = loss_EO_perturbed_adaptive(logit, label, sens, coef)
            else:
                assert False, f'Unsupport fairness matrix'
            loss.backward()
            adversary_optimizer.step()
            noise_overlay.clip_by_budget(master)
            # collecting performance information
            pred = to_prediction(logit, model_name='AgeModel')
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
                label, sens = filter_agemodel_label(label)
                data, label, sens = data.to(device), label.to(torch.float32).to(device), sens.to(device)
                adv_image, label = noise_overlay.apply(data, label, master)
                adv_image = normalize(adv_image)
                logit = model(adv_image)
                # collecting performance information
                pred = to_prediction(logit, model_name='AgeModel')
                stat = calc_groupcm(pred, label, sens)
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

    def get_stats_per_epoch(stat):
        # Input: stats for a single epochs, shape (1, attributes, 8)
        mtp, mfp, mfn, mtn = [stat[0,:,i] for i in range(0, 4)]
        ftp, ffp, ffn, ftn = [stat[0,:,i] for i in range(4, 8)]
        # accuracy
        macc = (mtp+mtn)/(mtp+mfp+mfn+mtn)
        facc = (ftp+ftn)/(ftp+ffp+ffn+ftn)
        tacc = (mtp+mtn+ftp+ftn)/(mtp+mfp+mfn+mtn+ftp+ffp+ffn+ftn)
        # fairness
        mtpr, mtnr = mtp/(mtp+mfn), mtn/(mtn+mfp)
        ftpr, ftnr = ftp/(ftp+ffn), ftn/(ftn+ffp)
        equality_of_opportunity = abs(mtpr-ftpr)
        equalized_odds = abs(mtpr-ftpr) + abs(mtnr-ftnr)
        stat_dict = {"male_acc": macc, "female_acc": facc, "total_acc": tacc, 
                     "equality_of_opportunity": equality_of_opportunity, "equalized_odds": equalized_odds}
        return stat_dict

    # record the stats for the whole training process
    train_stat = np.array([])
    val_stat = np.array([])
    print(f'Start training noise')
    start_time = time.time()
    # record the initial model accuracy and fairness status
    val_stat_per_epoch = val()
    stat_dict = get_stats_per_epoch(val_stat_per_epoch)
    if args.fairness_matrix == 'equality of opportunity':
        init_tacc, last_stus = stat_dict['total_acc'], stat_dict['equality_of_opportunity']
    elif args.fairness_matrix == 'equalized odds':
        init_tacc, last_stus = stat_dict['total_acc'], stat_dict['equalized_odds']
    else:
        assert False, f'Unsupport fairness matrix'
    # training loop
    for epoch in range(args.start_epoch, args.epochs):
        print(f'Training with coefficient {coef.clone().cpu().numpy()}')
        train_stat_per_epoch = train()
        # adversary_scheduler.step()
        val_stat_per_epoch = val()

        print(master[0,0,0:8,0:8])



        train_stat = np.concatenate((train_stat, train_stat_per_epoch), axis=0) if len(train_stat) else train_stat_per_epoch
        val_stat = np.concatenate((val_stat, val_stat_per_epoch), axis=0) if len(val_stat) else val_stat_per_epoch
        # update the utility loss coefficient
        stat_dict = get_stats_per_epoch(val_stat_per_epoch)
        if args.fairness_matrix == 'equality of opportunity':
            curr_tacc, curr_stus = stat_dict['total_acc'], stat_dict['equality_of_opportunity']
        elif args.fairness_matrix == 'equalized odds':
            curr_tacc, curr_stus = stat_dict['total_acc'], stat_dict['equalized_odds']
        else:
            assert False, f'Unsupport fairness matrix'
        for i in range(1):
            if curr_tacc[i] < init_tacc[i] - args.quality_target:
                coef[i] = coef[i]*1.2
            elif curr_stus[i] > args.fairness_target and curr_stus[i] > last_stus[i]:
                coef[i] = coef[i]*0.8
        last_stus = curr_stus
        #
        print(f'Epoch: {epoch:02d}')
        for a in range(1): # all attributes
            macc, facc, tacc, equality_of_opportunity, equalized_odds = get_basic_stat(a, val_stat_per_epoch)
            print(f'    {macc:.4f} - {facc:.4f} - {tacc:.4f} -- {equality_of_opportunity:.4f} - {equalized_odds:.4f}')
        # save the noise for each epoch
        noise = master.detach().cpu().numpy()
        save_stats(noise, name=f'{seed}_{args.name}_master_{epoch:04d}', root_folder='/tmp2/aislab/makila/noise')
    # save basic statistic
    save_stats(train_stat, f'{seed}_{args.name}_train', root_folder='/tmp2/aislab/makila/noise')
    save_stats(val_stat, f'{seed}_{args.name}_val', root_folder='/tmp2/aislab/makila/noise')
    total_time = time.time() - start_time
    print(f'Training time: {total_time/60:.4f} mins')

def get_args():
    import argparse

    parser = argparse.ArgumentParser(description="Noise training")
    parser.add_argument("-b", "--batch-size", default=128, type=int, help="batch size for model inputs")
    parser.add_argument("--epochs", default=125, type=int, help="number of epochs to run")
    parser.add_argument("--resume", default="", help="name of a adversarial element, without .npy")
    parser.add_argument("--start-epoch", default=0, type=int, help="start epoch, it won't do any check with the element loaded")

    parser.add_argument("--model", default="33907_UTKfaceAge_0124", help="name of a checkpoint, without .pth")
    parser.add_argument("--name", default="UTKfaceAge_noise_cm", help="name to save the stats")
    parser.add_argument("--lr", default=1e-4, type=float, help="step size for the adversarial element")

    parser.add_argument("--target-type", default="tp_fn", help="target cell be selected for fairness")
    parser.add_argument("--policy", default="buck_only", help="policy on how to mutiply the target cells")
    parser.add_argument("--indirect", default=False, help="boolean value to include cells that have negative label or not")

    parser.add_argument("--coef", default=[0.1,], type=float, nargs='+', help="coefficient multiply on utility loss, need to be match with the number of attributes")
    parser.add_argument("--fairness-target", default=0.01, type=float, help="Fairness target value")
    parser.add_argument("--quality-target", default=0.05, type=float, help="Max gap loss for prediction quaility")

    parser.add_argument("--fairness-matrix", default="equality of opportunity", help="how to measure fairness")
    return parser

if __name__ == '__main__':
    args = get_args().parse_args()
    main(args)