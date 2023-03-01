import time
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from lib.datasets import FairFace
from lib.models import AgeModel
from lib.fabricator import NoiseOverlay
from lib.fairnessBinary import *
from lib.utils import *

def main(args):
    print('Pytorch is running on version: ' + torch.__version__)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    start_time = time.time()
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    # dataset, dataloader (FairFace)
    fairface = FairFace(batch_size=args.batch_size)
    train_dataloader = fairface.train_dataloader
    val_dataloader = fairface.val_dataloader
    # base model, base model optimizer, and base model scheduler
    model = AgeModel(weights=None).to(device)
    optimizer = torch.optim.RMSprop(
        model.parameters(), lr=args.lr, alpha=0.9, eps=0.0316, 
        weight_decay=1e-5, momentum=0.9,
    )
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.921)
    model_ckpt_path = Path(args.model_ckpt_root)/args.model_name
    model, optimizer, scheduler = load_model(model, optimizer, scheduler, name=args.model_ckpt_name, root_folder=model_ckpt_path)
    # adversarial element, adversarial optimizer, and adversarial scheduler
    noise_ckpt_path = Path(args.noise_ckpt_root)/args.noise_name
    noise_stat_path = Path(args.noise_stat_root)/args.noise_name
    if args.resume:
        master = load_stats(name=args.resume, root_folder=noise_ckpt_path)
        master = torch.from_numpy(master).to(device)
        train_stat = load_stats(name=args.noise_name+'_train', root_folder=noise_stat_path)
        val_stat = load_stats(name=args.noise_name+'_val', root_folder=noise_stat_path)
    else:
        master = torch.zeros((1, 3, 224, 224)).to(device)
        train_stat, val_stat = np.array([]), np.array([])

    master = nn.Parameter(master)
    adversary_optimizer = torch.optim.SGD([master], lr=args.lr, )
    # adversary_scheduler = torch.optim.lr_scheduler.StepLR(adversary_optimizer, step_size=1, gamma=0.921)
    noise_overlay = NoiseOverlay()
    # coefficient of the recovery loss
    p_coef = torch.tensor(args.p_coef).to(device)
    n_coef = torch.tensor(args.n_coef).to(device)
    total_time = time.time() - start_time
    print(f'Preparation done in {total_time:.4f} secs')

    # training, and validation function
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
            if args.loss_type == 'direct':
                loss = loss_binary_direct(args.fairness_matrix, logit, label, sens)
            elif args.loss_type == 'BCE masking':
                loss = loss_binary_BCEmasking(args.fairness_matrix, logit, label, sens)
            elif args.loss_type == 'perturbOptim':
                loss = loss_binary_perturbOptim(args.fairness_matrix, logit, label, sens, p_coef, n_coef)
            elif args.loss_type == 'perturbOptimFull':
                loss = loss_binary_perturbOptim_full(args.fairness_matrix, logit, label, sens, p_coef, n_coef)
            else:
                assert False, f'Unsupport loss type'
            loss.backward()
            adversary_optimizer.step()
            noise_overlay.clip_by_budget(master)
            # collecting performance information
            pred = to_prediction(logit, model_name='AgeModel')
            stat = calc_groupcm(pred, label, sens)
            stat = stat[np.newaxis, :]
            train_stat = train_stat+stat if len(train_stat) else stat
        return train_stat # in shape (1, attribute, 8)
    def val(dataloader=val_dataloader):
        val_stat = np.array([])
        model.eval()
        with torch.no_grad():
            # validaton loop
            for batch_idx, (data, label) in enumerate(dataloader):
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

    print(f'Start training noise')
    start_time = time.time()

    if not args.resume:
        train_stat_per_epoch = val(train_dataloader)
        val_stat_per_epoch = val()
        train_stat = np.concatenate((train_stat, train_stat_per_epoch), axis=0) if len(train_stat) else train_stat_per_epoch
        val_stat = np.concatenate((val_stat, val_stat_per_epoch), axis=0) if len(val_stat) else val_stat_per_epoch
        init_tacc = get_stats_per_epoch(val_stat[0:1,:,:])["total_acc"]

    # training loop
    for epoch in range(args.start_epoch, args.epochs):
        print(f'Epoch: {epoch:02d}')
        train_stat_per_epoch = train()
        # adversary_scheduler.step()
        val_stat_per_epoch = val()
        train_stat = np.concatenate((train_stat, train_stat_per_epoch), axis=0) if len(train_stat) else train_stat_per_epoch
        val_stat = np.concatenate((val_stat, val_stat_per_epoch), axis=0) if len(val_stat) else val_stat_per_epoch
        stat_dict = get_stats_per_epoch(val_stat_per_epoch)
        # For special loss that adjust the coefficient base on fairness status
        if args.coef_mode == "dynamic":
            last_epoch_stat = val_stat[-2:-1,:,:]
            last_state_dict = get_stats_per_epoch(last_epoch_stat)
            curr_tacc, last_tacc = stat_dict["total_acc"], last_state_dict["total_acc"]
            curr_TPRd, curr_TNRd, last_TPRd, last_TNRd = stat_dict["tpr_diff"], stat_dict["tnr_diff"], last_state_dict["tpr_diff"], last_state_dict["tnr_diff"]
            p_coef_list, n_coef_list = p_coef.clone().cpu().numpy().tolist(), n_coef.clone().cpu().numpy().tolist()
            p_coef_list, n_coef_list = [f'{f:.4f}' for f in p_coef_list], [f'{f:.4f}' for f in n_coef_list]
            print(f'    p-coef: {" ".join(p_coef_list)} -- n-coef: {" ".join(n_coef_list)}')
            for a in range(1): # all attributes
                if curr_tacc[a] < init_tacc[a] - args.quality_target:
                    p_coef[a], n_coef[a] = min(p_coef[a]*1.05, 1e3), min(n_coef[a]*1.05, 1e3)
                elif args.fairness_matrix == "equality of opportunity":
                    if curr_TPRd[a] > args.fairness_target and curr_TPRd[a] > last_TPRd[a]:
                        p_coef[a] = max(p_coef[a]*0.95, 1e-3)
                elif args.fairness_matrix == "equalized odds":
                    if curr_TPRd[a] > args.fairness_target and curr_TPRd[a] > last_TPRd[a]:
                        p_coef[a] = max(p_coef[a]*0.95, 1e-3)
                    if curr_TNRd[a] > args.fairness_target and curr_TNRd[a] > last_TNRd[a]:
                        n_coef[a] = max(n_coef[a]*0.95, 1e-3 )
        #
        for a in range(1): # all attributes
            macc, facc, tacc, equality_of_opportunity, equalized_odds = stat_dict["male_acc"][a], stat_dict["female_acc"][a], stat_dict["total_acc"][a], stat_dict["equality_of_opportunity"][a], stat_dict["equalized_odds"][a]
            # print the training status and whether it meet the goal set by argument
            status = ''
            if args.fairness_matrix == 'equality of opportunity':
                if tacc < init_tacc[a] - args.quality_target:
                    status += ' [low accuracy]'
                if equality_of_opportunity > args.fairness_target:
                    status += ' [not fair enough]'
            elif args.fairness_matrix == 'equalized odds':
                if tacc < init_tacc[a] - args.quality_target:
                    status += ' [low accuracy]'
                if equalized_odds > args.fairness_target:
                    status += ' [not fair enough]'
            else:
                assert False, f'Unsupport loss type'
            print(f'    {macc:.4f} - {facc:.4f} - {tacc:.4f} -- {equality_of_opportunity:.4f} - {equalized_odds:.4f} {status}')
        # save the noise for each epoch
        noise = master.detach().cpu().numpy()
        save_stats(noise, f'{args.noise_name}_{epoch:04d}', root_folder=noise_ckpt_path)
    # save basic statistic
    save_stats(train_stat, f'{args.noise_name}_train', root_folder=noise_stat_path)
    save_stats(val_stat, f'{args.noise_name}_val', root_folder=noise_stat_path)
    total_time = time.time() - start_time
    print(f'Training time: {total_time/60:.4f} mins')

def get_args():
    import argparse

    parser = argparse.ArgumentParser(description="Noise training")
    # Training related arguments
    parser.add_argument("--seed", default=32138, type=int, help="seed for the adversarial instance training process")
    parser.add_argument("-b", "--batch-size", default=128, type=int, help="batch size for model inputs")
    parser.add_argument("--epochs", default=120, type=int, help="number of epochs to run")
    parser.add_argument("--lr", default=1e-4, type=float, help="step size for the adversarial element")
    # For base model
    parser.add_argument("--model-ckpt-root", default='/tmp2/aislab/makila/model_checkpoint', type=str, help='root path for model checkpoint')
    parser.add_argument("--model-stat-root", default='/tmp2/aislab/makila/model_stats', type=str, help='root path for model statistic')
    parser.add_argument("--model-name", default='default_model', type=str, help='name for this model trained')
    parser.add_argument("--model-ckpt-name", default='default_model', type=str, help='name for the model checkpoint, without .pth')
    # For adversarial element in training
    parser.add_argument("--noise-ckpt-root", default='/tmp2/aislab/makila/noise', type=str, help='root path for noise statistic')
    parser.add_argument("--noise-stat-root", default='/tmp2/aislab/makila/noise_stats', type=str, help='root path for noise itself')
    parser.add_argument("--noise-name", default='default_noise', type=str, help='name for the noise trained')
    parser.add_argument("--resume", default="", help="name of a adversarial element, without .npy")
    parser.add_argument("--start-epoch", default=0, type=int, help="start epoch, it won't do any check on the element loaded")
    # fairness parameter
    parser.add_argument("--fairness-matrix", default="equality of opportunity", help="how to measure fairness")
    parser.add_argument("--coef-mode", default="fix", type=str, help="method to adjust p-coef and n-coef durinig training")
    parser.add_argument("--p-coef", default=[0.1,], type=float, nargs='+', help="coefficient multiply on positive recovery loss, need to be match with the number of attributes")
    parser.add_argument("--n-coef", default=[0.1,], type=float, nargs='+', help="coefficient multiply on negative recovery loss, need to be match with the number of attributes")
    parser.add_argument("--fairness-target", default=0.03, type=float, help="Fairness target value")
    parser.add_argument("--quality-target", default=0.05, type=float, help="Max gap loss for prediction quaility")
    # method taken
    parser.add_argument("--loss-type", default='direct', type=str, help="Type of loss used")
    return parser

if __name__ == '__main__':
    args = get_args().parse_args()
    main(args)