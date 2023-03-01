import time
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from lib.datasets import UTKFace
from lib.models import UTKFaceModel
from lib.fabricator import NoiseOverlay
from lib.fairness import *
from lib.utils import *

def main(args):
    print('Pytorch is running on version: ' + torch.__version__)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    start_time = time.time()
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    # dataset, dataloader (UTKFace)
    utkface = UTKFace(batch_size=args.batch_size)
    train_dataloader = utkface.train_dataloader
    val_dataloader = utkface.val_dataloader
    # base model, base model optimizer, and base model scheduler
    model = UTKFaceModel(weights=None).to(device)
    optimizer = torch.optim.RMSprop(
        model.parameters(), lr=0.064, alpha=0.9, eps=0.0316, 
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
    coef = torch.tensor(args.coef).to(device)
    total_time = time.time() - start_time
    print(f'Preparation done in {total_time:.4f} secs')

    # training, and validation function
    def train():
        train_stat = np.array([])
        model.eval()
        # training loop
        for batch_idx, (data, label) in enumerate(train_dataloader):
            data, label = data.to(device), label.to(device)
            adv_image, label = noise_overlay.apply(data, label, master)
            adv_image = normalize(adv_image)
            adversary_optimizer.zero_grad()
            logit = model(adv_image)
            if args.loss_type == 'direct':
                loss = loss_categori_direct(logit, label, args.sens_type, args.attr_type, coef, 'UTKFace')
            elif args.loss_type == 'CE masking':
                loss = loss_categori_CEmasking(logit, label, args.sens_type, args.attr_type, 'UTKFace')
            elif args.loss_type == 'perturbOptim':
                loss = loss_categori_perturbOptim(logit, label, args.sens_type, args.attr_type, coef, 'UTKFace')
            elif args.loss_type == 'perturbOptimFull':
                loss = loss_categori_perturbOptim_full(logit, label, args.sens_type, args.attr_type, coef, 'UTKFace')
            else:
                assert False, f'Unsupport loss type'
            loss.backward()
            adversary_optimizer.step()
            noise_overlay.clip_by_budget(master)
            # collecting performance information
            pred = to_prediction(logit, model_name='UTKFace')
            stat = calc_grouppq(pred, label, args.sens_type)
            stat = stat[np.newaxis, :]
            train_stat = train_stat+stat if len(train_stat) else stat
        return train_stat # in shape (1, attribute+1, 4) 2 group, correct and wrong, the extra attribute is 'all'
    def val(dataloader=val_dataloader):
        val_stat = np.array([])
        model.eval()
        with torch.no_grad():
            # validaton loop
            for batch_idx, (data, label) in enumerate(dataloader):
                data, label = data.to(device), label.to(device)
                adv_image, label = noise_overlay.apply(data, label, master)
                adv_image = normalize(adv_image)
                logit = model(adv_image)
                # collecting performance information
                pred = to_prediction(logit, model_name='UTKFace')
                stat = calc_grouppq(pred, label, args.sens_type)
                stat = stat[np.newaxis, :]
                val_stat = val_stat+stat if len(val_stat) else stat
                return val_stat # in shape (1, attribute+1, 4), the extra attribute is 'all'
    # summarize the status in validation set for some adjustment
    def get_stats_per_epoch(stat):
        # Input: statistics for a single epochs, shape (1, attributes, 4)
        # 4 being group_1_correct, group_1_wrong, group_2_correct, group_2_wrong
        group_1_correct, group_1_wrong, group_2_correct, group_2_wrong = [stat[0,:,i] for i in range(4)]
        group_1_total, group_2_total = group_1_correct+group_1_wrong, group_2_correct+group_2_wrong
        group_1_acc, group_2_acc = group_1_correct/group_1_total, group_2_correct/group_2_total
        diff_prediction_quaility = abs(group_1_acc-group_2_acc) # in shape (A)
        total_acc = (group_1_correct+group_2_correct)/(group_1_total+group_2_total)
        stat_dict = {"group_1_acc": group_1_acc ,"group_2_acc": group_2_acc, "total_acc": total_acc,
                     "diff_pq": diff_prediction_quaility}
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
            curr_pq, last_pq = stat_dict["diff_pq"], last_state_dict["diff_pq"]
            # print the coefficient
            coef_list = coef.clone().cpu().numpy().tolist()
            coef_list = [f'{f:.4f}' for f in coef_list]
            print(f'    coef: {" ".join(coef_list)}')
            # adjust the coefficient by the loss type
            if coef.shape[0] == 1:
                attr2id = {'race': 0, 'gender': 1, 'age': 2, 'all': 3}
                if curr_tacc[attr2id[args.attr_type]] < init_tacc[attr2id[args.attr_type]] - args.quality_target:
                    coef[0] = min(coef[0]*1.05, 1e3)
                elif curr_pq[attr2id[args.attr_type]] > args.fairness_target and curr_pq[attr2id[args.attr_type]] > last_pq[attr2id[args.attr_type]]:
                    coef[0] = max(coef[0]*0.95, 1e-3)
            else:
                for a in range(coef.shape[0]): # all attributes
                    if curr_tacc[a] < init_tacc[a] - args.quality_target:
                        coef[a] = min(coef[a]*1.05, 1e3)
                    elif curr_pq[a] > args.fairness_target and curr_pq[a] > last_pq[a]:
                        coef[a] = max(coef[a]*0.95, 1e-3)
        #
        attr2id = {'race': 0, 'gender': 1, 'age': 2, 'all': 3}
        for a in range(4): # all attributes, include 'all'
            group_1_acc, group_2_acc, total_acc, diff_pq = stat_dict['group_1_acc'][a], stat_dict['group_2_acc'][a], stat_dict['total_acc'][a], stat_dict['diff_pq'][a]
            # print the training status and whether it meet the goal set by argument
            status = ''
            if total_acc < init_tacc[a] - args.quality_target:
                status += ' [low accuracy]'
            if diff_pq > args.fairness_target:
                status += ' [not fair enough]'
            if args.attr_type != 'all' and attr2id[args.attr_type] == a:
                status += ' *'
            if args.attr_type == 'all':
                status += ' *'
            print(f'    {group_1_acc:.4f} - {group_2_acc:.4f} - {total_acc:.4f} -- {diff_pq:.4f}{status}')
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
    # parser.add_argument("--fairness-matrix", default="prediction quaility", help="how to measure fairness")
    parser.add_argument("--sens-type", default="race", type=str, help="sensitive attribute to divide the dataset into 2 group")
    parser.add_argument("--attr-type", default="all", type=str, help="attribute that fairness is evaluated on")
    parser.add_argument("--coef-mode", default="fix", type=str, help="method to adjust coef durinig training")
    parser.add_argument("--coef", default=[0.1,], type=float, nargs='+', help="coefficient multiply on recovery loss, must match the number of losses used")
    parser.add_argument("--fairness-target", default=0.03, type=float, help="Fairness target value")
    parser.add_argument("--quality-target", default=0.05, type=float, help="Max gap loss for prediction quaility")
    # method taken
    parser.add_argument("--loss-type", default='direct', type=str, help="Type of loss used")
    return parser

if __name__ == '__main__':
    args = get_args().parse_args()
    main(args)