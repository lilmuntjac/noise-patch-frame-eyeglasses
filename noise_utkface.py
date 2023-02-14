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
            data, label = data.to(device), label.to(device)
            adv_image, label = noise_overlay.apply(data, label, master)
            adv_image = normalize(adv_image)
            adversary_optimizer.zero_grad()
            logit = model(adv_image)
            
    

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