import time
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.utils import save_image

from lib.datasets import CelebA
from lib.models import CelebAModel
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
    celeba = CelebA(batch_size=args.batch_size)
    train_dataloader = celeba.train_dataloader
    val_dataloader = celeba.val_dataloader

    # base model, base model optimizer, and base model scheduler
    model = CelebAModel(weights=None).to(device)
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
    adversary_optimizer = torch.optim.RMSprop(
        model.parameters(), lr=0.064, alpha=0.9, eps=0.0316, 
        weight_decay=1e-5, momentum=0.9,
    )
    adversary_scheduler = torch.optim.lr_scheduler.StepLR(adversary_optimizer, step_size=1, gamma=0.921)

    # NoiseOverlay
    noise_overlay = NoiseOverlay()

    def save_batch_image(image, name, root_folder='./toyexp'):
        folder = Path(root_folder)
        folder.mkdir(parents=True, exist_ok=True)
        
        for idx in range(image.shape[0]):
            path = folder / f"{name}_{idx}.png"
            save_image(image[idx:idx+1,:,:,:].detach().cpu(), path)

    def train():
        train_stat = np.array([])
        model.eval()
        # training loop
        for batch_idx, (data, label) in enumerate(train_dataloader):
            # label, sens = celeba.process_label(label)
            label, sens = filter_celeba_label(label)
            data, label, sens = data.to(device), label.to(torch.float32).to(device), sens.to(device)
            # save a batch of image
            save_batch_image(data, 'toyexp')
            # pass image into the model to get logit
            logit = model(normalize(data))
            # fake_logit = torch.tensor([[0.02, 0.02, 0.9, 0.9],
            #                            [0.02, 0.57, 0.01, 0.53],
            #                            [0.01, 0.93, 0.08, 0.97],
            #                            [0.96, 0.09, 0.09, 0.02],
            #                            [0.97, 0.99, 0.99, 0.99],
            #                            [0.49, 0.95, 0.95, 0.98],
            #                            [0.73, 0.36, 0.31, 0.25],
            #                            [0.98, 0.21, 0.02, 0.05]])
            # fake_sens = torch.tensor([[0],[0],[0],[1],[1],[1],[1],[1]])
            # fake_label = torch.tensor([[0,0,1,1],
            #                            [0,1,0,1],
            #                            [0,1,0,1],
            #                            [1,0,1,0],
            #                            [1,1,1,1],
            #                            [0,1,1,1],
            #                            [0,0,0,0],
            #                            [1,0,0,0]])
            # fake_label = fake_label.to(torch.float32)
            # bce_TPR_loss(fake_logit, fake_label, fake_sens)
            loss = bce_TPR_loss(logit, label, sens, args.target_type, args.policy, args.indirect)
            print(args.indirect)
            print(loss)




            break # run only once
    
    start_time = time.time()
    train()
    total_time = time.time() - start_time
    print(f'Running time: {total_time/60:.4f} mins')


def get_args():
    import argparse

    parser = argparse.ArgumentParser(description="Noise training")
    parser.add_argument("-b", "--batch-size", default=24, type=int, help="batch size for model inputs")
    parser.add_argument("--epochs", default=125, type=int, help="number of epochs to run")
    parser.add_argument("--resume", default="", help="name of a adversarial element, without .npy")
    parser.add_argument("--start-epoch", default=0, type=int, help="start epoch, it won't do any check with the element loaded")

    parser.add_argument("--model", default="33907_CelebA_0124", help="name of a checkpoint, without .pth")
    parser.add_argument("--name", default="CelebA_noise_cm", help="name to save the stats")
    parser.add_argument("--target-type", default="tp", help="target cell be selected for fairness")
    parser.add_argument("--policy", default="buck_only", help="policy on how to mutiply the target cells")
    parser.add_argument("--indirect", default=False, help="boolean value to include cells that have negative label or not")
    return parser

if __name__ == '__main__':
    args = get_args().parse_args()
    main(args)