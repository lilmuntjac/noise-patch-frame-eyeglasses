from pathlib import Path
import numpy as np
import pandas as pd
import random

import torch
from torchvision import transforms

imagenet_mean = (0.485, 0.456, 0.406)
imagenet_std  = (0.229, 0.224, 0.225)

def normalize(data, mean=imagenet_mean, std=imagenet_std):
    """Normalize batch of images"""

    transform = transforms.Normalize(mean=mean, std=std)
    return transform(data)

def denormalize(data, mean=imagenet_mean, std=imagenet_std):
    """Denormalize batch of images"""

    transform = transforms.Normalize(
        mean= [-m/s for m, s in zip(mean, std)],
        std= [1/s for s in std])
    return transform(data)

def save_model(model, optimizer, scheduler, name, root_folder='/tmp2/aislab/makila/model_checkpoint'):
    """Save the model weight, optimizer, scheduler, and random states"""

    folder = Path(root_folder)
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"{name}.pth"
    # save the model checkpoint
    save_dict = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'rng_state': torch.get_rng_state(),
        'cuda_rng_state': torch.cuda.get_rng_state(),
    }
    torch.save(save_dict, path)

def load_model(model, optimizer, scheduler, name, root_folder='/tmp2/aislab/makila/model_checkpoint'):
    """Load the model weight, optimizer, and random states"""

    folder = Path(root_folder)
    path = folder / f"{name}.pth"
    ckpt = torch.load(path)
    model.load_state_dict(ckpt['model_state_dict'])
    optimizer.load_state_dict(ckpt['optimizer_state_dict'])
    scheduler.load_state_dict(ckpt['scheduler_state_dict'])
    torch.set_rng_state(ckpt['rng_state'])
    torch.cuda.set_rng_state(ckpt['cuda_rng_state'])
    return model, optimizer, scheduler

def save_stats(nparray, name, root_folder='/tmp2/aislab/makila/model_stats'):
    """Save the numpy array"""

    folder = Path(root_folder)
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"{name}.npy"
    np.save(path, nparray)

def load_stats(name, root_folder='/tmp2/aislab/makila/model_stats'):
    """Load the numpy array"""

    folder = Path(root_folder)
    path = folder / f"{name}.npy"
    nparray = np.load(path)
    return nparray

def UTKFace_split(root_folder='/tmp2/dataset/UTKFace', seed=14798):
    """Split the UTKFace dataset into train and validation set"""
    root_dir = Path(root_folder)
    image_list = list((root_dir).glob('**/[0-9]*_[0-9]*_[0-9]*_[0-9]*.jpg.chip.jpg'))
    image_list.sort()
    random.Random(seed).shuffle(image_list)
    # split the list into 8:2
    train, val = image_list[:int(len(image_list)*0.8)], image_list[int(len(image_list)*0.8):]
    train_df = pd.DataFrame(columns=['file', 'age', 'gender', 'race'])
    for p in train:
        p_stem = str(p.stem).split('_')
        age, gender, race = p_stem[0], p_stem[1], p_stem[2]
        train_df.loc[len(train_df.index)] = [p.name, age, gender, race]
    val_df = pd.DataFrame(columns=['file', 'age', 'gender', 'race'])
    for p in val:
        p_stem = str(p.stem).split('_')
        age, gender, race = p_stem[0], p_stem[1], p_stem[2]
        val_df.loc[len(val_df.index)] = [p.name, age, gender, race]
    train_df.to_csv(root_dir/'utkface_train.csv', index=False)
    val_df.to_csv(root_dir/'utkface_val.csv', index=False)
