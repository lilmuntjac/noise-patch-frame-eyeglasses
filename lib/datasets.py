from dataclasses import dataclass, field
from typing import Tuple
from pathlib import Path
from PIL import Image
import numpy as np
import pandas as pd

import torch
from torchvision import datasets, transforms

@dataclass
class CelebA:
    """get train and vaildation dataloader from CelebA dataset"""

    batch_size: int = 128
    root: str = '/tmp2/dataset'
    mean: Tuple = (0.485, 0.456, 0.406)
    std: Tuple = (0.229, 0.224, 0.225)

    train_dataloader: torch.utils.data.DataLoader = field(init=False)
    val_dataloader: torch.utils.data.DataLoader = field(init=False)

    def __post_init__(self):
        train_transform = transforms.Compose([
            transforms.RandAugment(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            # To make clipping on adverserial element intuitively, 
            # only normalize before passing it into the model
            # transforms.Normalize(mean, std),
            # becauuse of this, random erasing pixel as imagenet mean, not zero
            # transforms.RandomErasing(0.2, value=self.mean),
        ])
        val_transform = transforms.Compose([
            transforms.Resize((224, 224)), 
            transforms.ToTensor(),
            # To make clipping on adverserial element intuitively
            # only normalize before passing it into the model
            # transforms.Normalize(mean, std),
        ])
        # dataset source from pytorch official
        train_dataset = datasets.CelebA(root=self.root, split='train', target_type='attr', 
            transform=train_transform, download=False)
        val_dataset = datasets.CelebA(root=self.root, split='valid', target_type='attr', 
            transform=val_transform, download=False)
        self.train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=self.batch_size,
            shuffle=True, num_workers=12, pin_memory=True, drop_last=True,)
        self.val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=self.batch_size, 
            shuffle=True, num_workers=12, pin_memory=True, drop_last=True,)

    # def process_label(self, label, attr_pred_idx=[2, 19, 21, 31], attr_sens_idx=[20,]):
    #     """Split the label into training label and sensitive attribute"""
    #     # Attractive: 2, High_Cheekbones: 19, Mouth_Slightly_Open: 21, Smiling: 31, Male: 20
    #     return label[:,attr_pred_idx], label[:,attr_sens_idx]

    # def process_pred(self, pred, label, sens):
    #     """
    #     split the prediction and the corresponding label by its sensitive attribute,
    #     then compute the confusion matrix for it
    #         Output:
    #             Numpy array in shape (attributes, 8) male& female confusion matrix
    #     """
    #     def confusion_matrix(pred, label, idx):
    #         tp = torch.mul(pred[:,idx], label[:,idx]).sum()
    #         fp = torch.mul(pred[:,idx], torch.sub(1, label[:,idx])).sum()
    #         fn = torch.mul(torch.sub(1, pred[:,idx]), label[:,idx]).sum()
    #         tn = torch.mul(torch.sub(1, pred[:,idx]), torch.sub(1, label[:,idx])).sum()
    #         return tp, fp, fn, tn

    #     m_pred, m_label = pred[sens[:,0]==1], label[sens[:,0]==1]
    #     f_pred, f_label = pred[sens[:,0]==0], label[sens[:,0]==0]
    #     stat = np.array([])
    #     for idx in range(label.shape[-1]):
    #         mtp, mfp, mfn, mtn = confusion_matrix(m_pred, m_label, idx)
    #         ftp, ffp, ffn, ftn = confusion_matrix(f_pred, f_label, idx)
    #         row = np.array([[mtp.item(), mfp.item(), mfn.item(), mtn.item(), ftp.item(), ffp.item(), ffn.item(), ftn.item()]])
    #         stat =  np.concatenate((stat, row), axis=0) if len(stat) else row
    #     return stat

class UTKfaceDataset(torch.utils.data.Dataset):
    """Custom pytorch Dataset class for UTKface dataset"""

    def __init__(self, root_dir, transform=None, race_def='default'):
        self.root_dir = Path(root_dir)
        self.data_path = list((self.root_dir).glob('*.jpg'))
        self.transform = transform
        self.race_def = race_def
        # Warning:
        # Latino_Hispanic & Middle Eastern will be mapped to Latino_Hispanic in fairface
        self.fairface_mapping = {'0': '0', '1': '1', '2': '3', '3': '5', '4': '2'}
        # self.race_dict = {'white': 0, 'black': 1, 'asian': 2, 'indian': 3, 'others': 4}
        # self.gender_dict = {'male': 0, 'female': 1}
        # remember to delete bad file in the dataset with re '^[0-9]*_[0-9]*_[0-9]*.jpg.chip.jpg' (missing race)
        # command for counting race=others (in terminal): ls | grep '^[0-9]*_[0-9]*_4_*' | tee /dev/tty | wc -l

    def __len__(self):
        return len(self.data_path)
    
    def __getitem__(self, index: int):
        X = Image.open(self.data_path[index])
        if self.transform:
            X = self.transform(X)
        
        filename = str(self.data_path[index].stem).split('_')
        age, gender, race = filename[0], filename[1], filename[2]
        if self.race_def == 'FairFace':
            race = self.fairface_mapping[race]
        # map age to interval
        mapping = [(2, 0), (9, 1), (19, 2), (29, 3), (39, 4), (49, 5), (59, 6), (69, 7), (200, 8)]
        for upper_bound, interval in mapping:
            if int(age) <= upper_bound:
                age_interval = interval
                break

        target = [int(race), int(gender), age_interval]
        target = torch.tensor(target)
        return X, target

@dataclass
class UTKface:
    """
    get train and validation dataloader from UTKface dataset
    source: https://susanqq.github.io/UTKFace/
    """

    batch_size: int = 128
    root: str = '/tmp2/dataset/UTKface'
    race_def: str = 'default'
    mean: Tuple = (0.485, 0.456, 0.406)
    std: Tuple = (0.229, 0.224, 0.225)

    train_dataloader: torch.utils.data.DataLoader = field(init=False)
    val_dataloader: torch.utils.data.DataLoader = field(init=False)

    def __post_init__(self):
        train_transform: transforms.transforms.Compose = \
            transforms.Compose([
                transforms.RandAugment(),
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                # To make clipping on adverserial element intuitively
                # only normalize before passing it into the model
                # transforms.Normalize(mean, std),
                # becauuse of this, random erasing pixel as imagenet mean, not zero
                # transforms.RandomErasing(0.2, value=self.mean),
            ])
        val_transform: transforms.transforms.Compose = \
            transforms.Compose([
                transforms.Resize((224, 224)), 
                transforms.ToTensor(),
                # To make clipping on adverserial element intuitively
                # only normalize before passing it into the model
                # transforms.Normalize(mean, std),
            ])
        # select the dataset race definition
        dataset = UTKfaceDataset(root_dir=self.root, transform=None, race_def=self.race_def)
        # no partition of dataset provided by the author, ramdom split it into two set
        train_size = int(len(dataset)*0.8878)
        val_size   = len(dataset) - train_size
        train_dataset, val_dataset = torch.utils.data.random_split(
            dataset = dataset, 
            lengths = [train_size, val_size], 
            generator=torch.Generator().manual_seed(74353)
        )
        train_dataset.dataset.transform, val_dataset.dataset.transform = train_transform, val_transform
        self.train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=self.batch_size, 
            shuffle=True, num_workers=12, pin_memory=True, drop_last=True,)
        self.val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=self.batch_size, 
            shuffle=True, num_workers=12, pin_memory=True, drop_last=True,)
    
    # def process_logit(self, logit, label, split):
    #     """
    #     Convert logit into prediction and compute accuracy by 
    #         race: white or non-white
    #         gender: male or female
    #         Output:
    #             Numpy array in shape (3, 4) 
    #                 4 being group 1 correct / group 1 wromg / group 2 correct /  group 2 wrong
    #     """

    #     _, race_pred = torch.max(logit[:,0:5], dim=1)
    #     _, gender_pred = torch.max(logit[:,5:7], dim=1)
    #     _, age_pred = torch.max(logit[:,7:16], dim=1)
    #     pred = torch.stack((race_pred, gender_pred, age_pred), dim=1)
    #     if split == 'race':
    #         w_pred, w_label = pred[label[:,0]==0], label[label[:,0]==0]
    #         n_pred, n_label = pred[label[:,0]!=0], label[label[:,0]!=0]
    #         w_result, n_result = torch.eq(w_pred, w_label), torch.eq(n_pred, n_label)
    #         w_correct, w_wrong = torch.sum(w_result, dim=0), torch.sum(torch.logical_not(w_result), dim=0)
    #         n_correct, n_wrong = torch.sum(n_result, dim=0), torch.sum(torch.logical_not(n_result), dim=0)
    #         result = torch.stack((w_correct, w_wrong, n_correct, n_wrong), dim=1)
    #         return result.detach().cpu().numpy()
    #     elif split == 'gender':
    #         m_pred, m_label = pred[label[:,1]==0], label[label[:,1]==0]
    #         f_pred, f_label = pred[label[:,1]==1], label[label[:,1]==1]
    #         m_result, f_result = torch.eq(m_pred, m_label), torch.eq(f_pred, f_label)
    #         m_correct, m_wrong = torch.sum(m_result, dim=0), torch.sum(torch.logical_not(m_result), dim=0)
    #         f_correct, f_wrong = torch.sum(f_result, dim=0), torch.sum(torch.logical_not(f_result), dim=0)
    #         result = torch.stack((m_correct, m_wrong, f_correct, f_wrong), dim=1)
    #         return result.detach().cpu().numpy()
    #     else:
    #         assert False, f'no such attribute'

    # # these 2 function are for age model
    # def process_label(self, label):
    #     """Split the label into training label and sensitive attribute"""
    #     # in age model, training label is age, sensitive attrribute is gender
    #     # also we need to binarized the age label
    #     binary_age = torch.where(label[:,2:3]>3, 1, 0) # keep the shape as (N, 1)
    #     return binary_age, label[:,1:2]

    # def process_pred(self, pred, label, sens):
    #     """
    #     split the prediction and the corresponding label by its sensitive attribute,
    #     then compute the confusion matrix for it
    #         Output:
    #             Numpy array in shape (attributes, 8) male& female confusion matrix
    #     """
    #     def confusion_matrix(pred, label, idx):
    #         tp = torch.mul(pred[:,idx], label[:,idx]).sum()
    #         fp = torch.mul(pred[:,idx], torch.sub(1, label[:,idx])).sum()
    #         fn = torch.mul(torch.sub(1, pred[:,idx]), label[:,idx]).sum()
    #         tn = torch.mul(torch.sub(1, pred[:,idx]), torch.sub(1, label[:,idx])).sum()
    #         return tp, fp, fn, tn

    #     m_pred, m_label = pred[sens[:,0]==1], label[sens[:,0]==1]
    #     f_pred, f_label = pred[sens[:,0]==0], label[sens[:,0]==0]
    #     stat = np.array([])
    #     for idx in range(label.shape[-1]):
    #         mtp, mfp, mfn, mtn = confusion_matrix(m_pred, m_label, idx)
    #         ftp, ffp, ffn, ftn = confusion_matrix(f_pred, f_label, idx)
    #         row = np.array([[mtp.item(), mfp.item(), mfn.item(), mtn.item(), ftp.item(), ffp.item(), ffn.item(), ftn.item()]])
    #         stat =  np.concatenate((stat, row), axis=0) if len(stat) else row
    #     return stat

class FairFaceDataset(torch.utils.data.Dataset):
    """Custom pytorch Dataset class for FairFace dataset"""

    def __init__(self, csv_file, root_dir, transform=None, race_def='default'):
        self.attributes = pd.read_csv(csv_file)
        self.root_dir = Path(root_dir)
        self.transform = transform
        if race_def == 'UTKface':
            self.race_dict = {'White': 0, 'Black': 1, 'Latino_Hispanic': 4, 'East Asian': 2, 
                'Southeast Asian': 4, 'Indian': 3, 'Middle Eastern': 4}
        else: 
            self.race_dict = {'White': 0, 'Black': 1, 'Latino_Hispanic': 2, 'East Asian': 3, 
                'Southeast Asian': 4, 'Indian': 5, 'Middle Eastern': 6}
        self.gender_dict = {'Male': 0, 'Female': 1}
        self.age_dict = {'0-2': 0, '3-9': 1, '10-19': 2, '20-29': 3,
            '30-39': 4, '40-49': 5, '50-59': 6, '60-69': 7, 'more than 70': 8}

    def __len__(self):
        return len(self.attributes.index)

    def __getitem__(self, index: int):
        X = Image.open(self.root_dir / self.attributes.iloc[index]['file'])
        if self.transform:
            X = self.transform(X)

        # raw attributes from csv file
        age = self.attributes.iloc[index]['age']
        gender = self.attributes.iloc[index]['gender']
        race = self.attributes.iloc[index]['race']
        # convert raw attributes to binary labels
        
        target = [self.race_dict[race], self.gender_dict[gender], self.age_dict[age]]
        target = torch.tensor(target)
        return X, target

@dataclass
class FairFace:
    """
    get train and validation dataloader from FairFace dataset
    source: https://github.com/joojs/fairface
    """

    batch_size: int = 128
    root: str = '/tmp2/dataset/fairface-img-margin025-trainval'
    train_csv: str = '/tmp2/dataset/fairface-img-margin025-trainval/fairface_label_train.csv'
    val_csv: str = '/tmp2/dataset/fairface-img-margin025-trainval/fairface_label_val.csv'
    race_def: str = 'default'
    mean: Tuple = (0.485, 0.456, 0.406)
    std: Tuple = (0.229, 0.224, 0.225)

    train_dataloader: torch.utils.data.DataLoader = field(init=False)
    val_dataloader: torch.utils.data.DataLoader = field(init=False)

    def __post_init__(self):
        train_transform: transforms.transforms.Compose = \
            transforms.Compose([
                transforms.RandAugment(),
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                # To make clipping on adverserial element intuitively
                # only normalize before passing it into the model
                # transforms.Normalize(mean, std),
                # becauuse of this, random erasing pixel as imagenet mean, not zero
                # transforms.RandomErasing(0.2, value=self.mean),
            ])
        val_transform: transforms.transforms.Compose = \
            transforms.Compose([
                transforms.Resize((224, 224)), 
                transforms.ToTensor(),
                # To make clipping on adverserial element intuitively
                # only normalize before passing it into the model
                # transforms.Normalize(mean, std),
            ])
        # select the dataset race definition
        train_dataset = FairFaceDataset(self.train_csv, self.root, transform=train_transform, race_def=self.race_def)
        val_dataset = FairFaceDataset(self.val_csv, self.root, transform=val_transform, race_def=self.race_def)
        self.train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=self.batch_size, \
            shuffle=True, num_workers=12, pin_memory=True, drop_last=True,)
        self.val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=self.batch_size, \
            shuffle=True, num_workers=12, pin_memory=True, drop_last=True,)

    # def process_logit(self, logit, label, split):
    #     """
    #     Convert logit into prediction and compute accuracy by 
    #         race: white or non-white
    #         gender: male or female
    #         Output:
    #             Numpy array in shape (3, 4) 
    #                 4 being group 1 correct / group 1 wromg / group 2 correct /  group 2 wrong
    #     """

    #     _, race_pred = torch.max(logit[:,0:7], dim=1)
    #     _, gender_pred = torch.max(logit[:,7:9], dim=1)
    #     _, age_pred = torch.max(logit[:,9:18], dim=1)
    #     pred = torch.stack((race_pred, gender_pred, age_pred), dim=1)
    #     if split == 'race':
    #         w_pred, w_label = pred[label[:,0]==0], label[label[:,0]==0]
    #         n_pred, n_label = pred[label[:,0]!=0], label[label[:,0]!=0]
    #         w_result, n_result = torch.eq(w_pred, w_label), torch.eq(n_pred, n_label)
    #         w_correct, w_wrong = torch.sum(w_result, dim=0), torch.sum(torch.logical_not(w_result), dim=0)
    #         n_correct, n_wrong = torch.sum(n_result, dim=0), torch.sum(torch.logical_not(n_result), dim=0)
    #         result = torch.stack((w_correct, w_wrong, n_correct, n_wrong), dim=1)
    #         return result.detach().cpu().numpy()
    #     elif split == 'gender':
    #         m_pred, m_label = pred[label[:,1]==0], label[label[:,1]==0]
    #         f_pred, f_label = pred[label[:,1]==1], label[label[:,1]==1]
    #         m_result, f_result = torch.eq(m_pred, m_label), torch.eq(f_pred, f_label)
    #         m_correct, m_wrong = torch.sum(m_result, dim=0), torch.sum(torch.logical_not(m_result), dim=0)
    #         f_correct, f_wrong = torch.sum(f_result, dim=0), torch.sum(torch.logical_not(f_result), dim=0)
    #         result = torch.stack((m_correct, m_wrong, f_correct, f_wrong), dim=1)
    #         return result.detach().cpu().numpy()
    #     else:
    #         assert False, f'no such attribute'

    # # these 2 function are for age model
    # def process_label(self, label):
    #     """Split the label into training label and sensitive attribute"""
    #     # in age model, training label is age, sensitive attrribute is gender
    #     # also we need to binarized the age label
    #     binary_age = torch.where(label[:,2:3]>3, 1, 0) # keep the shape as (N, 1)
    #     return binary_age, label[:,1:2]

    # def process_pred(self, pred, label, sens):
    #     """
    #     split the prediction and the corresponding label by its sensitive attribute,
    #     then compute the confusion matrix for it
    #         Output:
    #             Numpy array in shape (attributes, 8) male& female confusion matrix
    #     """
    #     def confusion_matrix(pred, label, idx):
    #         tp = torch.mul(pred[:,idx], label[:,idx]).sum()
    #         fp = torch.mul(pred[:,idx], torch.sub(1, label[:,idx])).sum()
    #         fn = torch.mul(torch.sub(1, pred[:,idx]), label[:,idx]).sum()
    #         tn = torch.mul(torch.sub(1, pred[:,idx]), torch.sub(1, label[:,idx])).sum()
    #         return tp, fp, fn, tn

    #     m_pred, m_label = pred[sens[:,0]==1], label[sens[:,0]==1]
    #     f_pred, f_label = pred[sens[:,0]==0], label[sens[:,0]==0]
    #     stat = np.array([])
    #     for idx in range(label.shape[-1]):
    #         mtp, mfp, mfn, mtn = confusion_matrix(m_pred, m_label, idx)
    #         ftp, ffp, ffn, ftn = confusion_matrix(f_pred, f_label, idx)
    #         row = np.array([[mtp.item(), mfp.item(), mfn.item(), mtn.item(), ftp.item(), ffp.item(), ffn.item(), ftn.item()]])
    #         stat =  np.concatenate((stat, row), axis=0) if len(stat) else row
    #     return stat