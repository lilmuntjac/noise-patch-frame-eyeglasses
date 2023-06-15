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
    root: str = '/tmp2/npfe'
    mean: Tuple = (0.485, 0.456, 0.406)
    std: Tuple = (0.229, 0.224, 0.225)

    train_dataloader: torch.utils.data.DataLoader = field(init=False)
    val_dataloader: torch.utils.data.DataLoader = field(init=False)

    def __post_init__(self):
        train_transform = transforms.Compose([
            transforms.TrivialAugmentWide(),
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


class UTKFaceDataset(torch.utils.data.Dataset):
    """Custom pytorch Dataset class for UTKFace dataset"""

    def __init__(self, csv_file, root_dir, transform=None, race_def='default'):
        self.attributes = pd.read_csv(csv_file)
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.race_def = race_def
        # Warning:
        # Latino_Hispanic & Middle Eastern will be mapped to Latino_Hispanic in fairface
        self.fairface_mapping = {'0': '0', '1': '1', '2': '3', '3': '5', '4': '2'}
        # self.race_dict = {'white': 0, 'black': 1, 'asian': 2, 'indian': 3, 'others': 4}
        # self.gender_dict = {'male': 0, 'female': 1}
        # there're bad files in the dataset with re '^[0-9]*_[0-9]*_[0-9]*.jpg.chip.jpg' (missing race)
        # command for counting race=others (in terminal): ls | grep '^[0-9]*_[0-9]*_4_*' | tee /dev/tty | wc -l

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
class UTKFace:
    """
    get train and validation dataloader from UTKface dataset
    source: https://susanqq.github.io/UTKFace/
    """

    batch_size: int = 128
    root: str = '/tmp2/dataset/UTKFace'
    train_csv: str = '/tmp2/dataset/UTKFace/utkface_train.csv'
    val_csv: str = '/tmp2/dataset/UTKFace/utkface_val.csv'
    race_def: str = 'default'
    mean: Tuple = (0.485, 0.456, 0.406)
    std: Tuple = (0.229, 0.224, 0.225)

    train_dataloader: torch.utils.data.DataLoader = field(init=False)
    val_dataloader: torch.utils.data.DataLoader = field(init=False)

    def __post_init__(self):
        train_transform: transforms.transforms.Compose = \
            transforms.Compose([
                transforms.TrivialAugmentWide(),
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
        train_dataset = UTKFaceDataset(self.train_csv, self.root, transform=train_transform, race_def=self.race_def)
        val_dataset = UTKFaceDataset(self.val_csv, self.root, transform=val_transform, race_def=self.race_def)
        self.train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=self.batch_size, 
            shuffle=True, num_workers=12, pin_memory=True, drop_last=True,)
        self.val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=self.batch_size, 
            shuffle=True, num_workers=12, pin_memory=True, drop_last=True,)

class FairFaceDataset(torch.utils.data.Dataset):
    """Custom pytorch Dataset class for FairFace dataset"""

    def __init__(self, csv_file, root_dir, transform=None, race_def='default'):
        self.attributes = pd.read_csv(csv_file)
        self.root_dir = Path(root_dir)
        self.transform = transform
        if race_def == 'UTKFace':
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
                transforms.TrivialAugmentWide(),
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
        
class MAADFaceHQDataset(torch.utils.data.Dataset):
    """Custom pytorch Dataset class for MAAD-Face-HQ dataset"""

    def __init__(self, csv_file, root_dir, transform=None):
        self.attributes = pd.read_csv(csv_file)
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.attribute_list = ['Filename','Identity','Male','Young','Middle_Aged','Senior','Asian','White','Black',
                               'Rosy_Cheeks','Shiny_Skin','Bald','Wavy_Hair','Receding_Hairline','Bangs','Sideburns','Black_Hair','Blond_Hair','Brown_Hair','Gray_Hair',
                               'No_Beard','Mustache','5_o_Clock_Shadow','Goatee','Oval_Face','Square_Face','Round_Face','Double_Chin','High_Cheekbones','Chubby',
                               'Obstructed_Forehead','Fully_Visible_Forehead','Brown_Eyes','Bags_Under_Eyes','Bushy_Eyebrows','Arched_Eyebrows',
                               'Mouth_Closed','Smiling','Big_Lips','Big_Nose','Pointy_Nose','Heavy_Makeup',
                               'Wearing_Hat','Wearing_Earrings','Wearing_Necktie','Wearing_Lipstick','No_Eyewear','Eyeglasses','Attractive']
    
    def __len__(self):
        return len(self.attributes.index)
    
    def __getitem__(self, index: int):
        X = Image.open(self.root_dir / self.attributes.iloc[index]['Filename'])
        if self.transform:
            X = self.transform(X)

        target = self.attributes.iloc[index][self.attribute_list[2:]] # binary attributes
        target = torch.tensor(target)
        return X, target

@dataclass
class MAADFaceHQ:
    """
    get train and test dataloader from MAAD-Face-HQ dataset
    attribute source: https://github.com/pterhoer/MAAD-Face
    image is from Vggface2-HQ: https://github.com/NNNNAI/VGGFace2-HQ
    """

    batch_size: int = 128
    root: str = '/tmp2/dataset/MAADFace_HQ'
    train_csv: str = '/tmp2/dataset/MAADFace_HQ/MAADFace_HQ_train.csv'
    test_csv: str = '/tmp2/dataset/MAADFace_HQ/MAADFace_HQ_test.csv'
    mean: Tuple = (0.485, 0.456, 0.406)  # same as ImageNet
    std: Tuple = (0.229, 0.224, 0.225)

    train_dataloader: torch.utils.data.DataLoader = field(init=False)
    val_dataloader: torch.utils.data.DataLoader = field(init=False)

    def __post_init__(self):
        train_transform: transforms.transforms.Compose = \
            transforms.Compose([
                transforms.TrivialAugmentWide(),
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                # To make clipping on adverserial element intuitively
                # only normalize before passing it into the model
                # transforms.Normalize(mean, std),
                # becauuse of this, random erasing pixel as imagenet mean, not zero
                # transforms.RandomErasing(0.2, value=self.mean),
            ])
        test_transform: transforms.transforms.Compose = \
            transforms.Compose([
                transforms.Resize((224, 224)), 
                transforms.ToTensor(),
                # To make clipping on adverserial element intuitively
                # only normalize before passing it into the model
                # transforms.Normalize(mean, std),
            ])
        # select the dataset source
        train_dataset = MAADFaceHQDataset(self.train_csv, self.root+'/train', transform=train_transform)
        test_dataset = MAADFaceHQDataset(self.test_csv, self.root+'/test', transform=test_transform)
        self.train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=self.batch_size, 
            shuffle=True, num_workers=12, pin_memory=True, drop_last=True,)
        self.test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=self.batch_size, 
            shuffle=True, num_workers=12, pin_memory=True, drop_last=True,)