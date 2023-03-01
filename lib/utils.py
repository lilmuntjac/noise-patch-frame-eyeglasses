from pathlib import Path
import numpy as np
import pandas as pd
import random

import torch
from torchvision import transforms
from fairnessCategori import regroup_categori

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

def filter_celeba_label(label, attr_pred_idx=[2, 19, 21, 31], attr_sens_idx=[20,]):
    """Split the CelebA label into training label and sensitive attribute"""
    # Attractive: 2, High_Cheekbones: 19, Mouth_Slightly_Open: 21, Smiling: 31, Male: 20
    return label[:,attr_pred_idx], label[:,attr_sens_idx]

def filter_agemodel_label(label):
    """Split the UTKFace, FairFace label into training label and sensitive attribute"""
    # in age model, training label is age, sensitive attrribute is gender
    # also we need to binarized the age label
    binary_age = torch.where(label[:,2:3]>3, 1, 0) # keep the shape as (N, 1)
    return binary_age, label[:,1:2]

def to_prediction(logit, model_name=None):
    """
    Convert the logit into prediction for CelebA, UTKface, FairFace, and Age model
        Input:
            logit: model output from CelebA, UTKface model, FairFace model, and Age model
            model_name: the name of the model, only "CelebA", "UTKface", "FairFace", "AgeModel" are allowed
        Output:
            A torch tensor in shape (N, attributes)
    """
    if model_name == 'CelebA' or model_name == 'AgeModel':
        pred = torch.where(logit> 0.5, 1, 0)
    elif model_name == 'UTKFace':
        _, race_pred = torch.max(logit[:,0:5], dim=1)
        _, gender_pred = torch.max(logit[:,5:7], dim=1)
        _, age_pred = torch.max(logit[:,7:16], dim=1)
        pred = torch.stack((race_pred, gender_pred, age_pred), dim=1)
    elif model_name == 'FairFace':
        _, race_pred = torch.max(logit[:,0:7], dim=1)
        _, gender_pred = torch.max(logit[:,7:9], dim=1)
        _, age_pred = torch.max(logit[:,9:18], dim=1)
        pred = torch.stack((race_pred, gender_pred, age_pred), dim=1)
    else:
        assert False, f'Unsupport model'
    return pred

def calc_groupcm(pred, label, sens):
    """
    Split the prediction and the corresponding label by its sensitive attribute,
    then compute the confusion matrix for them.
        Output:
            Numpy array in shape (attributes, 8) male & female confusion matrix
    """
    def confusion_matrix(pred, label, idx):
        tp = torch.mul(pred[:,idx], label[:,idx]).sum()
        fp = torch.mul(pred[:,idx], torch.sub(1, label[:,idx])).sum()
        fn = torch.mul(torch.sub(1, pred[:,idx]), label[:,idx]).sum()
        tn = torch.mul(torch.sub(1, pred[:,idx]), torch.sub(1, label[:,idx])).sum()
        return tp, fp, fn, tn
    m_pred, m_label = pred[sens[:,0]==1], label[sens[:,0]==1]
    f_pred, f_label = pred[sens[:,0]==0], label[sens[:,0]==0]
    stat = np.array([])
    for idx in range(label.shape[-1]):
        mtp, mfp, mfn, mtn = confusion_matrix(m_pred, m_label, idx)
        ftp, ffp, ffn, ftn = confusion_matrix(f_pred, f_label, idx)
        row = np.array([[mtp.item(), mfp.item(), mfn.item(), mtn.item(), ftp.item(), ffp.item(), ffn.item(), ftn.item()]])
        stat =  np.concatenate((stat, row), axis=0) if len(stat) else row
    return stat

def calc_groupacc(pred, label, split):
    """
    Split the prediction and the training label by either "race" or "gender" to 
    count the sum of right and worng predictions.
        For "race", split by white or non-white 
        For "gender", split by male or female
    Output:
        Numpy array of shape (3, 4)
        3: race, gender and age, 
        4: group 1 correct / group 1 wromg / group 2 correct /  group 2 wrong
    """
    if split == 'race':
        w_pred, w_label = pred[label[:,0]==0], label[label[:,0]==0]
        n_pred, n_label = pred[label[:,0]!=0], label[label[:,0]!=0]
        w_result, n_result = torch.eq(w_pred, w_label), torch.eq(n_pred, n_label)
        w_correct, w_wrong = torch.sum(w_result, dim=0), torch.sum(torch.logical_not(w_result), dim=0)
        n_correct, n_wrong = torch.sum(n_result, dim=0), torch.sum(torch.logical_not(n_result), dim=0)
        stat = torch.stack((w_correct, w_wrong, n_correct, n_wrong), dim=1)
    elif split == 'gender':
        m_pred, m_label = pred[label[:,1]==0], label[label[:,1]==0]
        f_pred, f_label = pred[label[:,1]==1], label[label[:,1]==1]
        m_result, f_result = torch.eq(m_pred, m_label), torch.eq(f_pred, f_label)
        m_correct, m_wrong = torch.sum(m_result, dim=0), torch.sum(torch.logical_not(m_result), dim=0)
        f_correct, f_wrong = torch.sum(f_result, dim=0), torch.sum(torch.logical_not(f_result), dim=0)
        stat = torch.stack((m_correct, m_wrong, f_correct, f_wrong), dim=1)
    else:
        assert False, f'Unsupport split attribute'
    return stat.detach().cpu().numpy()

def calc_grouppq(pred, label, sens_type):
    """
    Split the prediction and the corresponding label by its sensitive attribute type,
    then compute the prediciton result for them. It has nothing to do with the fairness.
    """
    # pred and label should in shape (N, 3) as a whole.
    group_1_pred, group_2_pred = regroup_categori(pred, label, sens_type)
    group_1_label, group_2_label = regroup_categori(label, label, sens_type)
    group_1_total, group_2_total = group_1_label.shape[0], group_2_label.shape[0]
    group_1_correct = torch.sum(torch.eq(group_1_pred, group_1_label), dim=0)
    group_1_correct_all = torch.all(torch.eq(group_1_pred, group_1_label), dim=1).sum().reshape(1)
    group_1_correct = torch.cat((group_1_correct, group_1_correct_all), 0)
    group_1_wrong = group_1_total-group_1_correct
    group_2_correct = torch.sum(torch.eq(group_2_pred, group_2_label), dim=0)
    group_2_correct_all = torch.all(torch.eq(group_2_pred, group_2_label), dim=1).sum().reshape(1)
    group_2_correct = torch.cat((group_2_correct, group_2_correct_all), 0)
    group_2_wrong = group_2_total-group_2_correct
    result = torch.stack((group_1_correct, group_1_wrong, group_2_correct, group_2_wrong), dim=1)
    # in shape (attribute+1, 4) NOT THE PERFORMANCE OF FAIRNESS, the extra attribute is 'all'
    return result.clone().detach().cpu().numpy()