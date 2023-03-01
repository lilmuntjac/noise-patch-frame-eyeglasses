from dataclasses import dataclass, field
import numpy as np

import torch
import torch.nn.functional as F
from .perturbations import *

def filter_logit_categori(logit, batch_dim=0, model_name=None):
    """
    Convert raw logit of UTKFace and FairFace into race, gender, and age respectively
    Input:
        logit: raw logit from UTKFace or FairFace model
        batch_dim: batch dimension of the logit tensor, 0 for typical, 1 for perturbed loss
        model_name: name of the model, only "UTKFace" and "FairFace" are allowed
    Output:
        3 tensor of race_logit, gender_logit, and age_logit, they are not in the same shape
    """
    # UTKFace
    if model_name == "UTKFace":
        if batch_dim == 0:
            return logit[:,0:5], logit[:,5:7], logit[:,7:16]
        elif batch_dim == 1:
            return logit[:,:,0:5], logit[:,:,5:7], logit[:,:,7:16]
        else:
            assert False, f'batch dimension must be 0 or 1'
    # FairFace
    elif model_name == "FairFace":
        if batch_dim == 0:
            return logit[:,0:7], logit[:,7:9], logit[:,9:18]
        elif batch_dim == 1:
            return logit[:,:,0:7], logit[:,:,7:9], logit[:,:,9:18]
        else:
            assert False, f'batch dimension must be 0 or 1'
    else:
        assert False, f'unsupported model name, only "UTKFace" and "FairFace" are allowed.'

def to_CEloss(race_logit, gender_logit, age_logit, label):
    """
    Get the cross entropy loss of race, gender, and age
    """
    loss_CE_race = F.cross_entropy(race_logit, label[:,0])
    loss_CE_gender = F.cross_entropy(gender_logit, label[:,1])
    loss_CE_age = F.cross_entropy(age_logit, label[:,2])
    return loss_CE_race, loss_CE_gender, loss_CE_age

def to_prediction_categori(race_logit, gender_logit, age_logit, batch_dim=0):
    """
    Convert race_logit, gender_logit, and age_logit into prediction
    """
    if batch_dim != 0 and batch_dim != 1:
        assert False, f'batch dimension must be 0 or 1'
    _, race_pred = torch.max(race_logit, dim=batch_dim+1)
    _, gender_pred = torch.max(gender_logit, dim=batch_dim+1)
    _, age_pred = torch.max(age_logit, dim=batch_dim+1)
    pred = torch.stack((race_pred, gender_pred, age_pred), dim=batch_dim+1)
    return pred

def regroup_categori(tensor, label, sens_type, batch_dim=0):
    """
    Regroup losses, logit, prediction, or label into 2 groups by sensitive attribute
    """
    if batch_dim == 0:
        if sens_type == "race":
            # white and non-white
            g1_tensor = tensor[label[:,0]==0]
            g2_tensor = tensor[label[:,0]!=0]
        elif sens_type == "gender":
            # male and female
            g1_tensor = tensor[label[:,1]==0]
            g2_tensor = tensor[label[:,1]!=0]
        elif sens_type == "age":
            # young and senior
            g1_tensor = tensor[label[:,2]<=3]
            g2_tensor = tensor[label[:,2]>3]
        else:
            assert False, f'no such sensitive attribute'
    elif batch_dim == 1:
        if sens_type == "race":
            # white and non-white
            g1_tensor = tensor[:, label[:,0]==0]
            g2_tensor = tensor[:, label[:,0]!=0]
        elif sens_type == "gender":
            # male and female
            g1_tensor = tensor[:, label[:,1]==0]
            g2_tensor = tensor[:, label[:,1]!=0]
        elif sens_type == "age":
            # young and senior
            g1_tensor = tensor[:, label[:,2]<=3]
            g2_tensor = tensor[:, label[:,2]>3]
        else:
            assert False, f'no such sensitive attribute'
    else:
        assert False, f'batch dimension must be 0 or 1'
    return g1_tensor, g2_tensor

"""
Fairness loss function (categorical attributes)
"""

def loss_categori_direct(logit, label, sens_type, attr_type, coef, model_name=None):
    """
    Write the fairness loss directly defined by the fairness matrix
    Input: 
        logit: model output from model that made categorical predictions
        label: training label from the dataset
        sens_type: sensitive attribute from the dataset,
            race: white or non-white 
            gander: male or female
            age: age <= 30 or age > 30
        attr_type: attribute type that fairness (accuracy) is evaluated on,
                   it could be "all", "race", "gender", or "age".
        coef: coeficient for cross entropy loss
        model_name: specify the model used. It could be "UTKFace" or "FairFace"
    Output:
        fairness loss in shape []
    """
    # cross entropy loss
    race_logit, gender_logit, age_logit = filter_logit_categori(logit, batch_dim=0, model_name=model_name)
    loss_CE_race, loss_CE_gender, loss_CE_age = to_CEloss(race_logit, gender_logit, age_logit, label)
    # prediction 
    pred = to_prediction_categori(race_logit, gender_logit, age_logit, batch_dim=0)
    # regroup logit and label by sensitive attribute type & compute accuracy
    group_1_pred, gropu_2_pred = regroup_categori(pred, label, sens_type, batch_dim=0)
    group_1_label, gropu_2_label = regroup_categori(label, label, sens_type, batch_dim=0)
    group_1_total, group_2_total = group_1_label.shape[0], gropu_2_label.shape[0]
    if attr_type == "all":
        group_1_correct = torch.all(torch.eq(group_1_pred, group_1_label), dim=1).sum()
        group_2_correct = torch.all(torch.eq(gropu_2_pred, gropu_2_label), dim=1).sum()
        loss_CE = loss_CE_race+loss_CE_gender+loss_CE_age
    elif attr_type == "race":
        group_1_correct = torch.eq(group_1_pred, group_1_label)[:,0].sum()
        group_2_correct = torch.eq(gropu_2_pred, gropu_2_label)[:,0].sum()
        loss_CE = loss_CE_race
    elif attr_type == "gender":
        group_1_correct = torch.eq(group_1_pred, group_1_label)[:,1].sum()
        group_2_correct = torch.eq(gropu_2_pred, gropu_2_label)[:,1].sum()
        loss_CE = loss_CE_gender
    elif attr_type == "age":
        group_1_correct = torch.eq(group_1_pred, group_1_label)[:,2].sum()
        group_2_correct = torch.eq(gropu_2_pred, gropu_2_label)[:,2].sum()
        loss_CE = loss_CE_age
    else:
        assert False, f'no such attribute'
    group_1_acc, group_2_acc = group_1_correct/(group_1_total+1e-9), group_2_correct/(group_2_total+1e-9)
    direct_loss = torch.abs(group_1_acc-group_2_acc)
    loss = direct_loss+coef*loss_CE
    return loss

def loss_categori_CEmasking(logit, label, sens_type, attr_type, model_name=None):
    """
    Write the fairness loss directly defined by the fairness matrix
    Input: 
        logit: model output from model that made categorical predictions
        label: training label from the dataset
        sens_type: sensitive attribute from the dataset,
            race: white or non-white 
            gander: male or female
            age: age <= 30 or age > 30
        attr_type: attribute type that fairness (accuracy) is evaluated on,
                   it could be "all", "race", "gender", or "age".
        model_name: specify the model used. It could be "UTKFace" or "FairFace"
    Output:
        fairness loss in shape []
    """
    race_logit, gender_logit, age_logit = filter_logit_categori(logit, batch_dim=0, model_name=model_name)
    pred = to_prediction_categori(race_logit, gender_logit, age_logit, batch_dim=0)
    loss_CE_race, loss_CE_gender, loss_CE_age = to_CEloss(race_logit, gender_logit, age_logit, label)
    # regroup the data by the sensitive attribute type
    group_1_pred, gropu_2_pred = regroup_categori(pred, label, sens_type, batch_dim=0)
    group_1_label, gropu_2_label = regroup_categori(label, label, sens_type, batch_dim=0)
    group_1_total, group_2_total = group_1_label.shape[0], gropu_2_label.shape[0]
    # find the advantage group
    if attr_type == "all":
        group_1_correct = torch.all(torch.eq(group_1_pred, group_1_label), dim=1).sum()
        group_2_correct = torch.all(torch.eq(gropu_2_pred, gropu_2_label), dim=1).sum()
        loss_CE = loss_CE_race+loss_CE_gender+loss_CE_age
    elif attr_type == "race":
        group_1_correct = torch.eq(group_1_pred, group_1_label)[:,0].sum()
        group_2_correct = torch.eq(gropu_2_pred, gropu_2_label)[:,0].sum()
        loss_CE = loss_CE_race
    elif attr_type == "gender":
        group_1_correct = torch.eq(group_1_pred, group_1_label)[:,1].sum()
        group_2_correct = torch.eq(gropu_2_pred, gropu_2_label)[:,1].sum()
        loss_CE = loss_CE_gender
    elif attr_type == "age":
        group_1_correct = torch.eq(group_1_pred, group_1_label)[:,2].sum()
        group_2_correct = torch.eq(gropu_2_pred, gropu_2_label)[:,2].sum()
        loss_CE = loss_CE_age
    else:
        assert False, f'no such attribute'
    group_1_acc, group_2_acc = group_1_correct/(group_1_total+1e-9), group_2_correct/(group_2_total+1e-9)
    group_1_coef, group_2_coef = torch.where(group_1_acc > group_2_acc, -1 ,0), torch.where(group_2_acc > group_1_acc, -1 ,0)
    group_1_CE, group_2_CE = regroup_categori(loss_CE, label, sens_type)
    loss = torch.cat((group_1_CE*group_1_coef, group_2_CE*group_2_coef), 0) # in shape (N)
    return torch.mean(loss)

def loss_categori_perturbOptim(logit, label, sens_type, attr_type, coef, model_name=None):
    """
    
    """
    # cross entropy loss
    race_logit, gender_logit, age_logit = filter_logit_categori(logit, batch_dim=0, model_name=model_name)
    loss_CE_race, loss_CE_gender, loss_CE_age = to_CEloss(race_logit, gender_logit, age_logit, label)
    if attr_type == "all":
        loss_CE = loss_CE_race+loss_CE_gender+loss_CE_age
    elif attr_type == "race":
        loss_CE = loss_CE_race
    elif attr_type == "gender":
        loss_CE = loss_CE_gender
    elif attr_type == "age":
        loss_CE = loss_CE_age
    else:
        assert False, f'no such attribute'
    # fairness function
    def perturbed_pq(x, label=label):
        race_logit, gender_logit, age_logit = filter_logit_categori(x, batch_dim=1, model_name=model_name)
        pred = to_prediction_categori(race_logit, gender_logit, age_logit, batch_dim=1)
        label_duped = label.repeat(x.shape[0], 1, 1)
        group_1_pred, gropu_2_pred = regroup_categori(pred, label, sens_type, batch_dim=1)
        group_1_label, gropu_2_label = regroup_categori(label_duped, label, sens_type, batch_dim=1)
        group_1_total, group_2_total = group_1_label.shape[0], gropu_2_label.shape[0]
        # compute fairness
        if attr_type == "all":
            group_1_correct = torch.sum(torch.all(torch.eq(group_1_pred, group_1_label), dim=2), dim=1)
            group_2_correct = torch.sum(torch.all(torch.eq(gropu_2_pred, gropu_2_label), dim=2), dim=1)
        elif attr_type == "race":
            group_1_correct = torch.sum(torch.eq(group_1_pred[:,:,0], group_1_label[:,:,0]), dim=1)
            group_2_correct = torch.sum(torch.eq(gropu_2_pred[:,:,0], gropu_2_label[:,:,0]), dim=1)
        elif attr_type == "gender":
            group_1_correct = torch.sum(torch.eq(group_1_pred[:,:,1], group_1_label[:,:,1]), dim=1)
            group_2_correct = torch.sum(torch.eq(gropu_2_pred[:,:,1], gropu_2_label[:,:,1]), dim=1)
        elif attr_type == "age":
            group_1_correct = torch.sum(torch.eq(group_1_pred[:,:,2], group_1_label[:,:,2]), dim=1)
            group_2_correct = torch.sum(torch.eq(gropu_2_pred[:,:,2], gropu_2_label[:,:,2]), dim=1)
        else:
            assert False, f'no such attribute'
        group_1_acc, group_2_acc = group_1_correct/(group_1_total+1e-9), group_2_correct/(group_2_total+1e-9)
        perturbed_loss = torch.abs(group_1_acc-group_2_acc)
        return perturbed_loss
    # Turns a function into a differentiable one via perturbations
    pret_pq = perturbed(perturbed_pq, 
                         num_samples=10000,
                         sigma=0.5,
                         noise='gumbel',
                         batched=False)
    # loss perturbed_loss
    loss = pret_pq(logit)+coef*loss_CE
    return loss

def loss_categori_perturbOptim_full(logit, label, sens_type, attr_type, coef, model_name=None):
    """
    
    """
    # fairness function
    def perturbed_pq(x, label=label):
        race_logit, gender_logit, age_logit = filter_logit_categori(x, batch_dim=1, model_name=model_name)
        pred = to_prediction_categori(race_logit, gender_logit, age_logit, batch_dim=1)
        label_duped = label.repeat(x.shape[0], 1, 1)
        group_1_pred, gropu_2_pred = regroup_categori(pred, label, sens_type, batch_dim=1)
        group_1_label, gropu_2_label = regroup_categori(label_duped, label, sens_type, batch_dim=1)
        group_1_total, group_2_total = group_1_label.shape[0], gropu_2_label.shape[0]
        # compute fairness
        if attr_type == "all":
            group_1_correct = torch.sum(torch.all(torch.eq(group_1_pred, group_1_label), dim=2), dim=1)
            group_2_correct = torch.sum(torch.all(torch.eq(gropu_2_pred, gropu_2_label), dim=2), dim=1)
        elif attr_type == "race":
            group_1_correct = torch.sum(torch.eq(group_1_pred[:,:,0], group_1_label[:,:,0]), dim=1)
            group_2_correct = torch.sum(torch.eq(gropu_2_pred[:,:,0], gropu_2_label[:,:,0]), dim=1)
        elif attr_type == "gender":
            group_1_correct = torch.sum(torch.eq(group_1_pred[:,:,1], group_1_label[:,:,1]), dim=1)
            group_2_correct = torch.sum(torch.eq(gropu_2_pred[:,:,1], gropu_2_label[:,:,1]), dim=1)
        elif attr_type == "age":
            group_1_correct = torch.sum(torch.eq(group_1_pred[:,:,2], group_1_label[:,:,2]), dim=1)
            group_2_correct = torch.sum(torch.eq(gropu_2_pred[:,:,2], gropu_2_label[:,:,2]), dim=1)
        else:
            assert False, f'no such attribute'
        group_1_acc, group_2_acc = group_1_correct/(group_1_total+1e-9), group_2_correct/(group_2_total+1e-9)
        perturbed_loss = torch.abs(group_1_acc-group_2_acc)
        return perturbed_loss
    def perturbed_ac(x, label=label):
        race_logit, gender_logit, age_logit = filter_logit_categori(x, batch_dim=1, model_name=model_name)
        pred = to_prediction_categori(race_logit, gender_logit, age_logit, batch_dim=1)
        label_duped = label.repeat(x.shape[0], 1, 1)
        group_1_pred, gropu_2_pred = regroup_categori(pred, label, sens_type, batch_dim=1)
        group_1_label, gropu_2_label = regroup_categori(label_duped, label, sens_type, batch_dim=1)
        group_1_total, group_2_total = group_1_label.shape[0], gropu_2_label.shape[0]
        #
        if attr_type == "all":
            group_1_correct = torch.sum(torch.all(torch.eq(group_1_pred, group_1_label), dim=2), dim=1)
            group_2_correct = torch.sum(torch.all(torch.eq(gropu_2_pred, gropu_2_label), dim=2), dim=1)
        elif attr_type == "race":
            group_1_correct = torch.sum(torch.eq(group_1_pred[:,:,0], group_1_label[:,:,0]), dim=1)
            group_2_correct = torch.sum(torch.eq(gropu_2_pred[:,:,0], gropu_2_label[:,:,0]), dim=1)
        elif attr_type == "gender":
            group_1_correct = torch.sum(torch.eq(group_1_pred[:,:,1], group_1_label[:,:,1]), dim=1)
            group_2_correct = torch.sum(torch.eq(gropu_2_pred[:,:,1], gropu_2_label[:,:,1]), dim=1)
        elif attr_type == "age":
            group_1_correct = torch.sum(torch.eq(group_1_pred[:,:,2], group_1_label[:,:,2]), dim=1)
            group_2_correct = torch.sum(torch.eq(gropu_2_pred[:,:,2], gropu_2_label[:,:,2]), dim=1)
        else:
            assert False, f'no such attribute'
        group_1_acc, group_2_acc = group_1_correct/(group_1_total+1e-9), group_2_correct/(group_2_total+1e-9)
        perturbed_loss = (1-group_1_acc)+(1-group_2_acc)
        return perturbed_loss
    # Turns a function into a differentiable one via perturbations
    pret_pq = perturbed(perturbed_pq, 
                         num_samples=10000,
                         sigma=0.5,
                         noise='gumbel',
                         batched=False)
    pert_ac = perturbed(perturbed_ac, 
                         num_samples=10000,
                         sigma=0.5,
                         noise='gumbel',
                         batched=False)
    # loss perturbed_loss
    loss = pret_pq(logit)+coef*pert_ac(logit)
    return loss