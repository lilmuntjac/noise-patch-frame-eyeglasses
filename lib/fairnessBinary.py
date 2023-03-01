from dataclasses import dataclass, field
import numpy as np

import torch
import torch.nn.functional as F
from .perturbations import *

def regroup(tensor, sens):
    # regroup logit, prediction, or label into 2 groups
    g1_tensor = tensor[sens[:,0]==0]
    g2_tensor = tensor[sens[:,0]!=0]
    return g1_tensor, g2_tensor

def regroup_perturbed(tensor, sens):
    # regroup prediction and label, 
    # beware of the new dimension added for the perturbed optimizers
    g1_tensor = tensor[:, sens[:,0]==0]
    g2_tensor = tensor[:, sens[:,0]!=0]
    return g1_tensor, g2_tensor

def get_TPR(pred, label, dim=0):
    """
    Compute TPR for the predicitons
    dim: batch dimension of the predictions/ label, for example
        0: the predictions have the shape (N, A) or similar
        1: the predictions have the shape (P, N, A)
    """
    numerator = torch.sum(pred*label, dim=dim) # TP
    denominator = torch.sum(label, dim=dim) # ~TP+FN
    TPR = torch.full_like(denominator, fill_value=1.0)
    TPR_mask = (denominator != 0)
    TPR[TPR_mask] = numerator[TPR_mask]/denominator[TPR_mask]
    return TPR

def get_TNR(pred, label, dim=0):
    """
    Compute TNR for the predicitons
    dim: batch dimension of the predictions/ label, for example
        0: the predictions have the shape (N, A) or similar
        1: the predictions have the shape (P, N, A)
    """
    numerator = torch.sum(torch.sub(1, pred)*torch.sub(1, label), dim=dim) # TN
    denominator = torch.sum(torch.sub(1, label), dim=dim) # ~FP+TN
    TNR = torch.full_like(denominator, fill_value=1.0)
    TNR_mask = (denominator != 0)
    TNR[TNR_mask] = numerator[TNR_mask]/denominator[TNR_mask]
    return TNR

def get_FNR(pred, label, dim=0):
    """
    Compute FNR for the predicitons
    dim: batch dimension of the predictions/ label, for example
        0: the predictions have the shape (N, A) or similar
        1: the predictions have the shape (P, N, A)
    """
    numerator = torch.sum(torch.sub(1, pred)*label, dim=dim) # FN
    denominator = torch.sum(label, dim=dim) # ~TP+FN
    FNR = torch.full_like(denominator, fill_value=1.0)
    FNR_mask = (denominator != 0)
    FNR[FNR_mask] = numerator[FNR_mask]/denominator[FNR_mask]
    return FNR

def get_FPR(pred, label, dim=0):
    """
    Compute FPR for the predicitons
    dim: batch dimension of the predictions/ label, for example
        0: the predictions have the shape (N, A) or similar
        1: the predictions have the shape (P, N, A)
    """
    numerator = torch.sum(pred*torch.sub(1, label), dim=dim) # FP
    denominator = torch.sum(label, dim=dim) # ~TP+FN
    FPR = torch.full_like(denominator, fill_value=1.0)
    FPR_mask = (denominator != 0)
    FPR[FPR_mask] = numerator[FPR_mask]/denominator[FPR_mask]
    return FPR

def get_acc(pred, label, dim=0):
    """
    Compute Accuracy for the predictions
    dim: batch dimension of the predictions/ label, for example
        0: the predictions have the shape (N, A) or similar
        1: the predictions have the shape (P, N, A)
    """
    numerator = torch.sum(pred*label, dim=dim)+torch.sum(torch.sub(1, pred)*torch.sub(1, label), dim=dim) #TP+TN
    denominator = label.shape[dim]
    acc = numerator/denominator
    return acc

"""
Fairness loss function (binary attributes)
"""
    
def loss_binary_direct(state_of_fairness, logit, label, sens_attr, p_coef=0, n_coef=0):
    """
    Write the fairness loss directly defined by the fairness matrix
    Input: 
        state_of_fairness: fairness matrix used
        logit: model output from model that made binary predictions
        label: training label from the dataset
        sens_attr: sensitive attribute from the dataset
    Output:
        fairness loss in shape []
    """
    loss_BCE = F.binary_cross_entropy(logit, label, reduction='none')
    positive_BCE, negative_BCE = loss_BCE*(label), loss_BCE*(torch.sub(1, label))
    # get and regroup the predictions
    # pred = torch.where(logit> 0.5, 1, 0) # gradient can't pass through this function
    pred = 1./(1+torch.exp(-1e5*logit))
    g1_pred, g2_pred = regroup(pred, sens_attr)
    g1_label, g2_label = regroup(label, sens_attr)
    if state_of_fairness == 'equality of opportunity':
        g1_TPR, g2_TPR = get_TPR(g1_pred, g1_label, dim=0), get_TPR(g2_pred, g2_label, dim=0)
        TPR_loss = torch.abs(g1_TPR-g2_TPR)
        loss = TPR_loss+p_coef*positive_BCE
        return torch.mean(loss)
    elif state_of_fairness == 'equalized odds':
        g1_TPR, g2_TPR = get_TPR(g1_pred, g1_label, dim=0), get_TPR(g2_pred, g2_label, dim=0)
        g1_TNR, g2_TNR = get_TNR(g1_pred, g1_label, dim=0), get_TNR(g2_pred, g2_label, dim=0)
        TPR_loss, TNR_loss = torch.abs(g1_TPR-g2_TPR),  torch.abs(g1_TNR-g2_TNR)
        loss = TPR_loss+p_coef*positive_BCE + TNR_loss+n_coef*negative_BCE
        return torch.mean(loss)
    elif state_of_fairness == 'prediction quality':
        g1_acc, g2_acc = get_acc(g1_pred, g1_label, dim=0), get_acc(g2_pred, g2_label, dim=0)
        acc_loss = torch.abs(g1_acc-g2_acc)
        return torch.mean(acc_loss)
    else:
        assert False, f'Unsupport state of fairness'

def loss_binary_BCEmasking(state_of_fairness, logit, label, sens_attr):
    """
    Batch based, buck the advantage group label positive cells for TPR, (negative for TNR)
    For prediction quaility, buck all the cells belong to the advantage group
    """
    loss_BCE = F.binary_cross_entropy(logit, label, reduction='none')
    g1_BCE, g2_BCE = regroup(loss_BCE, sens_attr)
    # find the advantage group and the target mask
    pred = torch.where(logit> 0.5, 1, 0)
    g1_pred, g2_pred = regroup(pred, sens_attr)
    g1_label, g2_label = regroup(label, sens_attr)
    if state_of_fairness ==  'equality of opportunity':
        g1_TPR, g2_TPR = get_TPR(g1_pred, g1_label, dim=0), get_TPR(g2_pred, g2_label, dim=0)
        g1_p_coef, g2_p_coef= torch.where(g1_TPR > g2_TPR, -1, 0), torch.where(g2_TPR > g1_TPR, -1, 0)
        g1_p_loss, g2_p_loss = g1_BCE*(g1_label*g1_p_coef), g2_BCE*(g2_label*g2_p_coef)
        loss = torch.cat((g1_p_loss, g2_p_loss), 0)
    elif state_of_fairness == 'equalized odds':
        g1_TPR, g2_TPR = get_TPR(g1_pred, g1_label, dim=0), get_TPR(g2_pred, g2_label, dim=0)
        g1_TNR, g2_TNR = get_TNR(g1_pred, g1_label, dim=0), get_TNR(g2_pred, g2_label, dim=0)
        g1_p_coef, g2_p_coef= torch.where(g1_TPR > g2_TPR, -1, 0), torch.where(g2_TPR > g1_TPR, -1, 0)
        g1_n_coef, g2_n_coef= torch.where(g1_TNR > g2_TNR, -1, 0), torch.where(g2_TNR > g1_TNR, -1, 0)
        g1_p_loss, g2_p_loss = g1_BCE*(g1_label*g1_p_coef), g2_BCE*(g2_label*g2_p_coef)
        g1_n_loss, g2_n_loss = g1_BCE*(torch.sub(1, g1_label)*g1_n_coef), g2_BCE*(torch.sub(1, g2_label)*g2_n_coef)
        loss = torch.cat((g1_p_loss+g1_n_loss, g2_p_loss+g2_n_loss), 0)
    elif state_of_fairness == 'prediction quality':
        g1_acc, g2_acc = get_acc(g1_pred, g1_label, dim=0), get_acc(g2_pred, g2_label, dim=0)
        g1_coef, g2_coef= torch.where(g1_acc > g2_acc, -1, 0), torch.where(g2_acc > g1_acc, -1, 0)
        g1_loss, g2_loss = g1_BCE*g1_coef, g2_BCE*g2_coef
        loss = torch.cat((g1_loss, g2_loss), 0)
    else:
        assert False, f'Unsupport state of fairness'
    return torch.mean(loss)

def loss_binary_perturbOptim(state_of_fairness, logit, label, sens_attr, p_coef=0, n_coef=0):
    """
    Epoch based, Write the fairness loss directly and warp it with perturbMAP
    """ 
    pred = torch.where(logit> 0.5, 1, 0)
    loss_BCE = F.binary_cross_entropy(logit, label, reduction='none')
    FN_mask, FP_mask = torch.sub(1, pred)*label, pred*torch.sub(1, label)
    FN_BCE, FP_BCE = torch.mean(loss_BCE*FN_mask, dim=0), torch.mean(loss_BCE*FP_mask, dim=0)
    # fairness function
    def perturbed_TPR(x, label=label, sens=sens_attr):
        pred = torch.where(x> 0.5, 1, 0)
        # dupe the label to have the same shape as x
        label_duped = label.repeat(x.shape[0], 1, 1)
        # regroup and compute TPR for both groups
        g1_pred, g2_pred = regroup_perturbed(pred, sens)
        g1_label, g2_label = regroup_perturbed(label_duped, sens)
        g1_TPR, g2_TPR = get_TPR(g1_pred, g1_label, dim=1), get_TPR(g2_pred, g2_label, dim=1)
        TPR_loss = torch.abs(g1_TPR-g2_TPR)
        return TPR_loss
    def perturbed_TNR(x, label=label, sens=sens_attr):
        pred = torch.where(x> 0.5, 1, 0)
        # dupe the label to have the same shape as x
        label_duped = label.repeat(x.shape[0], 1, 1)
        # regroup and compute TPR for both groups
        g1_pred, g2_pred = regroup_perturbed(pred, sens)
        g1_label, g2_label = regroup_perturbed(label_duped, sens)
        g1_TNR, g2_TNR = get_TNR(g1_pred, g1_label, dim=1), get_TNR(g2_pred, g2_label, dim=1)
        TNR_loss = torch.abs(g1_TNR-g2_TNR)
        return TNR_loss
    def perturbed_acc(x, label=label, sens=sens_attr):
        pred = torch.where(x> 0.5, 1, 0)
        # dupe the label to have the same shape as x
        label_duped = label.repeat(x.shape[0], 1, 1)
        # regroup and compute TPR for both groups
        g1_pred, g2_pred = regroup_perturbed(pred, sens)
        g1_label, g2_label = regroup_perturbed(label_duped, sens)
        g1_acc, g2_acc = get_acc(g1_pred, g1_label, dim=1), get_acc(g2_pred, g2_label, dim=1)
        acc_loss = torch.abs(g1_acc-g2_acc)
        return acc_loss
    # Turns a function into a differentiable one via perturbations
    pret_tpr = perturbed(perturbed_TPR, 
                         num_samples=10000,
                         sigma=0.5,
                         noise='gumbel',
                         batched=False)
    pret_tnr = perturbed(perturbed_TNR, 
                         num_samples=10000,
                         sigma=0.5,
                         noise='gumbel',
                         batched=False)
    pret_acc = perturbed(perturbed_acc, 
                         num_samples=10000,
                         sigma=0.5,
                         noise='gumbel',
                         batched=False)
    if state_of_fairness ==  'equality of opportunity':
        TPR_loss = pret_tpr(logit) # in shape (A)
        loss = TPR_loss+p_coef*FN_BCE
    elif state_of_fairness == 'equalized odds':
        TPR_loss = pret_tpr(logit) # in shape (A)
        TNR_loss = pret_tnr(logit) # in shape (A)
        loss = TPR_loss+p_coef*FN_BCE + TNR_loss+n_coef*FP_BCE
    elif state_of_fairness == 'prediction quality':
        acc_loss = pret_acc(logit)
        loss = acc_loss
    else:
        assert False, f'Unsupport state of fairness'
    return torch.mean(loss)

def loss_binary_perturbOptim_full(state_of_fairness, logit, label, sens_attr, p_coef=0, n_coef=0):
    """
    Epoch based, Write the fairness loss directly and warp it with perturbMAP
    """ 
    # fairness function
    def perturbed_TPR(x, label=label, sens=sens_attr):
        pred = torch.where(x> 0.5, 1, 0)
        # dupe the label to have the same shape as x
        label_duped = label.repeat(x.shape[0], 1, 1)
        # regroup and compute TPR for both groups
        g1_pred, g2_pred = regroup_perturbed(pred, sens)
        g1_label, g2_label = regroup_perturbed(label_duped, sens)
        g1_TPR, g2_TPR = get_TPR(g1_pred, g1_label, dim=1), get_TPR(g2_pred, g2_label, dim=1)
        TPR_loss = torch.abs(g1_TPR-g2_TPR)
        return TPR_loss
    def perturbed_TNR(x, label=label, sens=sens_attr):
        pred = torch.where(x> 0.5, 1, 0)
        # dupe the label to have the same shape as x
        label_duped = label.repeat(x.shape[0], 1, 1)
        # regroup and compute TPR for both groups
        g1_pred, g2_pred = regroup_perturbed(pred, sens)
        g1_label, g2_label = regroup_perturbed(label_duped, sens)
        g1_TNR, g2_TNR = get_TNR(g1_pred, g1_label, dim=1), get_TNR(g2_pred, g2_label, dim=1)
        TNR_loss = torch.abs(g1_TNR-g2_TNR)
        return TNR_loss
    def perturbed_acc(x, label=label, sens=sens_attr):
        pred = torch.where(x> 0.5, 1, 0)
        # dupe the label to have the same shape as x
        label_duped = label.repeat(x.shape[0], 1, 1)
        # regroup and compute TPR for both groups
        g1_pred, g2_pred = regroup_perturbed(pred, sens)
        g1_label, g2_label = regroup_perturbed(label_duped, sens)
        g1_acc, g2_acc = get_acc(g1_pred, g1_label, dim=1), get_acc(g2_pred, g2_label, dim=1)
        acc_loss = torch.abs(g1_acc-g2_acc)
        return acc_loss
    # recovery function
    def perturbed_rFNR(x, label=label, sens=sens_attr):
        pred = torch.where(x> 0.5, 1, 0)
        # dupe the label to have the same shape as x
        label_duped = label.repeat(x.shape[0], 1, 1)
        # regroup and compute TPR for both groups
        g1_pred, g2_pred = regroup_perturbed(pred, sens)
        g1_label, g2_label = regroup_perturbed(label_duped, sens)
        g1_FNR, g2_FNR = get_FNR(g1_pred, g1_label, dim=1), get_FNR(g2_pred, g2_label, dim=1)
        return g1_FNR+g2_FNR
    def perturbed_rFPR(x, label=label, sens=sens_attr):
        pred = torch.where(x> 0.5, 1, 0)
        # dupe the label to have the same shape as x
        label_duped = label.repeat(x.shape[0], 1, 1)
        # regroup and compute TPR for both groups
        g1_pred, g2_pred = regroup_perturbed(pred, sens)
        g1_label, g2_label = regroup_perturbed(label_duped, sens)
        g1_FPR, g2_FPR = get_FPR(g1_pred, g1_label, dim=1), get_FPR(g2_pred, g2_label, dim=1)
        return g1_FPR+g2_FPR
    def perturbed_rgFNR(x, label=label, sens=sens_attr):
        pred = torch.where(x> 0.5, 1, 0)
        # dupe the label to have the same shape as x
        label_duped = label.repeat(x.shape[0], 1, 1)
        # regroup and compute TPR for both groups
        g1_pred, g2_pred = regroup_perturbed(pred, sens)
        g1_label, g2_label = regroup_perturbed(label_duped, sens)
        g1_FNR, g2_FNR = get_FNR(g1_pred, g1_label, dim=1), get_FNR(g2_pred, g2_label, dim=1)
        g1_TPR, g2_TPR = get_TPR(g1_pred, g1_label, dim=1), get_TPR(g2_pred, g2_label, dim=1)
        g1_FNR_mask, g2_FNR_mask = torch.where(g1_TPR < g2_TPR, 1, 0), torch.where(g2_TPR < g1_TPR, 1, 0)
        return g1_FNR*g1_FNR_mask+g2_FNR*g2_FNR_mask
    def perturbed_rgFPR(x, label=label, sens=sens_attr):
        pred = torch.where(x> 0.5, 1, 0)
        # dupe the label to have the same shape as x
        label_duped = label.repeat(x.shape[0], 1, 1)
        # regroup and compute TPR for both groups
        g1_pred, g2_pred = regroup_perturbed(pred, sens)
        g1_label, g2_label = regroup_perturbed(label_duped, sens)
        g1_FPR, g2_FPR = get_FPR(g1_pred, g1_label, dim=1), get_FPR(g2_pred, g2_label, dim=1)
        g1_TNR, g2_TNR = get_TNR(g1_pred, g1_label, dim=1), get_TNR(g2_pred, g2_label, dim=1)
        g1_FNR_mask, g2_FNR_mask = torch.where(g1_TNR < g2_TNR, 1, 0), torch.where(g2_TNR < g1_TNR, 1, 0)
        return g1_FPR*g1_FNR_mask+g2_FPR*g2_FNR_mask

    # Turns a function into a differentiable one via perturbations
    pret_tpr = perturbed(perturbed_TPR, 
                         num_samples=10000,
                         sigma=0.5,
                         noise='gumbel',
                         batched=False)
    pret_tnr = perturbed(perturbed_TNR, 
                         num_samples=10000,
                         sigma=0.5,
                         noise='gumbel',
                         batched=False)
    pret_acc = perturbed(perturbed_acc, 
                         num_samples=10000,
                         sigma=0.5,
                         noise='gumbel',
                         batched=False)
    pret_rfnr = perturbed(perturbed_rFNR,
                          num_samples=10000,
                         sigma=0.5,
                         noise='gumbel',
                         batched=False)
    pret_rfpr = perturbed(perturbed_rFPR,
                          num_samples=10000,
                         sigma=0.5,
                         noise='gumbel',
                         batched=False)
    
    if state_of_fairness ==  'equality of opportunity':
        TPR_loss = pret_tpr(logit) # in shape (A)
        rFNR_loss = pret_rfnr(logit)
        loss = TPR_loss+p_coef*rFNR_loss
    elif state_of_fairness == 'equalized odds':
        TPR_loss = pret_tpr(logit) # in shape (A)
        TNR_loss = pret_tnr(logit) # in shape (A)
        rFNR_loss = pret_rfnr(logit)
        rFPR_loss = pret_rfpr(logit)
        loss = TPR_loss+p_coef*rFNR_loss + TNR_loss+n_coef*rFPR_loss
    elif state_of_fairness == 'prediction quality':
        acc_loss = pret_acc(logit)
        loss = acc_loss
    else:
        assert False, f'Unsupport state of fairness'
    return torch.mean(loss)
