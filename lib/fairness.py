from dataclasses import dataclass, field
import numpy as np

import torch
import torch.nn.functional as F
from .perturbations import *

"""
Fairness testing function
"""

def filter_celeba_label(label, attr_pred_idx=[2, 19, 21, 31], attr_sens_idx=[20,]):
    """Split the CelebA label into training label and sensitive attribute"""
    # Attractive: 2, High_Cheekbones: 19, Mouth_Slightly_Open: 21, Smiling: 31, Male: 20
    return label[:,attr_pred_idx], label[:,attr_sens_idx]

def filter_agemodel_label(label):
    """Split the UTKface, FairFace label into training label and sensitive attribute"""
    # in age model, training label is age, sensitive attrribute is gender
    # also we need to binarized the age label
    binary_age = torch.where(label[:,2:3]>3, 1, 0) # keep the shape as (N, 1)
    return binary_age, label[:,1:2]

def to_prediction(logit, model_name=None):
    """
    Convert the logit into prediction for UTKface and FairFace model
    Input:
        logit: model output from UTKface model and FairFace model
        model_name: the name of the model, only "CelebA", "UTKface", "FairFace", and "AgeModel" are allowed
    Output:
        A torch tensor in shape (N, attributes)
    """
    if model_name == 'CelebA' or model_name == 'AgeModel':
        pred = torch.where(logit> 0.5, 1, 0)
    elif model_name == 'UTKface':
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

"""
Fairness loss function
"""

def loss_pq_BCEdiscrepancy(logit, label, sens, loss_type='diff'):
    """
    Binary cross entropy based fairloss for prediction quailty
    Input:
        logit: model output from CelebA model or Age model
        label: training label for CelebA model or Age model
        sens: sensitive attribute for CelebA model or Age model
        type: types of fair loss, 
            "diff": difference of losses
            "pfrac": 1 - proper fractionof losses min(g1,g2)/max(g1,g2)
            "max": only the loss from disadvantage group
    Output: fairloss for prediction quailty in shape []
    """
    # raw binary cross entropy loss with shape (N, attributes)
    raw_bce = F.binary_cross_entropy(logit, label, reduction='none')
    # regroup the raw binary cross entropy loss into 2 groups
    g1_celoss = torch.mean(raw_bce[sens[:,0]==0], dim=0)
    g2_celoss = torch.mean(raw_bce[sens[:,0]!=0], dim=0)
    if loss_type == 'diff':
        fair_bce = torch.abs(g1_celoss-g2_celoss)
    elif loss_type == 'pfrac':
        fair_bce = torch.sub(1, torch.minimum(g1_celoss, g2_celoss)/torch.maximum(g1_celoss, g2_celoss))
    elif loss_type == 'max':
        fair_bce = torch.maximum(g1_celoss, g2_celoss)
    else:
        assert False, f'Unsupport loss type'
    # take mean of all attributes
    fair_bce = torch.mean(fair_bce)
    return fair_bce

def regroup(tensor, sens):
    # regroup logit, prediction, or jabel into 2 groups
    g1_tensor = tensor[sens[:,0]==0]
    g2_tensor = tensor[sens[:,0]!=0]
    return g1_tensor, g2_tensor

def calc_tpr(pred, label):
    # compute true positive rate for prediction from a group
    # pred, label should be in same shape of (attributes)
    numerator = torch.sum(pred*label, dim=0) # TP
    denominator = torch.sum(label, dim=0) # ~TP+FN
    TPR = torch.full_like(denominator, fill_value=1.0)
    TPR_mask = (denominator != 0)
    TPR[TPR_mask] = numerator[TPR_mask]/denominator[TPR_mask]
    return TPR

def calc_coef(g1_TPR, g2_TPR, buck, boost, ignore=0):
    # compute coefficient for target cell in 2 groups
    g1_boost = torch.where(g1_TPR < g2_TPR, boost, ignore)
    g1_buck = torch.where(g1_TPR > g2_TPR, buck, ignore)
    g1_coef = g1_boost+g1_buck
    g2_boost = torch.where(g2_TPR < g1_TPR, boost, ignore)
    g2_buck = torch.where(g2_TPR > g1_TPR, buck, ignore)
    g2_coef = g2_boost+g2_buck
    return g1_coef, g2_coef

def regroup_perturbed(tensor, sens):
    # regroup prediction and label, 
    # beware of the new dimension added for the perturbed optimizers
    g1_tensor = tensor[:, sens[:,0]==0]
    g2_tensor = tensor[:, sens[:,0]!=0]
    return g1_tensor, g2_tensor

def calc_perturbed_tpr(pred, label):
    # the pred and label should include perturbed dimension
    numerator = torch.sum(pred*label, dim=1) # TP
    denominator = torch.sum(label, dim=1) # ~TP+FN
    TPR = torch.full_like(denominator, fill_value=1.0)
    TPR_mask = (denominator != 0)
    TPR[TPR_mask] = numerator[TPR_mask]/denominator[TPR_mask]
    return TPR

def calc_perturbed_tnr(pred, label):
    # the pred and label should include perturbed dimension
    numerator = torch.sum(torch.sub(1, pred)*torch.sub(1, label), dim=1) # TN
    denominator = torch.sum(torch.sub(1, label), dim=1) # ~FP+TN
    TNR = torch.full_like(denominator, fill_value=1.0)
    TNR_mask = (denominator != 0)
    TNR[TNR_mask] = numerator[TNR_mask]/denominator[TNR_mask]
    return TNR

def loss_EOpp_BCEmasking(logit, label, sens, target_type='tp', policy='buck_only', indirect=False):
    """
    Binary cross entropy based fairloss for equality on TPR
    Input:
        logit: model output from CelebA model or Age model
        label: training label for CelebA model or Age model
        sens: sensitive attribute for CelebA model or Age model
        target_type: target cell be selected for fairness, only "tp", "fn", and "tp_fn" are allowed
                     default: "tp"
        policy: policy on how to mutiply the target cells
            "buck_only": x-1 on the target cell for the advantage group, x0 for the disadvantage group.
            "boost_only": x1 on the target cell for the disadvantage group, x0 for the advantage group.
            "buck_boost": x-1 on the target cell for the advantage group, x1 for the disadvantage group.
            default: "buck_only"
        indirect: boolean value to include cells that have negative label or not, default is False
    Output: fairloss for prediction quailty in shape []
    """

    # compute the coefficient for the BCE loss
    pred = torch.where(logit> 0.5, 1, 0) # just like prediction on CelebA and the Age model
    raw_bce = F.binary_cross_entropy(logit, label, reduction='none')
    g1_bce, g2_bce = regroup(raw_bce, sens)
    # extract the target cell mask
    if target_type == 'tp':
        target_cell = pred*label
    elif target_type == 'fn':
        target_cell = torch.sub(1, pred)*label
    elif target_type == 'tp_fn':
        target_cell = pred*label + torch.sub(1, pred)*label
    else:
        assert False, f'Unsupport target cell type'
    g1_target, g2_target = regroup(target_cell, sens)
    # find the advantage group and disadvantage group by comparing TPR
    g1_pred, g2_pred = regroup(pred, sens)
    g1_label, g2_label = regroup(label, sens)
    g1_TPR, g2_TPR = calc_tpr(g1_pred, g1_label), calc_tpr(g2_pred, g2_label)
    # fairness policy
    if policy == 'buck_only':
        g1_coef, g2_coef = calc_coef(g1_TPR, g2_TPR, buck=-1, boost=0)
    elif policy == 'boost_only':
        g1_coef, g2_coef = calc_coef(g1_TPR, g2_TPR, buck=0, boost=1)
    elif policy == 'buck_boost':
        g1_coef, g2_coef = calc_coef(g1_TPR, g2_TPR, buck=-1, boost=1)
    else:
        assert False, f'Unsupport policy'
    g1_loss = g1_bce*(g1_target*g1_coef)
    g2_loss = g2_bce*(g2_target*g2_coef)
    direct_loss = torch.cat((g1_loss, g2_loss), 0)
    # whether to include the false positive cells and true negative cell
    if indirect:
        false_positive = pred*torch.sub(1, label)
        true_negative = torch.sub(1, pred)*torch.sub(1, label)
        indir_cell = false_positive+true_negative
        g1_indir, g2_inder = regroup(indir_cell, sens)
        indir_loss = torch.cat((g1_bce*g1_indir, g2_bce*g2_inder), 0)
        loss = direct_loss+indir_loss
        return torch.mean(torch.mean(loss, dim=0))
    else:
        return torch.mean(torch.mean(direct_loss, dim=0))

def loss_EOpp_direct(logit, label, sens):
    # direct loss to encourage high TPR
    pred = torch.where(logit> 0.5, 1, 0) # just like prediction on CelebA and the Age model
    g1_pred, g2_pred = regroup(pred, sens)
    g1_label, g2_label = regroup(label, sens)
    g1_TPR, g2_TPR = calc_tpr(g1_pred, g1_label), calc_tpr(g2_pred, g2_label)
    loss = 1-g1_TPR + 1-g2_TPR
    return torch.mean(loss)

def loss_hTPR_perturbed(logit, label, sens):
    # perturbed loss that encourage higher TPR
    def hTPR_perturbed_loss(x, label=label, sens=sens):
        # get prediction from logit
        pred = torch.where(x> 0.5, 1, 0)
        # dupe the label to have the same shape as x
        label_duped = label.repeat(x.shape[0], 1, 1)
        # regroup prediction and label, 
        # beware of the new dimension added for the perturbed optimizers
        g1_pred, g2_pred = regroup_perturbed(pred, sens)
        g1_label, g2_label = regroup_perturbed(label_duped, sens)
        g1_TPR, g2_TPR = calc_perturbed_tpr(g1_pred, g1_label), calc_perturbed_tpr(g2_pred, g2_label)
        loss = 1-g1_TPR + 1-g2_TPR
        return loss
    # Turns a function into a differentiable one via perturbations
    pret_tpr = perturbed(hTPR_perturbed_loss,
                         num_samples=10000,
                         sigma=0.5,
                         noise='gumbel',
                         batched=False)
    loss = pret_tpr(logit)
    return torch.mean(loss)

def loss_EOpp_perturbed_epoch(logit, label, sens, loss_mask, composition='linear', coef=1.):
    # epoch based 
    # perturbed loss that reduce the difference in TPR (for equality of opportunity)
    def EOpp_perturbed_loss(x, label=label, sens=sens):
        # get prediction from logit
        pred = torch.where(x> 0.5, 1, 0)
        # dupe the label to have the same shape as x
        label_duped = label.repeat(x.shape[0], 1, 1)
        # regroup prediction and label, 
        # beware of the new dimension added for the perturbed optimizers
        g1_pred, g2_pred = regroup_perturbed(pred, sens)
        g1_label, g2_label = regroup_perturbed(label_duped, sens)
        g1_TPR, g2_TPR = calc_perturbed_tpr(g1_pred, g1_label), calc_perturbed_tpr(g2_pred, g2_label)
        loss = torch.abs(g1_TPR-g2_TPR)
        return loss
    # Turns a function into a differentiable one via perturbations
    pret_tpr = perturbed(EOpp_perturbed_loss,
                         num_samples=10000,
                         sigma=0.5,
                         noise='gumbel',
                         batched=False)
    raw_loss = pret_tpr(logit)
    # utility loss
    utility_loss = F.binary_cross_entropy(logit, label, reduction='none')
    utility_loss = torch.mean(utility_loss, dim=0)
    if composition == 'linear':
        loss = raw_loss+utility_loss*loss_mask*coef
    elif composition == 'exclusive':
        loss = raw_loss*torch.sub(1, loss_mask) + utility_loss*loss_mask*coef
    else:
        assert False, f'Unsupport loss composition'
    
    return torch.mean(loss)

def loss_EOpp_perturbed_batch(logit, label, sens, threshold=0.02, composition='linear', coef=1.):
    # batch based 
    # perturbed loss that reduce the difference in TPR (for equality of opportunity)
    def EOpp_perturbed_loss(x, label=label, sens=sens):
        # get prediction from logit
        pred = torch.where(x> 0.5, 1, 0)
        # dupe the label to have the same shape as x
        label_duped = label.repeat(x.shape[0], 1, 1)
        # regroup prediction and label, 
        # beware of the new dimension added for the perturbed optimizers
        g1_pred, g2_pred = regroup_perturbed(pred, sens)
        g1_label, g2_label = regroup_perturbed(label_duped, sens)
        g1_TPR, g2_TPR = calc_perturbed_tpr(g1_pred, g1_label), calc_perturbed_tpr(g2_pred, g2_label)
        loss = torch.abs(g1_TPR-g2_TPR)
        return loss
    # Turns a function into a differentiable one via perturbations
    pret_tpr = perturbed(EOpp_perturbed_loss,
                         num_samples=10000,
                         sigma=0.5,
                         noise='gumbel',
                         batched=False)
    raw_loss = pret_tpr(logit)
    # utility loss
    utility_loss = F.binary_cross_entropy(logit, label, reduction='none')
    utility_loss = torch.mean(utility_loss, dim=0)
    loss_mask = torch.where(raw_loss < threshold, 1, 0)

    if composition == 'linear':
        loss = raw_loss+utility_loss*loss_mask*coef
    elif composition == 'exclusive':
        loss = raw_loss*torch.sub(1, loss_mask) + utility_loss*loss_mask*coef
    else:
        assert False, f'Unsupport loss composition'
    
    return torch.mean(loss)

def loss_EO_perturbed_batch(logit, label, sens, coef):
    # batch based 
    # perturbed loss that reduce the difference in TPR (for equality of opportunity)
    def EO_perturbed_loss(x, label=label, sens=sens):
        # get prediction from logit
        pred = torch.where(x> 0.5, 1, 0)
        # dupe the label to have the same shape as x
        label_duped = label.repeat(x.shape[0], 1, 1)
        # regroup prediction and label, 
        # beware of the new dimension added for the perturbed optimizers
        g1_pred, g2_pred = regroup_perturbed(pred, sens)
        g1_label, g2_label = regroup_perturbed(label_duped, sens)
        g1_TPR, g2_TPR = calc_perturbed_tpr(g1_pred, g1_label), calc_perturbed_tpr(g2_pred, g2_label)
        g1_TNR, g2_TNR = calc_perturbed_tnr(g1_pred, g1_label), calc_perturbed_tnr(g2_pred, g2_label)
        loss = torch.abs(g1_TPR-g2_TPR) + torch.abs(g1_TNR-g2_TNR)
        return loss
    # Turns a function into a differentiable one via perturbations
    pret_tpr = perturbed(EO_perturbed_loss,
                         num_samples=10000,
                         sigma=0.5,
                         noise='gumbel',
                         batched=False)
    fair_loss = pret_tpr(logit)
    # utility loss
    utility_loss = F.binary_cross_entropy(logit, label, reduction='none')
    utility_loss = torch.mean(utility_loss, dim=0)
    # loss_mask = torch.where(raw_loss < threshold, 1, 0)

    loss = fair_loss + coef*utility_loss

    
    return torch.mean(loss)

def loss_EOpp_perturbed_epoch_b(logit, label, sens, loss_mask, composition='linear', coef=1.):
    # epoch based 
    # perturbed loss that reduce the difference in TPR (for equality of opportunity)
    def EOpp_perturbed_loss(x, label=label, sens=sens):
        # get prediction from logit
        pred = torch.where(x> 0.5, 1, 0)
        # dupe the label to have the same shape as x
        label_duped = label.repeat(x.shape[0], 1, 1)
        # regroup prediction and label, 
        # beware of the new dimension added for the perturbed optimizers
        g1_pred, g2_pred = regroup_perturbed(pred, sens)
        g1_label, g2_label = regroup_perturbed(label_duped, sens)
        g1_TPR, g2_TPR = calc_perturbed_tpr(g1_pred, g1_label), calc_perturbed_tpr(g2_pred, g2_label)
        loss = torch.abs(g1_TPR-g2_TPR)
        return loss
    # Turns a function into a differentiable one via perturbations
    pret_tpr = perturbed(EOpp_perturbed_loss,
                         num_samples=10000,
                         sigma=0.5,
                         noise='gumbel',
                         batched=False)
    fair_loss = pret_tpr(logit)
    # utility loss
    utility_loss = F.binary_cross_entropy(logit, label, reduction='none')
    utility_loss = torch.mean(utility_loss, dim=0)
    # balance the loss by its magnitude
    f ,u = utility_loss/(fair_loss+utility_loss), fair_loss/(fair_loss+utility_loss)
    f, u = f.detach(), u.detach()

    if composition == 'linear':
        loss = fair_loss*f+utility_loss*u*loss_mask*coef
    elif composition == 'exclusive':
        loss = fair_loss*torch.sub(1, loss_mask) + utility_loss*loss_mask*coef
    else:
        assert False, f'Unsupport loss composition'
    
    return torch.mean(loss)

def loss_EOpp_perturbed_adaptive(logit, label, sens, coef):
    # perturbed loss that reduce the difference in TPR (for equality of opportunity)
    def EOpp_perturbed_loss(x, label=label, sens=sens):
        # get prediction from logit
        pred = torch.where(x> 0.5, 1, 0)
        # dupe the label to have the same shape as x
        label_duped = label.repeat(x.shape[0], 1, 1)
        # regroup prediction and label, 
        # beware of the new dimension added for the perturbed optimizers
        g1_pred, g2_pred = regroup_perturbed(pred, sens)
        g1_label, g2_label = regroup_perturbed(label_duped, sens)
        g1_TPR, g2_TPR = calc_perturbed_tpr(g1_pred, g1_label), calc_perturbed_tpr(g2_pred, g2_label)
        loss = torch.abs(g1_TPR-g2_TPR)
        return loss
    # Turns a function into a differentiable one via perturbations
    pret_tpr = perturbed(EOpp_perturbed_loss,
                         num_samples=10000,
                         sigma=0.5,
                         noise='gumbel',
                         batched=False)
    fair_loss = pret_tpr(logit)
    # utility loss
    utility_loss = F.binary_cross_entropy(logit, label, reduction='none')
    utility_loss = torch.mean(utility_loss, dim=0)

    loss = fair_loss + coef*utility_loss
    return torch.mean(loss)

def loss_EO_perturbed_adaptive(logit, label, sens, coef):
    # perturbed loss that reduce the difference in TPR (for equalized odds)
    def EO_perturbed_loss(x, label=label, sens=sens):
        # get prediction from logit
        pred = torch.where(x> 0.5, 1, 0)
        # dupe the label to have the same shape as x
        label_duped = label.repeat(x.shape[0], 1, 1)
        # regroup prediction and label, 
        # beware of the new dimension added for the perturbed optimizers
        g1_pred, g2_pred = regroup_perturbed(pred, sens)
        g1_label, g2_label = regroup_perturbed(label_duped, sens)
        g1_TPR, g2_TPR = calc_perturbed_tpr(g1_pred, g1_label), calc_perturbed_tpr(g2_pred, g2_label)
        g1_TNR, g2_TNR = calc_perturbed_tnr(g1_pred, g1_label), calc_perturbed_tnr(g2_pred, g2_label)
        loss = torch.abs(g1_TPR-g2_TPR) + torch.abs(g1_TNR-g2_TNR)
        return loss
    # Turns a function into a differentiable one via perturbations
    pret_tpr_tnr = perturbed(EO_perturbed_loss,
                         num_samples=10000,
                         sigma=0.5,
                         noise='gumbel',
                         batched=False)
    fair_loss = pret_tpr_tnr(logit)
    # utility loss
    utility_loss = F.binary_cross_entropy(logit, label, reduction='none')
    utility_loss = torch.mean(utility_loss, dim=0)

    loss = fair_loss + coef*utility_loss
    return torch.mean(loss)

@dataclass
class EO_adaptive:
    """Loss function by batch"""

    
