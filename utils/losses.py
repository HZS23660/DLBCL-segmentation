import torch
from torch.nn import functional as F
import numpy as np
import torch.nn as nn
from torch.autograd import Variable
from torch.nn.modules.loss import _Loss


def dice_loss(score, target):
    target = target.float()
    smooth = 1e-5
    intersect = torch.sum(score * target)
    y_sum = torch.sum(target * target)
    z_sum = torch.sum(score * score)
    loss = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
    loss = 1 - loss
    return loss


def log_cosh_dice_loss(y_true, y_pred):
    x = dice_loss(y_true, y_pred)
    return torch.log((torch.exp(x) + torch.exp(-x)) / 2.0)


def dice_loss1(score, target):
    target = target.float()
    smooth = 1e-5
    intersect = torch.sum(score * target)
    y_sum = torch.sum(target)
    z_sum = torch.sum(score)
    loss = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
    loss = 1 - loss
    return loss


def entropy_loss(p, C=2):
    # p N*C*W*H*D
    y1 = -1*torch.sum(p*torch.log(p+1e-6), dim=1) / \
        torch.tensor(np.log(C)).cuda()
    ent = torch.mean(y1)

    return ent



class BCELosswithLogits(nn.Module):
    def __init__(self, pos_weight=0.999, reduction='mean', n_classes = 2):
        super(BCELosswithLogits, self).__init__()
        self.pos_weight = pos_weight
        self.reduction = reduction
        self.n_classes = n_classes

    def _one_hot_encoder(self, input_tensor):
        tensor_list = []
        for i in range(self.n_classes):
            temp_prob = input_tensor == i * torch.ones_like(input_tensor)
            tensor_list.append(temp_prob)
        output_tensor = torch.cat(tensor_list, dim=1)
        return output_tensor.float()

    def forward(self, logits, target):

        logits = F.sigmoid(logits)
        target = self._one_hot_encoder(target)
        weight = [1 - self.pos_weight, self.pos_weight]
        assert logits.size() == target.size(), 'predict & target shape do not match'

        loss = - weight[0] * target[:, 0] * torch.log(logits[:, 0]) -\
               weight[1] * target[:, 1] * torch.log(logits[:, 1])

        # loss = - self.pos_weight * target * torch.log(logits) - \
        #        (1 - self.pos_weight) * (1 - target) * torch.log(1 - logits)

        if self.reduction == 'mean':
            loss = loss.mean()
        elif self.reduction == 'sum':
            loss = loss.sum()
        return loss


class PCELoss(nn.Module):
    def __init__(self, pos_weight=0.5, reduction='mean'):
        super(PCELoss, self).__init__()
        self.pos_weight = pos_weight
        self.reduction = reduction

    def forward(self, logits, target):
        # logits: [N, *], target: [N, *]
        # logits = F.softmax(logits, dim=1)
        # print(target.shape, logits[:, 1].shape)
        # weight = [1 - self.pos_weight, self.pos_weight]
        loss = - target * torch.log(logits[:, 1]) #+\
               # target * torch.log(logits[:, 0])
        num = torch.nonzero(target).size(0)
        # print(num)

        if self.reduction == 'mean':
            loss = loss.sum() / (num + 1e-7)
        elif self.reduction == 'sum':
            loss = loss.sum()
        return loss



class N_PCELoss(nn.Module):
    def __init__(self, pos_weight=0.5, reduction='mean'):
        super(N_PCELoss, self).__init__()
        self.pos_weight = pos_weight
        self.reduction = reduction

    def forward(self, seg_pre, target):

        target = target.to(dtype=torch.float32)
        # print(target.dtype)
        bsize, ch_sz, sp_sd1, sp_sd2, sp_sd3 = seg_pre.size()[:]
        maxpool = nn.AdaptiveMaxPool3d((sp_sd1, sp_sd2, sp_sd3))
        target_new = maxpool(target)
        target_new = target_new.to(dtype=torch.int64)

        loss = - target_new * torch.log(seg_pre[:, 1])
        num = torch.nonzero(target).size(0)
        # print(num)

        if self.reduction == 'mean':
            loss = loss.sum() / (num + 1e-7)
        elif self.reduction == 'sum':
            loss = loss.sum()
        return loss



class CrossEntropyLoss(torch.nn.Module):
    def __init__(self, reduction='mean'):
        super(CrossEntropyLoss, self).__init__()
        self.reduction = reduction
    def forward(self, logits, target):
        # logits: [N, C, H, W], target: [N, H, W]
        # loss = sum(-y_i * log(c_i))
        if logits.dim() > 2:
            logits = logits.view(logits.size(0), logits.size(1), -1)  # [N, C, HW]
            logits = logits.transpose(1, 2)   # [N, HW, C]
            logits = logits.contiguous().view(-1, logits.size(2))    # [NHW, C]
        target = target.view(-1, 1)    # [NHW，1]

        logits = F.log_softmax(logits, 1)
        logits = logits.gather(1, target)   # [NHW, 1]
        loss = -1 * logits

        if self.reduction == 'mean':
            loss = loss.mean()
        elif self.reduction == 'sum':
            loss = loss.sum()
        return loss



class BCEFocalLosswithLogits(nn.Module):
    def __init__(self, gamma=0.2, alpha=0.6, reduction='mean'):
        super(BCEFocalLosswithLogits, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, logits, target):
        # logits: [N, H, W], target: [N, H, W]
        logits = F.sigmoid(logits)
        alpha = self.alpha
        gamma = self.gamma
        loss = - alpha * (1 - logits) ** gamma * target * torch.log(logits) - \
               (1 - alpha) * logits ** gamma * (1 - target) * torch.log(1 - logits)
        if self.reduction == 'mean':
            loss = loss.mean()
        elif self.reduction == 'sum':
            loss = loss.sum()
        return loss


class CrossEntropyFocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=0.2, reduction='mean'):
        super(CrossEntropyFocalLoss, self).__init__()
        self.reduction = reduction
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits, target):
        # logits: [N, C, H, W], target: [N, H, W]
        # loss = sum(-y_i * log(c_i))
        if logits.dim() > 2:
            logits = logits.view(logits.size(0), logits.size(1), -1)  # [N, C, HW]
            logits = logits.transpose(1, 2)  # [N, HW, C]
            logits = logits.contiguous().view(-1, logits.size(2))  # [NHW, C]
        target = target.view(-1, 1)  # [NHW，1]

        pt = F.softmax(logits, 1)
        pt = pt.gather(1, target).view(-1)  # [NHW]
        log_gt = torch.log(pt)

        if self.alpha is not None:
            # alpha: [C]
            alpha = self.alpha.gather(0, target.view(-1))  # [NHW]
            log_gt = log_gt * alpha

        loss = -1 * (1 - pt) ** self.gamma * log_gt

        if self.reduction == 'mean':
            loss = loss.mean()
        elif self.reduction == 'sum':
            loss = loss.sum()
        return loss



def softmax_dice_loss(input_logits, target_logits):
    """Takes softmax on both sides and returns MSE loss

    Note:
    - Returns the sum over all examples. Divide by the batch size afterwards
      if you want the mean.
    - Sends gradients to inputs but not the targets.
    """
    assert input_logits.size() == target_logits.size()
    input_softmax = F.softmax(input_logits, dim=1)
    target_softmax = F.softmax(target_logits, dim=1)
    n = input_logits.shape[1]
    dice = 0
    for i in range(0, n):
        dice += dice_loss1(input_softmax[:, i], target_softmax[:, i])
    mean_dice = dice / n

    return mean_dice


def entropy_loss_map(p, C=2):
    ent = -1*torch.sum(p * torch.log(p + 1e-6), dim=1,
                       keepdim=True)/torch.tensor(np.log(C)).cuda()
    return ent


def softmax_mse_loss(input_logits, target_logits, sigmoid=False):
    """Takes softmax on both sides and returns MSE loss

    Note:
    - Returns the sum over all examples. Divide by the batch size afterwards
      if you want the mean.
    - Sends gradients to inputs but not the targets.
    """
    assert input_logits.size() == target_logits.size()
    if sigmoid:
        input_softmax = torch.sigmoid(input_logits)
        target_softmax = torch.sigmoid(target_logits)
    else:
        input_softmax = F.softmax(input_logits, dim=1)
        target_softmax = F.softmax(target_logits, dim=1)

    mse_loss = (input_softmax-target_softmax)**2
    return mse_loss


def softmax_kl_loss(input_logits, target_logits, sigmoid=False):
    """Takes softmax on both sides and returns KL divergence

    Note:
    - Returns the sum over all examples. Divide by the batch size afterwards
      if you want the mean.
    - Sends gradients to inputs but not the targets.
    """
    assert input_logits.size() == target_logits.size()
    if sigmoid:
        input_log_softmax = torch.log(torch.sigmoid(input_logits))
        target_softmax = torch.sigmoid(target_logits)
    else:
        input_log_softmax = F.log_softmax(input_logits, dim=1)
        target_softmax = F.softmax(target_logits, dim=1)

    # return F.kl_div(input_log_softmax, target_softmax)
    kl_div = F.kl_div(input_log_softmax, target_softmax, reduction='mean')
    # mean_kl_div = torch.mean(0.2*kl_div[:,0,...]+0.8*kl_div[:,1,...])
    return kl_div


def symmetric_mse_loss(input1, input2):
    """Like F.mse_loss but sends gradients to both directions

    Note:
    - Returns the sum over all examples. Divide by the batch size afterwards
      if you want the mean.
    - Sends gradients to both input1 and input2.
    """
    assert input1.size() == input2.size()
    return torch.mean((input1 - input2)**2)


def tversky_index(y_true, y_pred):
    smooth = 1
    true = torch.sum(y_true * y_pred)
    false_neg = torch.sum(y_true * (1 - y_pred))
    false_pos = torch.sum((1 - y_true) * y_pred)
    alpha = 0.7
    return (true + smooth) / (true + alpha * false_neg + (
                1 - alpha) * false_pos + smooth)


def tversky_loss(y_true, y_pred):
    return 1 - tversky_index(y_true, y_pred)



class FocalLoss(nn.Module):
    def __init__(self, gamma = 2, alpha = 0.25, size_average=True):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        if isinstance(alpha, (float, int)):
            self.alpha = torch.Tensor([alpha, 1-alpha])
        if isinstance(alpha, list):
            self.alpha = torch.Tensor(alpha)
        self.size_average = size_average

    def forward(self, input, target):
        if input.dim() > 2:
            # N,C,H,W => N,C,H*W
            input = input.view(input.size(0), input.size(1), -1)
            input = input.transpose(1, 2)    # N,C,H*W => N,H*W,C
            input = input.contiguous().view(-1, input.size(2))   # N,H*W,C => N*H*W,C
        target = target.view(-1, 1)

        logpt = F.log_softmax(input, dim=1)
        logpt = logpt.gather(1, target)
        logpt = logpt.view(-1)
        pt = Variable(logpt.data.exp())

        if self.alpha is not None:
            if self.alpha.type() != input.data.type():
                self.alpha = self.alpha.type_as(input.data)
            at = self.alpha.gather(0, target.data.view(-1))
            logpt = logpt * Variable(at)

        loss = -1 * (1-pt)**self.gamma * logpt
        if self.size_average:
            return loss.mean()
        else:
            return loss.sum()


class DiceLoss(nn.Module):
    def __init__(self, n_classes=2):
        super(DiceLoss, self).__init__()
        self.n_classes = n_classes

    def _one_hot_encoder(self, input_tensor):
        tensor_list = []
        for i in range(self.n_classes):
            temp_prob = input_tensor == i * torch.ones_like(input_tensor)
            tensor_list.append(temp_prob)
        output_tensor = torch.cat(tensor_list, dim=1)
        return output_tensor.float()

    def _dice_loss(self, score, target):
        target = target.float()
        smooth = 1e-5
        intersect = torch.sum(score * target)
        y_sum = torch.sum(target * target)
        z_sum = torch.sum(score * score)
        loss = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
        loss = 1 - loss
        return loss

    def forward(self, inputs, target, weight=None, softmax=False):
        if softmax:
            inputs = torch.softmax(inputs, dim=1)
        target = self._one_hot_encoder(target)
        if weight is None:
            weight = [1] * self.n_classes
        # print(inputs.size(), target.size())
        assert inputs.size() == target.size(), 'predict & target shape do not match'
        # class_wise_dice = []
        loss = 0.0
        for i in range(0, self.n_classes):
            dice = self._dice_loss(inputs[:, i], target[:, i])
            #dice = log_cosh_dice_loss(target[:, i], inputs[:, i])
            #dice = tversky_loss(target[:, i], inputs[:, i])
            #class_wise_dice.append(1.0 - dice.item())
            loss += dice * weight[i]
        return loss / self.n_classes


def entropy_minmization(p):
    y1 = -1*torch.sum(p*torch.log(p+1e-6), dim=1)
    ent = torch.mean(y1)

    return ent


def entropy_map(p):
    ent_map = -1*torch.sum(p * torch.log(p + 1e-6), dim=1,
                           keepdim=True)
    return ent_map



class Cosine_Similarity_Loss(nn.Module):
    def __init__(self, n_classes=2, alpha = 0.5):
        super(Cosine_Similarity_Loss, self).__init__()
        self.n_classes = n_classes
        self.alpha = alpha

    def _one_hot_encoder(self, input_tensor):
        tensor_list = []
        for i in range(self.n_classes):
            temp_prob = input_tensor == i * torch.ones_like(input_tensor)
            tensor_list.append(temp_prob)
        output_tensor = torch.cat(tensor_list, dim=1)
        return output_tensor.float()

    def _cosine_similarity(self, feature, target):
        # 输入的feature大小为b,c,d,h,w   target大小为b,1,d1,h1,w1
        cosine_eps = 1e-7
        bsize, ch_sz, sp_sd1, sp_sd2, sp_sd3 = feature.size()[:]
        maxpool = nn.AdaptiveMaxPool3d((sp_sd1, sp_sd2, sp_sd3))
        target = maxpool(target.float())
        # target重采样为与feature一样大
        # target_new = F.interpolate(target_new, size=(sp_sd, sp_sh, sp_sw), \
        #                            mode='trilinear', align_corners=True)
        target_new = self._one_hot_encoder(target)
        feat_list = []
        for i in range(0, self.n_classes):
            if torch.sum(target_new[:, i, :, :, :]) != 0:
                pass
            else:
                # print("目标特征的种类不是两类，请重新确定待分割类别{}".format(i))
                pass
            tmp_supp_feat = feature * target_new[:, i, :, :, :].unsqueeze(1)
            feat_list.append(tmp_supp_feat)


        tmp_query, tmp_supp = feat_list[0], feat_list[1]

        tmp_query = tmp_query.contiguous().view(bsize, ch_sz, -1)
        tmp_query_norm = torch.norm(tmp_query, 2, 1, True)
        # tmp_query_T = tmp_query.contiguous().permute(0, 2, 1)
        # tmp_query_T_norm = torch.norm(tmp_query_T, 2, 2, True)

        tmp_supp = tmp_supp.contiguous().view(bsize, ch_sz, -1)
        tmp_supp_norm = torch.norm(tmp_supp, 2, 1, True)
        tmp_supp_T = tmp_supp.contiguous().permute(0, 2, 1)
        tmp_supp_T_norm = torch.norm(tmp_supp_T, 2, 2, True)

        cos_similarity_diff = torch.bmm(tmp_supp_T, tmp_query) / (torch.bmm(tmp_supp_T_norm, tmp_query_norm) + cosine_eps)
        cos_similarity_same = torch.bmm(tmp_supp_T, tmp_supp) / (torch.bmm(tmp_supp_T_norm, tmp_supp_norm) + cosine_eps) #\
                                # + torch.bmm(tmp_query_T, tmp_query) / (torch.bmm(tmp_query_T_norm, tmp_query_norm) + cosine_eps)

        return cos_similarity_diff, cos_similarity_same


    def forward(self, inputs, target):
        cosine_eps = 1e-7
        cos_similarity_diff, cos_similarity_same = self._cosine_similarity(inputs, target)
        num_nonzero_diff = torch.nonzero(cos_similarity_diff).size(0)
        cos_similarity_diff = torch.sum(cos_similarity_diff) / (num_nonzero_diff + cosine_eps)
        num_nonzero_same = torch.nonzero(cos_similarity_same).size(0)
        cos_similarity_same = torch.sum(cos_similarity_same) / (num_nonzero_same + cosine_eps)

        cos_similarity_loss = self.alpha * cos_similarity_diff + (1 - self.alpha) * (1 - cos_similarity_same)

        return cos_similarity_loss



class Re_SS_Cos_Loss(nn.Module):
    def __init__(self, n_classes=2, alpha = 0.5):
        super(Re_SS_Cos_Loss, self).__init__()
        self.n_classes = n_classes
        self.alpha = alpha

    def _cosine_similarity(self, feature, target):
        # 输入的feature大小为b,c,d,h,w   target大小为b,2,d1,h1,w1/ output
        cosine_eps = 1e-7
        bsize, ch_sz, sp_sd1, sp_sd2, sp_sd3 = feature.size()[:]
        maxpool = nn.AdaptiveMaxPool3d((sp_sd1, sp_sd2, sp_sd3))
        target_new = maxpool(target)
        target_pos = target_new[:, 1, :, :, :].unsqueeze(1)
        target_neg = target_new[:, 0, :, :, :].unsqueeze(1)
        # target重采样为与feature一样大
        # target_new = F.interpolate(target_new, size=(sp_sd, sp_sh, sp_sw), \
        #                            mode='trilinear', align_corners=True)

        tmp_supp = feature * target_pos
        tmp_query = feature * target_neg

        tmp_query = tmp_query.contiguous().view(bsize, ch_sz, -1)
        tmp_query_norm = torch.norm(tmp_query, 2, 1, True)
        #------------
        tmp_query_T = tmp_query.contiguous().permute(0, 2, 1)
        tmp_query_T_norm = torch.norm(tmp_query_T, 2, 2, True)
        #-------------
        tmp_supp = tmp_supp.contiguous().view(bsize, ch_sz, -1)
        tmp_supp_norm = torch.norm(tmp_supp, 2, 1, True)
        tmp_supp_T = tmp_supp.contiguous().permute(0, 2, 1)
        tmp_supp_T_norm = torch.norm(tmp_supp_T, 2, 2, True)

        cos_similarity_diff = torch.bmm(tmp_supp_T, tmp_query) / (torch.bmm(tmp_supp_T_norm, tmp_query_norm) + cosine_eps)
        cos_similarity_same = torch.bmm(tmp_supp_T, tmp_supp) / (torch.bmm(tmp_supp_T_norm, tmp_supp_norm) + cosine_eps) \
                                 + torch.bmm(tmp_query_T, tmp_query) / (torch.bmm(tmp_query_T_norm, tmp_query_norm) + cosine_eps)

        return cos_similarity_diff, cos_similarity_same


    def forward(self, inputs, target):
        cosine_eps = 1e-7
        cos_similarity_diff, cos_similarity_same = self._cosine_similarity(inputs, target)
        num_nonzero_diff = torch.nonzero(cos_similarity_diff).size(0)
        cos_similarity_diff = torch.sum(cos_similarity_diff) / (num_nonzero_diff + cosine_eps)
        num_nonzero_same = torch.nonzero(cos_similarity_same).size(0)
        cos_similarity_same = torch.sum(cos_similarity_same) / (num_nonzero_same + cosine_eps)

        cos_similarity_loss = self.alpha * cos_similarity_diff + (1 - self.alpha) * (1 - cos_similarity_same)

        return cos_similarity_loss



class Re_WS_Cos_Loss(nn.Module):
    def __init__(self, n_classes=2):
        super(Re_WS_Cos_Loss, self).__init__()
        self.n_classes = n_classes

    def _cosine_similarity(self, prior_feature, prior_label, feature, target):
        # 输入的feature大小为b,c,d,h,w  prior_label大小b,1,d1,h1,w1  target大小为b,2,d1,h1,w1/ output
        cosine_eps = 1e-7
        bsize, ch_sz, sp_sd1, sp_sd2, sp_sd3 = prior_feature.size()[:]
        maxpool = nn.AdaptiveMaxPool3d((sp_sd1, sp_sd2, sp_sd3))
        prior_label_new = maxpool(prior_label.float())
        target_new = maxpool(target)

        tmp_supp = prior_feature * prior_label_new
        tmp_query = feature * target_new[:, 1, :, :, :].unsqueeze(1)
        # target_new 可以

        tmp_query = tmp_query.contiguous().view(bsize, ch_sz, -1)
        tmp_query_norm = torch.norm(tmp_query, 2, 1, True)

        tmp_supp = tmp_supp.contiguous().view(bsize, ch_sz, -1)
        # tmp_supp_norm = torch.norm(tmp_supp, 2, 1, True)
        tmp_supp_T = tmp_supp.contiguous().permute(0, 2, 1)
        tmp_supp_T_norm = torch.norm(tmp_supp_T, 2, 2, True)

        cos_similarity_same = torch.bmm(tmp_supp_T, tmp_query) / (torch.bmm(tmp_supp_T_norm, tmp_query_norm) + cosine_eps)

        return cos_similarity_same


    def forward(self, prior_feature, prior_label, feature, target):
        cosine_eps = 1e-7
        cos_similarity_same = self._cosine_similarity(prior_feature, prior_label, feature, target)
        num_nonzero_same = torch.nonzero(cos_similarity_same).size(0)
        cos_similarity_same = torch.sum(cos_similarity_same) / (num_nonzero_same + cosine_eps)

        cos_similarity_loss = 1 - cos_similarity_same

        return cos_similarity_loss




class CE_Cosine_Similarity_Loss(nn.Module):

    def __init__(self, n_classes=2):
        super(CE_Cosine_Similarity_Loss, self).__init__()
        self.n_classes = n_classes
        # self.pool = nn.AdaptiveMaxPool3d()
        # self.pool = nn.AvgPool3d(kernel_size=4, stride=4)

    def forward(self, inputs, target):

        # target = self.pool(target.float())
        # target = self.pool(target.float())
        target1 = target.view(target.size(0), -1)
        criteria = nn.CrossEntropyLoss()
        cos_similarity_loss = criteria(inputs, target1.long())

        return cos_similarity_loss



class N_CrossEntropyLoss(nn.Module):
    def __init__(self, n_classes=2):
        super(N_CrossEntropyLoss, self).__init__()
        self.n_classes = n_classes
        self.criter = nn.CrossEntropyLoss()

    def n_CrossEntropyLoss(self, seg_pre, target):
        # 输入的feature大小为b,c,d,h,w  prior_label大小b,1,d1,h1,w1  target大小为b,2,d1,h1,w1/ output

        target = target.to(dtype=torch.float32)
        # print(target.dtype)
        bsize, ch_sz, sp_sd1, sp_sd2, sp_sd3 = seg_pre.size()[:]
        maxpool = nn.AdaptiveMaxPool3d((sp_sd1, sp_sd2, sp_sd3))
        target_new = maxpool(target)
        target_new = target_new.to(dtype=torch.int64)
        # print(seg_pre.shape, target_new.shape)

        loss = self.criter(seg_pre, target_new)
        return loss

    def forward(self, seg_pre, target):
        loss = self.n_CrossEntropyLoss(seg_pre, target)
        return loss


class N_MSELoss(nn.Module):
    def __init__(self, n_classes=2):
        super(N_MSELoss, self).__init__()
        self.n_classes = n_classes
        self.criter = nn.MSELoss()

    def n_MSELoss(self, seg_pre, target):
        # 输入的feature大小为b,c,d,h,w  prior_label大小b,1,d1,h1,w1  target大小为b,2,d1,h1,w1/ output

        bsize, ch_sz, sp_sd1, sp_sd2, sp_sd3 = seg_pre.size()[:]
        maxpool = nn.AdaptiveMaxPool3d((sp_sd1, sp_sd2, sp_sd3))
        target_new = maxpool(target)
        loss = self.criter(seg_pre, target_new)
        return loss

    def forward(self, seg_pre, target):
        loss = self.n_MSELoss(seg_pre, target)
        return loss

#---------------------------------------------------------------------------


class SoftDiceLoss(_Loss):
    '''
    Soft_Dice = 2*|dot(A, B)| / (|dot(A, A)| + |dot(B, B)| + eps)
    eps is a small constant to avoid zero division,
    '''

    def __init__(self, *args, **kwargs):
        super(SoftDiceLoss, self).__init__()

    def forward(self, y_pred, y_true, eps=1e-8):
        intersection = torch.sum(torch.mul(y_pred, y_true))
        union = torch.sum(torch.mul(y_pred, y_pred)) + torch.sum(torch.mul(y_true, y_true)) + eps

        dice = 2 * intersection / union
        dice_loss = 1 - dice

        return dice_loss


class CustomKLLoss(_Loss):
    '''
    KL_Loss = (|dot(mean , mean)| + |dot(std, std)| - |log(dot(std, std))| - 1) / N
    N is the total number of image voxels
    '''

    def __init__(self, *args, **kwargs):
        super(CustomKLLoss, self).__init__()

    def forward(self, mean, std):
        return torch.mean(torch.mul(mean, mean)) + torch.mean(torch.mul(std, std)) - torch.mean(
            torch.log(torch.mul(std, std))) - 1


class CombinedL2KLLoss(_Loss):
    '''
    Combined_loss = Dice_loss + k1 * L2_loss + k2 * KL_loss
    As default: k1=0.1, k2=0.1
    '''

    def __init__(self, *args, **kwargs):
        super(CombinedL2KLLoss, self).__init__()
        self.l2_loss = nn.MSELoss()
        self.kl_loss = CustomKLLoss()

    def forward(self, y_pred, y_true, y_mid):

        est_mean, est_std = (y_mid[:, :128], y_mid[:, 128:])
        vae_pred, vae_truth = y_pred, y_true
        l2_loss = self.l2_loss(vae_pred, vae_truth)
        kl_div = self.kl_loss(est_mean, est_std)
        combined_loss = l2_loss + kl_div

        return combined_loss



class CombinedL1KLLoss(_Loss):
    '''
    Combined_loss = Dice_loss + k1 * L2_loss + k2 * KL_loss
    As default: k1=0.1, k2=0.1
    '''

    def __init__(self, *args, **kwargs):
        super(CombinedL1KLLoss, self).__init__()
        self.l1_loss = nn.SmoothL1Loss()
        self.kl_loss = CustomKLLoss()

    def forward(self, y_pred, y_true, y_mid):

        est_mean, est_std = (y_mid[:, :128], y_mid[:, 128:])
        vae_pred, vae_truth = y_pred, y_true
        l2_loss = self.l1_loss(vae_pred, vae_truth)
        kl_div = self.kl_loss(est_mean, est_std)
        combined_loss = l2_loss + kl_div

        return combined_loss



class CombinedLoss(_Loss):
    '''
    Combined_loss = Dice_loss + k1 * L2_loss + k2 * KL_loss
    As default: k1=0.1, k2=0.1
    '''

    def __init__(self, k1=0.5, k2=0.5):
        super(CombinedLoss, self).__init__()
        self.k1 = k1
        self.k2 = k2
        # self.dice_loss = SoftDiceLoss()
        self.l2_loss = nn.MSELoss()
        self.kl_loss = CustomKLLoss()

        self.ce_loss = nn.CrossEntropyLoss()
        self.dice_loss = DiceLoss(2)

    def forward(self, y_pred, y_true, y_mid):
        est_mean, est_std = (y_mid[:, :128], y_mid[:, 128:])
        seg_pred, seg_truth = (y_pred[:, :2, :, :, :], y_true[:, 0, :, :, :].long())
        vae_pred, vae_truth = (y_pred[:, 2:, :, :, :], y_true[:, 1:, :, :, :])
        # vae_pred, vae_truth = (y_pred[:, :, :, :, :], y_true[:, :, :, :, :])
        seg_pred_soft = torch.softmax(seg_pred, dim=1)

        # print(seg_pred.size(), seg_truth.size())
        # print(vae_pred.size(), vae_truth.size())
        dice_loss = self.ce_loss(seg_pred, seg_truth) + self.dice_loss(seg_pred_soft, seg_truth.unsqueeze(1))

        l2_loss = self.l2_loss(vae_pred, vae_truth)
        kl_div = self.kl_loss(est_mean, est_std)
        combined_loss = dice_loss + self.k1 * l2_loss + self.k2 * kl_div
        # combined_loss = self.k1 * l2_loss + self.k2 * kl_div
        # print("dice_loss:%.4f, L2_loss:%.4f, KL_div:%.4f, combined_loss:%.4f"%(dice_loss,l2_loss,kl_div,combined_loss))

        return combined_loss, dice_loss, l2_loss, kl_div