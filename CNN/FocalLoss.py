import torch
import torch.nn as nn
from torch.nn import functional as F


class FocalLoss1(nn.Module):
    def __init__(self, alpha=0.25, gamma=2, reduction='mean', ignore_lb=255):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.ignore_lb = ignore_lb

    def forward(self, logits, label):
        """[summary]
        
        Args:
            logits ([type]): tensor of shape (N, C, H, W)
            label ([type]): tensor of shape(N, H, W)
        
        Returns:
            [type]: [description]
        """

        # overcome ignored label
        ignore = label.data.cpu() == self.ignore_lb
        n_valid = (ignore == 0).sum()
        label[ignore] = 0

        ignore = ignore.nonzero()
        _, M = ignore.size()
        a, *b = ignore.chunk(M, dim=1)
        mask = torch.ones_like(logits)
        mask[[a, torch.arange(mask.size(1)), *b]] = 0

        # compute loss
        probs = torch.sigmoid(logits)
        lb_one_hot = logits.data.clone().zero_().scatter_(1, label.unsqueeze(1), 1)
        pt = torch.where(lb_one_hot == 1, probs, 1 - probs)
        alpha = self.alpha * lb_one_hot + (1 - self.alpha) * (1 - lb_one_hot)
        loss = -alpha * ((1 - pt)**self.gamma) * torch.log(pt + 1e-12)
        loss[mask == 0] = 0
        if self.reduction == 'mean':
            loss = loss.sum(dim=1).sum() / n_valid
        return loss


class FocalLoss(nn.Module):
    """https://www.kaggle.com/c/human-protein-atlas-image-classification/discussion/78109"""

    def __init__(self, gamma=2):
        super().__init__()
        self.gamma = gamma

    def forward(self, logit, target):
        target = target.float()
        max_val = (-logit).clamp(min=0)
        loss = logit - logit * target + max_val + ((-max_val).exp() + (-logit - max_val).exp()).log()
        invprobs = F.logsigmoid(-logit * (target * 2.0 - 1.0))
        loss = (invprobs * self.gamma).exp() * loss
        loss = loss.sum(dim=1) if len(loss.size()) == 2 else loss
        return loss.mean()
