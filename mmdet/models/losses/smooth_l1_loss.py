import torch
import torch.nn as nn

from ..registry import LOSSES
from .utils import weighted_loss


@weighted_loss
def smooth_l1_loss(pred, target,weight, beta=1.0):
    assert beta > 0
    assert pred.size() == target.size() and target.numel() > 0
    diff = torch.abs(pred - target)
    loss = torch.where(diff < beta, 0.5 * diff * diff / beta,
                       diff - 0.5 * beta)
    loss = torch.mean(loss,dim = 1)
    loss = loss * weight
    return loss


@LOSSES.register_module
class SmoothL1Loss(nn.Module):

    def __init__(self, beta=1.0, reduction='mean', loss_weight=1.0):
        super(SmoothL1Loss, self).__init__()
        self.beta = beta
        self.reduction = reduction
        self.loss_weight = loss_weight

    def forward(self,
                pred,
                target,
                weight=None,
                avg_factor=None,
                reduction_override=None,
                **kwargs):

	    assert pred.size() == target.size() and target.numel() > 0
	    diff = torch.abs(pred - target)
	    loss = torch.where(diff < self.beta, 0.5 * diff * diff / self.beta,
		               diff - 0.5 * self.beta)
	    loss = torch.sum(loss,dim = 1)
	    loss = loss * weight / avg_factor
	    return loss