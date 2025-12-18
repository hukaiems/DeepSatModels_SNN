import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0, ignore_index=255):
        """
        Focal Loss for Semantic Segmentation
        Args:
            alpha (float or list): Class weights.
                                    If list, weights for each classes.
            gamma (float): Focusing parameter. (Gamma=2.0 is standard)
            ignore_index (int): Index to ignore (e.g., void class)
        """
        super(FocalLoss, self).__init__()
        self.gamma=gamma
        self.ignore_index = ignore_index

        if alpha is not None:
            if isinstance(alpha, list):
                self.alpha = torch.tensor(alpha)
            else:
                self.alpha = alpha
        else:
            self.alpha = None
    
    def forward(self, inputs, targets):
        # 1 move alpha to device
        if self.alpha is not None and self.alpha.device != inputs.device:
            self.alpha = self.alpha.to(inputs_device)
        
        # 2. Compute Cross Entropy Loss
        # We use log_softmax + nll_loss for stability
        log_pt = F.log_softmax(inputs, dim=1)
        # negative log likelihood
        ce_loss = F.nll_loss(log_pt, targets, weight=self.alpha,
                             ignore_index=self.ignore_index, reduction='none')
        
        # 3 Compute probabilities 
        pt = torch.exp(-ce_loss)

        # 4 Compute Focal term (1-pt)^gamma
        focal_term = (1- pt) ** self.gamma
        
        # 5 FInal loss
        loss = focal_term * ce_loss

        return loss.mean()
