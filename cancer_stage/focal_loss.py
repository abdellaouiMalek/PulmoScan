"""
Focal Loss Implementation for Class Imbalance

This module provides a PyTorch implementation of Focal Loss for addressing class imbalance
in classification tasks.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """
    Focal Loss for addressing class imbalance
    
    Focal Loss was proposed in the paper "Focal Loss for Dense Object Detection"
    by Lin et al. (https://arxiv.org/abs/1708.02002)
    """
    
    def __init__(self, weight=None, gamma=2.0, reduction='mean'):
        """
        Args:
            weight: Class weights for weighted cross-entropy
            gamma: Focusing parameter (higher gamma focuses more on hard examples)
            reduction: Reduction method ('mean', 'sum', or 'none')
        """
        super(FocalLoss, self).__init__()
        self.weight = weight
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, input, target):
        """
        Args:
            input: Model predictions (logits)
            target: Ground truth labels
            
        Returns:
            Focal loss
        """
        # Get cross entropy loss
        ce_loss = F.cross_entropy(
            input, target, weight=self.weight, 
            reduction='none'
        )
        
        # Get probabilities
        pt = torch.exp(-ce_loss)
        
        # Calculate focal loss
        focal_loss = (1 - pt) ** self.gamma * ce_loss
        
        # Apply reduction
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class CombinedLoss(nn.Module):
    """
    Combined loss function: weighted cross-entropy + focal loss
    """
    
    def __init__(self, weight=None, gamma=2.0, alpha=0.5, reduction='mean'):
        """
        Args:
            weight: Class weights for weighted cross-entropy
            gamma: Focusing parameter for focal loss
            alpha: Weight for balancing cross-entropy and focal loss
            reduction: Reduction method ('mean', 'sum', or 'none')
        """
        super(CombinedLoss, self).__init__()
        self.weight = weight
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction
        
        # Create loss functions
        self.ce_loss = nn.CrossEntropyLoss(weight=weight, reduction=reduction)
        self.focal_loss = FocalLoss(weight=weight, gamma=gamma, reduction=reduction)
    
    def forward(self, input, target):
        """
        Args:
            input: Model predictions (logits)
            target: Ground truth labels
            
        Returns:
            Combined loss
        """
        # Calculate cross-entropy loss
        ce = self.ce_loss(input, target)
        
        # Calculate focal loss
        focal = self.focal_loss(input, target)
        
        # Combine losses
        return self.alpha * ce + (1 - self.alpha) * focal
