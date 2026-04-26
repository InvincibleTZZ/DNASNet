"""
分离损失函数
用于神经架构搜索中的辅助损失
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvSeparateLoss(nn.Module):
    """
    卷积分离损失
    
    主损失 + 辅助损失（基于架构参数的L2正则）
    """
    def __init__(self, criterion, weight=1.0):
        """
        Args:
            criterion: 主损失函数（如CrossEntropy）
            weight: 辅助损失权重
        """
        super(ConvSeparateLoss, self).__init__()
        self.criterion = criterion
        self.weight = weight
    
    def forward(self, logits, target, aux_input=None):
        """
        Args:
            logits: 模型输出
            target: 目标标签
            aux_input: 辅助输入（架构参数的差值）
        
        Returns:
            total_loss, main_loss, aux_loss
        """
        # 主损失
        main_loss = self.criterion(logits, target)
        
        # 辅助损失（L2正则）
        if aux_input is not None:
            aux_loss = self.weight * torch.sum(aux_input ** 2)
        else:
            aux_loss = torch.tensor(0.0).to(logits.device)
        
        total_loss = main_loss + aux_loss
        
        return total_loss, main_loss, aux_loss


class TriSeparateLoss(nn.Module):
    """
    三元分离损失
    
    主损失 + 辅助损失（三元约束）
    """
    def __init__(self, criterion, weight=1.0):
        super(TriSeparateLoss, self).__init__()
        self.criterion = criterion
        self.weight = weight
    
    def forward(self, logits, target, aux_input=None):
        """计算总损失"""
        # 主损失
        main_loss = self.criterion(logits, target)
        
        # 辅助损失（三元约束）
        if aux_input is not None:
            aux_loss = self.weight * torch.mean(torch.abs(aux_input))
        else:
            aux_loss = torch.tensor(0.0).to(logits.device)
        
        total_loss = main_loss + aux_loss
        
        return total_loss, main_loss, aux_loss


class MseSeparateLoss(nn.Module):
    """
    MSE分离损失
    """
    def __init__(self, criterion, weight=1.0):
        super(MseSeparateLoss, self).__init__()
        self.criterion = criterion
        self.weight = weight
    
    def forward(self, logits, target, aux_input=None):
        """计算总损失"""
        # 主损失
        main_loss = self.criterion(logits, target)
        
        # 辅助损失
        if aux_input is not None:
            aux_loss = self.weight * F.mse_loss(aux_input, torch.zeros_like(aux_input))
        else:
            aux_loss = torch.tensor(0.0).to(logits.device)
        
        total_loss = main_loss + aux_loss
        
        return total_loss, main_loss, aux_loss


# =============================================================================
# 简化版本：直接使用timm.loss
# =============================================================================

class SimpleWrapper(nn.Module):
    """
    简单包装器，使timm.loss兼容三返回值格式
    """
    def __init__(self, criterion):
        super(SimpleWrapper, self).__init__()
        self.criterion = criterion
    
    def forward(self, logits, target, aux_input=None):
        """
        Args:
            logits: 模型输出
            target: 目标标签
            aux_input: 辅助输入（忽略）
        
        Returns:
            loss, loss, 0 (为了兼容三返回值)
        """
        loss = self.criterion(logits, target)
        return loss, loss, torch.tensor(0.0).to(logits.device)
