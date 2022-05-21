from locale import normalize
import torch
from torch import nn


class BarlowTwinsLoss(nn.Module):

    def __init__(self, coef=3.9e-3) -> None:
        super(BarlowTwinsLoss, self).__init__()
        self.coef = coef

    def forward(self, x, y):
        x_norm = (x - x.mean(dim=0))
        y_norm = (y - y.mean(dim=0))
        cov = x_norm.t() @ y_norm

        x_l2 = x.pow(2).sum(0, keepdim=True).sqrt()
        y_l2 = y.pow(2).sum(0, keepdim=True).sqrt()
        cov = cov / (x_l2.t() @ y_l2)

        ret = (cov - torch.eye(cov.shape[0], device=cov.device)).pow(2)
        coef = self.coef * torch.ones_like(cov, device=cov.device) + (1 - self.coef) * torch.eye(cov.shape[0], device=cov.device)
        ret = ret * coef

        loss = ret.sum()
        return loss


class VICRegLoss(nn.Module):

    def __init__(self, eps=1e-4, std_gamma=1., coef=[25., 25., 1.]) -> None:
        super(VICRegLoss, self).__init__()
        self.eps = eps
        self.std_gamma = std_gamma
        self.coef = coef
        self.invariance_term = nn.MSELoss(reduce='mean')

    def variance_term(self, features: torch.Tensor):
        # Feature: B * D
        std_vec = (features.var(dim=0) + self.eps).sqrt()
        hinge_vec = torch.relu(self.std_gamma - std_vec)
        return hinge_vec.mean()

    def covariance_term(self, features: torch.Tensor):
        normalized = features - features.mean(dim=0)
        cov_mat = normalized.t() @ normalized
        off_diagonal_mask = torch.ones_like(cov_mat) - torch.eye(cov_mat.shape[0], dtype=cov_mat.dtype, device=cov_mat.device)
        cov_mat = cov_mat * off_diagonal_mask
        return cov_mat.pow(2).sum() / features.shape[1]

    def forward(self, x, y):
        loss_variance = self.variance_term(x) + self.variance_term(y)
        loss_invariance = self.invariance_term(x, y)
        loss_covariance = self.covariance_term(x) + self.covariance_term(y)
        return self.coef[0] * loss_variance + self.coef[1] * loss_invariance + self.coef[2] * loss_covariance