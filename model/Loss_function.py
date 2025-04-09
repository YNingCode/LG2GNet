import torch
import torch.nn as nn
from model.utils import *

class DistanceDecayLoss(nn.Module):
    def __init__(self, alpha=0.1):
        super(DistanceDecayLoss, self).__init__()
        self.alpha = alpha  # 衰减因子，控制距离衰减的速率

    def forward(self, adj):
        N = adj.size(0)  # 获取邻接矩阵的节点数 N

        # 创建一个距离矩阵，记录各节点间的距离
        dist_matrix = torch.abs(torch.arange(N).unsqueeze(0) - torch.arange(N).unsqueeze(1)).to(adj.device)

        # 计算距离衰减项
        decay_matrix = torch.exp(-self.alpha * dist_matrix.float())

        # 计算损失：邻接矩阵与衰减矩阵的差异（L2范数）
        decay_loss = torch.sum((adj - decay_matrix) ** 2)

        return decay_loss
class AdjacencySmoothnessLoss(nn.Module):
    def __init__(self, lambda_smooth=0.1):
        super(AdjacencySmoothnessLoss, self).__init__()
        self.lambda_smooth = lambda_smooth  # 控制平滑性正则化的强度

    def forward(self, adj):
        N = adj.size(0)  # 获取邻接矩阵的节点数 N

        # 计算平滑性损失：邻接矩阵的邻近节点之间的差异
        smoothness_loss = 0.0
        for i in range(1, N-1):
            for j in range(1, N-1):
                smoothness_loss += torch.abs(adj[i, j] - (adj[i-1, j] + adj[i+1, j] + adj[i, j-1] + adj[i, j+1]) / 4)

        return self.lambda_smooth * smoothness_loss
class CombinedGraphLoss(nn.Module):
    def __init__(self, alpha=0.1, lambda_smooth=0.1, gamma=0.001):
        super(CombinedGraphLoss, self).__init__()
        self.distance_decay_loss = DistanceDecayLoss(alpha)
        self.smoothness_loss = AdjacencySmoothnessLoss(lambda_smooth)
        self.gamma = gamma  # 邻接矩阵的范数正则化权重

    def forward(self, adj):
        adj = normalize_A(adj)
        # 计算距离衰减损失
        decay_loss = self.distance_decay_loss(adj)

        # 计算平滑性损失
        smoothness_loss = self.smoothness_loss(adj)

        # 计算邻接矩阵的 Frobenius 范数损失
        f_norm = torch.norm(adj, p='fro') ** 2

        # 综合损失
        total_loss = decay_loss + smoothness_loss + self.gamma * f_norm

        return total_loss


class GraphStructureLoss(nn.Module):
    def __init__(self, lambda_=1.0, phi=1.0, alpha=1.0, gamma=0.001):
        """
        初始化图结构损失函数。

        参数:
        - lambda_: 特征平滑损失的权重
        - phi: 对称性损失的权重
        - alpha: L1 正则化的权重
        - gamma: 图拉普拉斯平滑的权重
        """
        super(GraphStructureLoss, self).__init__()
        self.lambda_ = lambda_
        self.phi = phi
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, adj):
        """
        计算图结构学习任务中的损失。

        参数:
        - adj: 邻接矩阵（形状为 N x N）

        返回:
        - 总损失
        """
        adj = normalize_A(adj)
        # L1 正则化（邻接矩阵的稀疏性）
        loss_l1 = torch.norm(adj, p=1)

        # 图拉普拉斯平滑损失
        loss_smooth_feat = self.feature_smoothing(adj)

        # 对称性损失（鼓励邻接矩阵对称）
        loss_symmetric = torch.norm(adj - adj.t(), p="fro")

        # 总损失函数
        total_loss = (self.lambda_ * loss_smooth_feat +
                      self.phi * loss_symmetric +
                      self.alpha * loss_l1)

        return total_loss

    def feature_smoothing(self, adj):
        """
        计算特征平滑性损失（图拉普拉斯正则化）。

        参数:
        - adj: 邻接矩阵（估计的邻接矩阵）。

        返回:
        - 平滑性损失
        """
        adj = (adj.t() + adj) / 2  # 确保邻接矩阵对称
        rowsum = adj.sum(1)
        r_inv = rowsum.flatten()
        D = torch.diag(r_inv)
        L = D - adj

        r_inv = r_inv + 1e-3  # 避免除零
        r_inv = r_inv.pow(-1 / 2).flatten()
        r_inv[torch.isinf(r_inv)] = 0.
        r_mat_inv = torch.diag(r_inv)

        L = r_mat_inv @ L @ r_mat_inv
        loss_smooth_feat = torch.trace(L)  # 使用拉普拉斯矩阵的迹作为平滑损失
        return loss_smooth_feat

