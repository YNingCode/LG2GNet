import torch.nn as nn
import torch.nn.functional as F

class GraphMapper(nn.Module):
    def __init__(self, node_num, in_dim, window_num, hidden_dim=64):
        """
        图映射器模型，用于从全局图映射到局部图
        :param node_num: 节点数量
        :param in_dim: 输入维度（全局图的邻接矩阵维度）
        :param window_num: 时间窗口数量
        :param hidden_dim: 映射器隐藏层维度
        """
        super(GraphMapper, self).__init__()
        self.node_num = node_num
        self.in_dim = in_dim
        self.window_num = window_num

        # 为每个时间窗口设置一个映射器（MLP），将全局图映射到局部图
        self.mappers = nn.ModuleList([nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, node_num * node_num)
        ) for _ in range(window_num)])

    def forward(self, global_adj):
        """
        前向传播，生成每个时间窗口的图结构
        :param global_adj: 全局图的邻接矩阵，形状为 (node_num, node_num)
        :return: 每个时间窗口的图邻接矩阵列表
        """
        global_adj_flat = global_adj.view(-1)  # 将全局图展开为向量
        window_adj_list = []

        for mapper in self.mappers:
            # 每个映射器处理全局图，生成局部图
            window_adj_flat = mapper(global_adj_flat)
            window_adj = window_adj_flat.view(self.node_num, self.node_num)  # 恢复为矩阵形状
            window_adj = F.relu(window_adj)  # 经过非线性变换确保非负性
            window_adj_list.append(window_adj)

        return window_adj_list

class Windows_Graph_Mapper(nn.Module):
    def __init__(self, node_num, window_num, in_dim, hidden_dim=64):
        """
        时间序列图结构学习模型，包括全局和局部图结构的学习
        :param node_num: 节点数量
        :param in_dim: 输入维度（全局图的邻接矩阵维度）
        :param hidden_dim: 隐藏层维度
        """
        super(Windows_Graph_Mapper, self).__init__()
        self.node_num = node_num
        self.window_num = window_num

        # 图映射器模块
        self.graph_mapper = GraphMapper(node_num, in_dim, self.window_num, hidden_dim)

    def forward(self, global_adj):
        """
        前向传播
        :param time_series: 多元时间序列数据，形状为 (batch_size, time_steps, node_num)
        :return: 每个时间窗口的图邻接矩阵和对应的时间窗口数据
        """
        # 通过全局图映射学习每个时间窗口的局部图结构
        window_adj_list = self.graph_mapper(global_adj)

        return window_adj_list
