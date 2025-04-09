import numpy as np
import torch

def pearson(data):
    mean_data = torch.mean(data, axis=0)  # shape: (num_nodes, dim)
    # 转换为 numpy 数组来计算皮尔逊相关系数
    mean_data_np = mean_data.cpu().numpy()  # 将 tensor 转为 numpy 数组
    corr_matrix = np.corrcoef(mean_data_np, rowvar=True)
    # 将 numpy 数组转换回 tensor
    corr_matrix_tensor = torch.tensor(corr_matrix, dtype=torch.float32)  # 转换为 float32 类型 tensor

    return corr_matrix_tensor

if __name__ == '__main__':
    data = np.random.rand(20, 90, 195)
    adj = pearson(data)
    print(adj.shape)