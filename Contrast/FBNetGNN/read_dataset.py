import numpy as np
from torch.utils.data import Dataset, DataLoader, TensorDataset
import torch
from PIL import ImageFile
from scipy.io import loadmat
from torch.utils.data import random_split
from sklearn.preprocessing import StandardScaler

def normalize(data):
    min_vals = np.min(data, axis=(1, 2), keepdims=True)
    max_vals = np.max(data, axis=(1, 2), keepdims=True)
    normalized_data = (data - min_vals) / (max_vals - min_vals +1e5)
    return normalized_data

def ADNI(mission):
    # data_path = "E:\Dataset\ADNI\ADNI_NC_SMC_EMCI_New.mat"
    data_path = "/lab/2023/yn/Dataset/fMRI/ADNI/ADNI.mat"
    m = loadmat(data_path)
    data = m['feas']  # (203,90,197)这是306个受试者的90个脑区在240时间点的血氧水平含量
    labels = m['label'][0]  # 有0、1、2三种

    if mission == '0_12':
        for i in range(labels.shape[0]):
            if labels[i] == 2:
                labels[i] = 1
    elif mission == '0_1':
        # 只取标签0, 1
        bool_idx = (labels == 0) | (labels == 1)
        data = data[bool_idx]
        labels = labels[bool_idx]
    elif mission == '0_2':
        # 只取标签0，2
        bool_idx = (labels == 0) | (labels == 2)
        data = data[bool_idx]
        labels = labels[bool_idx]
        for i in range(labels.shape[0]):
            if labels[i] == 2:
                labels[i] = 1
    elif mission == '1_2':
        # 只取标签1，2
        bool_idx = (labels == 1) | (labels == 2)
        data = data[bool_idx]
        labels = labels[bool_idx]
        for i in range(labels.shape[0]):
            if labels[i] == 1:
                labels[i] = 0
            if labels[i] == 2:
                labels[i] = 1

    data = normalize(data)
    index = [i for i in range(data.shape[0])]
    np.random.shuffle(index)
    data = data[index]
    labels = labels[index]

    data_tensor = torch.from_numpy(data).float()
    labels_tensor = torch.from_numpy(labels)
    num_nodes = data_tensor.size(1)
    seq_length = data_tensor.size(2)
    num_classes = torch.unique(labels_tensor).size(0)

    dataset = TensorDataset(data_tensor, labels_tensor)

    return dataset, num_nodes, seq_length, num_classes

def PD(mission):
    data_path = "/lab/2023/yn/Dataset/fMRI/PD/PD_dataset.mat"
    m = loadmat(data_path)
    data = m['feas']  # (162,116,220)
    labels = m['label'][0]  # 有0、1、2三种(int)

    if mission == '0_12':
        for i in range(labels.shape[0]):
            if labels[i] == 2:
                labels[i] = 1
    elif mission == '0_1':
        # 只取标签0, 1
        bool_idx = (labels == 0) | (labels == 1)
        data = data[bool_idx]
        labels = labels[bool_idx]
    elif mission == '0_2':
        # 只取标签0，2
        bool_idx = (labels == 0) | (labels == 2)
        data = data[bool_idx]
        labels = labels[bool_idx]
        for i in range(labels.shape[0]):
            if labels[i] == 2:
                labels[i] = 1
    elif mission == '1_2':
        # 只取标签1，2
        bool_idx = (labels == 1) | (labels == 2)
        data = data[bool_idx]
        labels = labels[bool_idx]
        for i in range(labels.shape[0]):
            if labels[i] == 1:
                labels[i] = 0
            if labels[i] == 2:
                labels[i] = 1

    data = normalize(data)
    index = [i for i in range(data.shape[0])]
    np.random.shuffle(index)
    data = data[index]
    labels = labels[index]

    data_tensor = torch.from_numpy(data).float()
    labels_tensor = torch.from_numpy(labels)
    num_nodes = data_tensor.size(1)
    seq_length = data_tensor.size(2)
    num_classes = torch.unique(labels_tensor).size(0)

    dataset = TensorDataset(data_tensor, labels_tensor)

    return dataset, num_nodes, seq_length, num_classes