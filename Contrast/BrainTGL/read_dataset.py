import numpy as np
from torch.utils.data import Dataset, DataLoader, TensorDataset
import torch
from PIL import ImageFile
from scipy.io import loadmat
from torch.utils.data import random_split
from sklearn.preprocessing import StandardScaler

class DynamicFCDataset(Dataset):
    def __init__(self, data, labels, window_length, stride, sub_seq_length):
        self.data = data
        self.labels = labels
        self.window_length = window_length
        self.stride = stride
        self.sub_seq_length = sub_seq_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        time_series = self.data[idx]
        # Crop the time series to a fixed length
        time_series = time_series[:self.sub_seq_length]
        dynamic_fc, bold_signals = self.compute_dynamic_fc_and_bold(time_series)
        return torch.FloatTensor(dynamic_fc), torch.FloatTensor(bold_signals), torch.LongTensor([self.labels[idx]])

    def compute_dynamic_fc_and_bold(self, time_series):
        num_windows = (time_series.shape[0] - self.window_length) // self.stride + 1
        dynamic_fc = np.zeros((num_windows, time_series.shape[1], time_series.shape[1]))
        bold_signals = np.zeros((num_windows, time_series.shape[1]))

        for i in range(num_windows):
            start = i * self.stride
            end = start + self.window_length
            window = time_series[start:end, :]
            fc = np.corrcoef(window.T)
            dynamic_fc[i] = fc
            bold_signals[i] = window.mean(axis=0)

        return dynamic_fc, bold_signals

def normalize(data):
    min_vals = np.min(data, axis=(1, 2), keepdims=True)
    max_vals = np.max(data, axis=(1, 2), keepdims=True)
    normalized_data = (data - min_vals) / (max_vals - min_vals +1e5)
    return normalized_data

def ADNI(mission, window_length, stride, sub_seq_length):
    # data_path = "E:\Dataset\ADNI\ADNI_NC_SMC_EMCI_New.mat"
    data_path = "/lab/2023/yn/Dataset/fMRI/ADNI/ADNI.mat"
    m = loadmat(data_path)
    data = m['feas']  # (203,90,197)这是306个受试者的90个脑区在240时间点的血氧水平含量
    labels = m['label'][0]  # 有0、1、2三种
    data = np.transpose(data, (0, 2, 1))

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

    dataset = DynamicFCDataset(data, labels, window_length, stride, sub_seq_length)

    labels_tensor = torch.from_numpy(labels)
    num_classes = torch.unique(labels_tensor).size(0)

    return dataset, num_classes

def PD(mission, window_length, stride, sub_seq_length):
    data_path = "/lab/2023/yn/Dataset/fMRI/PD/PD_dataset.mat"
    m = loadmat(data_path)
    data = m['feas']  # (162,116,220)
    labels = m['label'][0]  # 有0、1、2三种(int)
    data = np.transpose(data, (0, 2, 1))

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

    dataset = DynamicFCDataset(data, labels, window_length, stride, sub_seq_length)

    labels_tensor = torch.from_numpy(labels)
    num_classes = torch.unique(labels_tensor).size(0)

    return dataset, num_classes