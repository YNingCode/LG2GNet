import warnings
import numpy as np
import torch
from numpy import corrcoef
from scipy.io import loadmat
from sklearn.manifold import TSNE
from sklearn.metrics import confusion_matrix, accuracy_score, roc_auc_score, precision_score, recall_score, f1_score
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import KFold
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import os
from model.bnt import BrainNetworkTransformer
from torch.utils.data import Dataset, DataLoader
warnings.filterwarnings("ignore")
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
seed = 1

def save_roc_data(file_path, all_fpr, all_tpr):
    with open(file_path, 'w') as f:
        for fpr, tpr in zip(all_fpr, all_tpr):
            for fp, tp in zip(fpr, tpr):
                f.write(f"{fp}\t{tp}\n")

def MaxMinNormalization(x, Max, Min):
    x = (x - Min) / (Max - Min)
    return x

#########PD##########
# #########    0 vs 1,2   ##########
# m = loadmat('D:\YuanNing\Dataset\pd\PD_dataset.mat')  # fmri
# # m = loadmat('/lab/2023/yn/Dataset/fMRI/PD/PD_dataset.mat')
# feature = m['feas']  # (162,116,220)
# labels = m['label'][0]  # 有0、1、2三种
# net_all = []
# for i in range(feature.shape[0]):
#     net = corrcoef(feature[i])
#     net_all.append(net)
# fdata = np.array(net_all)
# for i in range(labels.shape[0]):
#     if labels[i] == 2:
#         labels[i] = 1

#########    0 vs 1   ##########
# m = loadmat('D:\YuanNing\Dataset\pd\PD_dataset.mat')  # fmri
# feature = m['feas']  # (162,116,220)
# labels = m['label'][0]  # 有0、1、2三种
# # 只取标签0, 1
# bool_idx = (labels == 0) | (labels == 1)
# feature = feature[bool_idx]
# labels = labels[bool_idx]
# net_all = []
# for i in range(feature.shape[0]):
#     net = corrcoef(feature[i])
#     net_all.append(net)
# fdata = np.array(net_all)

#########    0 vs 2   ##########
# m = loadmat('D:\YuanNing\Dataset\pd\PD_dataset.mat')  # fmri
# feature = m['feas']  # (162,116,220)
# labels = m['label'][0]  # 有0、1、2三种
# # 只取标签0, 1
# bool_idx = (labels == 0) | (labels == 2)
# feature = feature[bool_idx]
# labels = labels[bool_idx]
# for i in range(labels.shape[0]):
#     if labels[i] == 2:
#         labels[i] = 1
# net_all = []
# for i in range(feature.shape[0]):
#     net = corrcoef(feature[i])
#     net_all.append(net)
# fdata = np.array(net_all)

#########    1 vs 2   ##########
# m = loadmat('D:\YuanNing\Dataset\pd\PD_dataset.mat')  # fmri
# feature = m['feas']  # (162,116,220)
# labels = m['label'][0]  # 有0、1、2三种
# bool_idx = (labels == 1) | (labels == 2)
# feature = feature[bool_idx]
# labels = labels[bool_idx]
# for i in range(labels.shape[0]):
#     if labels[i] == 1:
#         labels[i] = 0
#     if labels[i] == 2:
#         labels[i] = 1
# net_all = []
# for i in range(feature.shape[0]):
#     net = corrcoef(feature[i])
#     net_all.append(net)
# fdata = np.array(net_all)

#########ADNI##########
#########    0 vs 1,2   ##########
# m = loadmat('D:\YuanNing\Dataset\ADNI\ADNI_NC_SMC_EMCI_New.mat')  # fmri
m = loadmat('/lab/2023/yn/Dataset/fMRI/ADNI/ADNI.mat')
feature = m['feas']  # (203,90,197)这是306个受试者的90个脑区在240时间点的血氧水平含量
labels = m['label'][0]  # 有0、1、2三种
net_all = []
for i in range(feature.shape[0]):
    net = corrcoef(feature[i])
    net_all.append(net)
fdata = np.array(net_all)
for i in range(labels.shape[0]):
    if labels[i] == 2:
        labels[i] = 1

#########    0 vs 1   ##########
# m = loadmat('D:\YuanNing\Dataset\ADNI\ADNI_NC_SMC_EMCI_New.mat')  # fmri
# m = loadmat('/lab/2023/yn/Dataset/fMRI/ADNI/ADNI.mat')
# feature = m['feas']  # (203,90,197)这是306个受试者的90个脑区在240时间点的血氧水平含量
# labels = m['label'][0]  # 有0、1、2三种
# # 只取标签0, 1
# bool_idx = (labels == 0) | (labels == 1)
# feature = feature[bool_idx]
# labels = labels[bool_idx]
# net_all = []
# for i in range(feature.shape[0]):
#     net = corrcoef(feature[i])
#     net_all.append(net)
# fdata = np.array(net_all)

##########    0 vs 2   ##########
# m = loadmat('D:\YuanNing\Dataset\ADNI\ADNI_NC_SMC_EMCI_New.mat')  # fmri
# m = loadmat('/lab/2023/yn/Dataset/fMRI/ADNI/ADNI.mat')
# feature = m['feas']  # (203,90,197)这是306个受试者的90个脑区在240时间点的血氧水平含量
# labels = m['label'][0]  # 有0、1、2三种
# # 只取0, 2
# bool_idx = (labels == 0) | (labels == 2)
# feature = feature[bool_idx]
# labels = labels[bool_idx]
# for i in range(labels.shape[0]):
#     if labels[i] == 2:
#         labels[i] = 1
# net_all = []
# for i in range(feature.shape[0]):
#     net = corrcoef(feature[i])
#     net_all.append(net)
# fdata = np.array(net_all)

#########    1 vs 2   ##########
# m = loadmat('D:\YuanNing\Dataset\ADNI\ADNI_NC_SMC_EMCI_New.mat')  # fmri
# m = loadmat('/lab/2023/yn/Dataset/fMRI/ADNI/ADNI.mat')
# feature = m['feas']  # (203,90,197)这是306个受试者的90个脑区在240时间点的血氧水平含量
# labels = m['label'][0]  # 有0、1、2三种
# # 只取0, 2
# bool_idx = (labels == 1) | (labels == 2)
# feature = feature[bool_idx]
# labels = labels[bool_idx]
# for i in range(labels.shape[0]):
#     if labels[i] == 1:
#         labels[i] = 0
#     if labels[i] == 2:
#         labels[i] = 1
# net_all = []
# for i in range(feature.shape[0]):
#     net = corrcoef(feature[i])
#     net_all.append(net)
# fdata = np.array(net_all)


index = [i for i in range(fdata.shape[0])]
np.random.shuffle(index)
fdata = fdata[index]
labels = labels[index]
feature = feature[index]


##############################################定义数据######################################
class Dianxian(Dataset):
    def __init__(self):
        super(Dianxian, self).__init__()
        self.nodes = feature
        self.edges = fdata
        self.labels = labels

    def __getitem__(self, item):
        node = self.nodes[item]
        edge = self.edges[item]
        label = self.labels[item]
        return node, edge, label

    def __len__(self):
        return self.nodes.shape[0]


########################训练部分##################################
# 定义测试函数
def stest(model, datasets_test):
    # 在测试集上检验效果
    eval_loss = 0
    eval_acc = 0
    pre_all = []
    labels_all = []
    pro_all = []
    tsne_all = []
    model.eval()  # 将模型改为预测模式
    with torch.no_grad():
        for node, edge, label in datasets_test:
            node, edge, label = node.to(DEVICE), edge.to(DEVICE), label.to(DEVICE)
            node = node.float()
            edge = edge.float()
            label = label.long()
            output, tsne = model(node, edge)
            loss = loss_fn(output, label)
            # 记录误差
            eval_loss += float(loss)
            # out = F.log_softmax(output, dim=-1)
            # 记录准确率
            _, pred = output.max(1)
            num_correct = (pred == label).sum()
            acc = int(num_correct) / node.shape[0]
            eval_acc += acc
            pre = pred.cpu().detach().numpy()
            pre_all.extend(pre)
            label_true = label.cpu().detach().numpy()
            labels_all.extend(label_true)
            pro_all.extend(output[:, 1].cpu().detach().numpy())
            tsne_all.append(tsne)
    tn, fp, fn, tp = confusion_matrix(labels_all, pre_all).ravel()
    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)
    eval_acc_epoch = accuracy_score(labels_all, pre_all)
    my_auc = roc_auc_score(labels_all, pro_all)
    precision = precision_score(labels_all, pre_all)
    recall = recall_score(labels_all, pre_all)
    f1 = f1_score(labels_all, pre_all)
    tsne = np.concatenate(tsne_all, axis=0)

    return eval_loss, eval_acc, eval_acc_epoch, precision, recall, f1, my_auc, sensitivity, specificity, pre_all, labels_all, pro_all, tsne


# 开始训练
dataset = Dianxian()
test_acc = []
test_sens = []
test_spec = []
test_auc = []
test_pre = []
test_recall = []
test_f1 = []
label_ten = []
pro_ten = []
i = 0
bs_train = 20
bs_test = 20
s = 0
KF = KFold(n_splits=10, shuffle=True, random_state=seed)
for lr in [5e-4]:
    s = s + 1
    for train_idx, test_idx in KF.split(dataset):
        train_subsampler = SubsetRandomSampler(train_idx)
        test_sunsampler = SubsetRandomSampler(test_idx)
        datasets_all = DataLoader(dataset, batch_size=1, shuffle=False)
        datasets_train = DataLoader(dataset, batch_size=2, shuffle=False, sampler=train_subsampler, drop_last=True)
        datasets_test = DataLoader(dataset, batch_size=4, shuffle=False, sampler=test_sunsampler, drop_last=True)
        epoch = 300
        losses = []  # 记录训练误差，用于作图分析
        acces = []
        eval_losses = []
        eval_acces = []
        patiences = 50
        min_acc = 0
        best_acc = 0
        model = BrainNetworkTransformer()
        model.to(DEVICE)
        loss_fn = torch.nn.CrossEntropyLoss(reduction='mean')
        optimizer = torch.optim.Adam(model.parameters(), lr=3e-5)
        for num in range(epoch):
            train_loss = 0
            train_acc = 0
            model.train()
            for node, edge, label in datasets_train:
                node, edge, label = node.to(DEVICE), edge.to(DEVICE), label.to(DEVICE)
                # 前向传播
                node = node.float()
                edge = edge.float()
                label = label.long()

                output, _ = model(node, edge)
                loss = loss_fn(output, label)

                # 反向传播
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                train_loss += float(loss)
                # out = F.log_softmax(output, dim=-1)
                _, pred = output.max(1)
                num_correct = (pred == label).sum()
                acc = int(num_correct) / node.shape[0]
                train_acc += acc
            losses.append(train_loss / len(datasets_train))
            acces.append(train_acc / len(datasets_train))

            # 测试集测试
            eval_loss, eval_acc, eval_acc_epoch, precision, recall, f1, my_auc, sensitivity, specificity, pre_all, labels_all, pro_all, _ = stest(
                model, datasets_test)
            if (eval_acc_epoch > min_acc) & (eval_acc_epoch <= 1):
                torch.save(model.state_dict(), f'./save/model{i}.pth')
                print("Model saved at epoch{}".format(num))
                min_acc = eval_acc_epoch
                pre_gd = precision
                recall_gd = recall
                f1_gd = f1
                auc_gd = my_auc
                sens_gd = sensitivity
                spec_gd = specificity
                labels_all_gd = labels_all
                pro_all_gd = pro_all
                patience = 0
            else:
                patience += 1
            if patience > patiences:
                break
            eval_losses.append(eval_loss / len(datasets_test))
            eval_acces.append(eval_acc / len(datasets_test))
            print(
                'i:{}, epoch:{}, TrainLoss:{:.6f}, TrainAcc:{:.6f}, EvalLoss:{:.6f}, EvalAcc:{:.6f},precision:{'
                ':.6f}, recall:{:.6f}, f1:{:.6f}, my_auc:{:.6f}'
                .format(i, num, train_loss / len(datasets_train), train_acc / len(datasets_train),
                        eval_loss / len(datasets_test), eval_acc_epoch, precision, recall, f1, my_auc))
        # model_best = BrainNetworkTransformer()
        # model_best.to(DEVICE)
        # model_best.load_state_dict(torch.load('model.pth'))
        # eval_loss, eval_acc, eval_acc_epoch, precision, recall, f1, my_auc, sensitivity, specificity, pre_all, labels_all, pro_all, _ = stest(
        #     model_best,
        #     datasets_all)
        # if eval_acc_epoch > best_acc:
        #     torch.Save(model.state_dict(), f'./model/Save/model{i}.pth')
        #     print("Best model saved")
        #     dataset_best = datasets_test
        #     best_acc = eval_acc_epoch

        test_acc.append(min_acc)
        test_pre.append(pre_gd)
        test_recall.append(recall_gd)
        test_f1.append(f1_gd)
        test_auc.append(auc_gd)
        test_sens.append(sens_gd)
        test_spec.append(spec_gd)
        label_ten.extend(labels_all_gd)
        pro_ten.extend(pro_all_gd)

        i = i + 1

    kk = 10
    print("test_acc", test_acc)
    print("test_pre", test_pre)
    print("test_recall", test_recall)
    print("test_f1", test_f1)
    print("test_auc", test_auc)
    print("test_sens", test_sens)
    print("test_spec", test_spec)
    avg_acc = sum(test_acc) / kk
    avg_pre = sum(test_pre) / kk
    avg_recall = sum(test_recall) / kk
    avg_f1 = sum(test_f1) / kk
    avg_auc = sum(test_auc) / kk
    avg_sens = sum(test_sens) / kk
    avg_spec = sum(test_spec) / kk
    print("*****************************************************")
    print('acc', avg_acc)
    print('pre', avg_pre)
    print('recall', avg_recall)
    print('f1', avg_f1)
    print('auc', avg_auc)
    print("sensitivity", avg_sens)
    print("specificity", avg_spec)

    acc_std = np.sqrt(np.var(test_acc))
    pre_std = np.sqrt(np.var(test_pre))
    recall_std = np.sqrt(np.var(test_recall))
    f1_std = np.sqrt(np.var(test_f1))
    auc_std = np.sqrt(np.var(test_auc))
    sens_std = np.sqrt(np.var(test_sens))
    spec_std = np.sqrt(np.var(test_spec))
    print("*****************************************************")
    print("acc_std", acc_std)
    print("pre_std", pre_std)
    print("recall_std", recall_std)
    print("f1_std", f1_std)
    print("auc_std", auc_std)
    print("sens_std", sens_std)
    print("spec_std", spec_std)
    print("*****************************************************")

    print("*****************************************************")
    print(label_ten)
    print(pro_ten)

