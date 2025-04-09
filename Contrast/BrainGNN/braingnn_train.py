import warnings
import torch
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.model_selection import KFold
import torch.nn as nn
import numpy as np
from numpy import random
from scipy.io import loadmat
import torch.nn.functional as F
from torch.utils.data import Dataset, SubsetRandomSampler, DataLoader
from model.braingnn import Network
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import os

def save_roc_data(file_path, all_fpr, all_tpr):
    with open(file_path, 'w') as f:
        for fpr, tpr in zip(all_fpr, all_tpr):
            for fp, tp in zip(fpr, tpr):
                f.write(f"{fp}\t{tp}\n")


os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
warnings.filterwarnings("ignore")
###############################################固定随机数种子####################################
seed = 183
torch.manual_seed(seed)  # 为CPU设置随机种子
torch.cuda.manual_seed(seed)  # 为当前GPU设置随机种子
torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU，为所有GPU设置随机种子
np.random.seed(seed)  # Numpy module.
random.seed(seed)  # Python random module.
k = 10



# 定义测试函数
def stest(model, datasets_test):
    # 在测试集上检验效果
    eval_loss = 0
    eval_acc = 0
    pre_all = []
    labels_all = []
    gailv_all = []
    pro_all = []
    tsne_all = []
    # criterion = nn.CrossEntropyLoss()
    model.eval()  # 将模型改为预测模式
    for net, data_feas, label in datasets_test:
        net, data_feas, label = net.to(DEVICE), data_feas.to(DEVICE), label.to(DEVICE)
        net = net.float()
        data_feas = data_feas.float()
        label = label.long()
        edge_list = []
        for i in range(data_feas.size(0)):
            adj = data_feas[i]
            row, col = torch.nonzero(adj, as_tuple=True)
            edges = torch.stack((row + i * 90, col + i * 90), dim=0)
            edge_list.append(edges)

        x = net.reshape(-1, 197)
        edge_index = torch.cat(edge_list, dim=1)
        edge_attr = torch.ones(edge_index.size(1), 1)
        batch = torch.repeat_interleave(torch.arange(net.size(0)), 90)
        pos = torch.stack([torch.diag(torch.ones(90)) for _ in range(data_feas.size(0))])
        pos = pos.view(-1, 90)
        x, edge_index, edge_attr, batch, pos = x.to(DEVICE), edge_index.to(DEVICE), edge_attr.to(DEVICE), batch.to(DEVICE), pos.to(DEVICE)
        x = x.float()
        edge_attr = edge_attr.float()
        pos = pos.float()

        outs,tsne = model(x, edge_index, batch,edge_attr,pos)

        losss = F.nll_loss(outs, label)
        # 记录误差
        eval_loss += float(losss)
        # 记录准确率
        gailv, pred = outs.max(1)
        num_correct = (pred == label).sum()
        acc = int(num_correct) / net.shape[0]
        eval_acc += acc
        pre = pred.cpu().detach().numpy()
        pre_all.extend(pre)
        label_true = label.cpu().detach().numpy()
        labels_all.extend(label_true)
        pro_all.extend(outs[:, 1].cpu().detach().numpy())
        tsne_all.append(tsne)
    tsnes = np.concatenate(tsne_all, axis=0)
    # tsnes = tsne_all
    tn, fp, fn, tp = confusion_matrix(labels_all, pre_all).ravel()
    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)
    eval_acc_epoch = accuracy_score(labels_all, pre_all)
    precision = precision_score(labels_all, pre_all)
    recall = recall_score(labels_all, pre_all)
    f1 = f1_score(labels_all, pre_all)
    my_auc = roc_auc_score(labels_all, pro_all)

    return eval_loss, eval_acc, eval_acc_epoch, precision, recall, f1, my_auc, sensitivity, specificity, pre_all, labels_all, pro_all,tsnes

def MaxMinNormalization(x, Max, Min):
    x = (x - Min) / (Max - Min)
    return x
batchsize = 20

# log = open('./record/record_ADNI.txt', mode='a', encoding='utf-8')
log = open('./record/record_PD.txt', mode='a', encoding='utf-8')
for lr1 in [1e-3]:
    for wd in [1e-2]:
        ################################# PD数据集    0  vs  1_2 ########################################
        # m = loadmat('E:\Dataset\pd\PD_dataset.mat')
        # fdata = m['feas']  # (162,116,220)
        # labels = m['label'][0]  # 有0、1、2三种(int)
        # net_all = []
        # for i in range(fdata.shape[0]):
        #     net = np.corrcoef(fdata[i])
        #     net_all.append(net)
        # net_all = np.array(net_all)
        # net_all = np.where(np.abs(net_all) >= 0.4, 1, 0)
        # for i in range(fdata.shape[0]):
        #     max_t = np.max(fdata[i])
        #     min_t = np.min(fdata[i])
        #     fdata[i] = MaxMinNormalization(fdata[i], max_t, min_t)
        # for i in range(labels.shape[0]):
        #     if labels[i] == 2:
        #         labels[i] = 1
        ################################# 获取数据集    0  vs  1  ########################################
        # m = loadmat('E:\Dataset\pd\PD_dataset.mat')
        # fdata = m['feas']  # (162,116,220)
        # labels = m['label'][0]  # 有0、1、2三种(int)
        # # 只取标签0, 1
        # bool_idx = (labels == 0) | (labels == 1)
        # fdata = fdata[bool_idx]
        # labels = labels[bool_idx]
        # net_all = []
        # for i in range(fdata.shape[0]):
        #     net = np.corrcoef(fdata[i])
        #     net_all.append(net)
        # net_all = np.array(net_all)
        # net_all = np.where(np.abs(net_all) >= 0.4, 1, 0)
        # for i in range(fdata.shape[0]):
        #     max_t = np.max(fdata[i])
        #     min_t = np.min(fdata[i])
        #     fdata[i] = MaxMinNormalization(fdata[i], max_t, min_t)
        ################################# 获取数据集    0  vs   2  ########################################
        # m = loadmat('E:\Dataset\pd\PD_dataset.mat')
        # fdata = m['feas']  # (162,116,220)
        # labels = m['label'][0]  # 有0、1、2三种(int)
        # # 只取标签0，2
        # bool_idx = (labels == 0) | (labels == 2)
        # fdata = fdata[bool_idx]
        # labels = labels[bool_idx]
        # for i in range(labels.shape[0]):
        #     if labels[i] == 2:
        #         labels[i] = 1
        # net_all = []
        # for i in range(fdata.shape[0]):
        #     net = np.corrcoef(fdata[i])
        #     net_all.append(net)
        # net_all = np.array(net_all)
        # net_all = np.where(np.abs(net_all) >= 0.4, 1, 0)
        # for i in range(fdata.shape[0]):
        #     max_t = np.max(fdata[i])
        #     min_t = np.min(fdata[i])
        #     fdata[i] = MaxMinNormalization(fdata[i], max_t, min_t)

        ################################# 获取数据集    1  vs   2  ########################################
        # m = loadmat('E:\Dataset\pd\PD_dataset.mat')
        # fdata = m['feas']  # (162,116,220)
        # labels = m['label'][0]  # 有0、1、2三种(int)
        # # 只取标签1，2
        # bool_idx = (labels == 1) | (labels == 2)
        # fdata = fdata[bool_idx]
        # labels = labels[bool_idx]
        # for i in range(labels.shape[0]):
        #     if labels[i] == 1:
        #         labels[i] = 0
        #     if labels[i] == 2:
        #         labels[i] = 1
        # net_all = []
        # for i in range(fdata.shape[0]):
        #     net = np.corrcoef(fdata[i])
        #     net_all.append(net)
        # net_all = np.array(net_all)
        # net_all = np.where(np.abs(net_all) >= 0.4, 1, 0)
        # for i in range(fdata.shape[0]):
        #     max_t = np.max(fdata[i])
        #     min_t = np.min(fdata[i])
        #     fdata[i] = MaxMinNormalization(fdata[i], max_t, min_t)


        #########ADNI##########
        # m = loadmat('E:\Dataset\ADNI\ADNI_NC_SMC_EMCI_New.mat')  # fmri
        m = loadmat('/lab/2023/yn/Dataset/fMRI/ADNI/ADNI.mat')  # fmri
        keysm = list(m.keys())
        fdata = m[keysm[3]]  # 特征数据
        net_all = []
        for i in range(fdata.shape[0]):
            net = np.corrcoef(fdata[i])
            net_all.append(net)
        net_all = np.array(net_all)

        labels = m['label'][0]
        for i in range(203):
            max_t = np.max(fdata[i])
            min_t = np.min(fdata[i])
            fdata[i] = MaxMinNormalization(fdata[i], max_t, min_t)
        for i in range(labels.shape[0]):
            if labels[i] == 2:
                labels[i] = 1
        ################################# 获取数据集    NC  vs   SMC########################################
        # m = loadmat('E:\Dataset\ADNI\ADNI_NC_SMC_EMCI_New.mat')  # fmri
        # data = m['feas']  # (203,90,197)这是306个受试者的90个脑区在240时间点的血氧水平含量
        # labels = m['label'][0]  # 有0、1、2三种
        # # 只取标签0, 1
        # bool_idx = (labels == 0) | (labels == 1)
        # fdata = data[bool_idx]
        # labels = labels[bool_idx]
        # for i in range(fdata.shape[0]):
        #     max_t = np.max(fdata[i])
        #     min_t = np.min(fdata[i])
        #     fdata[i] = MaxMinNormalization(fdata[i], max_t, min_t)
        # net_all = []
        # for i in range(fdata.shape[0]):
        #     net = np.corrcoef(fdata[i])
        #     net_all.append(net)
        # net_all = np.array(net_all)

        ################################# 获取数据集    NC  vs   EMCI########################################
        # m = loadmat('E:\Dataset\ADNI\ADNI_NC_SMC_EMCI_New.mat')  # fmri
        # fdata = m['feas']  # (203,90,197)这是306个受试者的90个脑区在240时间点的血氧水平含量
        # labels = m['label'][0]  # 有0、1、2三种
        # bool_idx = (labels == 0) | (labels == 2)
        # fdata = fdata[bool_idx]
        # labels = labels[bool_idx]
        # for i in range(labels.shape[0]):
        #     if labels[i] == 2:
        #         labels[i] = 1
        # for i in range(fdata.shape[0]):
        #     max_t = np.max(fdata[i])
        #     min_t = np.min(fdata[i])
        #     fdata[i] = MaxMinNormalization(fdata[i], max_t, min_t)
        # net_all = []
        # for i in range(fdata.shape[0]):
        #     net = np.corrcoef(fdata[i])
        #     net_all.append(net)
        # net_all = np.array(net_all)

        ################################# 获取数据集    SMC  vs   EMCI########################################
        # m = loadmat('E:\Dataset\ADNI\ADNI_NC_SMC_EMCI_New.mat')  # fmri
        # fdata = m['feas']  # (203,90,197)这是306个受试者的90个脑区在240时间点的血氧水平含量
        # labels = m['label'][0]  # 有0、1、2三种
        # bool_idx = (labels == 1) | (labels == 2)
        # fdata = fdata[bool_idx]
        # labels = labels[bool_idx]
        # for i in range(labels.shape[0]):
        #     if labels[i] == 1:
        #         labels[i] = 0
        #     if labels[i] == 2:
        #         labels[i] = 1
        # for i in range(fdata.shape[0]):
        #     max_t = np.max(fdata[i])
        #     min_t = np.min(fdata[i])
        #     fdata[i] = MaxMinNormalization(fdata[i], max_t, min_t)
        # net_all = []
        # for i in range(fdata.shape[0]):
        #     net = np.corrcoef(fdata[i])
        #     net_all.append(net)
        # net_all = np.array(net_all)



        # 对应打乱数据集
        index = [i for i in range(fdata.shape[0])]
        np.random.shuffle(index)
        fdata = fdata[index]
        net_all = net_all[index]
        labels = labels[index]


        class Dianxian(Dataset):
            def __init__(self):
                super(Dianxian, self).__init__()
                self.feas = fdata
                self.nets = net_all
                self.label = labels

            def __getitem__(self, item):
                fea = self.feas[item]
                net = self.nets[item]
                label = self.label[item]
                return fea, net, label

            def __len__(self):
                return self.feas.shape[0]


        DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        # 获取数据集、划分数据集，开始训练和测试
        i = 0
        test_acc = []
        test_pre = []
        test_recall = []
        test_f1 = []
        test_auc = []
        test_sens = []
        test_spec = []
        pro_ten = []
        label_ten = []
        kk = 10
        dataset = Dianxian()
        dataset_best = Dianxian()

        KF = KFold(n_splits=10, shuffle=True, random_state=seed)
        for train_idx, test_idx in KF.split(dataset):
            train_subsampler = SubsetRandomSampler(train_idx)
            test_sunsampler = SubsetRandomSampler(test_idx)
            train_size = len(train_idx)
            test_size = len(test_idx)
            full_size = train_size + test_size
            datasets_all = DataLoader(dataset, batch_size=full_size, shuffle=False)
            datasets_train = DataLoader(dataset, batch_size=int(train_size/4), shuffle=False, sampler=train_subsampler, drop_last=True)
            datasets_test = DataLoader(dataset, batch_size=test_size, shuffle=False, sampler=test_sunsampler, drop_last=True)
            epoch = 200
            losses = []  # 记录训练误差，用于作图分析
            acces = []
            eval_losses = []
            eval_acces = []
            patiences = 500
            min_acc = 0
            best_acc = 0

            # criterion = nn.CrossEntropyLoss()
            model = Network(197, 0.5, 2)
            model.to(DEVICE)
            optimizer = torch.optim.Adam(model.parameters(), lr=lr1, weight_decay=wd)  # 0.005
            # scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.98)
            for e in range(epoch):
                train_loss = 0
                train_cro_loss = 0
                train_con_loss = 0
                train_acc = 0
                model.train()
                for ot_net, cheb, label in datasets_train:
                    edge_list = []
                    for uu in range(cheb.size(0)):
                        adj = cheb[uu]
                        row, col = torch.nonzero(adj, as_tuple=True)
                        edges = torch.stack((row + i * 90, col + i * 90), dim=0)
                        edge_list.append(edges)

                    x = ot_net.reshape(-1, 197)
                    edge_index = torch.cat(edge_list, dim=1)
                    edge_attr = torch.ones(edge_index.size(1), 1)
                    batch = torch.repeat_interleave(torch.arange(ot_net.size(0)), 90)
                    pos = torch.stack([torch.diag(torch.ones(90)) for _ in range(cheb.size(0))])
                    pos = pos.view(-1, 90)

                    x, edge_index, edge_attr, batch, pos = x.to(DEVICE), edge_index.to(DEVICE), edge_attr.to(DEVICE), batch.to(DEVICE), pos.to(DEVICE)
                    x = x.float()
                    edge_attr = edge_attr.float()
                    pos = pos.float()



                    ot_net, cheb, label = ot_net.to(DEVICE), cheb.to(DEVICE), label.to(DEVICE)
                    # 前向传播
                    ot_net = ot_net.float()
                    cheb = cheb.float()
                    label = label.long()

                    out, _ = model(x, edge_index, batch, edge_attr,pos)
                    cro_loss = F.nll_loss(out, label)
                    # con_loss = get_pro_loss(fea, label)


                    loss = cro_loss

                    # 反向传播
                    optimizer.zero_grad()

                    with torch.autograd.detect_anomaly():
                        loss.backward()
                    # 梯度
                    nn.utils.clip_grad_norm(model.parameters(), max_norm=4.0)
                    optimizer.step()
                    train_loss += float(loss)
                    train_cro_loss += float(cro_loss)
                    _, pred = out.max(1)
                    num_correct = (pred == label).sum()
                    acc = num_correct / ot_net.shape[0]
                    train_acc += acc

                losses.append(train_loss / len(datasets_train))
                acces.append(train_acc / len(datasets_train))

                eval_loss, eval_acc, eval_acc_epoch, precision, recall, f1, my_auc, sensitivity, specificity, pre_all, labels_all, pro_all,_ = stest(
                    model,
                    datasets_test)
                if (eval_acc_epoch > min_acc) & (eval_acc_epoch <= 1):
                    torch.save(model.state_dict(), f'./save/model{i}.pth')
                    print("Model saved at epoch{}".format(e))

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
                #     print('Eval Loss: {:.6f}, Eval Acc: {:.6f}'
                #           .format(eval_loss / len(datasets_test), eval_acc / len(datasets_test)))
                # '''
                print(
                    'i:{},epoch: {}, Train Loss: {:.6f}, cro Loss: {:.6f}, con Loss: {:.6f}, Train Acc: {:.6f}, Eval Loss: {:.6f}, Eval Acc: {:.6f},precision : {'
                    ':.6f},recall : {:.6f},f1 : {:.6f},my_auc : {:.6f} '
                    .format(i, e, train_loss / len(datasets_train), train_cro_loss / len(datasets_train), train_con_loss / len(datasets_train), train_acc / len(datasets_train),
                            eval_loss / len(datasets_test), eval_acc_epoch, precision, recall, f1, my_auc))


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

        print('lr', lr1, 'wd', wd,
              "test_acc",
              test_acc, file=log)
        print('lr', lr1, 'wd', wd,
              "test_pre",
              test_pre, file=log)
        print('lr', lr1, 'wd', wd,
              "test_recall",
              test_recall, file=log)
        print('lr', lr1, 'wd', wd,
              "test_f1",
              test_f1, file=log)
        print('lr', lr1, 'wd', wd,
              "test_auc",
              test_auc, file=log)
        print('lr', lr1, 'wd', wd,
              "test_sens",
              test_sens, file=log)
        print('lr', lr1, 'wd', wd,
              "test_spec",
              test_spec, file=log)
        avg_acc = sum(test_acc) / k
        avg_pre = sum(test_pre) / k
        avg_recall = sum(test_recall) / k
        avg_f1 = sum(test_f1) / k
        avg_auc = sum(test_auc) / k
        avg_sens = sum(test_sens) / k
        avg_spec = sum(test_spec) / k
        print("*****************************************************", file=log)
        print('lr', lr1, 'wd', wd,
              'acc', avg_acc,
              file=log)
        print('lr', lr1, 'wd', wd,
              'pre', avg_pre,
              file=log)
        print('lr', lr1, 'wd', wd,
              'recall',
              avg_recall, file=log)
        print('lr', lr1, 'wd', wd,
              'f1', avg_f1,
              file=log)
        print('lr', lr1, 'wd', wd,
              'auc', avg_auc,
              file=log)
        print('lr', lr1, 'wd', wd,
              "sensitivity",
              avg_sens, file=log)
        print('lr', lr1, 'wd', wd,
              "specificity",
              avg_spec, file=log)

        acc_std = np.sqrt(np.var(test_acc))
        pre_std = np.sqrt(np.var(test_pre))
        recall_std = np.sqrt(np.var(test_recall))
        f1_std = np.sqrt(np.var(test_f1))
        auc_std = np.sqrt(np.var(test_auc))
        sens_std = np.sqrt(np.var(test_sens))
        spec_std = np.sqrt(np.var(test_spec))
        print("*****************************************************", file=log)
        print('lr', lr1, 'wd', wd,
              "acc_std",
              acc_std, file=log)
        print('lr', lr1, 'wd', wd,
              "pre_std",
              pre_std, file=log)
        print('lr', lr1, 'wd', wd,
              "recall_std",
              recall_std, file=log)
        print('lr', lr1, 'wd', wd,
              "f1_std",
              f1_std, file=log)
        print('lr', lr1, 'wd', wd,
              "auc_std",
              auc_std, file=log)
        print('lr', lr1, 'wd', wd,
              "sens_std",
              sens_std, file=log)
        print('lr', lr1, 'wd', wd,
              "spec_std",
              spec_std, file=log)
        print("*****************************************************", file=log)

        print('lr', lr1, 'wd', wd,
              label_ten,
              file=log)
        print('lr', lr1, 'wd', wd,
              pro_ten,
              file=log)
        print("*****************************************************", file=log)
        print("\n"*3, file=log)