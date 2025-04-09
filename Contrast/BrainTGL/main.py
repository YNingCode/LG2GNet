import warnings
import numpy as np
import random
import torch
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, accuracy_score, confusion_matrix
from torch.utils.data import DataLoader, SubsetRandomSampler
from read_dataset import ADNI, PD
from args import *
from sklearn.model_selection import KFold
from model import *
torch.backends.cudnn.deterministic = True
warnings.filterwarnings("ignore")
seed = 8
# 设定种子
np.random.seed(seed)
random.seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)  # 为了禁止hash随机化，使得实验可复现。
torch.manual_seed(seed)  # 为CPU设置随机种子
torch.cuda.manual_seed(seed)  # 为当前GPU设置随机种子（只用一块GPU）
torch.cuda.manual_seed_all(seed)  # 为所有GPU设置随机种子（多块GPU）


batch_size = 64
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
lr = 0.003
weight_decay = 3e-3
epochs = 200

input_dim = 116  # Number of brain regions

    # ADNI
# window_length = 19
# stride = 2
# sub_seq_length = 200
    # PD
window_length = 105
stride = 1
sub_seq_length = 220  # Length of sub-sequences

hidden_dim = 64
output_dim = 2  # Binary classification
num_gcn_layers = 2
num_lstm_layers = 1
max_skip = 3

def accuracy(output, labels):
    _, pred = torch.max(output, dim=1)
    # print(output)
    correct = pred.eq(labels)
    # print(pred)
    # print(labels)
    acc_num = correct.sum()

    return acc_num

def stest(model, datasets_test):
    eval_loss = 0
    eval_acc = 0
    pre_all = []
    labels_all = []
    gailv_all = []
    pro_all = []
    model.eval()
    for data_test, bold_test, label_test in datasets_test:
        batch_num_test = len(data_test)
        data_test = data_test.to(device)
        bold_test = bold_test.to(device)
        label_test = label_test.squeeze(1).to(device)

        output = model(data_test, bold_test)
        # print(output)
        losss = nn.CrossEntropyLoss()(output, label_test)
        eval_loss += float(losss)
        _, pred = torch.max(output, dim=1)
        num_correct = (pred == label_test).sum()
        acc = int(num_correct) / batch_num_test
        eval_acc += acc
        pre = pred.cpu().detach().numpy()
        pre_all.extend(pre)
        label_true = label_test.cpu().detach().numpy()
        labels_all.extend(label_true)
        pro_all.extend(output[:, 1].cpu().detach().numpy())


    tn, fp, fn, tp = confusion_matrix(labels_all, pre_all).ravel()
    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)
    eval_acc_epoch = accuracy_score(labels_all, pre_all)
    precision = precision_score(labels_all, pre_all)
    recall = recall_score(labels_all, pre_all)
    f1 = f1_score(labels_all, pre_all)
    my_auc = roc_auc_score(labels_all, pro_all)

    return eval_loss, eval_acc, eval_acc_epoch, precision, recall, f1, my_auc, sensitivity, specificity, pre_all, labels_all, pro_all

# 读取数据集
mission = "1_2"
# dataset, num_classes = ADNI(mission, window_length, stride, sub_seq_length)
dataset, num_classes = PD(mission, window_length, stride, sub_seq_length)

dataset_length = len(dataset)
train_ratio = 0.9
valid_ratio = 0.1
kk = 10
test_acc = []
test_pre = []
test_recall = []
test_f1 = []
test_auc = []
label_ten = []
test_sens = []
test_spec = []
pro_ten = []
i = 0
KF = KFold(n_splits=10, shuffle=True)
labels = [dataset[i][1] for i in range(dataset_length)]
print("mission is: ", mission)
for train_and_val_indices, test_indices in KF.split(dataset):
    print("*******{}-flod*********".format(i + 1))

    # 将数据集进一步划分成训练验证测试
    train_size = int(train_ratio * len(train_and_val_indices))
    valid_size = len(train_and_val_indices) - train_size
    train_indices, val_indices = train_and_val_indices[:train_size], train_and_val_indices[train_size:]

    # 提取训练、验证和测试的数据集
    datasets_train = DataLoader(dataset, batch_size=batch_size, shuffle=False, sampler=SubsetRandomSampler(train_indices))
    datasets_valid = DataLoader(dataset, batch_size=batch_size, shuffle=False, sampler=SubsetRandomSampler(val_indices))
    datasets_test = DataLoader(dataset, batch_size=batch_size, shuffle=False, sampler=SubsetRandomSampler(test_indices))

    model = BrainTGL(input_dim, hidden_dim, output_dim, num_gcn_layers, num_lstm_layers, max_skip)
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    closs = nn.CrossEntropyLoss()

    losses = []
    acces = []
    eval_losses = []
    eval_acces = []
    patiences = 500
    min_acc = 0

    for epoch in range(epochs):
        model.train()
        train_loss = 0
        train_acc = 0
        for data, bold, label in datasets_train:
            batch_num_train = len(data)
            data = data.to(device)
            bold = bold.to(device)
            label = label.squeeze(1).to(device)

            output = model(data, bold)
            batch_loss_train = closs(output, label)

            optimizer.zero_grad()
            batch_loss_train.backward()
            optimizer.step()
            # torch.cuda.empty_cache()
            acc_num = accuracy(output, label)
            train_acc += acc_num / batch_num_train
            train_loss += batch_loss_train

        losses.append(train_loss / len(datasets_train))
        acces.append(train_acc / len(datasets_train))

        eval_loss, eval_acc, eval_acc_epoch, precision, recall, f1, my_auc, sensitivity, specificity, pre_all, labels_all, pro_all = stest(
            model, datasets_test)
        if eval_acc_epoch > min_acc:
            min_acc = eval_acc_epoch
            torch.save(model.state_dict(), './Save/latest_1_' + str(i) + '.pth')
            print("Model saved at epoch{}".format(epoch))

            patience = 0
        else:
            patience += 1
        if patience > patiences:
            break
        eval_losses.append(eval_loss / len(datasets_test))
        eval_acces.append(eval_acc / len(datasets_test))

        print(
            'i:{},epoch: {}, Train Loss: {:.6f}, Train Acc: {:.6f}, Eval Loss: {:.6f}, Eval Acc: {:.6f},precision : {'
            ':.6f},recall : {:.6f},f1 : {:.6f},my_auc : {:.6f} '
            .format(i, epoch, train_loss / len(datasets_train), train_acc / len(datasets_train),
                    eval_loss / len(datasets_valid), eval_acc_epoch, precision, recall, f1, my_auc))

    model_test = BrainTGL(input_dim, hidden_dim, output_dim, num_gcn_layers, num_lstm_layers, max_skip)
    model_test = model_test.to(device)
    model_test.load_state_dict(torch.load('./Save/latest_1_' + str(i) + '.pth'))
    eval_loss, eval_acc, eval_acc_epoch, precision, recall, f1, my_auc, sensitivity, specificity, pre_all, labels_all, pro_all = stest (
        model_test, datasets_test)

    test_acc.append(eval_acc_epoch)
    test_pre.append(precision)
    test_recall.append(recall)
    test_f1.append(f1)
    test_auc.append(my_auc)
    test_sens.append(sensitivity)
    test_spec.append(specificity)
    label_ten.extend(labels_all)
    pro_ten.extend(pro_all)
    i = i + 1

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