import warnings
import numpy as np
import random
import torch
from matplotlib import pyplot as plt
from sklearn.manifold import TSNE
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, accuracy_score, confusion_matrix
from torch.utils.data import DataLoader, SubsetRandomSampler
from read_dataset import ADNI, PD, ABIDE
from args import *
from sklearn.model_selection import KFold
from model.Model import *
from stest import stest
torch.backends.cudnn.deterministic = True
warnings.filterwarnings("ignore")

def accuracy(output, labels):
    _, pred = torch.max(output, dim=1)
    # print(output)
    correct = pred.eq(labels)
    # print(pred)
    # print(labels)
    acc_num = correct.sum()

    return acc_num

args = parse_args()
# 更新窗口大小
args.window_size = args.time_length // args.window_num
# 设定种子
np.random.seed(args.seed)
random.seed(args.seed)
os.environ['PYTHONHASHSEED'] = str(args.seed)  # 为了禁止hash随机化，使得实验可复现。
torch.manual_seed(args.seed)  # 为CPU设置随机种子
torch.cuda.manual_seed(args.seed)  # 为当前GPU设置随机种子（只用一块GPU）
torch.cuda.manual_seed_all(args.seed)  # 为所有GPU设置随机种子（多块GPU）

# 读取数据集
if args.dataset == 'ADNI':
    dataset, num_classes = ADNI(args)
if args.dataset == 'PD':
    dataset, num_classes = PD(args)
if args.dataset == 'ABIDE':
    dataset, num_classes = ABIDE(args)

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
print("mission is: ", args.mission)
for train_and_val_indices, test_indices in KF.split(dataset):
    print("*******{}-flod*********".format(i + 1))

    # 将数据集进一步划分成训练验证测试
    train_size = int(train_ratio * len(train_and_val_indices))
    valid_size = len(train_and_val_indices) - train_size
    train_indices, val_indices = train_and_val_indices[:train_size], train_and_val_indices[train_size:]

    # 提取训练、验证和测试的数据集
    datasets_train = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, sampler=SubsetRandomSampler(train_indices))
    datasets_valid = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, sampler=SubsetRandomSampler(val_indices))
    datasets_test = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, sampler=SubsetRandomSampler(test_indices))

    model = Model(args)
    model = model.to(args.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    closs = nn.CrossEntropyLoss()

    losses = []
    acces = []
    eval_losses = []
    eval_acces = []
    patiences = args.patience
    min_acc = 0

    for epoch in range(args.epochs):
        model.train()
        train_loss = 0
        train_acc = 0
        for data, label in datasets_train:
            batch_num_train = len(data)
            data = data.to(args.device)
            label = label.long().to(args.device)

            output, g_loss, time_loss, adj, L_adj, _ = model(data, args)
            batch_loss_train = args.cross_loss * closs(output, label) + args.graph_loss * g_loss + args.time_loss * time_loss

            optimizer.zero_grad()
            batch_loss_train.backward()
            optimizer.step()
            # torch.cuda.empty_cache()
            acc_num = accuracy(output, label)
            train_acc += acc_num / batch_num_train
            train_loss += batch_loss_train

        losses.append(train_loss / len(datasets_train))
        acces.append(train_acc / len(datasets_train))

        eval_loss, eval_acc, eval_acc_epoch, precision, recall, f1, my_auc, sensitivity, specificity, pre_all, labels_all, pro_all, avg_adj, avg_L_adj, out_all = stest(
            args, model, datasets_test)
        if eval_acc_epoch > min_acc:
            min_acc = eval_acc_epoch
            torch.save(model.state_dict(), './Save/model/latest_'+str(args.mission)+'-' + str(i) + '.pth')
            avg_adj_np = avg_adj.detach().cpu().numpy()
            avg_L_adj_np = avg_L_adj.detach().cpu().numpy()
            # 保存邻接矩阵为npy文件
            if args.save_adj == True:
                np.save('./Save/Adj/'+str(args.mission)+'/adj_' + str(i) + '.npy', avg_adj_np)
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

    model_test = Model(args)
    model_test = model_test.to(args.device)
    model_test.load_state_dict(torch.load('./Save/model/latest_'+str(args.mission)+'-' + str(i) + '.pth'))
    eval_loss, eval_acc, eval_acc_epoch, precision, recall, f1, my_auc, sensitivity, specificity, pre_all, labels_all, pro_all, avg_adj, avg_L_adj, out_all = stest(
        args, model_test, datasets_test)

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