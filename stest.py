import torch
from torch import nn
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, accuracy_score, confusion_matrix


def stest(args, model, datasets_test):
    eval_loss = 0
    eval_acc = 0
    pre_all = []
    labels_all = []
    gailv_all = []
    pro_all = []
    out_all = []
    adj_sum = torch.zeros((116, 116), device=args.device)
    L_adj_sum = torch.zeros((116, 116), device=args.device)
    batch_count = 0
    model.eval()
    for data_test, label_test in datasets_test:
        batch_num_test = len(data_test)
        data_test = data_test.to(args.device)
        label_test = label_test.long().to(args.device)

        output, g_loss,time_loss, adj, L_adj, out = model(data_test, args)
        # print(output)
        losss = args.cross_loss * nn.CrossEntropyLoss()(output, label_test)+args.graph_loss*g_loss+args.time_loss*time_loss
        eval_loss += float(losss)
        _, pred = torch.max(output, dim=1)
        out_all.append(out.cpu().detach().numpy())  # 保存out_logits
        num_correct = (pred == label_test).sum()
        acc = int(num_correct) / batch_num_test
        eval_acc += acc
        pre = pred.cpu().detach().numpy()
        pre_all.extend(pre)
        label_true = label_test.cpu().detach().numpy()
        labels_all.extend(label_true)
        pro_all.extend(output[:, 1].cpu().detach().numpy())
        # 累加 adj 和 L_adj
        adj_sum += adj
        L_adj_sum += L_adj
        batch_count += 1


    tn, fp, fn, tp = confusion_matrix(labels_all, pre_all).ravel()
    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)
    eval_acc_epoch = accuracy_score(labels_all, pre_all)
    precision = precision_score(labels_all, pre_all)
    recall = recall_score(labels_all, pre_all)
    f1 = f1_score(labels_all, pre_all)
    my_auc = roc_auc_score(labels_all, pro_all)
    # 计算平均的 adj 和 L_adj
    avg_adj = adj_sum / batch_count
    avg_L_adj = L_adj_sum / batch_count

    return eval_loss, eval_acc, eval_acc_epoch, precision, recall, f1, my_auc, sensitivity, specificity, pre_all, labels_all, pro_all, avg_adj, avg_L_adj, out_all