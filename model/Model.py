import torch
import torch.nn as nn
import torch.nn.functional as F
from fvcore.nn import FlopCountAnalysis

from model.Mapper import *
from model.GNN import *
from model.utils import Linear, normalize_A
from model.Loss_function import *
from args import *
from thop import profile
from model.PCC import pearson
import torch.profiler

class Model(nn.Module):
    def __init__(self, args):
        super(Model, self).__init__()
        self.window_size = args.window_size
        self.window_num = args.window_num

        # self.args = args
        # self.args.window_size = self.args.time_length // self.args.window_num

        if args.if_graph == True:
            self.A = nn.Parameter(torch.FloatTensor(args.num_node, args.num_node))
            nn.init.xavier_normal_(self.A)

        if args.if_time_mapper == True:
            self.time_GSL = Windows_Graph_Mapper(node_num=args.num_node, window_num=args.window_num, in_dim=args.num_node * args.num_node)

        self.gnn = GNN(input=args.window_size, num_out=args.gcn_output)

        if args.if_time_graph == True:
            self.fc1 = Linear(args.num_node * args.gcn_output, args.windows_input)
            self.time_adj = nn.Parameter(torch.FloatTensor(args.window_num, args.window_num))
            nn.init.xavier_normal_(self.time_adj)
            self.time_gnn = GNN(input=args.windows_input, num_out=args.windows_output)
            self.fc2 = Linear(args.windows_output*args.window_num, 8)
            self.fc3 = Linear(8, args.num_class)
        elif args.if_time_graph == False:
            # self.fc1 = Linear(args.num_node * args.gcn_output, args.windows_input)
            # self.fc2 = Linear(args.windows_input * args.window_num, 8)
            # self.fc3 = Linear(8, args.num_class)
            self.fc1 = Linear(args.num_node * args.gcn_output, args.windows_output)
            self.fc2 = Linear(args.windows_output, 8)
            self.fc3 = Linear(8, args.num_class)


        self.time_loss = CombinedGraphLoss(alpha=0.1, lambda_smooth=0.1, gamma=0.001)
        self.graph_loss = GraphStructureLoss(lambda_=1.0, phi=1.0, alpha=1.0, gamma=0.001)

    def forward(self, x, args):
        # bs,90,195
        # 划分时间窗口
        window_size = self.window_size
        window_num = self.window_num
        window_data_list = [x[:, :, i * window_size:(i + 1) * window_size]
                            for i in range(window_num)]
        window_data = torch.stack(window_data_list, dim=0)
        win_num, bs, node_num, dim = window_data.shape

        # 窗口邻接矩阵映射器
        if args.if_graph == False:
            global_adj = pearson(x).to(args.device)
            if args.if_time_mapper == True:
                window_adj_list = self.time_GSL(global_adj)
        elif args.if_graph == True:
            if args.if_time_mapper == True:
                window_adj_list = self.time_GSL(self.A)

        # 每个窗口使用global_adj
        global_gcn_outputs = []
        # 消融实验：不使用可学习的邻接矩阵
        if args.if_graph == False:
            for i in range(win_num):
                # 使用全局邻接矩阵进行GCN
                gcn_output = self.gnn(window_data[i], global_adj)
                global_gcn_outputs.append(gcn_output)
        elif args.if_graph == True:
            for i in range(win_num):
                # 使用全局邻接矩阵进行GCN
                gcn_output = self.gnn(window_data[i], self.A)
                global_gcn_outputs.append(gcn_output)
        global_gcn_outputs = torch.stack(global_gcn_outputs, dim=0)  # win, bs, 11520

        # 每个窗口使用window_adj
        if args.if_time_mapper == True:
            windows_gcn_outputs = []
            for i in range(win_num):
                # 使用当前窗口的邻接矩阵进行 GCN
                window_adj = window_adj_list[i]
                gcn_window_output = self.gnn(window_data[i], window_adj)
                windows_gcn_outputs.append(gcn_window_output)
            windows_gcn_outputs = torch.stack(windows_gcn_outputs, dim=0)  # win, bs, 11520
            combined_outputs = global_gcn_outputs + windows_gcn_outputs
        elif args.if_time_mapper == False:
            combined_outputs = global_gcn_outputs

        # 时间图卷积
        if args.if_time_graph == True:
            result = F.relu(self.fc1(combined_outputs))
            result = result.transpose(0,1)
            result = self.time_gnn(result, self.time_adj)     # 20, 384
            out_logits = F.relu(self.fc2(result))
            out_logits = self.fc3(out_logits)
            out = F.softmax(out_logits, dim=1)
        elif args.if_time_graph == False:
            # result = F.relu(self.fc1(combined_outputs))
            # result = result.view(result.size(1), -1)
            # result = F.relu(self.fc2(result))
            # result = self.fc3(result)
            # result = F.softmax(result, dim=1)
            result = torch.sum(combined_outputs, dim=0)
            result = F.relu(self.fc1(result))
            out_logits = F.relu(self.fc2(result))
            out_logits = self.fc3(out_logits)
            out = F.softmax(out_logits, dim=1)


        if args.if_time_graph == True:
            time_loss = self.time_loss(self.time_adj)
        elif args.if_time_graph == False:
            time_loss = 0
        if args.if_graph == True:
            graph_loss = self.graph_loss(self.A)
        elif args.if_graph == False:
            graph_loss = 0

        return out, graph_loss, time_loss, self.A, normalize_A(self.A), result

if __name__ == "__main__":
    args = parse_args()
    # 更新窗口大小
    args.window_size = args.time_length // args.window_num
    model = Model(args).to(args.device)
    sample_shape = torch.randn(20,116,220).to(args.device)
    # output, g_loss, time_loss, adj, L_adj, _ = model(sample_shape, args)

    # 使用 torch.profiler 来分析模型的性能
    with torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
            record_shapes=True,
            profile_memory=True,
            with_stack=True
    ) as prof:
        # 运行前向传播
        output, g_loss, time_loss, adj, L_adj, _ = model(sample_shape, args)

    # 输出分析结果
    prof.export_chrome_trace("trace.json")  # 导出为 Chrome Trace 格式，方便在 Chrome 浏览器中查看
    print(prof.key_averages().table(sort_by="cpu_time_total"))  # 打印按 CPU 时间排序的操作表


    # flops, params = profile(model, (sample_shape, args))
    # print('flops: ', flops, 'params: ', params)
    # print('flops: %.3f M, params: %.3f M' % (flops / 2000000.0, params / 1000000.0))
    # 使用 fvcore 计算 FLOPS 和参数量
    # flops = FlopCountAnalysis(model, sample_shape)
    #
    # # 打印 FLOPS 和参数量
    # print(f"FLOPS: {flops.total() / 1e6}M")
    # print(f"Parameters: {sum(p.numel() for p in model.parameters()) / 1e6}M")  # 参数量（以百万为单位）