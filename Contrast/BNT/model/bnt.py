import torch
import torch.nn as nn
from Contrast.BNT.model.basemodel import BaseModel
from Contrast.BNT.model.dec import DEC
from Contrast.BNT.model.modelbnt import InterpretableTransformerEncoder


class TransPoolingEncoder(nn.Module):
    """
    Transformer encoder with Pooling mechanism.
    Input size: (batch_size, input_node_num, input_feature_size)
    Output size: (batch_size, output_node_num, input_feature_size)
    """

    def __init__(self, input_feature_size, input_node_num, hidden_size, output_node_num, pooling=True, orthogonal=True,
                 freeze_center=False, project_assignment=True):
        super().__init__()
        self.transformer = InterpretableTransformerEncoder(d_model=input_feature_size, nhead=3,
                                                           dim_feedforward=hidden_size,
                                                           batch_first=True)

        self.pooling = pooling
        if pooling:
            encoder_hidden_size = 32
            self.encoder = nn.Sequential(
                nn.Linear(input_feature_size *
                          input_node_num, encoder_hidden_size),
                nn.LeakyReLU(),
                nn.Linear(encoder_hidden_size, encoder_hidden_size),
                nn.LeakyReLU(),
                nn.Linear(encoder_hidden_size,
                          input_feature_size * input_node_num),
            )
            self.dec = DEC(cluster_number=output_node_num, hidden_dimension=input_feature_size, encoder=self.encoder,
                           orthogonal=orthogonal, freeze_center=freeze_center, project_assignment=project_assignment)

    def is_pooling_enabled(self):
        return self.pooling

    def forward(self, x):
        x = self.transformer(x)
        if self.pooling:
            x, assignment = self.dec(x)
            return x, assignment
        return x, None

    def get_attention_weights(self):
        return self.transformer.get_attention_weights()

    def loss(self, assignment):
        return self.dec.loss(assignment)

#
# model = TransPoolingEncoder(90,90,90,90)
# x = torch.randn(20,90,90)
# out,_ = model(x)
# print(out.shape)


class BrainNetworkTransformer(BaseModel):

    def __init__(self):

        super().__init__()

        self.attention_list = nn.ModuleList()
        # forward_dim = config.dataset.node_sz
        forward_dim = 90

        # self.pos_encoding = config.model.pos_encoding
        # if self.pos_encoding == 'identity':
        #     self.node_identity = nn.Parameter(torch.zeros(
        #         config.dataset.node_sz, config.model.pos_embed_dim), requires_grad=True)
        #     forward_dim = config.dataset.node_sz + config.model.pos_embed_dim
        #     nn.init.kaiming_normal_(self.node_identity)

        sizes = [360, 90]
        sizes[0] = 90
        in_sizes = [90] + sizes[:-1]
        do_pooling = [False, True]
        self.do_pooling = do_pooling
        for index, size in enumerate(sizes):
            self.attention_list.append(
                TransPoolingEncoder(input_feature_size=forward_dim,
                                    input_node_num=in_sizes[index],
                                    hidden_size=1024,
                                    output_node_num=size,
                                    pooling=do_pooling[index],
                                    orthogonal=True,
                                    freeze_center=True,
                                    project_assignment=True))

        self.dim_reduction = nn.Sequential(
            nn.Linear(forward_dim, 8),
            nn.LeakyReLU()
        )

        self.fc = nn.Sequential(
            nn.Linear(8 * sizes[-1], 256),
            nn.BatchNorm1d(256),
            nn.Dropout(p=0.5),
            nn.Linear(256, 32),
            nn.BatchNorm1d(32),
            nn.Dropout(p=0.5),

        )
        self.ff1 = nn.Linear(32, 2)

    def forward(self,
                time_seires: torch.tensor,
                node_feature: torch.tensor):

        bz, _, _, = node_feature.shape

        # if self.pos_encoding == 'identity':
        #     pos_emb = self.node_identity.expand(bz, *self.node_identity.shape)
        #     node_feature = torch.cat([node_feature, pos_emb], dim=-1)

        assignments = []

        for atten in self.attention_list:
            node_feature, assignment = atten(node_feature)
            assignments.append(assignment)

        node_feature = self.dim_reduction(node_feature)

        node_feature = node_feature.reshape((bz, -1))
        out = self.fc(node_feature)
        tsne = out.detach().cpu().numpy()
        out = self.ff1(out)

        return out,tsne

    def get_attention_weights(self):
        return [atten.get_attention_weights() for atten in self.attention_list]

    def get_cluster_centers(self) -> torch.Tensor:
        """
        Get the cluster centers, as computed by the encoder.

        :return: [number of clusters, hidden dimension] Tensor of dtype float
        """
        return self.dec.get_cluster_centers()

    def loss(self, assignments):
        """
        Compute KL loss for the given assignments. Note that not all encoders contain a pooling layer.
        Inputs: assignments: [batch size, number of clusters]
        Output: KL loss
        """
        decs = list(
            filter(lambda x: x.is_pooling_enabled(), self.attention_list))
        assignments = list(filter(lambda x: x is not None, assignments))
        loss_all = None

        for index, assignment in enumerate(assignments):
            if loss_all is None:
                loss_all = decs[index].loss(assignment)
            else:
                loss_all += decs[index].loss(assignment)
        return loss_all


if __name__ == '__main__':
    model = BrainNetworkTransformer()
    x = torch.randn(20, 90, 90)
    time_series = torch.randn(20, 90, 197)
    out, _ = model(time_series, x)
    print(out.shape)



#
# flops, params = profile(model, (time_series, x,))
# print('flops: ', flops, 'params: ', params)
# print('flops: %.5f M, params: %.5f M' % (flops / 1000000.0, params / 1000000.0))