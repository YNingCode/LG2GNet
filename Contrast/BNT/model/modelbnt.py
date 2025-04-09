import torch
from torch.nn import TransformerEncoderLayer
from torch import Tensor
from typing import Optional
import torch.nn.functional as F

class InterpretableTransformerEncoder(TransformerEncoderLayer):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation=F.relu,
                 layer_norm_eps=1e-5, batch_first=False, norm_first=False,
                 device=None, dtype=None) -> None:
        super().__init__(d_model, nhead, dim_feedforward, dropout, activation,
                         layer_norm_eps, batch_first, norm_first, device, dtype)
        self.attention_weights: Optional[Tensor] = None

    def _sa_block(self, x: Tensor,
                  attn_mask: Optional[Tensor], key_padding_mask: Optional[Tensor], is_causal: Optional[bool] = False) -> Tensor:
        x, weights = self.self_attn(x, x, x,
                                    attn_mask=attn_mask,
                                    key_padding_mask=key_padding_mask,
                                    need_weights=True)
        self.attention_weights = weights
        return self.dropout1(x)

    def get_attention_weights(self) -> Optional[Tensor]:
        return self.attention_weights  # torch.Size([20, 90, 90])


# x = torch.randn(20, 90, 90)
#
# model = InterpretableTransformerEncoder(d_model=90, nhead=3, dim_feedforward=90, batch_first=True)
# out = model(x)
# print(out.shape)


# cheb_polynomials = torch.randn(1, 6, 90, 90).cuda()
# sample_shape = torch.randn(1, 3, 90, 90).cuda()
# model = STGCN_model(3, 6, 45, 10, 1, 3, 90, 90, 1024, 256, 2).cuda()
# #
# # total = sum([param.nelement() for param in model.parameters()])
# # print('Number of parameter: %.2fM '% (total/1e6))
# #
# flops, params = profile(model, (cheb_polynomials, sample_shape,))
# print('flops: ', flops, 'params: ', params)
# print('flops: %.3f M, params: %.3f M' % (flops / 1000000.0, params / 1000000.0))
