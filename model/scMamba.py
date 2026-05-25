import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from einops import rearrange, repeat, einsum
import math
from typing import Union

from model.GCNLayer import MyGCN3


@dataclass
class ModelArgs:
    # gcn_hidden_dim: int
    d_model: int  # 隐藏层维度
    n_layer: int  # 层数
    num_genes: int  # 基因数量
    num_classes: int  # 分类数量
    d_state: int = 16  # 状态空间维度
    expand: int = 2  # 扩展因子
    dt_rank: Union[int, str] = 'auto'  # dt_rank
    d_conv: int = 4  # 卷积核维度
    pad_vocab_size_multiple: int = 8  # 最小公倍数
    conv_bias: bool = True  # 卷积层偏置
    bias: bool = False  # 其他层偏置

    def __post_init__(self):
        self.d_inner = int(self.expand * self.d_model)
        if self.dt_rank == 'auto':
            self.dt_rank = math.ceil(self.d_model / 16)


class Mamba(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.embedding = nn.Linear(args.num_genes, args.d_model)
        # self.gcn = MyGCN3(args.num_genes, args.gcn_hidden_dim, args.d_model)
        self.layers = nn.ModuleList([ResidualBlock(args) for _ in range(args.n_layer)])
        self.norm_f = RMSNorm(args.d_model)
        self.classifier = nn.Linear(args.d_model, args.num_classes, bias=False)

    def forward(self, x):
        x = self.embedding(x)
        # x = self.gcn(x)
        x = x.unsqueeze(1)  # 增加一个维度，使其与模型预期的输入格式匹配
        for layer in self.layers:
            x = layer(x)
        # print(x, x.shape)
        # x = self.norm_f(x)
        logits = self.classifier(x.mean(dim=1))  # 用均值池化代替序列长度的缩减
        return logits


class ResidualBlock(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.mixer = MambaBlock(args)
        self.norm = RMSNorm(args.d_model)

    def forward(self, x):
        return self.mixer(self.norm(x)) + x


class MambaBlock(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.in_proj = nn.Linear(args.d_model, args.d_inner * 2, bias=args.bias)
        self.conv1d = nn.Conv1d(
            in_channels=args.d_inner,
            out_channels=args.d_inner,
            bias=args.conv_bias,
            kernel_size=args.d_conv,
            groups=args.d_inner,
            padding=args.d_conv - 1,
        )
        self.x_proj = nn.Linear(args.d_inner, args.dt_rank + args.d_state * 2, bias=False)
        self.dt_proj = nn.Linear(args.dt_rank, args.d_inner, bias=True)
        A = repeat(torch.arange(1, args.d_state + 1).float(), 'n -> d n', d=args.d_inner)
        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(args.d_inner))
        self.out_proj = nn.Linear(args.d_inner, args.d_model, bias=args.bias)

    def forward(self, x):
        (b, _, d) = x.shape
        x_and_res = self.in_proj(x)
        (x, res) = x_and_res.split(split_size=[self.args.d_inner, self.args.d_inner], dim=-1)
        x = rearrange(x, 'b l d_in -> b d_in l')
        x = self.conv1d(x)[:, :, :x.shape[2]]
        x = rearrange(x, 'b d_in l -> b l d_in')
        x = F.silu(x)
        y = self.ssm(x)
        y = y * F.silu(res)
        output = self.out_proj(y)
        return output

    def ssm(self, x):
        (d_in, n) = self.A_log.shape
        A = -torch.exp(self.A_log.float())
        D = self.D.float()
        x_dbl = self.x_proj(x)
        (delta, B, C) = x_dbl.split(split_size=[self.args.dt_rank, n, n], dim=-1)
        delta = F.softplus(self.dt_proj(delta))
        y = self.selective_scan(x, delta, A, B, C, D)
        return y

    def selective_scan(self, u, delta, A, B, C, D):
        (b, l, d_in) = u.shape
        n = A.shape[1]
        deltaA = torch.exp(einsum(delta, A, 'b l d_in, d_in n -> b l d_in n'))
        deltaB_u = einsum(delta, B, u, 'b l d_in, b l n, b l d_in -> b l d_in n')
        x = torch.zeros((b, d_in, n), device=deltaA.device)
        ys = []
        for i in range(l):
            x = deltaA[:, i] * x + deltaB_u[:, i]
            y = einsum(x, C[:, i, :], 'b d_in n, b n -> b d_in')
            ys.append(y)
        y = torch.stack(ys, dim=1)
        y = y + u * D
        return y


class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))

    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight


if __name__ == '__main__':
    labels_num = 13
    cells_num = 7204
    genes_num = 17317
    # 创建模型参数
    args = ModelArgs(
        d_model=128,  # 示例隐藏层维度
        n_layer=2,  # 示例层数
        num_genes=genes_num,  # 基因数量
        num_classes=labels_num  # 分类数量
    )

    # 创建模型
    model = Mamba(args)

    # 示例输入
    input_data = torch.randn(cells_num, genes_num)  # 假设有cells_num个细胞，每个细胞有genes_num个基因表达值
    print(input_data.shape)

    # 前向传播
    output = model(input_data)
    print(input_data)
    print(output)
    print(output.shape)  # 输出形状应为 (32, num_classes)，即32个样本对应的num_classes类分类得分
