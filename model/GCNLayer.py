import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data


class MyGCN(torch.nn.Module):
    def __init__(self, num_node_features, gcn_hidden_dim, gcn_output_dim):
        super(MyGCN, self).__init__()
        self.conv1 = GCNConv(num_node_features, gcn_hidden_dim)
        self.conv2 = GCNConv(gcn_hidden_dim, gcn_output_dim)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        return x


def filter_cell_edges(edge_index, num_cells):
    """
    过滤 `edge_index` 以只保留细胞之间的边。
    """
    mask = (edge_index[0] < num_cells) & (edge_index[1] < num_cells)
    return edge_index[:, mask]


class MyGCN1(torch.nn.Module):
    def __init__(self, num_node_features, gcn_hidden_dim, gcn_output_dim):
        super(MyGCN1, self).__init__()
        self.conv1 = GCNConv(num_node_features, gcn_hidden_dim)
        self.conv2 = GCNConv(gcn_hidden_dim, gcn_output_dim)

    def forward(self, data):
        # 获取细胞之间的边
        edge_index = filter_cell_edges(data.edge_index, num_cells=2715)
        x = data.x[:2715]  # 只选择前 2715 个细胞的特征
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        return x


class MyGCN3(torch.nn.Module):
    def __init__(self, num_node_features, gcn_hidden_dim, gcn_output_dim):
        super(MyGCN3, self).__init__()
        # 定义两层GCN卷积层
        self.conv1 = GCNConv(num_node_features, gcn_hidden_dim)
        self.conv2 = GCNConv(gcn_hidden_dim, gcn_output_dim)

    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr

        # 第一层GCN卷积，使用边权重进行加权卷积
        x = self.conv1(x, edge_index, edge_weight=edge_attr)
        x = F.relu(x)  # 激活函数

        # 第二层GCN卷积
        x = self.conv2(x, edge_index, edge_weight=edge_attr)
        return x

