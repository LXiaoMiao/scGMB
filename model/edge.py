import pandas as pd
import torch
from torch_geometric.data import Data
from sklearn.metrics.pairwise import cosine_similarity


def preprocess(expression_matrix_path):
    # 计数矩阵
    expression_matrix = pd.read_csv(expression_matrix_path, index_col=0)

    # 获取细胞数量和基因数量
    num_cells = expression_matrix.shape[0]
    num_genes = expression_matrix.shape[1]
    # print(num_cells, num_genes)

    node_features = torch.tensor(expression_matrix.values, dtype=torch.float)
    # print(node_features)

    # 创建细胞和基因名称到整数索引的映射
    cell_idx_map = {cell: idx for idx, cell in enumerate(expression_matrix.index)}
    gene_idx_map = {gene: idx + num_cells for idx, gene in enumerate(expression_matrix.columns)}
    # {'AAACCCACAGGTTCGC-1': 0, ..., 'TTTGTTGTCATCACTT-1': 3393} {'MIR1302-2HG': 3394, ..., 'AC007325.2': 39994}
    # print(cell_idx_map, gene_idx_map)

    # 创建加权边索引
    edges = []
    weights = []
    for cell_name, row in expression_matrix.iterrows():
        cell_idx = cell_idx_map[cell_name]
        for gene_name, expr_value in row.items():
            gene_idx = gene_idx_map[gene_name]
            if expr_value > 0:  # 只考虑非零表达值
                edges.append([cell_idx, gene_idx])
                weights.append(expr_value)
                # print([cell_idx, gene_idx], expr_value) [170, 3613] 13

    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    edge_attr = torch.tensor(weights, dtype=torch.float)

    # 创建Data对象
    data = Data(x=node_features, edge_index=edge_index, edge_attr=edge_attr)
    # print(data.edge_index, data.edge_attr)

    return data


def preprocess2(expression_matrix_path):
    # 计数矩阵
    expression_matrix = pd.read_csv(expression_matrix_path, index_col=0)

    # 获取细胞数量和基因数量
    num_cells = expression_matrix.shape[0]
    num_genes = expression_matrix.shape[1]

    # 创建细胞特征 (num_cells, num_genes)
    cell_features = torch.tensor(expression_matrix.values, dtype=torch.float)

    # 创建基因特征，假设基因没有额外的特征，可以用全 0 初始化
    gene_features = torch.zeros((num_genes, num_genes), dtype=torch.float)

    # 将细胞特征和基因特征拼接在一起 (num_cells + num_genes, num_genes)
    node_features = torch.cat([cell_features, gene_features], dim=0)

    # 创建细胞和基因名称到整数索引的映射
    cell_idx_map = {cell: idx for idx, cell in enumerate(expression_matrix.index)}
    gene_idx_map = {gene: idx + num_cells for idx, gene in enumerate(expression_matrix.columns)}

    # 创建加权边索引
    edges = []
    weights = []
    for cell_name, row in expression_matrix.iterrows():
        cell_idx = cell_idx_map[cell_name]
        for gene_name, expr_value in row.items():
            gene_idx = gene_idx_map[gene_name]
            if expr_value > 0:  # 只考虑非零表达值
                edges.append([cell_idx, gene_idx])
                weights.append(expr_value)

    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    edge_attr = torch.tensor(weights, dtype=torch.float)

    # 创建Data对象，x 包含细胞和基因的特征
    data = Data(x=node_features, edge_index=edge_index, edge_attr=edge_attr)

    return data


def preprocess3(expression_matrix_path, threshold=0.1):
    # 读取基因表达矩阵
    expression_matrix = pd.read_csv(expression_matrix_path, index_col=0)

    # 将表达矩阵转换为 PyTorch 张量
    cell_features = torch.tensor(expression_matrix.values, dtype=torch.float)

    # 获取细胞数量和基因数量
    num_cells = expression_matrix.shape[0]

    # 初始化 edge_index 和 edge_attr
    edge_index = []
    edge_attr = []

    # 遍历每对细胞，构建基于非零表达值的边
    for i in range(num_cells):
        print(i/num_cells)
        for j in range(i + 1, num_cells):  # 遍历每对细胞，避免自连接
            # 找到两个细胞在表达矩阵中非零的基因表达值位置
            cell_i_expr = expression_matrix.iloc[i]
            cell_j_expr = expression_matrix.iloc[j]

            # 计算共有的非零表达值（两个细胞都有非零表达的基因）
            common_expr = (cell_i_expr > threshold) & (cell_j_expr > threshold)
            common_expr_sum = (cell_i_expr[common_expr] * cell_j_expr[common_expr]).sum()

            if common_expr_sum > 0:  # 仅为有非零共有基因表达的细胞构建边
                # 记录双向边
                edge_index.append([i, j])
                edge_index.append([j, i])

                # 将共有的表达值加和作为边权重
                edge_attr.append(common_expr_sum)
                edge_attr.append(common_expr_sum)

    # 将边索引和边权重转换为 PyTorch 张量
    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    edge_attr = torch.tensor(edge_attr, dtype=torch.float)

    # 返回构建好的 Data 对象，x 是细胞特征，edge_index 是边，edge_attr 是边的权重
    return Data(x=cell_features, edge_index=edge_index, edge_attr=edge_attr)


