import pandas as pd
import torch
from sklearn.preprocessing import LabelEncoder
import torch.nn as nn
import torch.optim as optim
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data
from sklearn.model_selection import train_test_split
from torch_geometric.utils import subgraph

from scMamba import Mamba, ModelArgs
from edge import preprocess3


def adjust_edge_index(batch_data):
    # 使用 batch 的索引来重新映射 edge_index
    edge_index, edge_attr = batch_data.edge_index, batch_data.edge_attr
    batch = batch_data.batch

    # 获取当前批次的节点索引
    mask = torch.unique(batch)
    edge_index, edge_attr = subgraph(mask, edge_index, edge_attr=edge_attr, relabel_nodes=True)

    return edge_index, edge_attr


best = 0

# file_path = r'F:\Intra-dataset\Pancreatic_data\Baron Human\Filtered_Baron_HumanPancreas_data.csv'
# file_path = r'F:\Intra-dataset\Pancreatic_data\Baron Mouse\Filtered_MousePancreas_data.csv'
# file_path = r'F:\Intra-dataset\Zheng sorted\Filtered_DownSampled_SortedPBMC_data.csv'
# file_path = r'F:\Intra-dataset\Zheng 68K\Filtered_68K_PBMC_data.csv'
file_path = r'F:\Intra-dataset\AMB\Filtered_mouse_allen_brain_data.csv'
# labels_path = r'F:\Intra-dataset\Pancreatic_data\Baron Human\Labels.csv'
# labels_path = r'F:\Intra-dataset\Pancreatic_data\Baron Mouse\Labels.csv'
# labels_path = r'F:\Intra-dataset\Zheng sorted\Labels.csv'
# labels_path = r'F:\Intra-dataset\Zheng 68K\Labels.csv'
labels_path = r'F:\Intra-dataset\AMB\Labels.csv'

labels = pd.read_csv(labels_path)
label_encoder = LabelEncoder()
# integer_labels = label_encoder.fit_transform(labels['x'])  # other
integer_labels = label_encoder.fit_transform(labels['Subclass'])  # AMB

integer_labels_df = pd.DataFrame(integer_labels, columns=['cell_type_int'])
labels_tensor = torch.tensor(integer_labels_df['cell_type_int'].values, dtype=torch.long)
# print(labels_tensor.shape)

label_mapping = dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))

data = preprocess3(file_path)
data.y = labels_tensor
print(data)

print("GO")

train_mask, test_mask = train_test_split(range(len(data.y)), test_size=0.2, stratify=data.y)

# 训练集和测试集的数据加载器
train_data = Data(x=data.x[train_mask], edge_index=data.edge_index, y=data.y[train_mask])
test_data = Data(x=data.x[test_mask], edge_index=data.edge_index, y=data.y[test_mask])

# 创建训练集和测试集的 DataLoader
train_loader = DataLoader([train_data], batch_size=len(train_mask), shuffle=True)
test_loader = DataLoader([test_data], batch_size=len(test_mask), shuffle=False)


# 创建模型参数和模型
args = ModelArgs(
    gcn_hidden_dim=512,
    d_model=128,
    n_layer=2,
    num_genes=data.x.shape[1],
    num_classes=len(torch.unique(labels_tensor))
)

model = Mamba(args)
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

# 训练模型
def train():
    model.train()
    total_loss = 0
    for batch in train_loader:
        optimizer.zero_grad()
        batch.edge_index, batch.edge_attr = adjust_edge_index(batch)
        out = model(batch)
        loss = criterion(out, batch.y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(train_loader)


# 测试函数（改进）
def evacuate(loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():  # 避免在测试时计算梯度
        for batch in loader:
            # 调整 edge_index 和 edge_attr 为当前批次
            batch.edge_index, batch.edge_attr = adjust_edge_index(batch)

            out = model(batch)
            pred = out.argmax(dim=1)
            correct += (pred == batch.y).sum().item()
            total += batch.y.size(0)  # 计算当前 batch 的样本数量
    return correct / total  # 返回准确率


# 训练过程
for epoch in range(1, 300):
    train_loss = train()
    test_acc = evacuate(test_loader)
    print(f'Epoch: {epoch}, Loss: {train_loss:.4f}, Test Accuracy: {test_acc:.4f}')
