import pandas as pd
import torch
from sklearn.preprocessing import LabelEncoder
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import precision_recall_fscore_support
from joblib import dump, load
from torch.utils.data import DataLoader, TensorDataset, random_split

from model.scMamba import Mamba, ModelArgs

file_path = r'..\Intra-dataset\Zheng 68K\Filtered_68K_PBMC_data.csv'
labels_path = r'..\Intra-dataset\Zheng 68K\Labels.csv'

labels = pd.read_csv(labels_path)
label_encoder = LabelEncoder()
integer_labels = label_encoder.fit_transform(labels['x'])

integer_labels_df = pd.DataFrame(integer_labels, columns=['cell_type_int'])
labels_tensor = torch.tensor(integer_labels_df['cell_type_int'].values, dtype=torch.long)
print(labels_tensor.shape)
print(labels_tensor[:30000].shape)

label_mapping = dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))
print(label_mapping)

# 直接读一整个
# data = pd.read_csv(file_path, index_col=0)
# features = data.iloc[30000:, :].values
# print(features.shape)
# print(features)
# 分块读取数据
chunk_size = 30000  # 根据你的内存调整块大小
chunks = []
i = 0
model_filename = 'model.joblib'  # 模型保存文件名
for chunk in pd.read_csv(file_path, index_col=0, chunksize=chunk_size):
    print(chunk.shape)
    print(chunk.head())
    i += 1
    if i == 3:
        # chunks.append(chunk)

        # 将所有块合并为一个 DataFrame
        # data = pd.concat(chunks, axis=0)

        # print(data.shape)
        print("开始了")
        features = chunk.values
        features_tensor = torch.tensor(features, dtype=torch.float32)

        # 创建 TensorDataset
        # dataset = TensorDataset(features_tensor, labels_tensor[chunk_size * i: chunk_size * i + len(chunk)*(i+1)])
        dataset = TensorDataset(features_tensor, labels_tensor[60000:])

        # 划分数据集
        num_train = int(0.8 * len(dataset))  # 80% 训练数据
        num_test = len(dataset) - num_train  # 20% 测试数据
        print(num_train, 'train samples')
        print(num_test, 'test samples')
        train_dataset, test_dataset = random_split(dataset, [num_train, num_test])
        print(train_dataset, test_dataset)
        # 创建 DataLoader
        train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
        # 检查 CUDA 是否可用
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("Using device:", device)
        #
        labels_num = 11
        #
        # 创建模型参数和模型
        args = ModelArgs(
            d_model=128,
            n_layer=2,
            num_genes=features.shape[1],
            num_classes=labels_num
        )
        # model = Mamba(args)
        model = load(model_filename)
        # 将模型移动到 GPU
        model = model.to(device)

        # 定义损失函数和优化器
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)

        # 训练模型
        model.train()
        num_epochs = 20
        for epoch in range(num_epochs):
            for batch_features, batch_labels in train_loader:
                # 将数据移动到 GPU
                batch_features = batch_features.to(device)
                batch_labels = batch_labels.to(device)

                optimizer.zero_grad()
                outputs = model(batch_features)
                loss = criterion(outputs, batch_labels)
                loss.backward()
                optimizer.step()
            print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {loss.item()}")
        # 保存模型
        dump(model, model_filename)
        print(f'Model saved to {model_filename} after processing {len(chunk)} samples.')

        # 评估模型
        model.eval()
        total_correct = 0
        total_samples = 0
        all_labels = []
        all_predicted = []
        with torch.no_grad():
            for batch_features, batch_labels in test_loader:
                # 将数据移动到 GPU
                batch_features = batch_features.to(device)
                batch_labels = batch_labels.to(device)

                outputs = model(batch_features)
                _, predicted = torch.max(outputs, 1)
                total_correct += (predicted == batch_labels).sum().item()
                total_samples += batch_labels.size(0)
                # 将结果添加到列表中
                all_labels.extend(batch_labels.cpu().numpy())
                all_predicted.extend(predicted.cpu().numpy())
        accuracy = total_correct / total_samples
        print(f"Accuracy: {accuracy}")

        # 计算精确率、召回率和F1分数
        precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_predicted, average='weighted')
        print(f"Precision: {precision}")
        print(f"Recall: {recall}")
        print(f"F1 Score: {f1}")
