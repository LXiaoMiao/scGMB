import numpy as np
from matplotlib import pyplot as plt
import matplotlib.patches as mpatches
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib.lines import Line2D
from sklearn.manifold import TSNE
from sklearn.metrics import confusion_matrix, roc_curve, auc
from sklearn.preprocessing import label_binarize
import seaborn as sns


def pic(all_features, label_mapping, all_labels):
    tsne = TSNE(n_components=2, random_state=0)
    features_2d = tsne.fit_transform(all_features)

    # 将label_mapping的键值对翻转以便通过编码值找到名称
    inv_label_mapping = {v: k for k, v in label_mapping.items()}
    unique_labels = np.unique(all_labels)
    num_labels = len(unique_labels)
    colors = plt.cm.tab20(np.linspace(0, 1, num_labels))  # 使用鲜艳的'tab20'色彩方案

    # 绘制t-SNE散点图
    plt.figure(figsize=(8, 6))
    for idx, label in enumerate(unique_labels):
        mask = np.array(all_labels) == label
        cell_type_name = inv_label_mapping[label]  # 获取细胞类型名称
        plt.scatter(
            features_2d[mask, 0], features_2d[mask, 1],
            color=colors[idx],
            label=f"{cell_type_name}",  # 显示细胞名称和编号
            alpha=0.7,
            s=50,
            edgecolors='w', linewidth=0.2
        )

    # 添加图例
    legend_handles = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor=colors[idx], markersize=8,
               label=f"{inv_label_mapping[label]}")
        for idx, label in enumerate(unique_labels)
    ]

    # 添加图例
    plt.legend(handles=legend_handles, bbox_to_anchor=(1, 1), loc='upper left', frameon=False, handletextpad=0)

    # 设置图标题和轴标签
    plt.title('Completed Data T-SNE Visualization', fontsize=14)
    plt.xlabel('tSNE-1', fontsize=12)
    plt.ylabel('tSNE-2', fontsize=12)
    plt.grid(False)
    plt.tight_layout(rect=[0, 0, 1, 1])
    plt.show()


def confusion(all_labels, all_predicted, label_encoder):
    cm = confusion_matrix(all_labels, all_predicted)

    # 绘制混淆矩阵
    plt.figure(figsize=(8, 6))
    ax = sns.heatmap(
        cm, annot=True, fmt='d', cmap='Blues',
        xticklabels=range(len(cm)), yticklabels=range(len(cm)),
        cbar=True  # 显示色条
    )
    ax.set_xlabel('Predicted label')
    ax.set_ylabel('True label')
    ax.set_title('AMB Confusion Matrix')

    # 为右侧色条添加黑色边框
    colorbar = ax.collections[0].colorbar
    colorbar.outline.set_edgecolor('black')
    colorbar.outline.set_linewidth(1)

    # 为混淆矩阵的外框添加黑色边框
    ax.patch.set_edgecolor('black')  # 设置外框颜色
    ax.patch.set_linewidth(1)  # 设置外框线宽

    plt.tight_layout()
    plt.show()


def roc(all_labels, label_mapping, all_probabilities):
    all_labels_onehot = label_binarize(all_labels, classes=range(len(label_mapping)))
    class_names = list(label_mapping.keys())

    fig, ax_main = plt.subplots(figsize=(9, 7))
    for i in range(len(label_mapping)):
        fpr, tpr, _ = roc_curve(all_labels_onehot[:, i], [prob[i] for prob in all_probabilities])
        roc_auc = auc(fpr, tpr)
        ax_main.plot(fpr, tpr, label=f'{class_names[i]} ({roc_auc:.4f})')

    # 绘制参考线
    ax_main.plot([0, 1], [0, 1], color='gray', linestyle='--', label='Random Guess')
    ax_main.set_xlabel('False Positive Rate')
    ax_main.set_ylabel('True Positive Rate')
    ax_main.set_title('Baron Mouse ROC Curves')
    ax_main.legend(loc='lower right')

    # 定义插入图位置和大小
    ax_inset = inset_axes(
        ax_main,
        width="40%",
        height="40%",
        loc="center",
        bbox_to_anchor=(-0.1, 0.1, 1, 1),  # 调整 x 和 y 偏移量
        bbox_transform=ax_main.transAxes  # 使用主图的坐标系
    )

    # 在插入图中重新绘制选定的曲线（可以选择多个类别）
    for i in range(len(label_mapping)):
        fpr, tpr, _ = roc_curve(all_labels_onehot[:, i], [prob[i] for prob in all_probabilities])
        if i in range(len(label_mapping)):  # 选择放大的类别，例如类别 0, 1, 2
            ax_inset.plot(fpr, tpr, label=f'{class_names[i]}')

    # 放大插入图的某个区域
    ax_inset.set_xlim(-0.01, 0.2)
    ax_inset.set_ylim(0.7, 1.02)
    ax_inset.set_xticks([0.0, 0.1, 0.2])
    ax_inset.set_yticks([0.7, 0.8, 0.9, 1.0])

    # 添加参考线
    ax_inset.plot([0, 1], [0, 1], color='gray', linestyle='--')

    plt.show()
