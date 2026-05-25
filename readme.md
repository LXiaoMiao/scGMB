====
                                   scGMB
            单细胞 RNA 测序细胞类型分类：GCN + Mamba 状态空间模型
===

一、项目简介
--------------------------------------------------------------------------------
scGMB（single-cell GCN-Mamba）将图卷积网络（GCN）与 Mamba 状态空间模型相结合，
用于单细胞 RNA 测序的细胞类型分类。流程为：先通过 GCN 在细胞-基因图上聚合邻居
信息，再由 Mamba 对表达特征进行序列建模，最后输出细胞类型预测。

二、模型架构
--------------------------------------------------------------------------------
  输入（基因表达向量）
      │
      ▼
  GCN 层 —— 在细胞-基因二分图 / 细胞相似度图上聚合邻居信息
      │
      ▼
  scMamba —— 线性嵌入 → 堆叠 Mamba 残差块（SSM + Conv1d + SiLU）→ 均值池化
      │
      ▼
  分类头 → 细胞类型 logits

三、项目结构
--------------------------------------------------------------------------------
  scGMB/
  ├── model/                          # 模型核心代码
  │   ├── scMamba.py                  # scMamba 分类器（主要模型）
  │   │   ├── Mamba                   # 主分类网络
  │   │   ├── ResidualBlock           # 残差块（MambaBlock + RMSNorm）
  │   │   ├── MambaBlock              # Mamba 基本块（SSM + Conv1d + SiLU）
  │   │   ├── RMSNorm                 # RMS 归一化层
  │   │   └── ModelArgs               # 模型超参数（dataclass）
  │   ├── Mamba.py                    # 原始 Mamba 参考实现（NLP 版本）
  │   ├── GCNLayer.py                 # 图卷积层
  │   │   ├── MyGCN                   # 通用 GCN
  │   │   ├── MyGCN1                  # 固定维度 GCN
  │   │   └── MyGCN3                  # 带边权重的 GCN
  │   └── edge.py                     # 图数据预处理
  │       ├── preprocess()            # 基础细胞-基因二分图
  │       ├── preprocess2()           # 带基因特征的二分图
  │       └── preprocess3()           # 细胞-细胞相似度图
  │
  ├── exp/                            # 实验脚本
  │   ├── out.py                      # 对比实验（纯 Mamba，Zheng 68K 数据集）
  │   ├── withgcn.py                  # 消融实验（Mamba + GCN，AMB 数据集）
  │   ├── draw.py                     # 可视化（t-SNE / 混淆矩阵 / ROC 曲线）
  │   └── model.joblib                # 已训练模型文件
  │
  ├── Intra-dataset/                  # 单细胞数据集
  │   ├── Zheng 68K/                  # 68K 人类 PBMC（11 种免疫细胞）
  │   ├── Zheng sorted/               # 排序 PBMC 子集
  │   ├── Pancreatic_data/
  │   │   ├── Baron Human/            # 人类胰腺（13 种细胞类型）
  │   │   └── Baron Mouse/            # 小鼠胰腺
  │   └── AMB/                        # Allen 小鼠脑（22 种子类）
  │
  └── img/                            # 实验结果图

四、环境依赖
--------------------------------------------------------------------------------
  Python >= 3.8
  PyTorch >= 2.0, torch_geometric, einops
  pandas, numpy, scikit-learn, joblib, matplotlib, seaborn

五、运行方式
--------------------------------------------------------------------------------
  # 纯 Mamba 对比实验
  cd exp/ && python out.py

  # Mamba + GCN 消融实验
  cd exp/ && python withgcn.py

  # 可视化
  cd exp/ && python draw.py

六、数据集概览
--------------------------------------------------------------------------------
  Zheng 68K    人 PBMC         ~68,000 细胞    11 类
  Zheng sorted 人 PBMC 子集    ~20,000 细胞    11 类
  Baron Human  人胰腺            8,569 细胞    13 类
  Baron Mouse  小鼠胰腺          1,886 细胞    13 类
  AMB          小鼠脑            2,715 细胞    22 类

七、模型参数（ModelArgs）
--------------------------------------------------------------------------------
  d_model      : 隐藏层维度（默认 128）
  n_layer      : Mamba 层数（默认 2）
  num_genes    : 输入基因数
  num_classes  : 输出类别数
  d_state      : SSM 状态维度（默认 16）
  expand       : 扩展因子（默认 2）
  d_conv       : 卷积核大小（默认 4）
