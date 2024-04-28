## 优化算法
import numpy as np


def pca(X, n_components):
    # 1. 数据标准化
    X_mean = np.mean(X, axis=0)
    X_std = np.std(X, axis=0)
    X_normalized = (X - X_mean) / X_std

    # 2. 计算协方差矩阵
    cov_matrix = np.cov(X_normalized.T)

    # 3. 计算特征值和特征向量
    eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)

    # 4. 选择主成分
    idx = np.argsort(eigenvalues)[::-1]  # 按特征值大小降序排序
    selected_eigenvectors = eigenvectors[:, idx[:n_components]]  # 选取前n_components个特征向量

    # 5. 数据转换
    transformed_data = np.dot(X_normalized, selected_eigenvectors)

    return transformed_data