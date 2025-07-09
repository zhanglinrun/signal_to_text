import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib import rcParams
from scipy.io import savemat, loadmat
from sklearn.metrics import confusion_matrix
import os


def plot_confusion_matrix(y_true, y_pred, class_names, output_path, title="Confusion Matrix"):
    """
    绘制混淆矩阵并保存为图片。

    Args:
        y_true (array-like): 真实标签
        y_pred (array-like): 预测标签
        class_names (list): 类别名称列表
        output_path (str): 保存图片的完整路径
        title (str, optional): 图表标题. 默认为 "Confusion Matrix"
    """
    # 计算混淆矩阵
    cm = confusion_matrix(y_true, y_pred)

    # 创建图形
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)

    # 添加标签和标题
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(title)
    plt.tight_layout()

    # 保存图形
    plt.savefig(output_path)
    plt.close()
