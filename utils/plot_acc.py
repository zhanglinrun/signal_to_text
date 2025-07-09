import matplotlib.pyplot as plt
import os
import numpy as np


def plot_accuracy(history, output_dir='./plots'):
    """
    绘制训练过程中的损失曲线和准确率曲线
    """
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    # 设置字体和样式
    plt.rcParams.update({'font.size': 12, 'font.family': 'Times New Roman'})

    # 绘制准确率曲线
    plt.figure(figsize=(10, 5))
    plt.plot(history['val_acc'], label='Validation Accuracy', color='green', linewidth=2)
    plt.xlabel('Epoch', fontsize=14)
    plt.ylabel('Accuracy (%)', fontsize=14)
    plt.title('Validation Accuracy')
    plt.ylim(0, 100)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "accuracy_curve.png"), dpi=300, bbox_inches='tight')
    plt.close()