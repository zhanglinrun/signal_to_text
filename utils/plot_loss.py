import matplotlib.pyplot as plt
import os



def plot_loss(history, output_dir='./plots'):
    """
    绘制训练过程中的损失曲线和准确率曲线
    """
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    # 设置字体和样式
    plt.rcParams.update({'font.size': 12, 'font.family': 'Times New Roman'})

    # 绘制损失曲线
    plt.figure(figsize=(10, 5))
    plt.plot(history['train_loss'], label='Training Loss', linewidth=2)
    plt.plot(history['val_loss'], label='Validation Loss', linewidth=2, linestyle='--')
    plt.xlabel('Epoch', fontsize=14)
    plt.ylabel('Loss', fontsize=14)
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "loss_curves.png"), dpi=300, bbox_inches='tight')
    plt.close()