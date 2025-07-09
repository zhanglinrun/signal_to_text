import pickle
import numpy as np
import matplotlib
matplotlib.use('Agg')  # 设置后端为Agg，适合服务器环境
import matplotlib.pyplot as plt
from scipy.signal import stft
import os

# 创建保存文件夹（如果不存在）
save_dir = 'data'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

# 读取RML2016.10a_dict.pkl文件
with open('RML2016.10a_dict.pkl', 'rb') as f:
    data = pickle.load(f, encoding='latin1')


def plot_and_save_spectrogram(signal, save_path, fs=1.0, window='hann', nperseg=64, noverlap=32):
    # 合并I和Q分量为复数信号
    complex_signal = signal[0, :] + 1j * signal[1, :]

    # 计算短时傅里叶变换（STFT）
    f, t, Zxx = stft(complex_signal, fs=fs, window=window, nperseg=nperseg, noverlap=noverlap)

    # 绘制时频图
    plt.figure(figsize=(10, 6))
    plt.pcolormesh(t, f, np.abs(Zxx), shading='gouraud')
    plt.title('Spectrogram')
    plt.ylabel('Frequency/hz')
    plt.xlabel('Time/s')
    plt.colorbar(label='Intensity')

    # 保存图像
    plt.savefig(save_path)
    plt.close()


# 遍历数据集并绘制时频图
for key in data.keys():
    modulation = key[0]  # 调制方式
    snr = key[1]  # 信噪比
    signals = data[key]

    # 遍历所有样本
    for i in range(len(signals)):
        signal = signals[i]
        # 格式为 "调制方式_信噪比_样本索引.png"
        save_path = os.path.join(save_dir, f"{modulation}_{snr}_{i}.png")
        print(f"Processing: {save_path}")
        plot_and_save_spectrogram(signal, save_path)