# 定义时频图像生成文本的网络模型

import torch
import torch.nn as nn


class TimeFreqToText(nn.Module):
    def __init__(self, feature_dim=512, hidden_dim=256, vocab_size=33, max_seq_len=15):
        super(TimeFreqToText, self).__init__()

        # 卷积层用于提取图像特征
        self.conv1 = nn.Conv2d(1, 64, kernel_size=(1, 3), stride=1, padding=(0, 1))  # 输入通道=1，输出通道=64
        self.conv2 = nn.Conv2d(64, 128, kernel_size=(1, 3), stride=1, padding=(0, 1))  # 输入通道=64，输出通道=128
        self.conv3 = nn.Conv2d(128, 256, kernel_size=(1, 3), stride=1, padding=(0, 1))  # 输入通道=128，输出通道=256
        self.pool = nn.MaxPool2d(kernel_size=(1, 2), stride=2, padding=0)  # 对宽度进行池化

        # 特征降维到指定维度
        self.fc = nn.Linear(4096, feature_dim)  # 假设经过池化后的宽度为32

        # 文本解码器部分，使用双向LSTM
        self.embedding = nn.Embedding(vocab_size, hidden_dim)
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, num_layers=2, bidirectional=True, batch_first=True)
        self.hidden_dim = hidden_dim

        # 输出层，用于生成词概率
        self.fc_out = nn.Linear(hidden_dim * 2, vocab_size)

        # 生成的最大序列长度
        self.max_seq_len = max_seq_len

    def forward(self, images, captions=None):
        """
        :param images: 输入的时频图像，形状为[B, C, H, W]，这里 C=1，H=2, W=128
        :param captions: 输入的真实文本序列（用于训练），形状为[B, T]
        :return: 生成的文本序列或概率分布
        """
        # 图像特征提取
        x = self.pool(torch.relu(self.conv1(images)))  # [B, 64, 2, 128] -> [B, 64, 2, 64]
        x = self.pool(torch.relu(self.conv2(x)))  # [B, 128, 2, 64] -> [B, 128, 2, 32]
        x = self.pool(torch.relu(self.conv3(x)))  # [B, 256, 2, 32] -> [B, 256, 2, 16]
        x = x.view(x.size(0), -1)  # Flatten -> [B, 256*2*16] = [B, 8192]

        # 特征降维到指定维度
        features = self.fc(x)  # [B, feature_dim]

        # 初始解码输入
        batch_size = features.size(0)
        if captions is not None:
            embeddings = self.embedding(captions)  # [B, T, hidden_dim]
        else:
            embeddings = torch.zeros(batch_size, self.max_seq_len, self.hidden_dim, device=images.device)

        # 解码LSTM
        lstm_out, _ = self.lstm(embeddings)  # [B, T, hidden_dim*2]

        # 输出词概率分布
        outputs = self.fc_out(lstm_out)  # [B, T, vocab_size]

        return outputs


# 测试模型
if __name__ == "__main__":
    # 创建模型实例
    model = TimeFreqToText()

    # 假设输入的时频图像形状为[B, C, H, W]
    dummy_images = torch.randn(4, 3, 224, 224)  # 4个样本，3通道图像

    # 假设输入的真实文本序列形状为[B, T]
    dummy_captions = torch.randint(0, 10000, (4, 10))  # 每个样本10个词，词汇大小为10000

    # 前向传播
    outputs = model(dummy_images, dummy_captions)

    # 打印输出形状
    print(outputs.shape)  # [B, T, vocab_size]
