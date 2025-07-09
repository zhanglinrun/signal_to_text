import torch
import torch.nn as nn
import torch.nn.functional as F


class MCLDNNEncoder(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        # 修改为论文中的通道数
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 256, kernel_size=(1, 3), padding=(0, 1)),
            nn.BatchNorm2d(256),
            nn.SELU(inplace=True)
        )
        self.conv2 = nn.Sequential(
            nn.Conv1d(1, 256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256),
            nn.SELU(inplace=True)
        )
        self.conv3 = nn.Sequential(
            nn.Conv1d(1, 256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256),
            nn.SELU(inplace=True)
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=(1, 3), padding=(0, 1)),
            nn.BatchNorm2d(256),
            nn.SELU(inplace=True)
        )
        self.conv5 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=(2, 3), padding=0),
            nn.BatchNorm2d(512),
            nn.SELU(inplace=True)
        )

        # LSTM部分
        self.lstm = nn.LSTM(
            input_size=512,
            hidden_size=128,
            num_layers=2,
            bidirectional=False,
            batch_first=True
        )

        # 分类头
        self.classifier = nn.Sequential(
            nn.Linear(128, 128),
            nn.SELU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        # x shape: [B, 1, 2, 128]
        I = x[:, :, 0, :].squeeze(1)  # [B, 128]
        Q = x[:, :, 1, :].squeeze(1)  # [B, 128]

        # 多通道处理
        x = self.conv1(x)  # [B, 256, 2, 128]

        # I/Q单独处理
        I = self.conv2(I.unsqueeze(1))  # [B, 256, 128]
        Q = self.conv3(Q.unsqueeze(1))  # [B, 256, 128]

        # 合并I/Q
        IQ = torch.stack((I, Q), dim=2)  # [B, 256, 2, 128]
        IQ = self.conv4(IQ)  # [B, 256, 2, 128]

        # 最终合并
        x = torch.cat((IQ, x), dim=1)  # [B, 512, 2, 128]
        x = self.conv5(x)  # [B, 512, 1, 126]

        # LSTM处理
        x = x.squeeze(2).transpose(1, 2)  # [B, 126, 512]
        x, _ = self.lstm(x)  # [B, 126, 128]
        x = x[:, -1, :]  # 取最后一个时间步 [B, 128]

        # 分类
        # return self.classifier(x)
        return x