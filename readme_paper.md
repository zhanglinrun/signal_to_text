好的！以下是一个完整的 `readme.md` 文件内容。

我将直接以 Markdown 格式输出，你可以复制保存为 **readme.md** 文件使用：

---

# CLASP: 基于对比语言-信号预测的自动调制识别

本项目实现了论文：

> Yunpeng Qu, Zhilin Lu, Bingyu Hui, Jintao Wang, Jian Wang.
> *Contrastive Language-Signal Prediction for Automatic Modulation Recognition*, IEEE Wireless Communications Letters, 2024.

CLASP 结合了对比学习、预训练语言模型和 SNR 预测子任务，提升了自动调制识别（AMR）在低信噪比环境下的鲁棒性和泛化能力。

---

## 📖 方法简介

CLASP 将无线电信号特征与人类语言描述通过对比学习映射到统一的特征空间，并同时预测调制类型和 SNR。

主要思路：

* 用时序建模提取信号特征。
* 将 (调制类型, SNR) 标签转化为文本模板，使用 CLIP 编码。
* 通过余弦相似度计算信号特征与文本特征的相似性进行分类。
* 引入 SNR 预测作为子任务，提供噪声先验。

---

## 🗂️ 数据集

实验采用两个开源数据集：

* [RadioML2016.10a](https://www.deepsig.ai/datasets)
* [RadioML2018.01a](https://www.deepsig.ai/datasets)

这些数据集基于 GNU Radio 生成，包含真实信道损伤（采样率偏移、频率偏移、多径衰落、AWGN 等）。

### 数据处理

* 划分比例：**训练 : 验证 : 测试 = 6 : 2 : 2**
* 每个样本是长度为 $N$ 的复数时序信号：

  * 原始：$r(t) = s(t) * h(t, \tau) + n(t)$
  * 离散化后：$r[n] = r_I[n] + j r_Q[n]$
  * 转换为振幅/相位形式：$r_A[n], r_P[n]$
  * 拼接为输入矩阵：$X = [r_A, r_P] ∈ ℝ^{N×2}$

---

## 🧪 实验设置

| 参数                | RadioML2016.10a | RadioML2018.01a |
| ----------------- | --------------- | --------------- |
| 优化器               | AdamW           | AdamW           |
| 初始学习率             | 0.001           | 0.001           |
| SNR 损失权重 $λ$      | 0.3             | 0.3             |
| 硬件                | NVIDIA RTX 3090 | NVIDIA RTX 3090 |
| 训练时间              | \~1 小时          | \~13 小时         |
| batch size & 训练轮数 | 论文未明确，可根据验证集调整  |                 |

---

## 📐 模型架构

CLASP 由以下三个模块组成：

---

### 🔷 1. 语言分支

目标：生成人类语言先验，用作特征空间的 anchor。

**输入**：

* 所有类别的文本模板：

  ```
  Modulation: {调制类型}, SNR: {SNR值} dB
  ```

  例如：

  ```
  Modulation: Binary Phase Shift Keying, SNR: -10 dB
  ```

**处理步骤**：

* 用 **冻结的 CLIP 文本编码器 E** 得到文本特征：

  ```
  t_c,s = E(T_c,s)  ∈ ℝ^C
  ```
* 为了和信号特征维度一致，引入一个可训练的投影器 $P$（线性层），将 CLIP 输出映射到 $d$ 维：

  ```
  t_c,s = P(E(T_c,s)) ∈ ℝ^d
  ```

**输出**：

* 所有 $M = |C| × |S|$ 个类别的文本特征 $\{t_c,s\}$。
* 这些特征在推理阶段提前计算并缓存。

---

### 🔷 2. 信号分支

目标：提取信号的判别特征。

**输入**：

* 信号矩阵 $X ∈ ℝ^{N×2}$

**处理步骤**：

* 使用骨干网络 $ψ$ 提取信号特征，默认使用 **双层 LSTM（LSTM2）**：

  * 输入维度：2
  * 隐藏维度：$h$ （需实验确定）
  * 层数：2
  * 取最后时间步的隐藏状态作为信号特征 $x$。
* 信号特征归一化。

**可选骨干网络**：

* LSTM2（默认）
* ResNet（CNN）
* MCLDNN（CNN + LSTM）

**输出**：

* 信号特征向量 $x ∈ ℝ^d$

---

### 🔷 3. 对比学习模块

目标：计算信号和各类别文本特征之间的相似度，并进行分类。

**步骤**：
1️⃣ 归一化信号特征 $x$ 和文本特征 $t_c,s$：

```
x ← x / ||x||_2
t_c,s ← t_c,s / ||t_c,s||_2
```

2️⃣ 计算每个类别 $(c,s)$ 的余弦相似度：

```
q_c,s = x · t_c,s
```

3️⃣ 使用 softmax 转为联合概率：

```
p̂(c,s|X) = softmax(q_c,s / τ)
```

其中 $τ$ 是温度系数（可调）。

4️⃣ 多任务优化：

* 调制分类损失：

  ```
  L_mod = - ∑_c p(c|X) log p̂(c|X)
  ```
* SNR 预测损失：

  ```
  L_snr = - ∑_s p(s|X) log p̂(s|X)
  ```
* 总损失：

  ```
  L = L_mod + λ * L_snr
  ```

  $λ=0.3$

---

### 🔷 推理过程

1️⃣ 预先计算并缓存所有文本特征 $t_c,s$。
2️⃣ 对输入信号 $X$：

* 得到信号特征 $x$。
* 计算相似度 $q_c,s$。
* 取最大概率对应的 $c$ 作为预测调制类型：

  ```
  ĉ = argmax_c p̂(c|X)
  ```

---

## 🚀 训练与评估

* 单卡 NVIDIA RTX 3090。
* 训练时间：

  * RadioML2016.10a: \~1 小时
  * RadioML2018.01a: \~13 小时
* 评估指标：调制分类准确率（特别关注低 SNR 场景）。

---

## 📊 对比基线

| 方法类型 | 代表方法                                                               |
| ---- | ------------------------------------------------------------------ |
| 深度学习 | ResNet, LSTM2, LSTM-DAE, TransGroupNet, ThreeStream, MCLDNN, FEA-T |
| 特征工程 | Logistic Regression, SVM, Random Forest                            |

---



## 💻 示例命令

```bash
# 安装依赖
pip install torch torchvision ftfy regex tqdm
git clone https://github.com/openai/CLIP
cd CLIP
pip install -e .

# 训练
python train_clasp.py --dataset RadioML2016.10a --backbone LSTM2 --lambda_snr 0.3 --lr 1e-3

# 测试
python evaluate_clasp.py --dataset RadioML2016.10a --checkpoint path_to_model.pt
```

---

## 📝 注意事项

* 推理阶段无需调用 CLIP 编码器，文本特征可以提前缓存。
* 论文未给出温度参数 $τ$ 和 batch size，请根据实验经验选择（如 batch size 128, $τ$ 在 0.1\~1.0 之间调优）。
* LSTM 隐藏维度、投影器维度等超参数需要实验选择。

---



