import os
import re
import json
import pickle
import torch
import numpy as np
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from sklearn.model_selection import train_test_split
from transformers import CLIPTokenizer


class RadioMLSignalTextDataset(Dataset):
    def __init__(self, data_path, knowledge_file, type="train", tokenizer=None,
                 snr_filter=None, min_snr=None, max_snr=None):
        """
        初始化数据集
        :param data_path: RML2016.10a_dict.pkl 文件路径
        :param knowledge_file: 知识文本文件路径
        :param type: 训练还是测试
        :param tokenizer: 文本编码器
        :param snr_filter: 特定的信噪比列表，如 [0, 2, 4, 6]
        :param min_snr: 最小信噪比值（包含此值）
        :param max_snr: 最大信噪比值（包含此值）
        """
        if type != "train" and type != "test":
            raise ValueError(f"type {type} not supported")

        self.data_path = data_path
        self.type = type
        self.tokenizer = tokenizer

        # 加载原始信号数据
        with open(data_path, 'rb') as f:
            data_dict = pickle.load(f, encoding='latin1')

        # 存储样本和标签信息
        self.samples = []
        self.modulation_types = set()
        self.snr_values = set()
        self.mod_type_to_idx = {}

        # 解析数据字典
        for key in data_dict.keys():
            mod_type, snr = key
            signals = data_dict[key]  # 形状为 (n_samples, 2, 128)

            # 应用信噪比过滤
            if self._filter_snr(snr, snr_filter, min_snr, max_snr):
                self.modulation_types.add(mod_type)
                self.snr_values.add(snr)

                # 为每个信号样本添加条目
                for i in range(signals.shape[0]):
                    self.samples.append((signals[i], mod_type, snr))

        # 为每种调制类型分配一个唯一索引
        self.modulation_types = sorted(self.modulation_types)
        for idx, mod_type in enumerate(self.modulation_types):
            self.mod_type_to_idx[mod_type] = idx

        if len(self.samples) == 0:
            print("警告: 在应用信噪比过滤后没有剩余数据!")
        else:
            print(f"应用信噪比过滤后剩余 {len(self.samples)} 个样本")
            print(f"调制方式: {self.modulation_types}")
            print(f"调制方式索引映射: {self.mod_type_to_idx}")
            print(f"信噪比值: {sorted(self.snr_values)}")

        # 按调制类型和信噪比对数据进行划分
        self.train_data = []
        self.test_data = []
        self._split_data()

        # 加载知识文本
        self.knowledge_texts = {}
        try:
            with open(knowledge_file, 'r', encoding='utf-8') as file:
                modulation_knowledge = json.load(file)

                for mod_type in self.modulation_types:
                    if mod_type in modulation_knowledge:
                        info = modulation_knowledge[mod_type]
                        full_description = f"{mod_type}调制: {info.get('产生方式', '')} {info.get('时频域特征', '')}"
                        self.knowledge_texts[mod_type] = full_description
                    else:
                        self.knowledge_texts[mod_type] = f"{mod_type}调制信号"

            print(f"已加载 {len(self.knowledge_texts)} 种调制方式的描述")
        except Exception as e:
            print(f"加载知识文件时出错: {e}")
            for mod_type in self.modulation_types:
                self.knowledge_texts[mod_type] = f"{mod_type}调制信号"

    def _filter_snr(self, snr, snr_filter, min_snr, max_snr):
        """
        根据给定条件过滤信噪比
        """
        if snr_filter is not None:
            return snr in snr_filter

        if min_snr is not None and snr < min_snr:
            return False
        if max_snr is not None and snr > max_snr:
            return False

        return True

    def _split_data(self):
        grouped = {}
        for signal, mod, snr in self.samples:
            key = (mod, snr)
            if key not in grouped:
                grouped[key] = []
            grouped[key].append((signal, mod, snr))

        self.train_data = []
        self.test_data = []
        for key in grouped:
            train, test = train_test_split(
                grouped[key],
                test_size=0.2,
                random_state=42
            )
            self.train_data.extend(train)
            self.test_data.extend(test)

    def __len__(self):
        if self.type == "train":
            return len(self.train_data)
        elif self.type == "test":
            return len(self.test_data)
        else:
            raise ValueError(f"type {self.type} not supported")

    def __getitem__(self, idx):
        """
        获取指定索引的数据和标签
        :param idx: 索引
        :return: (文本编码, 信号数据, 调制类型标签)
        """
        if self.type == 'train':
            signal, mod_type, snr = self.train_data[idx]
        elif self.type == 'test':
            signal, mod_type, snr = self.test_data[idx]
        else:
            raise ValueError(f"type {self.type} is not supported")

        # 将信号数据转换为torch张量
        signal_tensor = torch.from_numpy(signal).float()

        # 获取调制类型的索引作为标签
        label = self.mod_type_to_idx[mod_type]

        # 获取该调制类型的文本描述
        text = self.knowledge_texts[mod_type]

        if self.tokenizer:
            text_tokens = self.tokenizer(text, padding="max_length", max_length=77,
                                         truncation=True, add_special_tokens=True,
                                         return_tensors="pt", return_attention_mask=False)
            return text_tokens, signal_tensor, label
        else:
            return text, signal_tensor, label


def load_radioml_signal_text_dataset(data_path, knowledge_file, batch_size=64, num_workers=0,
                                     snr_filter=None, min_snr=None, max_snr=None):
    """
    加载 RadioML 信号和文本数据集并生成 DataLoader
    :param data_path: RML2016.10a_dict.pkl 文件路径
    :param knowledge_file: 知识文本文件路径
    :param batch_size: 每批次样本数
    :param num_workers: 数据加载的并行线程数
    :param snr_filter: 特定的信噪比列表，如 [0, 2, 4, 6]
    :param min_snr: 最小信噪比值（包含此值）
    :param max_snr: 最大信噪比值（包含此值）
    :return: 训练集和测试集的 DataLoader，以及调制类型映射
    """
    try:
        tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
        print("成功加载 CLIP tokenizer")
    except Exception as e:
        print(f"加载 CLIP tokenizer 失败: {e}")
        print("将使用原始文本而不进行编码")
        tokenizer = None

    train_dataset = RadioMLSignalTextDataset(
        data_path=data_path,
        knowledge_file=knowledge_file,
        type="train",
        tokenizer=tokenizer,
        snr_filter=snr_filter,
        min_snr=min_snr,
        max_snr=max_snr
    )

    test_dataset = RadioMLSignalTextDataset(
        data_path=data_path,
        knowledge_file=knowledge_file,
        type="test",
        tokenizer=tokenizer,
        snr_filter=snr_filter,
        min_snr=min_snr,
        max_snr=max_snr
    )

    def custom_collate_fn(batch):
        if tokenizer:
            labels = torch.tensor([item[2] for item in batch])

            text_encodings = [item[0] for item in batch]
            merged_encodings = {}
            for key in text_encodings[0].keys():
                tensors = [enc[key].squeeze(0) for enc in text_encodings]
                merged_encodings[key] = torch.stack(tensors)

            signals = torch.stack([item[1] for item in batch])

            return merged_encodings, signals, labels
        else:
            texts = [item[0] for item in batch]
            signals = torch.stack([item[1] for item in batch])
            labels = torch.tensor([item[2] for item in batch])

            return texts, signals, labels

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=custom_collate_fn
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=custom_collate_fn
    )

    mod_type_mapping = train_dataset.mod_type_to_idx
    mod_types = train_dataset.modulation_types
    knowledge_texts = train_dataset.knowledge_texts

    return train_loader, test_loader, mod_type_mapping, mod_types, knowledge_texts


if __name__ == "__main__":
    # 指定数据文件和知识文本文件路径
    data_path = "./RML2016.10a_dict.pkl"  # 替换为实际路径
    knowledge_file = "./knowledge.txt"  # 替换为实际路径

    # 示例：加载所有数据
    train_loader, test_loader, mod_mapping, mod_types, knowledge_texts = load_radioml_signal_text_dataset(
        data_path=data_path,
        knowledge_file=knowledge_file,
        batch_size=32,
        snr_filter=[16]
    )

    print("\n示例：加载所有数据")
    print(f"调制类型: {mod_types}")
    print(f"调制类型映射: {mod_mapping}")

    # 显示知识文本示例
    print("\n调制类型知识文本示例:")
    for mod_type in list(mod_types)[:3]:
        print(f"{mod_type}: {knowledge_texts[mod_type]}")

    # 显示批次数据
    for batch_idx, (text_data, signals, labels) in enumerate(train_loader):
        print(f"\n批次 {batch_idx + 1}")
        print(f"信号形状: {signals.shape}")
        print(f"标签形状: {labels.shape}")

        if isinstance(text_data, dict):
            print(f"文本编码类型: {type(text_data)}")
            print(f"文本编码包含键: {text_data.keys()}")
            print(f"文本编码张量形状: {text_data['input_ids'].shape}")
        else:
            print(f"文本类型: {type(text_data)}")
            print(f"文本数量: {len(text_data)}")
            print(f"示例文本: {text_data[0]}")

        for i in range(min(3, len(labels))):
            mod_idx = labels[i].item()
            mod_type = mod_types[mod_idx]
            print(f"样本 {i + 1}: 调制类型 = {mod_type}")

        break