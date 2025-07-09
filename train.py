import os
import argparse
import logging
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from transformers import CLIPTextModel, CLIPTokenizer
import torch.nn.functional as F
from data.rml_loader import load_radioml_signal_text_dataset
from utils.plot_confusion_matrix import plot_confusion_matrix
from utils.plot_loss import plot_loss
from utils.plot_acc import plot_accuracy
from utils.plot_tsne import plot_tsne
from loss.KLDivergenceLoss import KLDivergenceLoss
from args import parse_args
from model.MCLDNN import MCLDNNEncoder


class RadioMLSignalTextModel(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.signal_encoder = MCLDNNEncoder(num_classes)
        self.text_encoder = CLIPTextModel.from_pretrained("openai/clip-vit-base-patch32")
        for param in self.text_encoder.parameters():
            param.requires_grad = False

        # 增加投影层的维度
        self.text_projection = nn.Sequential(
            nn.Linear(self.text_encoder.config.hidden_size, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, 256)
        )

        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

        # 修改信号投影层结构
        self.signal_projection = nn.Sequential(
            nn.Linear(128, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, 256)
        )
        self.class_projection = nn.Linear(256, num_classes)
    def forward(self, text_data, signals, labels=None):
        signals = signals.unsqueeze(1)
        signal_features = self.signal_encoder(signals)
        signal_features = self.signal_projection(signal_features)
        class_logits = self.class_projection(signal_features)
        signal_features = F.normalize(signal_features, p=2, dim=1)

        text_outputs = self.text_encoder(**text_data)
        text_features = text_outputs.last_hidden_state.mean(dim=1)
        text_features = self.text_projection(text_features)
        text_features = F.normalize(text_features, p=2, dim=1)

        logit_scale = self.logit_scale.exp()
        logits_per_signal = logit_scale * signal_features @ text_features.t()
        logits_per_text = logits_per_signal.t()

        return logits_per_signal, logits_per_text, signal_features, text_features, class_logits


class AlignedCLIPLoss(nn.Module):
    def __init__(self, temperature=0.07, kl_weight=0.2, align_weight=0.7):
        super().__init__()
        self.temperature = temperature
        self.kl_weight = kl_weight
        self.align_weight = align_weight

    def forward(self, signal_features, text_features, labels, text_centers):
        # 原始CLIP对比损失
        logits = torch.matmul(signal_features, text_features.t()) / self.temperature
        targets = torch.zeros_like(logits)
        for i in range(labels.size(0)):
            for j in range(labels.size(0)):
                if labels[i] == labels[j]:
                    targets[i, j] = 1.0
        contrastive_loss = F.binary_cross_entropy_with_logits(logits, targets)

        # 增强的对齐损失：信号特征与对应类别的文本中心对齐
        # 使用更严格的中心对齐损失
        align_loss = self.align_weight * F.mse_loss(
            signal_features, text_centers[labels], reduction='mean')

        # KL散度损失
        kl_loss = KLDivergenceLoss()(signal_features, text_features, labels)

        # 组合损失 - 增加对齐损失的权重
        total_loss = (1 - self.kl_weight - self.align_weight) * contrastive_loss + \
                     self.kl_weight * kl_loss + \
                     self.align_weight * align_loss
        return total_loss, contrastive_loss, kl_loss, align_loss


def create_experiment_dir(args):
    """Create experiment directory"""
    timestamp = datetime.now().strftime('%m%d_%H%M')
    exp_name = f"exp_radioml_{args.model}_{timestamp}"
    base_dir = os.path.join(args.output_dir, exp_name)
    os.makedirs(base_dir, exist_ok=True)
    os.makedirs(os.path.join(base_dir, "weights"), exist_ok=True)
    os.makedirs(os.path.join(base_dir, "plots"), exist_ok=True)
    return base_dir


def setup_logger(exp_dir):
    """Set up logging"""
    log_file = os.path.join(exp_dir, "training.log")
    logging.basicConfig(
        filename=log_file,
        level=logging.INFO,
        format="%(asctime)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s - %(message)s")
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)


def create_class_text_descriptions(descriptions, tokenizer, mod_types, device, max_length=77):
    """Create text descriptions for modulation types"""
    class_texts = {}
    for mod_type in mod_types:
        description = descriptions.get(mod_type, f"A signal with {mod_type} modulation")
        inputs = tokenizer(
            description,
            padding="max_length",
            max_length=max_length,
            truncation=True,
            add_special_tokens=True,
            return_tensors="pt",
            return_attention_mask=False,
        )
        for k, v in inputs.items():
            inputs[k] = v.to(device)
        class_texts[mod_type] = inputs
    return class_texts


def compute_text_features(model, class_texts, mod_types):
    """Compute text features as tensor ordered by class index"""
    model.eval()
    features = []
    with torch.no_grad():
        for mod_type in sorted(mod_types):  # 确保顺序与label一致
            text_outputs = model.text_encoder(**class_texts[mod_type])
            feature = text_outputs.last_hidden_state.mean(dim=1)
            feature = model.text_projection(feature)
            features.append(F.normalize(feature, dim=1))
    return torch.cat(features, dim=0)  # [num_classes, feature_dim]


def compute_accuracy(logits, labels):
    """Compute accuracy from logits and labels"""
    _, preds = torch.max(logits, 1)
    correct = (preds == labels).sum().item()
    accuracy = 100.0 * correct / labels.size(0)
    return accuracy, preds.cpu().numpy(), labels.cpu().numpy()


def validate(model, test_loader, text_centers, mod_type_to_idx, criterion, device):
    """Validation function"""
    model.eval()
    total_clip_accuracy = 0.0
    total_supervised_accuracy = 0.0
    total_combined_accuracy = 0.0
    total_loss = 0.0
    total_contrast_loss = 0.0
    total_kl_loss = 0.0
    total_align_loss = 0.0
    total_supervised_preds = []
    total_labels = []
    num_batches = 0
    classification_criterion = nn.CrossEntropyLoss()

    with torch.no_grad():
        for text_data, signals, labels in test_loader:
            signals = signals.to(device)
            labels = labels.to(device)

            if isinstance(text_data, dict):
                for k, v in text_data.items():
                    text_data[k] = v.to(device)

            logits_per_signal, _, signal_features, text_features, class_logits = model(text_data, signals)

            clip_loss, contrast_loss, kl_loss, align_loss = criterion(
                signal_features, text_features, labels, text_centers
            )
            classification_loss = classification_criterion(class_logits, labels)
            loss = clip_loss + classification_loss

            signal_features_norm = F.normalize(signal_features, dim=1)
            logit_scale = model.logit_scale.exp()
            similarity_scores = logit_scale * (signal_features_norm @ text_centers.t())

            clip_acc, _, _ = compute_accuracy(similarity_scores, labels)
            supervised_acc, _, _ = compute_accuracy(class_logits, labels)
            combined_acc, batch_preds, batch_labels = compute_accuracy(similarity_scores + class_logits, labels)

            total_clip_accuracy += clip_acc
            total_supervised_accuracy += supervised_acc
            total_combined_accuracy += combined_acc
            total_loss += loss.item()
            total_contrast_loss += contrast_loss.item()
            total_kl_loss += kl_loss.item()
            total_align_loss += align_loss.item()
            total_supervised_preds.extend(batch_preds)
            total_labels.extend(batch_labels)
            num_batches += 1

    avg_clip_accuracy = total_clip_accuracy / num_batches if num_batches > 0 else 0.0
    avg_supervised_accuracy = total_supervised_accuracy / num_batches if num_batches > 0 else 0.0
    avg_combined_accuracy = total_combined_accuracy / num_batches if num_batches > 0 else 0.0
    avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
    avg_contrast_loss = total_contrast_loss / num_batches
    avg_kl_loss = total_kl_loss / num_batches
    avg_align_loss = total_align_loss / num_batches

    logging.info(f"CLIP Accuracy: {avg_clip_accuracy:.2f}%, Supervised Accuracy: {avg_supervised_accuracy:.2f}%, "
                 f"Combined Accuracy: {avg_combined_accuracy:.2f}%")
    logging.info(
        f"Contrast Loss: {avg_contrast_loss:.4f}, KL Loss: {avg_kl_loss:.4f}, Align Loss: {avg_align_loss:.4f}")

    return avg_loss, avg_combined_accuracy, total_supervised_preds, total_labels


def train(args):
    """Main training function"""
    exp_dir = create_experiment_dir(args)
    setup_logger(exp_dir)
    logging.info(f"Arguments: {args}")
    device = torch.device(args.device)
    logging.info(f"Using device: {device}")

    # Load datasets
    logging.info("Loading datasets...")
    train_loader, test_loader, mod_mapping, mod_types, knowledge_texts = load_radioml_signal_text_dataset(
        data_path=args.data,
        knowledge_file=args.knowledge_file,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        snr_filter=args.snr_filter,
        min_snr=args.min_snr,
        max_snr=args.max_snr
    )

    logging.info(f"Modulation types: {mod_types}")
    logging.info(f"Modulation mapping: {mod_mapping}")

    # Load tokenizer
    tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")

    # Initialize model
    logging.info(f"Initializing {args.model} model...")
    model = RadioMLSignalTextModel(num_classes=len(mod_types))
    model = model.to(device)

    # Create text descriptions
    logging.info("Creating text descriptions...")
    class_texts = create_class_text_descriptions(knowledge_texts, tokenizer, mod_types, device)

    # Compute initial text features (fixed during training)
    text_centers = compute_text_features(model, class_texts, mod_types).to(device)
    logging.info(f"Text centers shape: {text_centers.shape}")  # 应为[num_classes, feature_dim]

    # Initialize loss and optimizer
    clip_criterion = AlignedCLIPLoss(temperature=args.temperature, kl_weight=0.4, align_weight=0.5)
    classification_criterion = nn.CrossEntropyLoss()

    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

    scheduler = ReduceLROnPlateau(
        optimizer,
        mode='max',
        factor=args.lr_factor,
        patience=args.lr_patience,
        verbose=True
    )

    # Training history
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': [],
        'contrast_loss': [],
        'kl_loss': [],
        'align_loss': []
    }
    best_val_acc = 0.0
    logging.info("Starting training...")

    for epoch in range(args.epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_clip_loss = 0.0
        train_cls_loss = 0.0
        train_contrast_loss = 0.0
        train_kl_loss = 0.0
        train_align_loss = 0.0
        train_acc = 0.0
        num_batches = 0

        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{args.epochs}")
        for text_data, signals, labels in train_pbar:
            signals = signals.to(device)
            labels = labels.to(device)

            if isinstance(text_data, dict):
                for k, v in text_data.items():
                    text_data[k] = v.to(device)

            logits_per_signal, _, signal_features, text_features, class_logits = model(text_data, signals)

            clip_loss, contrast_loss, kl_loss, align_loss = clip_criterion(
                signal_features, text_features, labels, text_centers
            )
            cls_loss = classification_criterion(class_logits, labels)
            loss = clip_loss + cls_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Compute training accuracy
            acc, _, _ = compute_accuracy(class_logits, labels)
            train_acc += acc

            train_loss += loss.item()
            train_clip_loss += clip_loss.item()
            train_cls_loss += cls_loss.item()
            train_contrast_loss += contrast_loss.item()
            train_kl_loss += kl_loss.item()
            train_align_loss += align_loss.item()
            num_batches += 1

            train_pbar.set_postfix({
                'loss': loss.item(),
                'contrast': contrast_loss.item(),
                'kl': kl_loss.item(),
                'align': align_loss.item(),
                'cls': cls_loss.item(),
                'acc': f"{acc:.2f}%"
            })

        avg_train_loss = train_loss / num_batches
        avg_train_clip_loss = train_clip_loss / num_batches
        avg_train_cls_loss = train_cls_loss / num_batches
        avg_train_contrast_loss = train_contrast_loss / num_batches
        avg_train_kl_loss = train_kl_loss / num_batches
        avg_train_align_loss = train_align_loss / num_batches
        avg_train_acc = train_acc / num_batches

        # Validation
        val_loss, val_accuracy, val_preds, val_labels = validate(
            model, test_loader, text_centers, mod_mapping, clip_criterion, device
        )
        scheduler.step(val_accuracy)

        logging.info(f"Epoch {epoch + 1}/{args.epochs}: "
                     f"Train Loss: {avg_train_loss:.4f} "
                     f"(Contrast: {avg_train_contrast_loss:.4f}, KL: {avg_train_kl_loss:.4f}, "
                     f"Align: {avg_train_align_loss:.4f}, CLS: {avg_train_cls_loss:.4f}), "
                     f"Train Acc: {avg_train_acc:.2f}%, "
                     f"Val Loss: {val_loss:.4f}, "
                     f"Val Accuracy: {val_accuracy:.2f}%")

        history['train_loss'].append(avg_train_loss)
        history['train_acc'].append(avg_train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_accuracy)
        history['contrast_loss'].append(avg_train_contrast_loss)
        history['kl_loss'].append(avg_train_kl_loss)
        history['align_loss'].append(avg_train_align_loss)

        if val_accuracy > best_val_acc:
            best_val_acc = val_accuracy
            torch.save(model.state_dict(), os.path.join(exp_dir, "weights", "best_model.pth"))
            logging.info(f"Saved best model with validation accuracy: {best_val_acc:.2f}%")

        if (epoch + 1) % 10 == 0 or epoch == args.epochs - 1:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'history': history
            }, os.path.join(exp_dir, "weights", f"checkpoint_epoch_{epoch + 1}.pth"))

    # Plot results
    plot_dir = os.path.join(exp_dir, "plots")
    plot_loss(history, output_dir=plot_dir)
    plot_accuracy(history, output_dir=plot_dir)

    # Generate confusion matrix
    logging.info("Generating confusion matrix...")
    model.load_state_dict(torch.load(os.path.join(exp_dir, "weights", "best_model.pth")))
    _, _, val_preds, val_labels = validate(model, test_loader, text_centers, mod_mapping, clip_criterion, device)
    confusion_matrix_path = os.path.join(plot_dir, 'confusion_matrix.png')

    plot_confusion_matrix(val_labels, val_preds, mod_types, confusion_matrix_path)

    # Generate t-SNE visualization
    logging.info("Generating t-SNE visualization...")
    tsne_path = os.path.join(plot_dir, 'tsne_visualization.png')
    plot_tsne(model, test_loader, text_centers, mod_types, device, tsne_path)

    logging.info(f"Training completed. Results saved to {exp_dir}")
    return history, model


if __name__ == "__main__":
    args = parse_args()
    history, model = train(args)