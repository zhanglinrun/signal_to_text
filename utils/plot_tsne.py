import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import torch
import torch.nn.functional as F
from matplotlib.lines import Line2D

def plot_tsne(model, test_loader, text_centers, mod_types, device, output_path):
    """Generate enhanced t-SNE visualization of signal and text features"""
    model.eval()

    # Collect features and labels
    all_signal_features = []
    all_labels = []

    with torch.no_grad():
        for text_data, signals, labels in test_loader:
            signals = signals.to(device)
            labels = labels.cpu().numpy()

            # 确保text_data在正确的设备上
            if isinstance(text_data, dict):
                for k, v in text_data.items():
                    text_data[k] = v.to(device)
            elif isinstance(text_data, torch.Tensor):
                text_data = text_data.to(device)

            # Extract signal features
            _, _, signal_features, _, _ = model(text_data, signals)
            signal_features = F.normalize(signal_features, dim=1)

            # Collect features
            all_signal_features.append(signal_features.cpu().numpy())
            all_labels.extend(labels)

    # Concatenate features
    all_signal_features = np.concatenate(all_signal_features, axis=0)
    text_features = text_centers.cpu().numpy()  # [num_classes, feature_dim]

    # Combine features for t-SNE (only include one text feature per class)
    combined_features = np.concatenate([all_signal_features, text_features], axis=0)

    # Perform t-SNE
    tsne = TSNE(
        n_components=2,
        random_state=42,
        perplexity=min(30, len(all_signal_features) // 3),  # Only consider signal features for perplexity
        early_exaggeration=12,
        learning_rate='auto',
        max_iter=1000
    )
    tsne_features = tsne.fit_transform(combined_features)

    # Split results
    signal_tsne = tsne_features[:len(all_signal_features)]
    text_tsne = tsne_features[len(all_signal_features):]  # Only num_classes points

    # Create plot
    plt.figure(figsize=(20, 12))
    scatter_kwargs = {
        's': 80,
        'alpha': 0.8,
        'linewidths': 0.5,
        'edgecolors': 'w'
    }

    # Create color map
    cmap = plt.cm.get_cmap('tab20', len(mod_types))

    # Plot signal features
    for i in range(len(mod_types)):
        mask = np.array(all_labels) == i
        plt.scatter(
            signal_tsne[mask, 0],
            signal_tsne[mask, 1],
            color=cmap(i),
            marker='o',
            label=f'Signal: {mod_types[i]}',
            **scatter_kwargs
        )

    # Plot text features (only one per class)
    for i in range(len(mod_types)):
        plt.scatter(
            text_tsne[i, 0],
            text_tsne[i, 1],
            color=cmap(i),
            marker='^',
            s=200,  # Larger size for text centers
            edgecolor='k',
            linewidth=1.5,
            label=f'Text: {mod_types[i]}'
        )

    # Add connection lines from each signal to its class text center
    for i in range(len(signal_tsne)):
        class_idx = all_labels[i]
        plt.plot(
            [signal_tsne[i, 0], text_tsne[class_idx, 0]],
            [signal_tsne[i, 1], text_tsne[class_idx, 1]],
            color=cmap(class_idx),
            alpha=0.05,  # More transparent lines
            linewidth=0.3
        )

    # Create legends
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', label='Signal Features',
               markerfacecolor='gray', markersize=10),
        Line2D([0], [0], marker='^', color='w', label='Text Centers',
               markerfacecolor='gray', markersize=12)
    ]
    mod_elements = [Line2D([0], [0], marker='s', color='w', label=mod_types[i],
                   markerfacecolor=cmap(i), markersize=10)
                for i in range(len(mod_types))]

    first_legend = plt.legend(handles=legend_elements, loc='upper right')
    plt.gca().add_artist(first_legend)
    plt.legend(handles=mod_elements, loc='center right',
               bbox_to_anchor=(1.15, 0.5), title="Modulation Types")

    plt.title('t-SNE Visualization: Signal Features and Text Centers', pad=20)
    plt.xlabel('t-SNE Dimension 1')
    plt.ylabel('t-SNE Dimension 2')
    plt.grid(True, alpha=0.2)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
