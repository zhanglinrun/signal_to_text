import argparse
import json
import torch


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Radio Modulation Image-to-Text Training')

    # Dataset parameters
    parser.add_argument('--data', type=str, default='data/RML2016.10a_dict.pkl', help='Path to the time-frequency images')
    parser.add_argument('--knowledge_file', type=str, default='data/knowledge.txt', help='Path to knowledge text file')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size for training')
    parser.add_argument('--num_workers', type=int, default=0, help='Number of data loading workers')

    # SNR filtering parameters
    parser.add_argument('--snr_filter', type=list, default=[2], help='Specific SNR values to use')
    parser.add_argument('--min_snr', type=int, default=None, help='Minimum SNR value to include')
    parser.add_argument('--max_snr', type=int, default=None, help='Maximum SNR value to include')

    # Model parameters
    parser.add_argument('--model', type=str, default='MCLDNN',
                        choices=['MCLDNN', 'resnet18', 'resnet34', 'resnet50'], help='Image encoder model')
    parser.add_argument('--projection_dim', type=int, default=512, help='Dimension for embedding projection')
    parser.add_argument('--num_classes', type=int, default=11, help='classes for number of modulation types')
    parser.add_argument('--temperature', type=float, default=0.07, help='Temperature parameter for contrastive loss')

    # Training parameters
    parser.add_argument('--epochs', type=int, default=20, help='Number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Initial learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='Weight decay regularization')
    parser.add_argument('--momentum', type=float, default=0.9, help='Momentum for SGD optimizer')

    # Scheduler parameters
    parser.add_argument('--lr_patience', type=int, default=5, help='Patience for learning rate scheduler')
    parser.add_argument('--lr_factor', type=float, default=0.5, help='Factor by which to reduce learning rate')

    # Device parameters
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='Device to use (cuda or cpu)')

    # Output parameters
    parser.add_argument('--output_dir', type=str, default='./experiments', help='Directory to save results')

    return parser.parse_args()
