import torch
import torch.nn as nn





class KLDivergenceLoss(nn.Module):
    """
    KL Divergence loss for image-text matching.
    """

    def __init__(self):
        super().__init__()

    def forward(self, image_features, text_features, lables):
        """
        Calculate KL Divergence loss.

        Args:
            image_features: The features from the image encoder [batch_size, feature_dim]
            text_features: The features from the text encoder [batch_size, feature_dim]

        Returns:
            total_loss: KL Divergence loss
        """
        # Normalize features
        image_features = nn.functional.softmax(image_features, dim=1)  # Softmax to create a probability distribution
        text_features = nn.functional.softmax(text_features, dim=1)

        # KL Divergence from image to text
        kl_div_image_to_text = nn.functional.kl_div(image_features.log(), text_features, reduction='batchmean')

        # KL Divergence from text to image
        kl_div_text_to_image = nn.functional.kl_div(text_features.log(), image_features, reduction='batchmean')

        # Total loss is the sum of both KL divergences
        total_loss = (kl_div_image_to_text + kl_div_text_to_image) / 2
        return total_loss
