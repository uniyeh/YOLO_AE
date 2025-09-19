"""
Assisted Excitation (AE) implementation for YOLO
Based on "Assisted Excitation of Activations: A Learning Technique to Improve Object Detectors"

This module implements the Assisted Excitation technique that uses ground truth information
during training to enhance feature activations in object regions, improving detection performance.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class AssistedExcitation(nn.Module):
    """
    Assisted Excitation module that enhances feature activations using ground truth bounding boxes.

    During training, this module:
    1. Creates binary masks from ground truth bounding boxes
    2. Applies Gaussian smoothing to create soft attention maps
    3. Uses cosine annealing to gradually reduce excitation strength
    4. Enhances features in object regions to improve learning

    Args:
        channels (int): Number of input channels
        sigma (float): Standard deviation for Gaussian smoothing. Default: 0.5
        alpha_max (float): Maximum excitation strength. Default: 1.0
        alpha_min (float): Minimum excitation strength. Default: 0.0
        max_iters (int): Maximum training iterations for annealing. Default: 50000
    """

    def __init__(self, channels, sigma=0.5, alpha_max=1.0, alpha_min=0.0, max_iters=50000):
        super(AssistedExcitation, self).__init__()
        self.channels = channels
        self.sigma = sigma
        self.alpha_max = alpha_max
        self.alpha_min = alpha_min
        self.max_iters = max_iters
        self.current_iter = 0

        # Pre-compute Gaussian kernel for efficiency
        self.register_buffer('gaussian_kernel', self._create_gaussian_kernel())

    def _create_gaussian_kernel(self, kernel_size=5):
        """Create 2D Gaussian kernel for smoothing attention maps."""
        ax = torch.arange(-kernel_size // 2 + 1., kernel_size // 2 + 1.)
        xx, yy = torch.meshgrid(ax, ax, indexing='ij')
        kernel = torch.exp(-(xx**2 + yy**2) / (2 * self.sigma**2))
        kernel = kernel / kernel.sum()
        return kernel.unsqueeze(0).unsqueeze(0)  # Shape: (1, 1, K, K)

    def _cosine_annealing_alpha(self):
        """
        Compute current alpha using cosine annealing schedule.
        Alpha decreases from alpha_max to alpha_min over max_iters iterations.
        """
        if self.current_iter >= self.max_iters:
            return self.alpha_min

        # Cosine annealing: starts at alpha_max, ends at alpha_min
        progress = self.current_iter / self.max_iters
        alpha = self.alpha_min + (self.alpha_max - self.alpha_min) * \
                (1 + math.cos(math.pi * progress)) / 2
        return alpha

    def _create_attention_map(self, targets, height, width, device):
        """
        Create attention maps from ground truth bounding boxes.

        Args:
            targets (list): List of target dictionaries containing bounding boxes
            height (int): Feature map height
            width (int): Feature map width
            device (torch.device): Device to create tensors on

        Returns:
            torch.Tensor: Attention maps of shape (batch_size, 1, height, width)
        """
        batch_size = len(targets)
        attention_maps = torch.zeros(batch_size, 1, height, width, device=device)

        for batch_idx, target in enumerate(targets):
            if target is None:
                continue

            # Extract bounding boxes (handle different target formats)
            if isinstance(target, dict):
                boxes = target.get('boxes', None)
                if boxes is None:
                    boxes = target.get('bboxes', None)
            elif hasattr(target, 'boxes'):
                boxes = target.boxes
            else:
                boxes = target

            if boxes is None or len(boxes) == 0:
                continue

            # Convert to tensor if needed
            if not isinstance(boxes, torch.Tensor):
                boxes = torch.tensor(boxes, device=device, dtype=torch.float32)

            # Ensure boxes are 2D
            if boxes.dim() == 1:
                boxes = boxes.unsqueeze(0)

            # Create binary mask for this batch
            mask = torch.zeros(height, width, device=device)

            for box in boxes:
                if len(box) >= 4:
                    # Handle both YOLO format (x_center, y_center, width, height)
                    # and COCO format (x_min, y_min, x_max, y_max)
                    if box.max() <= 1.0:  # YOLO format (normalized)
                        x_center, y_center, box_width, box_height = box[:4]
                        x1 = max(0, int((x_center - box_width/2) * width))
                        y1 = max(0, int((y_center - box_height/2) * height))
                        x2 = min(width-1, int((x_center + box_width/2) * width))
                        y2 = min(height-1, int((y_center + box_height/2) * height))
                    else:  # COCO format (absolute coordinates)
                        x1, y1, x2, y2 = box[:4].int()
                        x1 = max(0, min(x1, width-1))
                        y1 = max(0, min(y1, height-1))
                        x2 = max(0, min(x2, width-1))
                        y2 = max(0, min(y2, height-1))

                    # Fill bounding box region
                    if x2 > x1 and y2 > y1:
                        mask[y1:y2+1, x1:x2+1] = 1.0

            attention_maps[batch_idx, 0] = mask

        return attention_maps

    def _smooth_attention_maps(self, attention_maps):
        """Apply Gaussian smoothing to attention maps."""
        # Pad attention maps for convolution
        padding = self.gaussian_kernel.shape[-1] // 2
        attention_maps = F.pad(attention_maps, (padding, padding, padding, padding), mode='reflect')

        # Apply Gaussian smoothing
        smoothed = F.conv2d(attention_maps, self.gaussian_kernel, padding=0)
        return smoothed

    def forward(self, x, targets=None):
        """
        Forward pass with assisted excitation.

        Args:
            x (torch.Tensor): Input feature maps of shape (B, C, H, W)
            targets (list, optional): Ground truth targets for training

        Returns:
            torch.Tensor: Enhanced feature maps of same shape as input
        """
        # During inference or when no targets provided, return input unchanged
        if not self.training or targets is None:
            return x

        batch_size, channels, height, width = x.shape
        device = x.device

        # Get current excitation strength
        alpha = self._cosine_annealing_alpha()

        # If alpha is very small, skip excitation for efficiency
        if alpha < 1e-6:
            return x

        # Create attention maps from ground truth
        attention_maps = self._create_attention_map(targets, height, width, device)

        # Apply Gaussian smoothing to create soft attention
        if self.sigma > 0:
            attention_maps = self._smooth_attention_maps(attention_maps)

        # Expand attention maps to match input channels
        attention_maps = attention_maps.expand(-1, channels, -1, -1)

        # Apply assisted excitation: enhanced_features = x + alpha * attention * x
        enhanced_features = x + alpha * attention_maps * x

        # Update iteration counter
        self.current_iter += 1

        return enhanced_features

    def reset_iteration(self):
        """Reset iteration counter (useful for new training epochs)."""
        self.current_iter = 0

    def set_iteration(self, iteration):
        """Set current iteration explicitly."""
        self.current_iter = iteration

    def get_current_alpha(self):
        """Get current excitation strength."""
        return self._cosine_annealing_alpha()


# Example usage and testing
if __name__ == "__main__":
    # Test basic AssistedExcitation module
    print("Testing AssistedExcitation module...")

    batch_size, channels, height, width = 2, 256, 32, 32
    x = torch.randn(batch_size, channels, height, width)

    # Create sample targets (YOLO format: normalized coordinates)
    targets = [
        {'boxes': torch.tensor([[0.5, 0.5, 0.3, 0.4], [0.2, 0.8, 0.15, 0.2]])},  # 2 boxes
        {'boxes': torch.tensor([[0.7, 0.3, 0.25, 0.35]])}  # 1 box
    ]

    # Test AssistedExcitation
    ae_module = AssistedExcitation(channels=channels, sigma=0.5, alpha_max=1.0)
    ae_module.train()

    print(f"Input shape: {x.shape}")
    output = ae_module(x, targets)
    print(f"Output shape: {output.shape}")
    print(f"Current alpha: {ae_module.get_current_alpha():.6f}")

    # Test iteration management
    print(f"\nIteration management:")
    print(f"Initial iteration: {ae_module.current_iter}")

    ae_module.set_iteration(1000)
    print(f"Alpha after setting to 1000: {ae_module.get_current_alpha():.6f}")

    ae_module.set_iteration(25000)  # Half of max_iters
    print(f"Alpha at half training: {ae_module.get_current_alpha():.6f}")

    ae_module.set_iteration(50000)  # Max iterations
    print(f"Alpha at max iterations: {ae_module.get_current_alpha():.6f}")

    print("\nAssistedExcitation implementation complete!")