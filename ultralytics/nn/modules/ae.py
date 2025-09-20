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
    Based on "Assisted Excitation of Activations: A Learning Technique to Improve Object Detectors"

    During training, this module:
    1. Creates binary masks from ground truth bounding boxes
    2. Uses cosine annealing to gradually reduce excitation strength
    3. Enhances features in object regions to improve learning

    Args:
        in_channels (int): Number of input channels
        alpha_max (float): Maximum excitation strength. Default: 1.0
        alpha_min (float): Minimum excitation strength. Default: 0.0
        max_iter (int): Maximum training iterations for annealing. Default: 50000
    """

    def __init__(self, in_channels, alpha_max=0.5, alpha_min=0.0, max_iters=50000):
        super(AssistedExcitation, self).__init__()
        self.in_channels = in_channels
        self.alpha_max = alpha_max
        self.alpha_min = alpha_min
        self.max_iter = max_iters
        self.current_iter = 0


    def _calc_excitation_factor(self):
        """
        Compute current alpha using cosine annealing schedule.
        Alpha decreases from alpha_max to alpha_min over max_iters iterations.
        """
        if self.current_iter >= self.max_iter:
            return self.alpha_min
        alpha = self.alpha_max * (1 + math.cos(math.pi * self.current_iter)) /  self.max_iter
        return alpha

    def _create_bbox_map(self, targets, height, width, batch_size, device):
        """
        Create binary masks from ground truth bounding boxes as described in the paper.

        Args:
            targets (list): List of target dictionaries containing bounding boxes
            height (int): Feature map height
            width (int): Feature map width
            batch_size(int): Batch size of each input
            device (torch.device): Device to create tensors on
        Returns:
            torch.Tensor: Binary masks of shape (batch_size, 1, height, width)
        """
        bbox_map = torch.zeros(batch_size, 1, height, width, device=device)

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
            gt = torch.zeros(height, width, device=device)

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
                        gt[y1:y2+1, x1:x2+1] = 1.0

            bbox_map[batch_idx, 0] = gt

        return bbox_map

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

        # Get current excitation strength using cosine annealing schedule
        alpha = self._calc_excitation_factor()

        # If alpha is very small, skip excitation for efficiency
        if alpha < 1e-6:
            return x

        # Create binary masks from ground truth bounding boxes
        bbox_map = self._create_bbox_map(targets, height, width, batch_size, device)

        # Compute channel-wise average to summarize all feature information
        c_avg = torch.mean(x, dim=1, keepdim=True)

        # Apply bbox mask to averaged features with alpha scaling
        excitation = alpha * bbox_map * c_avg

        # Expand excitation to match all feature channels for addition
        excitation = excitation.expand(-1, channels, -1, -1)

        # Add channel-averaged spatial enhancement to original features
        x += excitation

        # Update iteration counter for cosine annealing schedule
        self.current_iter += 1

        return x


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
    ae_module = AssistedExcitation(in_channels=channels, alpha_max=1.0)
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