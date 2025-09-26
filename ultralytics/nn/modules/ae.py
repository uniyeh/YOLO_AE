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
        alpha_max (float): Maximum excitation strength. Default: 0.5
        alpha_min (float): Minimum excitation strength. Default: 0.0
        max_iters (int): Maximum training iterations for annealing. Default: 100
    """

    def __init__(self, in_channels, alpha_max=0.5, alpha_min=0.0, max_iters=100):
        super(AssistedExcitation, self).__init__()
        self.in_channels = in_channels
        self.alpha_max = alpha_max
        self.alpha_min = alpha_min
        self.max_iters = max_iters
        self.current_iter = 0


    def _calc_excitation_factor(self):
        """
        Compute current alpha using cosine annealing schedule.
        Alpha decreases from alpha_max to alpha_min over max_iters iterations.

        Uses modified cosine annealing: alpha = alpha_max * (1 + cos(π * progress))
        where progress = current_iter / max_iters.
        """
        if self.max_iters <= 0 or self.current_iter >= self.max_iters:
            return self.alpha_min
        alpha = self.alpha_max * (1 + math.cos(math.pi * self.current_iter / self.max_iters)) 
        return alpha

    def _create_bbox_map(self, batch_data, height, width, batch_size, device):
        """
        Create binary masks from ground truth bounding boxes for each image in the batch.

        Processes YOLO's flattened bounding box format where all boxes from all images
        are stored in a single tensor with batch indices. For each image in the batch,
        creates a binary mask where pixels inside any ground truth bounding box are set
        to 1.0, and pixels outside all bounding boxes are set to 0.0. Multiple bounding
        boxes in the same image are accumulated using clamping to maintain binary values.

        Args:
            batch_data (dict): Batch data containing 'bboxes' and 'batch_idx' tensors.
                              bboxes: tensor of shape [total_boxes, 4] in YOLO format
                              [x_center, y_center, width, height] with normalized coordinates (0-1).
                              batch_idx: tensor of shape [total_boxes] indicating which image each box belongs to.
            height (int): Feature map height for the binary mask
            width (int): Feature map width for the binary mask
            batch_size (int): Number of images in the batch
            device (torch.device): Device to create tensors on (CPU/GPU)

        Returns:
            torch.Tensor: Binary masks of shape (batch_size, 1, height, width) where
                         each mask[i, 0] contains the accumulated binary mask for image i.
        """
        bbox_map = torch.zeros(batch_size, 1, height, width, device=device)
        batch_idx = batch_data['batch_idx']
        bboxes = batch_data['bboxes']
        bboxes_amt = len(bboxes)
        current_img = batch_idx[0].int()
        gt = torch.zeros(height, width, device=device)
        for i in range(bboxes_amt):
            if i >= len(batch_idx):
                break  # Safety check for out of bounds
            if batch_idx[i].int() != current_img:
                # Switched to new image, reset ground truth mask
                gt = torch.zeros(height, width, device=device)
                current_img = batch_idx[i].int()
            
            bbox = bboxes[i]
            if bbox is None:
                continue
            # Extract YOLO format coordinates: [x_center, y_center, width, height] (normalized)
            x_center, y_center, box_width, box_height = bbox

            # Convert to pixel coordinates
            x1 = max(0, int((x_center - box_width/2) * width))
            y1 = max(0, int((y_center - box_height/2) * height))
            x2 = min(width-1, int((x_center + box_width/2) * width))
            y2 = min(height-1, int((y_center + box_height/2) * height))

            # Fill bounding box region in ground truth mask
            if x2 > x1 and y2 > y1:
                gt[y1:y2+1, x1:x2+1] = 1

            # Accumulate ground truth into bbox_map with clamping to maintain binary values
            bbox_map[current_img, 0] = torch.clamp(bbox_map[current_img, 0] + gt, 0, 1)
        return bbox_map

    def forward(self, x, ground_truth=None):
        print(f"AE forward: training={self.training}, targets={'None' if ground_truth is None else len(ground_truth)}")
        """
        Forward pass with assisted excitation.

        Args:
            x (torch.Tensor): Input feature maps of shape (B, C, H, W)
            ground_truth (dict, optional): Ground truth batch data containing 'bboxes' and 'batch_idx'

        Returns:
            torch.Tensor: Enhanced feature maps of same shape as input
        """
        # Skip excitation during inference or when no ground truth is provided
        if not self.training or ground_truth is None:
            print("AE: Skipping excitation (inference or no targets)")
            return x

        batch_size, channels, height, width = x.shape
        device = x.device

        # Get current excitation strength using cosine annealing schedule
        alpha = self._calc_excitation_factor()
        print(f"AE Layer Start: Current alpha={alpha:.6f}, iteration={self.current_iter}")

        # Skip excitation if alpha is negligible for computational efficiency
        if alpha < 1e-6:
            return x

        # Create binary masks from ground truth bounding boxes
        bbox_map = self._create_bbox_map(ground_truth, height, width, batch_size, device)
        print(f"AE bbox_map generated: {len(bbox_map)}")

        # Compute channel-wise average to summarize all feature information
        c_avg = torch.mean(x, dim=1, keepdim=True)

        # Apply spatial mask to channel-averaged features with alpha scaling
        excitation = alpha * bbox_map * c_avg

        # Expand excitation to match all input feature channels
        excitation = excitation.expand(-1, channels, -1, -1)

        # Add spatial enhancement to original features (element-wise addition)
        x += excitation

        # Update iteration counter for cosine annealing schedule
        self.current_iter += 1
        print("AE layer completed.")
        return x

