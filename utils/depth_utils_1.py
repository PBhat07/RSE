import torch
import numpy as np
from transformers import pipeline
from PIL import Image
import torch.nn.functional as F

def initialize_depth_estimator(device="cuda"):
    """
    Initializes and returns the Depth Anything V2 model.
    """
    print("Initializing Depth Anything V2 model...")
    model = pipeline(task="depth-estimation", model="depth-anything/Depth-Anything-V2-Small-hf", device=device)
    print("Depth Anything V2 model initialized.")
    return model

def get_depth_map(depth_estimator, image_tensor):
    """
    Estimates the depth map for a given image tensor using the depth estimator.
    """
    # Convert tensor -> numpy (0-1) -> uint8 -> PIL
    image_np = (image_tensor.permute(1, 2, 0).cpu().numpy() * 255).astype("uint8")
    image_pil = Image.fromarray(image_np)

    # Run depth estimator
    gt_depth = depth_estimator(image_pil)["depth"]
    gt_depth_np = np.array(gt_depth)

    # Convert to torch tensor on GPU and add both batch and channel dimensions
    # so that it is (1, 1, H, W).
    return torch.from_numpy(gt_depth_np).float().cuda().unsqueeze(0).unsqueeze(0)

def depth_loss(rendered_depth, rendered_weight, gt_depth):
    """
    Calculates a weighted L1 loss between the rendered depth and ground truth depth.
    The ground truth depth is resized to match the rendered depth's dimensions.
    The loss is masked by the rendered weight map.
    """
    # Ensure rendered tensors have a consistent 4D shape (1, 1, H, W)
    # The rendered tensors from the rasterizer might be (H, W) or (1, H, W).
    # We enforce a consistent (1, 1, H, W) shape for all tensors before processing.
    if len(rendered_depth.shape) == 2:
        rendered_depth = rendered_depth.unsqueeze(0).unsqueeze(0)
    elif len(rendered_depth.shape) == 3:
        rendered_depth = rendered_depth.unsqueeze(0)
    
    if len(rendered_weight.shape) == 2:
        rendered_weight = rendered_weight.unsqueeze(0).unsqueeze(0)
    elif len(rendered_weight.shape) == 3:
        rendered_weight = rendered_weight.unsqueeze(0)

    # Size Alignment Check: Interpolate gt_depth to match rendered_depth's size.
    # gt_depth is already (1, 1, H, W) from get_depth_map, so we can interpolate directly.
    if gt_depth.shape[-2:] != rendered_depth.shape[-2:]:
        gt_depth = F.interpolate(
            gt_depth,
            size=rendered_depth.shape[-2:],
            mode="bilinear",
            align_corners=False
        )

    # Create a mask to only consider pixels with a minimum opacity (e.g., > 0.5)
    mask = rendered_weight > 0.5

    # Calculate the L1 loss only for the masked pixels
    # Ensure tensors are on the same device
    rendered_depth = rendered_depth.to(gt_depth.device)
    loss = torch.abs(rendered_depth - gt_depth)
    masked_loss = loss[mask]

    # Return the mean of the masked loss. If no pixels are masked, return 0.
    if masked_loss.numel() == 0:
        return torch.zeros_like(rendered_depth).mean()
    else:
        return masked_loss.mean()