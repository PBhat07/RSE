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
    # NOTE: Using the Depth Anything V2 Small model
    model = pipeline(task="depth-estimation", model="depth-anything/Depth-Anything-V2-Small-hf", device=device)
    print("Depth Anything V2 model initialized.")
    return model

def get_depth_map(depth_estimator, image_tensor):
    """
    Estimates the depth map for a given image tensor using the depth estimator.
    The output is a single-channel, uncalibrated inverse-disparity map (closer = smaller value),
    which is then prepared as a torch tensor.
    
    NOTE: We explicitly DO NOT perform per-image normalization here.
    The Scale-Invariant Log Loss (SILog) handles the scale mismatch.
    """
    # Convert tensor (C, H, W) -> numpy (H, W, C) -> uint8 -> PIL
    image_np = (image_tensor.permute(1, 2, 0).cpu().numpy() * 255).astype("uint8")
    image_pil = Image.fromarray(image_np)

    # Run depth estimator: gt_depth is a PIL Image of raw uncalibrated disparity
    gt_depth = depth_estimator(image_pil)["depth"]
    gt_depth_np = np.array(gt_depth, dtype=np.float32)

    # Convert to torch tensor on GPU and add dimensions (1, 1, H, W).
    return torch.from_numpy(gt_depth_np).float().cuda().unsqueeze(0).unsqueeze(0)

def depth_loss(rendered_depth, rendered_weight, gt_depth):
    """
    Calculates a weighted, Scale-Invariant Logarithmic Loss (SILog) 
    between the rendered depth and uncalibrated GT depth.
    
    This is the robust solution for uncalibrated/disparity depth, 
    as it accounts for scale and shift differences.
    """
    # 1. Ensure consistent 4D shape (1, 1, H, W) and device alignment
    if len(rendered_depth.shape) == 2:
        rendered_depth = rendered_depth.unsqueeze(0).unsqueeze(0)
    elif len(rendered_depth.shape) == 3:
        rendered_depth = rendered_depth.unsqueeze(0)
    
    if len(rendered_weight.shape) == 2:
        rendered_weight = rendered_weight.unsqueeze(0).unsqueeze(0)
    elif len(rendered_weight.shape) == 3:
        rendered_weight = rendered_weight.unsqueeze(0)
        
    rendered_depth = rendered_depth.to(gt_depth.device)

    # 2. Size Alignment Check: Interpolate gt_depth to match rendered_depth's size.
    if gt_depth.shape[-2:] != rendered_depth.shape[-2:]:
        gt_depth = F.interpolate(
            gt_depth,
            size=rendered_depth.shape[-2:],
            mode="bilinear",
            align_corners=False
        )

    # 3. Create Mask: Focus loss only where geometry is rendered (high opacity)
    mask = rendered_weight > 0.5
    
    # Check if there are any valid pixels to calculate loss
    if not torch.any(mask):
        return torch.zeros_like(rendered_depth).mean()

    # --- CRITICAL: Scale-Invariant Log Loss (SILog) ---
    
    # 4. Extract valid depth and GT values
    # Add a small epsilon to prevent log(0) - crucial for stability
    epsilon = 1e-6 
    
    d_rendered = rendered_depth[mask] + epsilon
    d_gt = gt_depth[mask] + epsilon

    # 5. Calculate the log differences
    log_diff = torch.log(d_rendered) - torch.log(d_gt)
    
    # The SILog loss is the mean of the squared log difference.
    # It removes the constant translation (shift) and rotation (scale) factors 
    # between the two depth maps.
    silog_loss = torch.mean(log_diff ** 2) - 0.5 * torch.mean(log_diff) ** 2

    # You may also want to combine this with a small penalty for the variance of the log difference.
    # We use the most common, simplified SILog form here.
    
    return silog_loss

