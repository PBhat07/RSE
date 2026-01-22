import torch
from typing import NamedTuple, Optional

# --- 1. MOCK CLASSES AND FUNCTIONS ---

# Since NamedTuple is immutable, use a mutable class to simulate viewpoint_cam
class MockCamera:
    def __init__(self, original_depth):
        # original_depth will be a tensor on 'cpu' loaded from cache
        self.original_depth = original_depth
        self.image_height = 10 
        self.image_width = 10

# Simplified depth loss function (Mean Absolute Error for demonstration)
def mock_depth_loss(rendered_depth, rendered_weight, gt_depth):
    # Assume rendered_weight is all ones for simplicity
    mask = rendered_weight > 0.5 
    # Use L1 (MAE) for a simple loss
    return torch.abs(rendered_depth[mask] - gt_depth[mask]).mean()

# --- 2. SETUP MOCK DATA ---

if not torch.cuda.is_available():
    print("❌ ERROR: CUDA is not available. Cannot run GPU transfer verification.")
    exit()

# Mock Ground Truth Depth (Simulates loaded from disk on CPU)
MOCK_GT_DEPTH_CPU = torch.arange(100).reshape(1, 1, 10, 10).float().cpu() + 1.0 
# Mock Rendered Data (Simulates output of render, must be on CUDA)
MOCK_RENDERED_DEPTH = torch.ones_like(MOCK_GT_DEPTH_CPU).cuda() * 50.0 
MOCK_RENDERED_WEIGHT = torch.ones_like(MOCK_RENDERED_DEPTH) 

# Create the Mock Camera
viewpoint_cam = MockCamera(original_depth=MOCK_GT_DEPTH_CPU)

# Define mock options (assuming lambda_depth is positive to run the loss)
class MockOpt(NamedTuple):
    lambda_depth: float

opt = MockOpt(lambda_depth=1.0) # Enable depth loss

print(f"--- STARTING DEPTH LOSS VERIFICATION ---")
print(f"Initial GT Tensor Device: {viewpoint_cam.original_depth.device}")
print(f"Rendered Tensor Device: {MOCK_RENDERED_DEPTH.device}")
print("------------------------------------------")


# --- 3. EXECUTE VERIFIED LOSS LOGIC FROM TRAIN.PY ---

# Only attempt depth loss if a GT depth tensor was loaded and is NOT None
gt_depth_tensor = viewpoint_cam.original_depth
losses = {}

if gt_depth_tensor is not None:
    # CRITICAL: Ensure the tensor is on CUDA for loss calculation
    if gt_depth_tensor.device != torch.device("cuda"):
        # This is the line that performs the transfer!
        gt_depth = gt_depth_tensor.to(torch.device("cuda"), non_blocking=True)
        print(f"INFO: GT tensor moved from CPU to {gt_depth.device}")
    else:
        gt_depth = gt_depth_tensor
    
    # Check the actual loss calculation
    if opt.lambda_depth > 0:
        raw_loss = mock_depth_loss(MOCK_RENDERED_DEPTH, MOCK_RENDERED_WEIGHT, gt_depth)
        losses['depth'] = raw_loss * opt.lambda_depth

    print("\n✅ Verification SUCCESSFUL.")
    print(f"Final GT Tensor Device: {gt_depth.device}")
    print(f"GT Depth Min/Max: {gt_depth.min().item():.2f} / {gt_depth.max().item():.2f}")
    print(f"Calculated Depth Loss: {losses['depth'].item():.4f}")
    
else:
    print("\n❌ Verification FAILED: viewpoint_cam.original_depth was None (This should not happen after filtering!)")

print("------------------------------------------")