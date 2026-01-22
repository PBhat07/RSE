import torch
import torch.nn.functional as F
import random
from diffusers import StableDiffusionPipeline, DDPMScheduler
from transformers import CLIPTextModel, CLIPTokenizer
from tqdm import tqdm

# --- Configuration for SDS ---
SDS_MODEL_ID = "runwayml/stable-diffusion-v1-5"
VAE_SCALE_FACTOR = 0.18215 # Standard VAE scaling factor for SD 1.x models

# A simple wrapper class to hold the pre-initialized SDS components
class SDSModel:
    def __init__(self, vae, tokenizer, text_encoder, unet, scheduler, device):
        self.vae = vae.to(device)
        self.tokenizer = tokenizer
        self.text_encoder = text_encoder.to(device)
        self.unet = unet.to(device)
        self.scheduler = scheduler
        self.device = device
        self.prompt_embeds = {}

        # Set scheduler timesteps once upon initialization
        self.scheduler.set_timesteps(1000, device=device)
        self.timesteps = self.scheduler.timesteps
        self.alphas = self.scheduler.alphas_cumprod.to(device)


def initialize_sds_model(device: str = "cuda") -> SDSModel:
    """
    Initializes the necessary components for Score Distillation Sampling (SDS).
    Loads the VAE, Tokenizer, Text Encoder, UNet, and DDPMScheduler.
    """
    print(f"Loading SDS Model components from {SDS_MODEL_ID}...")
    
    try:
        # Load components with half-precision to save VRAM
        pipeline = StableDiffusionPipeline.from_pretrained(
            SDS_MODEL_ID, 
            torch_dtype=torch.float16,
        )
    except Exception as e:
        print(f"Failed to load StableDiffusionPipeline: {e}")
        print("Please ensure you have `diffusers` and `transformers` installed and access to the model ID.")
        # Return a dummy model if loading fails to allow code structure analysis
        class DummyModule:
            def to(self, *args): return self
            def __call__(self, *args, **kwargs): return torch.zeros(1)
            def alphas_cumprod(self): return torch.zeros(1000)
            def set_timesteps(self, *args, **kwargs): return
            def timesteps(self): return torch.zeros(1000)

        return SDSModel(DummyModule(), DummyModule(), DummyModule(), DummyModule(), DDPMScheduler(), device)
        
    vae = pipeline.vae
    tokenizer = pipeline.tokenizer
    text_encoder = pipeline.text_encoder
    unet = pipeline.unet
    
    # Use DDPMScheduler, which is standard for SDS/DreamFusion
    scheduler = DDPMScheduler.from_pretrained(SDS_MODEL_ID, subfolder="scheduler")

    # Clean up pipeline object
    del pipeline
    torch.cuda.empty_cache()

    print("SDS Model components loaded successfully.")
    
    return SDSModel(vae, tokenizer, text_encoder, unet, scheduler, device)


def get_text_embeds(sds_model: SDSModel, prompt: str, negative_prompt: str):
    """
    Calculates and caches the conditional (c) and unconditional (u) text embeddings.
    """
    device = sds_model.device
    
    # Check cache first
    cache_key = (prompt, negative_prompt)
    if cache_key in sds_model.prompt_embeds:
        return sds_model.prompt_embeds[cache_key]

    # Tokenize and encode
    prompt_ids = sds_model.tokenizer(
        prompt,
        padding="max_length",
        max_length=sds_model.tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt"
    ).input_ids.to(device)
    
    uncond_ids = sds_model.tokenizer(
        negative_prompt,
        padding="max_length",
        max_length=sds_model.tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt"
    ).input_ids.to(device)

    # Encode embeddings
    with torch.no_grad():
        prompt_embeds = sds_model.text_encoder(prompt_ids)[0]
        uncond_embeds = sds_model.text_encoder(uncond_ids)[0]

    # Concatenate for single UNet forward pass (B*2, SeqLen, EmbDim)
    text_embeds = torch.cat([uncond_embeds, prompt_embeds])
    
    # Cache and return
    sds_model.prompt_embeds[cache_key] = text_embeds
    return text_embeds


def sds_loss(rendered_image: torch.Tensor, sds_model: SDSModel, opt, iteration: int) -> torch.Tensor:
    """
    Computes the Score Distillation Sampling (SDS) loss.

    Args:
        rendered_image (torch.Tensor): The RGB image rendered by Gaussian Splatting, 
                                       shape [3, H, W], in [0, 1] range, on CUDA.
        sds_model (SDSModel): The initialized diffusion model components.
        opt: The OptimizationParams namespace (must contain sds_prompt, sds_negative_prompt, 
             sds_cfg_scale, sds_min_step, sds_max_step).
        iteration (int): The current training iteration (unused, but kept for signature alignment).

    Returns:
        torch.Tensor: The scalar SDS loss.
    """
    
    # --- 1. Get CFG parameters from opt ---
    prompt = getattr(opt, 'sds_prompt', "a professional photograph of a beautiful 3D object")
    negative_prompt = getattr(opt, 'sds_negative_prompt', "")
    cfg_scale = getattr(opt, 'sds_cfg_scale', 7.5)
    min_step = getattr(opt, 'sds_min_step', 0.02)
    max_step = getattr(opt, 'sds_max_step', 0.98)
    
    device = sds_model.device
    
    # --- 2. Preprocess rendered image ---
    # a. Add batch dimension: [3, H, W] -> [1, 3, H, W]
    image_batch = rendered_image.unsqueeze(0).to(device) 
    
    # b. Normalize to [-1, 1] for VAE input
    image_batch = image_batch * 2.0 - 1.0 

    # --- 3. VAE Encode (Image -> Latent) ---
    with torch.no_grad():
        # Encode to latent space
        latent_dist = sds_model.vae.encode(image_batch.half()).latent_dist
        # Sample from the posterior (or just use the mean)
        latents = latent_dist.sample() * VAE_SCALE_FACTOR
    
    # --- 4. Get Text Embeddings ---
    text_embeds = get_text_embeds(sds_model, prompt, negative_prompt)
    
    # --- 5. Sample Timestep and Noise ---
    # Select the range for sampling (based on min/max step as percentage of 1000 total steps)
    min_t = int(min_step * 1000)
    max_t = int(max_step * 1000)
    
    # Randomly select a timestep index within the valid range
    t_idx = random.randint(min_t, max_t)
    t = sds_model.timesteps[t_idx] 
    
    # Sample random Gaussian noise
    noise = torch.randn_like(latents)

    # Add noise to the latents
    latents_noisy = sds_model.scheduler.add_noise(latents, noise, t).to(device)

    # --- 6. Predict Noise (UNet forward pass) ---
    # UNet input needs (B*2, 4, H/8, W/8) for CFG
    latent_model_input = torch.cat([latents_noisy] * 2) 
    t_batch = torch.cat([t.unsqueeze(0)] * 2).to(device)

    # Model predicts noise (epsilon_theta)
    noise_pred = sds_model.unet(latent_model_input.half(), t_batch, encoder_hidden_states=text_embeds.half()).sample.to(latents.dtype)

    # --- 7. Perform Classifier-Free Guidance (CFG) ---
    # Separate unconditional and conditional predictions
    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2) 
    
    # Guided noise estimate: \hat{\epsilon} = \epsilon_u + w (\epsilon_c - \epsilon_u)
    noise_pred_guided = noise_pred_uncond + cfg_scale * (noise_pred_text - noise_pred_uncond)

    # --- 8. Compute Loss and Weighting (Standard SDS Objective) ---
    
    # Get alpha_t (a_t) and sigma_t (s_t) for the current timestep
    alphas = sds_model.alphas
    sqrt_alpha_prod = alphas[t_idx].sqrt()
    sqrt_one_minus_alpha_prod = (1.0 - alphas[t_idx]).sqrt()
    
    # Weighting: w(t) = \frac{\sigma_t}{\sqrt{\alpha_t}} (a common variant of SDS/v-prediction scaling)
    # The original paper uses the DDIM objective weight, but this variance-scaled weight
    # is common in modern implementations to balance the loss.
    weight = sqrt_one_minus_alpha_prod / sqrt_alpha_prod

    # Compute loss (L2 distance between the guided noise prediction and the sampled noise)
    sds_loss_value = weight * F.mse_loss(noise_pred_guided, noise, reduction="mean")
    
    return sds_loss_value