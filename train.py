import os
import torch
import numpy as np
from PIL import Image
from torch.utils.data import DataLoader
import torch.nn.functional as F
from random import randint
from utils.sds_loss import initialize_sds_model, sds_loss # Correct import
from utils.loss_utils import l1_loss, ssim
# We keep initialize_depth_estimator/get_depth_map/depth_loss for potential future use or if depth_loss is used
from utils.depth_utils import depth_loss 
from gaussian_renderer import render, network_gui
from mesh_renderer import NVDiffRenderer
import sys
from scene import Scene, GaussianModel, FlameGaussianModel
from utils.general_utils import safe_state, PILtoTorch
from torchvision.utils import save_image
import uuid
from tqdm import tqdm
from utils.image_utils import psnr, error_map
from lpipsPyTorch import lpips
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams

try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

# =========================================================================
# NEW HELPER FUNCTION: Direct and Simple GT Depth Fetch
# This simplifies the entire data pipeline for depth.
# =========================================================================
def get_gt_depth_on_cuda(viewpoint_cam):
    """
    Attempts to load the GT depth tensor from the expected path and moves it to CUDA.
    Returns the CUDA tensor (1, 1, H, W) or None if the file is missing.
    """
    # CRITICAL: We rely on the camera object having the 'original_depth_path' property set.
    depth_path = None
    if hasattr(viewpoint_cam, 'original_depth_path') and viewpoint_cam.original_depth_path:
        depth_path = viewpoint_cam.original_depth_path
    
    # If the camera object didn't set the explicit path, try to deduce it (common structure)
    elif hasattr(viewpoint_cam, 'image_path'):
        # Fallback deduction based on image path (adjust if your data structure is different)
        img_path = viewpoint_cam.image_path
        # Example: change 'images/xxx.png' to 'depths/xxx.pt' or similar
        depth_path = img_path.replace('images', 'depths').rsplit('.', 1)[0] + '.pt'

    if depth_path and os.path.exists(depth_path):
        try:
            # Load the tensor directly from disk (it will be on CPU)
            # Use weights_only=True for safety, though it often defaults to False for general torch.load
            gt_depth_tensor = torch.load(depth_path, map_location='cpu', weights_only=True) 
            # Ensure it is moved to CUDA for loss calculation
            return gt_depth_tensor.to(torch.device("cuda"), non_blocking=True)
        except Exception as e:
            print(f"[ERROR] Could not load and move depth map from {depth_path}: {e}")
            return None
    else:
        return None

def training(dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from):
    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset)
    if dataset.bind_to_mesh:
        gaussians = FlameGaussianModel(dataset.sh_degree, dataset.disable_flame_static_offset, dataset.not_finetune_flame_params)
        mesh_renderer = NVDiffRenderer()
    else:
        gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians)
    gaussians.training_setup(opt)

    # --- SIMPLIFIED DATA LOADING: REMOVED ALL PRELOADING AND FILTERING ---
    # The depth map will be fetched on-demand inside the training loop.
    
    if checkpoint:
        ckpt = torch.load(checkpoint, map_location='cpu')
        (model_params, first_iter) = (ckpt["model"], ckpt["iteration"]) if isinstance(ckpt, dict) else ckpt
        gaussians.restore(model_params, opt)
        # Removed: Depth cache restoration from checkpoint.

    # --- REMOVED ALL CAMERA FILTERING LOGIC ---

    NUM_VIS_FRAMES = 10
    all_test_cameras = scene.getTestCameras()
    
    # Randomly select 10 unique indices and fix them for consistent visualization
    fixed_vis_camera_indices = torch.randperm(len(all_test_cameras))[:NUM_VIS_FRAMES].tolist()
    print(f"Selected fixed camera indices for visualization: {fixed_vis_camera_indices}")

    # Pass the actual camera objects for the report function to fetch depth on demand
    vis_cameras_for_report = [all_test_cameras[i] for i in fixed_vis_camera_indices]
    print("[INFO] Visualization setup complete.")
    

    # NEW CODE: Initialize the SDS model if the loss weight is positive
    sds_model = None
    if opt.lambda_sds > 0:
        print("[INFO] Initializing SDS model for Score Distillation Sampling...")
        # Assuming initialize_sds_model is imported from utils.sds_loss
        sds_model = initialize_sds_model(device="cuda")
        print("[INFO] SDS model initialized.")

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    # Increase num_workers for faster camera loading
    # NOTE: scene.getTrainCameras() returns the full list now.
    loader_camera_train = DataLoader(scene.getTrainCameras(), batch_size=None, shuffle=True, num_workers=8, pin_memory=True, persistent_workers=True)
    iter_camera_train = iter(loader_camera_train)
    
    ema_loss_for_log = 0.0
    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1
    for iteration in range(first_iter, opt.iterations + 1):
        if network_gui.conn == None:
            network_gui.try_connect()
        while network_gui.conn != None:
            try:
                # receive data
                net_image = None
                custom_cam, msg = network_gui.receive()

                # ... (GUI rendering logic remains unchanged) ...

                # render
                if custom_cam != None:
                    # mesh selection by timestep
                    if gaussians.binding != None:
                        gaussians.select_mesh_by_timestep(custom_cam.timestep, msg['use_original_mesh'])

                    # gaussian splatting rendering
                    if msg['show_splatting']:
                        net_image = render(custom_cam, gaussians, pipe, background, msg['scaling_modifier'])["render"]

                    # mesh rendering
                    if gaussians.binding != None and msg['show_mesh']:
                        out_dict = mesh_renderer.render_from_camera(gaussians.verts, gaussians.faces, custom_cam)

                        rgba_mesh = out_dict['rgba'].squeeze(0).permute(2, 0, 1)  # (C, W, H)
                        rgb_mesh = rgba_mesh[:3, :, :]
                        alpha_mesh = rgba_mesh[3:, :, :]

                        mesh_opacity = msg['mesh_opacity']
                        if net_image is None:
                            net_image = rgb_mesh
                        else:
                            net_image = rgb_mesh * alpha_mesh * mesh_opacity  + net_image * (alpha_mesh * (1 - mesh_opacity) + (1 - alpha_mesh))

                    # send data
                    net_dict = {'num_timesteps': gaussians.num_timesteps, 'num_points': gaussians._xyz.shape[0]}
                    network_gui.send(net_image, net_dict)
                if msg['do_training'] and ((iteration < int(opt.iterations)) or not msg['keep_alive']):
                    break
            except Exception as e:
                # print(e)
                network_gui.conn = None

        iter_start.record()

        gaussians.update_learning_rate(iteration)

        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        try:
            viewpoint_cam = next(iter_camera_train)
            
        except StopIteration:
            iter_camera_train = iter(loader_camera_train)
            viewpoint_cam = next(iter_camera_train)

        if gaussians.binding != None:
            gaussians.select_mesh_by_timestep(viewpoint_cam.timestep)

        # Render
        if (iteration - 1) == debug_from:
            pipe.debug = True
        render_pkg = render(viewpoint_cam, gaussians, pipe, background)
        image, viewspace_point_tensor, visibility_filter, radii, rendered_depth, rendered_weight = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"], render_pkg["rendered_depth"], render_pkg["rendered_weight"]

        # Loss
        gt_image = viewpoint_cam.original_image.cuda()

        losses = {}
        losses['l1'] = l1_loss(image, gt_image) * (1.0 - opt.lambda_dssim)
        losses['ssim'] = (1.0 - ssim(image, gt_image)) * opt.lambda_dssim


        #print("--- DEPTH LOSS SANITY CHECK (On-Demand Fetch) ---")
        #print(f"Rendered Depth (Min/Max): {rendered_depth.min().item():.2f} / {rendered_depth.max().item():.2f}")
        
        # --- SIMPLIFIED DEPTH LOSS LOGIC ---
        gt_depth = None
        if opt.lambda_depth > 0:
            # Fetch GT depth directly from disk and move to CUDA
            gt_depth = get_gt_depth_on_cuda(viewpoint_cam)
        
        if gt_depth is not None:
            #print(f"GT Depth (Min/Max): {gt_depth.min().item():.2f} / {gt_depth.max().item():.2f}")
            # Calculate depth loss (assuming depth_loss is imported from utils.depth_utils)
            losses['depth'] = depth_loss(rendered_depth, rendered_weight, gt_depth) * opt.lambda_depth
       # else:
            # This will print when the GT depth file is missing for a camera
           # print("GT Depth (Min/Max): N/A (Missing Cache File)") 
            
       # print("-------------------------------------------------")
        # --- END SIMPLIFIED DEPTH LOSS LOGIC ---


        # NEW CODE: Calculate SDS loss if enabled and model is initialized
        if opt.lambda_sds > 0 and sds_model is not None:
            losses['sds'] = sds_loss(image, sds_model, opt, iteration) * opt.lambda_sds

        if gaussians.binding != None:
            if opt.metric_xyz:
                losses['xyz'] = F.relu((gaussians._xyz*gaussians.face_scaling[gaussians.binding])[visibility_filter] - opt.threshold_xyz).norm(dim=1).mean() * opt.lambda_xyz
            else:
                # losses['xyz'] = gaussians._xyz.norm(dim=1).mean() * opt.lambda_xyz
                losses['xyz'] = F.relu(gaussians._xyz[visibility_filter].norm(dim=1) - opt.threshold_xyz).mean() * opt.lambda_xyz

            if opt.lambda_scale != 0:
                if opt.metric_scale:
                    losses['scale'] = F.relu(gaussians.get_scaling[visibility_filter] - opt.threshold_scale).norm(dim=1).mean() * opt.lambda_scale
                else:
                    # losses['scale'] = F.relu(gaussians._scaling).norm(dim=1).mean() * opt.lambda_scale
                    losses['scale'] = F.relu(torch.exp(gaussians._scaling[visibility_filter]) - opt.threshold_scale).norm(dim=1).mean() * opt.lambda_scale

            if opt.lambda_dynamic_offset != 0:
                losses['dy_off'] = gaussians.compute_dynamic_offset_loss() * opt.lambda_dynamic_offset

            if opt.lambda_dynamic_offset_std != 0:
                ti = viewpoint_cam.timestep
                t_indices =[ti]
                if ti > 0:
                    t_indices.append(ti-1)
                if ti < gaussians.num_timesteps - 1:
                    t_indices.append(ti+1)
                losses['dynamic_offset_std'] = gaussians.flame_param['dynamic_offset'].std(dim=0).mean() * opt.lambda_dynamic_offset_std

            if opt.lambda_laplacian != 0:
                losses['lap'] = gaussians.compute_laplacian_loss() * opt.lambda_laplacian

        losses['total'] = sum([v for k, v in losses.items()])
        losses['total'].backward()

        iter_end.record()

        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * losses['total'].item() + 0.6 * ema_loss_for_log
            if iteration % 10 == 0:
                postfix = {"Loss": f"{ema_loss_for_log:.{7}f}"}
                if 'xyz' in losses:
                    postfix["xyz"] = f"{losses['xyz']:.{7}f}"
                if 'scale' in losses:
                    postfix["scale"] = f"{losses['scale']:.{7}f}"
                if 'dy_off' in losses:
                    postfix["dy_off"] = f"{losses['dy_off']:.{7}f}"
                if 'lap' in losses:
                    postfix["lap"] = f"{losses['lap']:.{7}f}"
                if 'dynamic_offset_std' in losses:
                    postfix["dynamic_offset_std"] = f"{losses['dynamic_offset_std']:.{7}f}"
                # NEW CODE: Add SDS loss to progress bar
                if 'sds' in losses:
                    postfix["sds"] = f"{losses['sds']:.{7}f}"
                progress_bar.set_postfix(postfix)
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            # Log and save
            # We pass the list of visualization cameras instead of a pre-cached depth list (vis_cameras_for_report)
            training_report(tb_writer, iteration, losses, iter_start.elapsed_time(iter_end), testing_iterations, scene, render, (pipe, background), vis_cameras_for_report, fixed_vis_camera_indices)
            if (iteration in saving_iterations):
                print("[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)

            # Densification
            if iteration < opt.densify_until_iter:
                # Keep track of max radii in image-space for pruning
                gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                    gaussians.densify_and_prune(opt.densify_grad_threshold, 0.005, scene.cameras_extent, size_threshold)

                if iteration % opt.opacity_reset_interval == 0 or (dataset.white_background and iteration == opt.densify_from_iter):
                    gaussians.reset_opacity()

            # Optimizer step
            if iteration < opt.iterations:
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none = True)

            if (iteration in checkpoint_iterations):
                print(f"[ITER {iteration}] Saving Checkpoint (Without Depth Cache)")

                # --- SIMPLIFIED CHECKPOINT SAVE ---
                checkpoint_path = f"{scene.model_path}/chkpnt{iteration}.pth"
                torch.save({
                    "model": gaussians.capture(),
                    "iteration": iteration
                }, checkpoint_path)

                print(f"[INFO] Checkpoint saved: {checkpoint_path}")

def prepare_output_and_logger(args):
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str=os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])

    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer

# NOTE: The signature has been updated to accept the list of visualization cameras
def training_report(tb_writer, iteration, losses, elapsed, testing_iterations, scene : Scene, renderFunc, renderArgs, vis_cameras_for_report, fixed_vis_camera_indices):
    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/l1_loss', losses['l1'].item(), iteration)
        tb_writer.add_scalar('train_loss_patches/ssim_loss', losses['ssim'].item(), iteration)
        if 'xyz' in losses:
            tb_writer.add_scalar('train_loss_patches/xyz_loss', losses['xyz'].item(), iteration)
        if 'scale' in losses:
            tb_writer.add_scalar('train_loss_patches/scale_loss', losses['scale'].item(), iteration)
        if 'dy_off' in losses:
            tb_writer.add_scalar('train_loss_patches/dy_off', losses['dy_off'].item(), iteration)
        if 'lap' in losses:
            tb_writer.add_scalar('train_loss_patches/lap', losses['lap'].item(), iteration)
        if 'dynamic_offset_std' in losses:
            tb_writer.add_scalar('train_loss_patches/dynamic_offset_std', losses['dynamic_offset_std'].item(), iteration)
        # NEW CODE: Add SDS loss to TensorBoard
        if 'sds' in losses:
            tb_writer.add_scalar('train_loss_patches/sds_loss', losses['sds'].item(), iteration)
        
        tb_writer.add_scalar('train_loss_patches/total_loss', losses['total'].item(), iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)
        # NEW CODE: Add depth loss to TensorBoard
        if 'depth' in losses:
            tb_writer.add_scalar('train_loss_patches/depth_loss', losses['depth'].item(), iteration)

    # Report test and samples of training set
    if iteration in testing_iterations:
        print("[ITER {}] Evaluating".format(iteration))
        torch.cuda.empty_cache()

        # =========================================================================
        # NEW BLOCK: Save 3-way visualization (RGB | Rendered Depth | GT Depth)
        # We now fetch the GT depth on demand inside the report function.
        # =========================================================================
        TARGET_VIS_ITERATIONS = [1, 6000, 300000, 360000, 420000, 540000]
        
        # START of the critical TRY/EXCEPT block
        try:
            if iteration in TARGET_VIS_ITERATIONS:
                
                print(f"[ITER {iteration}] Saving 3-way depth visualization for fixed frames.")
                
                save_dir = os.path.join(scene.model_path, f"depth_vis_iter_{iteration}")
                os.makedirs(save_dir, exist_ok=True)
                
                # Iterate over the 10 FIXED random camera views
                for i, viewpoint in enumerate(vis_cameras_for_report):
                    
                    if scene.gaussians.num_timesteps > 1:
                        scene.gaussians.select_mesh_by_timestep(viewpoint.timestep)
                    
                    # --- 1. RENDER (from Gaussian Model) ---
                    rendering = renderFunc(viewpoint, scene.gaussians, *renderArgs)
                    rendered_rgb = torch.clamp(rendering["render"], 0.0, 1.0).cpu() # [3, H, W]
                    rendered_depth_tensor = rendering["rendered_depth"].cpu()

                    # Ensure rendered_depth is 3D (1, H, W) before normalization
                    if rendered_depth_tensor.dim() == 2:
                        rendered_depth_tensor = rendered_depth_tensor.unsqueeze(0)

                    # --- 2. GT DATA (On-Demand Fetch) ---
                    original_rgb = viewpoint.original_image.cpu() # [3, H, W]
                    
                    # Fetch GT depth on demand (CUDA tensor, then move to CPU for visualization)
                    gt_depth_tensor = get_gt_depth_on_cuda(viewpoint)
                    
                    # If depth is missing, use a black/zero image for visualization
                    if gt_depth_tensor is not None:
                         gt_depth_tensor_cpu = gt_depth_tensor.cpu() 
                    else:
                        H = viewpoint.image_height
                        W = viewpoint.image_width
                        gt_depth_tensor_cpu = torch.zeros((1, H, W), dtype=torch.float32)

                    # --- 3. NORMALIZE DEPTHS FOR VISUALIZATION ---
                    
                    # Normalize rendered depth
                    d_r_min, d_r_max = rendered_depth_tensor.min(), rendered_depth_tensor.max()
                    rendered_depth_norm = torch.clamp((rendered_depth_tensor - d_r_min) / (d_r_max - d_r_min + 1e-8), 0.0, 1.0)
                    rendered_depth_vis = rendered_depth_norm.squeeze().unsqueeze(0).repeat(3, 1, 1) # 1ch -> 3ch
                    
                    
                    # Normalize GT depth
                    gt_depth_tensor_cpu = gt_depth_tensor_cpu.clone()
                    if gt_depth_tensor_cpu.dim() == 2:
                        gt_depth_tensor_cpu = gt_depth_tensor_cpu.unsqueeze(0)
                    
                    d_gt_min = gt_depth_tensor_cpu.min()
                    d_gt_max = gt_depth_tensor_cpu.max()
                    
                    if (d_gt_max - d_gt_min).abs() < 1e-8:
                        # Depth is uniform or zero
                        gt_depth_norm = torch.zeros_like(gt_depth_tensor_cpu)
                    else:
                        gt_depth_norm = torch.clamp((gt_depth_tensor_cpu - d_gt_min) / (d_gt_max - d_gt_min), 0.0, 1.0)

                    gt_depth_vis = gt_depth_norm.repeat(3, 1, 1) # 1ch -> 3ch

                    # --- 4. STACK AND SAVE: RGB | Rendered Depth | GT Depth ---
                    comparison_image = torch.cat([original_rgb, rendered_depth_vis, gt_depth_vis], dim=2)
                    
                    image_path = os.path.join(save_dir, f"frame_{i:02d}_comparison.png")
                    save_image(comparison_image, image_path)

        except Exception as e:
            # This catches a failure that isn't isolated to a single frame
            print(f"\n[CRITICAL WARNING] Depth Visualization FAILED completely at iteration {iteration}: {e}")
            print("Ignoring visualization errors for this iteration and CONTINUING training...")
            pass 
        # END of the critical TRY/EXCEPT block
        
        # ... (rest of the training_report function remains unchanged) ...
        validation_configs = (
            {'name': 'val', 'cameras' : scene.getValCameras()},
            {'name': 'test', 'cameras' : scene.getTestCameras()},
        )

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                ssim_test = 0.0
                lpips_test = 0.0
                num_vis_img = 10
                image_cache = []
                gt_image_cache = []
                vis_ct = 0
                for idx, viewpoint in tqdm(enumerate(DataLoader(config['cameras'], shuffle=False, batch_size=None, num_workers=8)), total=len(config['cameras'])):
                    if scene.gaussians.num_timesteps > 1:
                        scene.gaussians.select_mesh_by_timestep(viewpoint.timestep)
                    image = torch.clamp(renderFunc(viewpoint, scene.gaussians, *renderArgs)["render"], 0.0, 1.0)
                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    if tb_writer and (idx % (len(config['cameras']) // num_vis_img) == 0):
                        tb_writer.add_images(config['name'] + "_{}/render".format(vis_ct), image[None], global_step=iteration)
                        error_image = error_map(image, gt_image)
                        tb_writer.add_images(config['name'] + "_{}/error".format(vis_ct), error_image[None], global_step=iteration)
                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(config['name'] + "_{}/ground_truth".format(vis_ct), gt_image[None], global_step=iteration)
                        vis_ct += 1
                    l1_test += l1_loss(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()
                    ssim_test += ssim(image, gt_image).mean().double()

                    image_cache.append(image)
                    gt_image_cache.append(gt_image)

                    if idx == len(config['cameras']) - 1 or len(image_cache) == 16:
                        batch_img = torch.stack(image_cache, dim=0)
                        batch_gt_img = torch.stack(gt_image_cache, dim=0)
                        lpips_test += lpips(batch_img, batch_gt_img).sum().double()
                        image_cache = []
                        gt_image_cache = []

                psnr_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])
                lpips_test /= len(config['cameras'])
                ssim_test /= len(config['cameras'])
                print("[ITER {}] Evaluating {}: L1 {:.4f} PSNR {:.4f} SSIM {:.4f} LPIPS {:.4f}".format(iteration, config['name'], l1_test, psnr_test, ssim_test, lpips_test))
                if tb_writer:
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - ssim', ssim_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - lpips', lpips_test, iteration)

        if tb_writer:
            tb_writer.add_histogram("scene/opacity_histogram", scene.gaussians.get_opacity, iteration)
            tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)
        torch.cuda.empty_cache()


if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--interval", type=int, default=60_000, help="A shared iteration interval for test and saving results and checkpoints.")
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default = None)
    #parser.add_argument("--lambda_sds", type=float, default=0.0, help="Weight for Score Distillation Sampling loss.")
    parser.add_argument("--sds_prompt", type=str, default="A beautiful digital rendering of a subject.", help="Text prompt for SDS loss.")
    parser.add_argument("--sds_negative_prompt", type=str, default="", help="Negative prompt for SDS loss.")
    parser.add_argument("--sds_cfg_scale", type=float, default=0.0, help="CFG scale for SDS guidance.")
    parser.add_argument("--sds_min_step", type=float, default=0.02, help="Minimum timestep percentage for SDS sampling.")
    parser.add_argument("--sds_max_step", type=float, default=0.98, help="Maximum timestep percentage for SDS sampling.")
    args = parser.parse_args(sys.argv[1:])
    if args.interval > op.iterations:
        args.interval = op.iterations // 5
    if len(args.test_iterations) == 0:
        args.test_iterations.extend(list(range(args.interval, args.iterations+1, args.interval)))
    if len(args.save_iterations) == 0:
        args.save_iterations.extend(list(range(args.interval, args.iterations+1, args.interval)))
    if len(args.checkpoint_iterations) == 0:
        args.checkpoint_iterations.extend(list(range(args.interval, args.iterations+1, args.interval)))
    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    # Start GUI server, configure and run training
    network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, args.debug_from)

    # All done
    print("\nTraining complete.")