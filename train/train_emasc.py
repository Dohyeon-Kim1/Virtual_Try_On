import os
import torch
import torch.nn.functional as F
import diffusers
import transformers
from accelerate import Accelerator
from diffusers.optimization import get_scheduler
from tqdm.auto import tqdm

from models.ladi_vton.AutoencoderKL import AutoencoderKL
from models import BodyPoseEstimation, FashionSegmentation
from utils.vgg_loss import VGGLoss
from utils.data_utils import mask_features
from utils.data_preprocessing import create_mask, remove_background

def train_emasc(dataloader, emasc, optimizer_emasc, 
                epochs, save_dir, device="cpu"):
    
    # Setup accelerator.
    accelerator = Accelerator(
        gradient_accumulation_steps=1,
        mixed_precision="fp16"
    )

    # Make one log on every process with the configuration for debugging.
    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    # Load VAE model.
    vae = AutoencoderKL.from_pretrained("stabilityai/stable-diffusion-2-inpainting", subfolder="vae")
    vae.eval()

    body_pose_model = BodyPoseEstimation(device=device)
    seg_model = FashionSegmentation(device=device)

    # Enable TF32 for faster training on Ampere GPUs,
    # cf https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
    torch.backends.cuda.matmul.allow_tf32 = True

    # Scheduler and math around the number of training steps.
    lr_scheduler = get_scheduler(
        "constant_with_warmup",
        optimizer=optimizer_emasc,
        num_warmup_steps=500,
        num_training_steps=epochs * len(dataloader)
    )

    # Define VGG loss
    criterion_vgg = VGGLoss()

    # Prepare everything with our `accelerator`.
    emasc, vae, dataloader, lr_scheduler, criterion_vgg = accelerator.prepare(
        emasc, vae, dataloader, lr_scheduler, criterion_vgg)
    
    save_path = "model_zoo/emasc"
    if not(os.path.exists(save_path)):
        os.makedirs(save_path)
    
    # Train!
    global_step = 0
    int_layers = [1, 2, 3, 4, 5]

    # Only show the progress bar once on each machine.
    progress_bar = tqdm(range(global_step, epochs * len(dataloader)), disable=not accelerator.is_local_main_process)
    progress_bar.set_description("Steps")

    for epoch in range(epochs):
        emasc.train()
        for step, batch in enumerate(dataloader):
            image = batch[0].to(device)
            category = batch[2]

            seg_maps = seg_model.predict(image)
            image = remove_background(image, seg_maps)

            key_pts = body_pose_model.predict(image)
            inpaint_mask, im_mask = create_mask(image, seg_maps, key_pts, category)

            inpaint_mask = inpaint_mask.to(device)
            im_mask = im_mask.to(device)

            with accelerator.accumulate(emasc):
                # Convert images to latent space
                with torch.no_grad():
                    # take latents from the encoded image and intermediate features from the encoded masked image
                    posterior_im, _ = vae.encode(image)
                    _, intermediate_features = vae.encode(im_mask)

                    intermediate_features = [intermediate_features[i] for i in int_layers]

                # Use EMASC to process the intermediate features
                processed_intermediate_features = emasc(intermediate_features)

                # Mask the features
                processed_intermediate_features = mask_features(processed_intermediate_features, inpaint_mask)

                # Decode the image from the latent space use the EMASC module
                latents = posterior_im.latent_dist.sample()
                reconstructed_image = vae.decode(z=latents,
                                                intermediate_features=processed_intermediate_features,
                                                int_layers=int_layers).sample

                # Compute the loss
                with accelerator.autocast():
                    loss_f1 = F.l1_loss(reconstructed_image, image, reduction="mean")
                    loss_vgg = criterion_vgg(reconstructed_image, image)
                    loss = loss_f1 + loss_vgg * 0.5

                # Backpropagate and update gradients
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(emasc.parameters(), 1.0)
                optimizer_emasc.step()
                lr_scheduler.step()
                optimizer_emasc.zero_grad()

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1
                accelerator.log({"train_loss": loss.detach().item()}, step=global_step)

                # Save checkpoint every checkpointing_steps steps
                if global_step % 1000 == 0:
                    if accelerator.is_main_process:
                        # Save model checkpoint
                        # accelerator_state_path = f"{save_path}/accelerator_{global_step}"
                        # accelerator.save_state(accelerator_state_path)

                        # Unwrap the EMASC model
                        unwrapped_emasc = accelerator.unwrap_model(emasc, keep_fp32_wrapper=True)

                        # Save EMASC model
                        emasc_path = f"{save_path}/emasc_chcekpoint_{global_step}.pth"
                        accelerator.save(unwrapped_emasc.state_dict(), emasc_path)
                        del unwrapped_emasc

            logs = {"step_loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
            progress_bar.set_postfix(**logs)

    # Unwrap the EMASC model
    unwrapped_emasc = accelerator.unwrap_model(emasc, keep_fp32_wrapper=True)

    # Save EMASC model
    emasc_path = f"{save_path}/emasc_checkpoint_last.pth"
    accelerator.save(unwrapped_emasc.state_dict(), emasc_path)

    # End of training
    accelerator.wait_for_everyone()
    accelerator.end_training()