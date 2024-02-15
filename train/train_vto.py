import os
import torch
import torch.nn.functional as F
import torchvision
import diffusers
from diffusers import DDPMScheduler, DDIMScheduler
from diffusers.optimization import get_scheduler
from diffusers.utils.import_utils import is_xformers_available
import transformers
from transformers import CLIPTextModel, CLIPTokenizer, CLIPVisionModelWithProjection, AutoProcessor
from accelerate import Accelerator
from tqdm.auto import tqdm

from models.ladi_vton.AutoencoderKL import AutoencoderKL
from models import BodyPoseEstimation, FashionSegmentation
from utils.encode_text_word_embedding import encode_text_word_embedding
from utils.data_preprocessing import keypoint_to_heatmap, create_mask

def train_vto(dataloader, unet, inversion_adapter, tps, refinement, optimizer_unet,
              epochs, save_dir, device="cpu"):
    torch.multiprocessing.set_sharing_strategy('file_system')

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

    # Load scheduler, tokenizer and models.
    noise_scheduler = DDPMScheduler.from_pretrained("stabilityai/stable-diffusion-2-inpainting", subfolder="scheduler")
    val_scheduler = DDIMScheduler.from_pretrained("stabilityai/stable-diffusion-2-inpainting", subfolder="scheduler")
    val_scheduler.set_timesteps(50, device=accelerator.device)

    tokenizer = CLIPTokenizer.from_pretrained("stabilityai/stable-diffusion-2-inpainting", subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained("stabilityai/stable-diffusion-2-inpainting", subfolder="text_encoder")
    vae = AutoencoderKL.from_pretrained("stabilityai/stable-diffusion-2-inpainting", subfolder="vae")
    vision_encoder = CLIPVisionModelWithProjection.from_pretrained("laion/CLIP-ViT-H-14-laion2B-s32B-b79K")
    processor = AutoProcessor.from_pretrained("laion/CLIP-ViT-H-14-laion2B-s32B-b79K")

    body_pose_model = BodyPoseEstimation(device=device)
    seg_model = FashionSegmentation(device=device)

    tps.to(device)
    refinement.to(device)

    new_in_channels = 31
    # the posemap has 18 channels, the (encoded) cloth has 4 channels, the standard SD inpaining has 9 channels
    with torch.no_grad():
        # Replace the first conv layer of the unet with a new one with the correct number of input channels
        conv_new = torch.nn.Conv2d(
            in_channels=new_in_channels,
            out_channels=unet.conv_in.out_channels,
            kernel_size=3,
            padding=1,
        )

        torch.nn.init.kaiming_normal_(conv_new.weight)  # Initialize new conv layer
        conv_new.weight.data = conv_new.weight.data * 0.  # Zero-initialize new conv layer

        conv_new.weight.data[:, :9] = unet.conv_in.weight.data  # Copy weights from old conv layer
        conv_new.bias.data = unet.conv_in.bias.data  # Copy bias from old conv layer

        unet.conv_in = conv_new  # replace conv layer in unet
        unet.config['in_channels'] = new_in_channels  # update config

    # Freeze vae, text_encoder, vision_encoder, inversion adpter
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    vision_encoder.requires_grad_(False)
    inversion_adapter.requires_grad_(False)
    
    inversion_adapter.eval()

    if is_xformers_available():
        unet.enable_xformers_memory_efficient_attention()
    else:
        raise ValueError("xformers is not available. Make sure it is installed correctly")

    # Enable gradient checkpointing for memory efficient training
    unet.enable_gradient_checkpointing()
    text_encoder.gradient_checkpointing_enable()

    # Enable TF32 for faster training on Ampere GPUs,
    # cf https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
    # if args.allow_tf32:
    #     torch.backends.cuda.matmul.allow_tf32 = True

    # Scheduler and math around the number of training steps.
    lr_scheduler = get_scheduler(
        "constant_with_warmup",
        optimizer=optimizer_unet,
        num_warmup_steps=500,
        num_training_steps=epochs * len(dataloader)
    )

    weight_dtype = torch.float16

    # Prepare everything with our `accelerator`.
    unet, inversion_adapter, text_encoder, optimizer_unet, dataloader, lr_scheduler = accelerator.prepare(
        unet, inversion_adapter, text_encoder, optimizer_unet, dataloader, lr_scheduler)

    # Move and vae to gpu and cast to weight_dtype
    vae.to(accelerator.device, dtype=weight_dtype)
    vision_encoder.to(accelerator.device, dtype=weight_dtype)

    save_path = f"unet_ckpt/{save_dir}"
    if not(os.path.exists(save_path)):
        os.makedirs(save_path)

    # Train!
    global_step = 0

    # Only show the progress bar once on each machine.
    progress_bar = tqdm(range(epochs * len(dataloader)), disable=not accelerator.is_local_main_process)
    progress_bar.set_description("Steps")

    for epoch in range(epochs):
        unet.train()

        for step, batch in enumerate(dataloader):
            image = batch[0].to(device)
            cloth = batch[1].to(device)
            category = batch[2]

            key_pts = body_pose_model.predict(image)
            seg_maps = seg_model.predict(image)
            pose_map = keypoint_to_heatmap(key_pts, (512,384))
            inpaint_mask, im_mask = create_mask(image, seg_maps, key_pts, category)

            pose_map = pose_map.to(device)
            inpaint_mask = inpaint_mask.to(device)
            im_mask = im_mask.to(device)

            with accelerator.accumulate(unet):
                # Convert images to latent space
                latents = vae.encode(image.to(weight_dtype))[0].latent_dist.sample()
                latents = latents * vae.config.scaling_factor

                # Get the pose map and resize it to the same size as the latents
                pose_map_resize = torch.nn.functional.interpolate(pose_map, size=(pose_map.shape[2] // 8, pose_map.shape[3] // 8), mode="bilinear")

                # Sample noise that we'll add to the latents
                noise = torch.randn_like(latents)
                bsz = latents.shape[0]
                # Sample a random timestep for each image
                timesteps = torch.randint(0, noise_scheduler.num_train_timesteps, (bsz,), device=latents.device)
                timesteps = timesteps.long()

                # Add noise to the latents according to the noise magnitude at each timestep
                # (this is the forward diffusion process)
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

                # Get the text embedding for conditioning
                # Generate the text for training the inversion adapter, '$' will be replaced with the PTEs during the
                # textual encoding process
                category_text = {
                    'dresses': 'a dress',
                    'upper_body': 'an upper body garment',
                    'lower_body': 'a lower body garment',
                }
                text = [f'a photo of a model wearing {category_text[c]} {" $ " * 16}' for c in category]

                with torch.no_grad():
                    # Compute the visual features of the in-shop cloths
                    input_image = torchvision.transforms.functional.resize((cloth + 1) / 2, (224, 224),antialias=True).clamp(0,1)
                    processed_images = processor(images=input_image, return_tensors="pt")
                    clip_cloth_features = vision_encoder(processed_images.pixel_values.to(accelerator.device).to(weight_dtype)).last_hidden_state

                    # Compute the predicted PTEs
                    word_embeddings = inversion_adapter(clip_cloth_features.to(accelerator.device))
                    word_embeddings = word_embeddings.reshape((bsz, 16, -1))

                target = noise

                # Compute the mask
                mask = inpaint_mask.to(weight_dtype)
                mask = torch.nn.functional.interpolate(mask, size=(512 // 8, 384 // 8))

                # Get the masked image
                masked_image = im_mask.to(weight_dtype)
                masked_image_latents = vae.encode(masked_image)[0].latent_dist.sample() * vae.config.scaling_factor

                # Get the warped cloths latents
                with torch.no_grad():
                    low_cloth = torchvision.transforms.functional.resize(cloth, (256, 192),
                                                                        torchvision.transforms.InterpolationMode.BILINEAR,
                                                                        antialias=True)
                    low_im_mask = torchvision.transforms.functional.resize(im_mask, (256, 192),
                                                                        torchvision.transforms.InterpolationMode.BILINEAR,
                                                                        antialias=True)
                    low_pose_map = torchvision.transforms.functional.resize(pose_map, (256, 192),
                                                                            torchvision.transforms.InterpolationMode.BILINEAR,
                                                                            antialias=True)
                    
                    agnostic = torch.cat([low_im_mask, low_pose_map], 1)
                    low_grid, theta, rx, ry, cx, cy, rg, cg = tps(low_cloth, agnostic)
                    
                    highres_grid = torchvision.transforms.functional.resize(low_grid.permute(0, 3, 1, 2),
                                                                            size=(512,384),
                                                                            interpolation=torchvision.transforms.InterpolationMode.BILINEAR,
                                                                            antialias=True).permute(0, 2, 3, 1)

                    warped_cloth = F.grid_sample(cloth, highres_grid, padding_mode='border')
                    warped_cloth = torch.cat([im_mask, pose_map, warped_cloth], 1)
                    warped_cloth = refinement(warped_cloth)

                    cloth_latents = vae.encode(warped_cloth.to(weight_dtype))[0].latent_dist.sample()
                    cloth_latents = cloth_latents * vae.config.scaling_factor

                # Randomly mask out some of the inputs
                uncond_mask_text = torch.rand(bsz, device=latents.device) < 0.2
                uncond_mask_cloth = torch.rand(bsz, device=latents.device) < 0.2
                uncond_mask_pose = torch.rand(bsz, device=latents.device) < 0.2
                text = [t if not uncond_mask_text[i] else "" for i, t in enumerate(text)]
                pose_map_resize[uncond_mask_pose] = torch.zeros_like(pose_map_resize[0])
                cloth_latents[uncond_mask_cloth] = torch.zeros_like(cloth_latents[0])

                # Encode the text
                tokenized_text = tokenizer(text, max_length=tokenizer.model_max_length, padding="max_length",
                                           truncation=True, return_tensors="pt").input_ids
                tokenized_text = tokenized_text.to(accelerator.device)

                # Encode the text using the PTEs extracted from the in-shop cloths
                encoder_hidden_states = encode_text_word_embedding(text_encoder, tokenized_text,
                                                                    word_embeddings,
                                                                    16).last_hidden_state

                # Predict the noise residual and compute loss
                unet_input = torch.cat([noisy_latents, mask, masked_image_latents, pose_map_resize.to(weight_dtype), cloth_latents], dim=1)
                model_pred = unet(unet_input, timesteps, encoder_hidden_states).sample

                # loss in accelerator.autocast according to docs https://huggingface.co/docs/accelerate/v0.15.0/quicktour#mixed-precision-training
                with accelerator.autocast():
                    loss = F.mse_loss(model_pred, target, reduction="mean")

                # Backpropagate
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(unet.parameters(), 1.0)

                optimizer_unet.step()
                lr_scheduler.step()
                optimizer_unet.zero_grad()

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1
                accelerator.log({"train_loss": loss.detach().item()}, step=global_step)

                # Save checkpoint every checkpointing_steps steps
                if global_step % 50 == 0:
                    if accelerator.is_main_process:
                        accelerator_state_path = f"{save_path}/accelerator_{global_step}"
                        accelerator.save_state(accelerator_state_path)

                        # Unwrap the Unet
                        unwrapped_unet = accelerator.unwrap_model(unet, keep_fp32_wrapper=True)

                        # Save the unet
                        unet_path = f"{save_path}/unet_{global_step}.pth"
                        accelerator.save(unwrapped_unet.state_dict(), unet_path)

                        del unwrapped_unet

            logs = {"step_loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
            progress_bar.set_postfix(**logs)

    # End of training
    accelerator.wait_for_everyone()
    accelerator.end_training()