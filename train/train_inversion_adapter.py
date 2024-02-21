import os
import torch
import torch.nn.functional as F
import torchvision
import diffusers
from diffusers import DDPMScheduler, UNet2DConditionModel, DDIMScheduler, AutoencoderKL
from diffusers.optimization import get_scheduler
from diffusers.utils.import_utils import is_xformers_available
import transformers
from transformers import CLIPTextModel, CLIPTokenizer, CLIPVisionModelWithProjection, AutoProcessor
from accelerate import Accelerator
from tqdm.auto import tqdm

from models import BodyPoseEstimation, FashionSegmentation, ClothCategoryClassfication
from utils.data_utils import create_mask, remove_background
from utils.encode_text_word_embedding import encode_text_word_embedding


def train_inversion_adapter(dataloader, inversion_adapter, optimizer_inversion_adapter,
                            epochs, device="cuda"):
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

    # Load the vision encoder and get the CLIP processor
    vision_encoder = CLIPVisionModelWithProjection.from_pretrained("laion/CLIP-ViT-H-14-laion2B-s32B-b79K")
    processor = AutoProcessor.from_pretrained("laion/CLIP-ViT-H-14-laion2B-s32B-b79K")

    # Load the VAE and UNet
    vae = AutoencoderKL.from_pretrained("stabilityai/stable-diffusion-2-inpainting", subfolder="vae")
    unet = UNet2DConditionModel.from_pretrained("stabilityai/stable-diffusion-2-inpainting", subfolder="unet")

    # Freeze vae and vision encoder
    vae.requires_grad_(False)
    vision_encoder.requires_grad_(False)

    body_pose_model = BodyPoseEstimation(device=device)
    seg_model = FashionSegmentation(device=device)
    category_cls_model = ClothCategoryClassfication(device=device)

    if is_xformers_available():
        unet.enable_xformers_memory_efficient_attention()
    else:
        raise ValueError("xformers is not available. Make sure it is installed correctly")
    
    # Enable TF32 for faster training on Ampere GPUs,
    # cf https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
    torch.backends.cuda.matmul.allow_tf32 = True

    # Scheduler and math around the number of training steps.
    lr_scheduler = get_scheduler(
        "constant_with_warmup",
        optimizer=optimizer_inversion_adapter,
        num_warmup_steps=500,
        num_training_steps=epochs * len(dataloader),
    )

    # Prepare everything with our `accelerator`.
    inversion_adapter, text_encoder, unet, optimizer_inversion_adapter, dataloader, lr_scheduler = accelerator.prepare(
        inversion_adapter, text_encoder, unet, optimizer_inversion_adapter, dataloader, lr_scheduler)

    # For mixed precision training we cast the text_encoder and vae weights to half-precision
    # as these models are only used for inference, keeping weights in full precision is not required.
    weight_dtype = torch.float16

    # Move text_encode and vae to gpu and cast to weight_dtype
    vae.to(accelerator.device, dtype=weight_dtype)
    vision_encoder.to(accelerator.device, dtype=weight_dtype)

    save_path = "model_zoo/inversion_adapter"
    if not(os.path.exists(save_path)):
        os.makedirs(save_path)

    # Train!
    global_step = 0

    # Only show the progress bar once on each machine.
    progress_bar = tqdm(range(epochs * len(dataloader)), disable=not accelerator.is_local_main_process)
    progress_bar.set_description("Steps")

    # Training loop
    for epoch in range(epochs):
        inversion_adapter.train()
        for step, batch in enumerate(dataloader):
            image = batch[0].to(device)
            cloth = batch[1].to(device)
            category = batch[2]
            
            subcategory = category_cls_model.predict(cloth, category)
            seg_maps = seg_model.predict(image)
            key_pts = body_pose_model.predict(image)
            inpaint_mask, im_mask = create_mask(image, seg_maps, key_pts, category)

            inpaint_mask = inpaint_mask.to(device)
            im_mask = im_mask.to(device)

            with accelerator.accumulate(inversion_adapter):
                # Convert images to latent space
                latents = vae.encode(image.to(weight_dtype)).latent_dist.sample()
                latents = latents * vae.config.scaling_factor

                # Sample noise that we'll add to the latents
                noise = torch.randn_like(latents)
                bsz = latents.shape[0]
                # Sample a random timestep for each image
                timesteps = torch.randint(0, noise_scheduler.num_train_timesteps, (bsz,), device=latents.device)
                timesteps = timesteps.long()

                # Add noise to the latents according to the noise magnitude at each timestep
                # (this is the forward diffusion process)
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

                # Generate the text for training the inversion adapter, '$' will be replaced with the PTEs during the
                # textual encoding process
                category_text = {
                    'dresses': 'a suit of dress',
                    'short_sleeved': 'a short sleeve tee',
                    'long_sleeved': 'a long sleeve tee',
                    'short_pants': 'a pair of shorts',
                    'long_pants': 'a pair of pant',
                    'skirts': 'a skirt'
                }
                text = [f'a photo of a model wearing {category_text[c]} {" $ " * 16}' for c in subcategory]

                # Get the target for loss
                target = noise

                # Compute the mask
                mask = inpaint_mask.to(weight_dtype)
                mask = torch.nn.functional.interpolate(mask, size=(512 // 8, 384 // 8))

                # Get the masked image and encode it
                masked_image = im_mask.to(weight_dtype)
                masked_image_latents = vae.encode(masked_image).latent_dist.sample() * vae.config.scaling_factor

                # Tokenize the text
                tokenized_text = tokenizer(text, max_length=tokenizer.model_max_length, padding="max_length",
                                           truncation=True, return_tensors="pt").input_ids
                tokenized_text = tokenized_text.to(accelerator.device)

                # Get the visual features of the in-shop cloths
                with torch.no_grad():
                    input_image = remove_background(cloth, seg_maps)
                    input_image = torchvision.transforms.functional.resize((input_image + 1) / 2, (224, 224),
                                                                            antialias=True).clamp(0, 1)
                    processed_images = processor(images=input_image, return_tensors="pt")
                    clip_cloth_features = vision_encoder(
                        processed_images.pixel_values.to(accelerator.device).to(weight_dtype)).last_hidden_state

                # Compute the predicted PTEs
                word_embeddings = inversion_adapter(clip_cloth_features.to(accelerator.device))
                word_embeddings = word_embeddings.reshape((bsz, 16, -1))

                # Encode the text using the PTEs extracted from the in-shop cloths
                encoder_hidden_states = encode_text_word_embedding(text_encoder, tokenized_text,
                                                                   word_embeddings,
                                                                   num_vstar=16).last_hidden_state

                # Predict the noise residual and compute loss
                unet_input = torch.cat([noisy_latents, mask, masked_image_latents], dim=1)
                model_pred = unet(unet_input, timesteps, encoder_hidden_states).sample

                # loss in accelerator.autocast according to docs https://huggingface.co/docs/accelerate/v0.15.0/quicktour#mixed-precision-training
                with accelerator.autocast():
                    loss = F.mse_loss(model_pred, target, reduction="mean")

                # Backpropagate
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(inversion_adapter.parameters(), 1.0)
                optimizer_inversion_adapter.step()
                lr_scheduler.step()
                optimizer_inversion_adapter.zero_grad()

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1
                accelerator.log({"train_loss": loss.detach().item()}, step=global_step)

                # Save checkpoint every checkpointing_steps steps
                if global_step % 1000 == 0:
                    if accelerator.is_main_process:
                        # accelerator_state_path = f"{save_path}/accelerator_{global_step}"
                        # accelerator.save_state(accelerator_state_path)

                        # Unwrap the inversion adapter
                        unwrapped_adapter = accelerator.unwrap_model(inversion_adapter, keep_fp32_wrapper=True)

                        # Save inversion adapter model
                        inversion_adapter_path = f"{save_path}/inversion_adapter_checkpoint_{global_step}.pth"
                        accelerator.save(unwrapped_adapter.state_dict(), inversion_adapter_path)
                        del unwrapped_adapter

            logs = {"step_loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
            progress_bar.set_postfix(**logs)

    # Unwrap the inversion adapter
    unwrapped_adapter = accelerator.unwrap_model(inversion_adapter, keep_fp32_wrapper=True)

    # Save inversion adapter model
    inversion_adapter_path = f"{save_path}/inversion_adapter_checkpoint_last.pth"
    accelerator.save(unwrapped_adapter.state_dict(), inversion_adapter_path)

    # End of training
    accelerator.wait_for_everyone()
    accelerator.end_training()
