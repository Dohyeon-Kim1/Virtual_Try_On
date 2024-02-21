import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from tqdm import tqdm

from models import BodyPoseEstimation, FashionSegmentation
from utils import create_mask, keypoint_to_heatmap
from utils.data_utils import extract_cloth, remove_background
from utils.vgg_loss import VGGLoss


def training_loop_tps(dataloader, tps, optimizer_tps, criterion_l1, scaler,
                      body_pose_model, seg_model, device="cpu"):
    tps.train()
    running_loss = 0.
    running_l1_loss = 0.
    running_const_loss = 0.
    progress = tqdm(dataloader)
    for step, inputs in enumerate(progress):  # Yield images with low resolution (256x192)
        image = inputs[0].to(device)
        cloth = inputs[1].to(device)
        category = inputs[2]

        seg_maps = seg_model.predict(image)
        image = remove_background(image, seg_maps)
        cloth = remove_background(cloth, seg_maps)

        key_pts = body_pose_model.predict(image)
        pose_map = keypoint_to_heatmap(key_pts, (512,384))

        _, im_mask = create_mask(image, seg_maps, key_pts, category)
        im_cloth = extract_cloth(cloth, seg_maps, category)

        im_cloth = im_cloth.to(device)
        im_mask = im_mask.to(device)
        pose_map = pose_map.to(device)

        low_cloth = torchvision.transforms.functional.resize(cloth, (256, 192),
                                                             torchvision.transforms.InterpolationMode.BILINEAR,
                                                             antialias=True)
        low_im_cloth = torchvision.transforms.functional.resize(im_cloth, (256, 192),
                                                             torchvision.transforms.InterpolationMode.BILINEAR,
                                                             antialias=True)
        low_im_mask = torchvision.transforms.functional.resize(im_mask, (256, 192),
                                                               torchvision.transforms.InterpolationMode.BILINEAR,
                                                               antialias=True)
        low_pose_map = torchvision.transforms.functional.resize(pose_map, (256, 192),
                                                                torchvision.transforms.InterpolationMode.BILINEAR,
                                                                antialias=True)
        
        with torch.cuda.amp.autocast():
            # TPS parameters prediction
            agnostic = torch.cat([low_im_mask, low_pose_map], 1)
            low_grid, theta, rx, ry, cx, cy, rg, cg = tps(low_cloth, agnostic)

            # Warp the cloth using the predicted TPS parameters
            low_warped_cloth = F.grid_sample(low_cloth, low_grid, padding_mode='border')

            # Compute the loss
            l1_loss = criterion_l1(low_warped_cloth, low_im_cloth)
            const_loss = torch.mean(rx + ry + cx + cy + rg + cg)

            loss = l1_loss + const_loss * 0.01

        # Update the parameters
        optimizer_tps.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer_tps)
        scaler.update()

        running_loss += loss.item()
        running_l1_loss += l1_loss.item()
        running_const_loss += const_loss.item()

        progress.set_postfix({"loss": loss.item()})

    loss = running_loss / (step + 1)
    l1_loss = running_l1_loss / (step + 1)
    const_loss = running_const_loss / (step + 1)
    return loss, l1_loss, const_loss


def training_loop_refinement(dataloader, tps, refinement, optimizer_ref, criterion_l1, criterion_vgg, scaler,
                             body_pose_model, seg_model, device="cpu"):
    """
    Training loop for the refinement network. Note that the refinement network is trained on a high resolution image
    """
    tps.eval()
    refinement.train()
    running_loss = 0.
    running_l1_loss = 0.
    running_vgg_loss = 0.
    progress = tqdm(dataloader)
    for step, inputs in enumerate(progress):
        image = inputs[0].to(device)
        cloth = inputs[1].to(device)
        category = inputs[2]

        seg_maps = seg_model.predict(image)
        image = remove_background(image, seg_maps)
        cloth = remove_background(cloth, seg_maps)

        key_pts = body_pose_model.predict(image)
        pose_map = keypoint_to_heatmap(key_pts, (512,384))

        _, im_mask = create_mask(image, seg_maps, key_pts, category)
        im_cloth = extract_cloth(cloth, seg_maps, category)

        im_cloth = im_cloth.to(device)
        im_mask = im_mask.to(device)
        pose_map = pose_map.to(device)

        # Resize the inputs to the low resolution for the TPS network
        low_cloth = torchvision.transforms.functional.resize(cloth, (256, 192),
                                                             torchvision.transforms.InterpolationMode.BILINEAR,
                                                             antialias=True)
        low_im_mask = torchvision.transforms.functional.resize(im_mask, (256, 192),
                                                               torchvision.transforms.InterpolationMode.BILINEAR,
                                                               antialias=True)
        low_pose_map = torchvision.transforms.functional.resize(pose_map, (256, 192),
                                                                torchvision.transforms.InterpolationMode.BILINEAR,
                                                                antialias=True)

        with torch.cuda.amp.autocast():
            # TPS parameters prediction
            agnostic = torch.cat([low_im_mask, low_pose_map], 1)
            low_grid, theta, rx, ry, cx, cy, rg, cg = tps(low_cloth, agnostic)

            # We upsample the grid to the original image size and warp the cloth using the predicted TPS parameters
            highres_grid = torchvision.transforms.functional.resize(low_grid.permute(0, 3, 1, 2),
                                                                    size=(512,384),
                                                                    interpolation=torchvision.transforms.InterpolationMode.BILINEAR,
                                                                    antialias=True).permute(0, 2, 3, 1)

            warped_cloth = F.grid_sample(cloth, highres_grid, padding_mode='border')

            # Refine the warped cloth using the refinement network
            warped_cloth = torch.cat([im_mask, pose_map, warped_cloth], 1)
            warped_cloth = refinement(warped_cloth)

            # Compute the loss
            l1_loss = criterion_l1(warped_cloth, im_cloth)
            vgg_loss = criterion_vgg(warped_cloth, im_cloth)

            loss = l1_loss * 1.0 + vgg_loss * 0.25

        # Update the parameters
        optimizer_ref.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer_ref)
        scaler.update()

        running_loss += loss.item()
        running_l1_loss += l1_loss.item()
        running_vgg_loss += vgg_loss.item()

        progress.set_postfix({"loss": loss.item()})

    loss = running_loss / (step + 1)
    l1_loss = running_l1_loss / (step + 1)
    vgg_loss = running_vgg_loss / (step + 1)
    return loss, l1_loss, vgg_loss


def train_tps(dataloader, tps, refinement, optimizer_tps, optimizer_ref, 
              epochs, device="cuda"):
    tps.to(device)
    refinement.to(device)
    
    body_pose_model = BodyPoseEstimation(device=device)
    seg_model = FashionSegmentation(device=device)

    scaler = torch.cuda.amp.GradScaler()
    criterion_l1 = nn.L1Loss()
    criterion_vgg = VGGLoss().to(device)

    for epoch in range(epochs):
        train_loss, train_l1_loss, train_const_loss = training_loop_tps(dataloader, tps, optimizer_tps, criterion_l1, scaler,
                                                                        body_pose_model, seg_model, device)
        print(f"Epoch {epoch+1}/{epochs}  loss: {round(train_loss, 4)}  l1_loss: {round(train_l1_loss, 4)}  const_loss: {round(train_const_loss, 4)}")
    
    scaler = torch.cuda.amp.GradScaler()

    for epoch in range(epochs):
        train_loss, train_l1_loss, train_vgg_loss = training_loop_refinement(dataloader, tps, refinement, optimizer_ref, criterion_l1, criterion_vgg, scaler,
                                                                             body_pose_model, seg_model, device)
        print(f"Epoch {epoch+1}/{epochs}  loss: {round(train_loss, 4)}  l1_loss: {round(train_l1_loss, 4)}  vgg_loss: {round(train_vgg_loss, 4)}")
    
    save_path = "model_zoo/tps"
    if not(os.path.exists(save_path)):
        os.makedirs(save_path)
    tps_path = f"{save_path}/tps_checkpoint_last.pth"
    torch.save({"tps": tps.state_dict(), "refinement": refinement.state_dict()}, tps_path)