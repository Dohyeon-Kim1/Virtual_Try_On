import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from tqdm import tqdm

from models.ConvNet_TPS import ConvNet_TPS
from models.UNet import UNetVanilla
from utils.vgg_loss import VGGLoss
from utils import create_mask, keypoint_to_heatmap
from utils.data_preprocessing import extract_cloth


def training_loop_tps(dataloader, tps, optimizer_tps, criterion_l1, scaler, const_weight,
                      body_pose_model, seg_model, device="cpu"):
    tps.train()
    running_loss = 0.
    running_l1_loss = 0.
    running_const_loss = 0.
    for step, inputs in enumerate(tqdm(dataloader)):  # Yield images with low resolution (256x192)
        image = inputs[0].to(device)
        cloth = inputs[1].to(device)
        category = inputs[2]

        key_pts = body_pose_model.predict(image)
        seg_maps = seg_model.predict(image)
        pose_map = keypoint_to_heatmap(key_pts)
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

            loss = l1_loss + const_loss * const_weight

        # Update the parameters
        optimizer_tps.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer_tps)
        scaler.update()

        running_loss += loss.item()
        running_l1_loss += l1_loss.item()
        running_const_loss += const_loss.item()

    loss = running_loss / (step + 1)
    l1_loss = running_l1_loss / (step + 1)
    const_loss = running_const_loss / (step + 1)
    return loss, l1_loss, const_loss


def training_loop_refinement(dataloader, tps, refinement, optimizer_ref, criterion_l1, criterion_vgg,
                             l1_weight, vgg_weight, scaler, body_pose_model, seg_model, height=512, width=384, device="cpu"):
    """
    Training loop for the refinement network. Note that the refinement network is trained on a high resolution image
    """
    tps.eval()
    refinement.train()
    running_loss = 0.
    running_l1_loss = 0.
    running_vgg_loss = 0.
    for step, inputs in enumerate(tqdm(dataloader)):
        image = inputs[0].to(device)
        cloth = inputs[1].to(device)
        category = inputs[2]

        key_pts = body_pose_model.predict(image)
        seg_maps = seg_model.predict(image)
        pose_map = keypoint_to_heatmap(key_pts)
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
            low_warped_cloth = F.grid_sample(cloth, low_grid, padding_mode='border')

            # We upsample the grid to the original image size and warp the cloth using the predicted TPS parameters
            highres_grid = torchvision.transforms.functional.resize(low_grid.permute(0, 3, 1, 2),
                                                                    size=(height, width),
                                                                    interpolation=torchvision.transforms.InterpolationMode.BILINEAR,
                                                                    antialias=True).permute(0, 2, 3, 1)

            warped_cloth = F.grid_sample(cloth, highres_grid, padding_mode='border')

            # Refine the warped cloth using the refinement network
            warped_cloth = torch.cat([im_mask, pose_map, warped_cloth], 1)
            warped_cloth = refinement(warped_cloth)

            # Compute the loss
            l1_loss = criterion_l1(warped_cloth, im_cloth)
            vgg_loss = criterion_vgg(warped_cloth, im_cloth)

            loss = l1_loss * l1_weight + vgg_loss * vgg_weight

        # Update the parameters
        optimizer_ref.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer_ref)
        scaler.update()

        running_loss += loss.item()
        running_l1_loss += l1_loss.item()
        running_vgg_loss += vgg_loss.item()

    loss = running_loss / (step + 1)
    l1_loss = running_l1_loss / (step + 1)
    vgg_loss = running_vgg_loss / (step + 1)
    return loss, l1_loss, vgg_loss


@torch.no_grad()
def extract_images(dataloader: DataLoader, tps: ConvNet_TPS, refinement: UNetVanilla, save_path: str, height: int = 512,
                   width: int = 384) -> None:
    """
    Extracts the images using the trained networks and saves them to the save_path
    """
    tps.eval()
    refinement.eval()

    # running_loss = 0.
    for step, inputs in enumerate(tqdm(dataloader)):
        c_name = inputs['c_name']
        im_name = inputs['im_name']
        cloth = inputs['cloth'].to(device)
        category = inputs.get('category')
        im_mask = inputs['im_mask'].to(device)
        pose_map = inputs.get('dense_uv')
        if pose_map is None:
            pose_map = inputs['pose_map']
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

        # TPS parameters prediction
        agnostic = torch.cat([low_im_mask, low_pose_map], 1)

        low_grid, theta, rx, ry, cx, cy, rg, cg = tps(low_cloth, agnostic)

        # We upsample the grid to the original image size and warp the cloth using the predicted TPS parameters
        highres_grid = torchvision.transforms.functional.resize(low_grid.permute(0, 3, 1, 2),
                                                                size=(height, width),
                                                                interpolation=torchvision.transforms.InterpolationMode.BILINEAR,
                                                                antialias=True).permute(0, 2, 3, 1)

        warped_cloth = F.grid_sample(cloth, highres_grid, padding_mode='border')

        # Refine the warped cloth using the refinement network
        warped_cloth = torch.cat([im_mask, pose_map, warped_cloth], 1)
        warped_cloth = refinement(warped_cloth)

        warped_cloth = (warped_cloth + 1) / 2
        warped_cloth = warped_cloth.clamp(0, 1)

        # Save the images
        for cname, iname, warpclo, cat in zip(c_name, im_name, warped_cloth, category):
            if not os.path.exists(os.path.join(save_path, cat)):
                os.makedirs(os.path.join(save_path, cat))
            save_image(warpclo, os.path.join(save_path, cat, iname.replace(".jpg", "") + "_" + cname),
                       quality=95)


def train_tps_parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=16, help='train/test batch size')
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--width", type=int, default=384)
    parser.add_argument('--const_weight', type=float, default=0.01, help='weight for the TPS constraint loss')
    parser.add_argument('--lr', type=float, default=1e-4, help='initial learning rate for adam')
    parser.add_argument('--vgg_weight', type=float, default=0.25, help='weight for the VGG loss (refinement network)')
    parser.add_argument('--l1_weight', type=float, default=1, help='weight for the L1 loss (refinement network)')
    parser.add_argument('--save_path', type=str, help='path to save the warped cloth images (if not provided, '
                                                      'the images will be saved in the data folder)')
    parser.add_argument('--epochs_tps', type=int, default=50, help='number of epochs to train the TPS network')
    parser.add_argument('--epochs_refinement', type=int, default=50,
                        help='number of epochs to train the refinement network')
    args = parser.parse_args()
    return args


def main():
    args = train_tps_parse_args()

    device = "cuda"

    # Training dataset and dataloader


    # Define TPS and refinement network
    input_nc = 21
    n_layer = 3
    tps = ConvNet_TPS(256, 192, input_nc, n_layer).to(device)
    refinement = UNetVanilla(n_channels=24, n_classes=3, bilinear=True).to(device)

    # Define optimizer, scaler and loss
    optimizer_tps = torch.optim.Adam(tps.parameters(), lr=args.lr, betas=(0.5, 0.99))
    optimizer_ref = torch.optim.Adam(list(refinement.parameters()), lr=args.lr, betas=(0.5, 0.99))

    scaler = torch.cuda.amp.GradScaler()
    criterion_l1 = nn.L1Loss()
    criterion_vgg = VGGLoss().to(device)

    start_epoch = 0

    # Training loop for TPS training
    # Set training dataset height and width to (256, 192) since the TPS is trained using a lower resolution
    dataset_train.height = 256
    dataset_train.width = 192
    for e in range(start_epoch, args.epochs_tps):
        print(f"Epoch {e}/{args.epochs_tps}")
        print('train')
        train_loss, train_l1_loss, train_const_loss, visual = training_loop_tps(
            dataloader_train,
            tps,
            optimizer_tps,
            criterion_l1,
            scaler,
            args.const_weight)

        # Save checkpoint
        os.makedirs(os.path.join(args.checkpoints_dir, args.exp_name), exist_ok=True)
        torch.save({
            'epoch': e + 1,
            'tps': tps.state_dict(),
            'refinement': refinement.state_dict(),
            'optimizer_tps': optimizer_tps.state_dict(),
            'optimizer_ref': optimizer_ref.state_dict(),
        }, os.path.join(args.checkpoints_dir, args.exp_name, f"checkpoint_last.pth"))

    scaler = torch.cuda.amp.GradScaler()  # Initialize scaler again for refinement

    # Training loop for refinement
    # Set training dataset height and width to (args.height, args.width) since the refinement is trained using a higher resolution
    dataset_train.height = args.height
    dataset_train.width = args.width
    for e in range(max(start_epoch, args.epochs_tps), max(start_epoch, args.epochs_tps) + args.epochs_refinement):
        print(f"Epoch {e}/{max(start_epoch, args.epochs_tps) + args.epochs_refinement}")
        train_loss, train_l1_loss, train_vgg_loss, visual = training_loop_refinement(
            dataloader_train,
            tps,
            refinement,
            optimizer_ref,
            criterion_l1,
            criterion_vgg,
            args.l1_weight,
            args.vgg_weight,
            scaler,
            args.height,
            args.width)

        # Save checkpoint
        os.makedirs(os.path.join(args.checkpoints_dir, args.exp_name), exist_ok=True)
        torch.save({
            'epoch': e + 1,
            'tps': tps.state_dict(),
            'refinement': refinement.state_dict(),
            'optimizer_tps': optimizer_tps.state_dict(),
            'optimizer_ref': optimizer_ref.state_dict(),
        }, os.path.join(args.checkpoints_dir, args.exp_name, f"checkpoint_last.pth"))

    # Extract warped cloth images at the end of training
    print("Extracting warped cloth images...")
    extraction_dataset_paired = torch.utils.data.ConcatDataset([dataset_test_paired, dataset_train])
    extraction_dataloader_paired = DataLoader(batch_size=args.batch_size,
                                              dataset=extraction_dataset_paired,
                                              shuffle=False,
                                              num_workers=args.workers,
                                              drop_last=False)

    if args.save_path:
        warped_cloth_root = args.save_path
    else:
        warped_cloth_root = PROJECT_ROOT / 'data'

    save_name_paired = warped_cloth_root / 'warped_cloths' / args.dataset
    extract_images(extraction_dataloader_paired, tps, refinement, save_name_paired, args.height, args.width)

    extraction_dataset = dataset_test_unpaired
    extraction_dataloader_paired = DataLoader(batch_size=args.batch_size,
                                              dataset=extraction_dataset,
                                              shuffle=False,
                                              num_workers=args.workers)

    save_name_unpaired = warped_cloth_root / 'warped_cloths_unpaired' / args.dataset
    extract_images(extraction_dataloader_paired, tps, refinement, save_name_unpaired, args.height, args.width)


if __name__ == '__main__':
    main()
