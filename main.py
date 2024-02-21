import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvison.models import resnet50, ResNet50_Weights
from diffusers import UNet2DConditionModel
from transformers import CLIPVisionModelWithProjection
from PIL import Image

from dataset import BodyClothPairDataset
from models.ladi_vton import ConvNet_TPS, EMASC, UNetVanilla, InversionAdapter
from train import train_category_classifier, train_tps, train_emasc, train_inversion_adapter, train_vto
from inference import Inferencer


def parse_argument():
    parser = argparse.ArgumentParser(description="Virtual Try On")
    parser.add_argument("--train", type=int, default=0, 
                        help="whether to train model or not")
    parser.add_argument("--tps_ckpt", type=str, 
                        default="model_zoo/tps/tps_checkpoint_last.pth")
    parser.add_argument("--emasc_ckpt", type=str, 
                        default="model_zoo/emasc/emasc_checkpoint_last.pth")
    parser.add_argument("--inversion_adapter_ckpt", type=str, 
                        default="model_zoo/inversion_adapter/inversion_adapter_checkpoint_last.pth")
    parser.add_argument("--unet_ckpt", type=str, 
                        default="model_zoo/unet/unet_checkpoint_last.pth")
    parser.add_argument("--device", type=str, default="cuda", 
                        help="the location in which model train or infernce")
    
    ## train options
    parser.add_argument("--model_kind", type=str, default=None, 
                        help="what kind of model you want to train")
    parser.add_argument("--pretrained", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--weight_decay", type=float, default=1e-2)
    parser.add_argument("--save_dir", type=str, default="checkpoint",
                        help="the location in which model's checkpoints are saved")


    ## inference options
    parser.add_argument("--category", type=str, default=None, 
                        help="the category of cloth")
    parser.add_argument("--body_img", type=str, default=None, 
                        help="the path of body image")
    parser.add_argument("--cloth_img", type=str, default=None, 
                        help="the path of cloth image")
    parser.add_argument("--guidance_scale", type=float, default=5.0)
    parser.add_argument("--num_inference_steps", type=int, default=50)

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_argument()

    if args.train:
        if args.model_kind == "category_classifier":
            category_classifier = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
            category_classifier.fc = nn.Linear(2048,6)

            optimizer_category_classifier = torch.optim.Adam(category_classifier.parameters(), lr=args.lr)

            train_category_classifier(category_classifier, optimizer_category_classifier,
                                      args.batch_size, args.epochs, args.device)
        else:
            dataset = BodyClothPairDataset()
            dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
            if args.model_kind == "tps":
                if args.pretrained:
                    tps, refinement = torch.hub.load(repo_or_dir='miccunifi/ladi-vton', source='github', 
                                                    model='warping_module', dataset="dresscode")
                else:
                    tps = ConvNet_TPS(256, 192, 21, 3)
                    refinement = UNetVanilla(n_channels=24, n_classes=3, bilinear=True)
                
                optimizer_tps = torch.optim.Adam(tps.parameters(), lr=args.lr, betas=(0.5, 0.99))
                optimizer_ref = torch.optim.Adam(list(refinement.parameters()), lr=args.lr, betas=(0.5, 0.99))

                train_tps(dataloader, tps, refinement, optimizer_tps, optimizer_ref,
                        args.epochs, args.device)
            elif args.model_kind == "emasc":
                if args.pretrained:
                    emasc = torch.hub.load(repo_or_dir='miccunifi/ladi-vton', source='github', 
                                        model='emasc', dataset="dresscode")
                else:
                    in_feature_channels = [128, 128, 128, 256, 512]
                    out_feature_channels = [128, 256, 512, 512, 512]
                    int_layers = [1, 2, 3, 4, 5]
                    emasc = EMASC(in_feature_channels, out_feature_channels,
                                    kernel_size=3, padding=1, stride=1, type="nonlinear")

                optimizer_emasc = torch.optim.AdamW(emasc.parameters(), lr=args.lr, betas=(0.9,0.999), 
                                                    weight_decay=args.weight_decay, eps=1e-08)
                    
                train_emasc(dataloader, emasc, optimizer_emasc,
                            args.epochs, args.device)
            elif args.model_kind == "inversion_adapter":
                if args.pretrained:
                    inversion_adapter = torch.hub.load(repo_or_dir='miccunifi/ladi-vton', source='github', 
                                                    model='inversion_adapter', dataset="dresscode")  
                else:
                    vision_encoder = CLIPVisionModelWithProjection.from_pretrained("laion/CLIP-ViT-H-14-laion2B-s32B-b79K")
                    inversion_adapter = InversionAdapter(input_dim=1280, hidden_dim=1280 * 4, output_dim=1024 * 16,
                                                        num_encoder_layers=1, config=vision_encoder.config)

                optimizer_inversion_adapter = torch.optim.AdamW(inversion_adapter.parameters(), lr=args.lr, betas=(0.9,0.999),
                                                                weight_decay=args.weight_decay, eps=1e-08)
                    
                train_inversion_adapter(dataloader, inversion_adapter, optimizer_inversion_adapter,
                                        args.epochs, args.device)
            elif args.model_kind == "vto":
                if args.pretrained:
                    tps = ConvNet_TPS(256, 192, 21, 3)
                    refinement = UNetVanilla(n_channels=24, n_classes=3, bilinear=True)
                    vision_encoder = CLIPVisionModelWithProjection.from_pretrained("laion/CLIP-ViT-H-14-laion2B-s32B-b79K")
                    inversion_adapter = InversionAdapter(input_dim=1280, hidden_dim=1280 * 4, output_dim=1024 * 16,
                                                        num_encoder_layers=1, config=vision_encoder.config)
                    
                    tps_checkpoint = torch.load(args.tps_ckpt)
                    inversion_adapter_checkpoint = torch.load(args.inversion_adapter_ckpt)
                    tps.load_state_dict(tps_checkpoint["tps"])
                    refinement.load_state_dict(tps_checkpoint["refinement"])
                    inversion_adapter.load_state_dict(inversion_adapter_checkpoint)
                else:
                    tps, refinement = torch.hub.load(repo_or_dir='miccunifi/ladi-vton', source='github', 
                                                    model='warping_module', dataset="dresscode")
                    inversion_adapter = torch.hub.load(repo_or_dir='miccunifi/ladi-vton', source='github', 
                                                    model='inversion_adapter', dataset="dresscode")
                unet = UNet2DConditionModel.from_pretrained("stabilityai/stable-diffusion-2-inpainting", subfolder="unet")

                optimizer_unet = torch.optim.AdamW(unet.parameters(), lr=args.lr, betas=(0.9, 0.999),
                                                weight_decay=args.weight_decay, eps=1e-8)
                
                train_vto(dataloader, unet, inversion_adapter, tps, refinement, optimizer_unet,
                        args.epochs, args.device)
    else:
        inferencer = Inferencer(device=args.device)
        body_img = Image.open(args.body_img)
        cloth_img = Image.open(args.cloth_img)
        
        output = inferencer.inference(body_img, cloth_img, args.category, 
                                      args.guidance_scale, args.num_inference_steps)
        output.save("images/output.png")