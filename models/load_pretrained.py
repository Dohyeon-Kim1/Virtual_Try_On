import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision.models import resnet50
import numpy as np
from diffusers import DDIMScheduler, UNet2DConditionModel
from transformers import CLIPTextModel, CLIPTokenizer, CLIPVisionModelWithProjection, AutoProcessor
from transformers import SegformerImageProcessor, AutoModelForSemanticSegmentation
from mmpose.apis import inference_bottomup, init_model

from models.ladi_vton import ConvNet_TPS, UNetVanilla, EMASC, InversionAdapter, AutoencoderKL, StableDiffusionTryOnePipeline
from utils.data_utils import tensor_to_arr, coco_keypoint_mapping
from utils.encode_text_word_embedding import encode_text_word_embedding
    

class ClothCategoryClassfication():
    def __init__(self, category_classifier_ckpt, device="cuda"):
        ckpt = torch.load(category_classifier_ckpt, map_location="cpu")
        if device == "cuda":
            assert torch.cuda.is_available()
        self.device = device
        self.label_mapping = ckpt["label"]

        self.model = resnet50(weights=None)
        self.model.fc = nn.Linear(2048,6)
        self.model.load_state_dict(ckpt["model_sd"])
        self.model.to(device)
        self.model.eval()
    
    ## input: torch.Tensor (N,3,H,W), List[str] / output: List[str]
    @torch.no_grad()
    def predict(self, imgs, category=None):
        imgs = torchvision.transforms.functional.resize(imgs, (128,96), antialias=True)
        imgs = imgs.to(self.device)
        pred = self.model(imgs).cpu()

        if category is None:
            subcartgory = [self.label_mapping[int(i)] for i in pred.argmax(dim=-1)]
        else:
            subcartgory = []
            for i, c in enumerate(category):
                if c == "upper_body":
                    pred[i,[0,2,4,5]] = -torch.inf
                    subcartgory.append(self.label_mapping[pred[i].argmax(dim=-1).item()])
                elif c == "lower_body":
                    pred[i,[1,3,5]] = -torch.inf
                    subcartgory.append(self.label_mapping[pred[i].argmax(dim=-1).item()])
                elif c == "dresses":
                    subcartgory.append("dresses")
        return subcartgory


class BodyPoseEstimation():
    def __init__(self, device="cuda"):
        cfg = "models/mmpose/configs/body_2d_keypoint/rtmo/coco/rtmo-s_8xb32-600e_coco-640x640.py"
        ckpt = "https://download.openmmlab.com/mmpose/v1/projects/rtmo/rtmo-s_8xb32-600e_coco-640x640-8db55a59_20231211.pth"
        if device == "cuda":
            assert torch.cuda.is_available()
        self.device = device

        self.model = init_model(cfg, ckpt, device=device)
        self.model.eval()
    
    ## input: torch.Tensor (N,3,H,W) / output: np.ndarray (N,18,2)
    @torch.no_grad()
    def predict(self, imgs):
        imgs = tensor_to_arr(imgs, scope=[-1,1], batch=True)
        key_pts = []
        for img in imgs:
            pred = inference_bottomup(self.model, img)[0]
            key_pt = pred.pred_instances.keypoints[0]
            key_pts.append(coco_keypoint_mapping(key_pt))
        return np.stack(key_pts)
        
    
class FashionSegmentation():
    def __init__(self, device="cuda"):
        if device == "cuda":
            assert torch.cuda.is_available()
        self.device = device

        self.processor = SegformerImageProcessor.from_pretrained("mattmdjaga/segformer_b2_clothes")
        self.model = AutoModelForSemanticSegmentation.from_pretrained("mattmdjaga/segformer_b2_clothes")
        self.model.to(device)
        self.model.eval()

    ## input: torch.Tensor (N,3,H,W) / output: torch.Tensor (N,H,W)
    @torch.no_grad()
    def predict(self, imgs):
        imgs = tensor_to_arr(imgs, scope=[-1,1], batch=True)
        input = self.processor(imgs, return_tensors="pt").to(self.device)
        low_output = self.model(**input).logits
        output = nn.functional.interpolate(
            low_output,
            size=(512,384),
            mode="bilinear",
            align_corners=False
        ).argmax(dim=1)
        return output
    

class LadiVTON():
    def __init__(self, category_classifier_ckpt, tps_ckpt, emasc_ckpt, inversion_adapter_ckpt, unet_ckpt, 
                 weight_dtype=torch.float16, device="cuda"):
        if device == "cuda":
            assert torch.cuda.is_available()
        self.device = device
        self.weight_dtype = weight_dtype

        pretrained_model_name_or_path = "stabilityai/stable-diffusion-2-inpainting"
        self.val_scheduler = DDIMScheduler.from_pretrained(pretrained_model_name_or_path, subfolder="scheduler")
        self.val_scheduler.set_timesteps(50, device=device)
        self.text_encoder = CLIPTextModel.from_pretrained(pretrained_model_name_or_path, subfolder="text_encoder")
        self.vae = AutoencoderKL.from_pretrained(pretrained_model_name_or_path, subfolder="vae")
        self.vision_encoder = CLIPVisionModelWithProjection.from_pretrained("laion/CLIP-ViT-H-14-laion2B-s32B-b79K")
        self.processor = AutoProcessor.from_pretrained("laion/CLIP-ViT-H-14-laion2B-s32B-b79K")
        self.tokenizer = CLIPTokenizer.from_pretrained(pretrained_model_name_or_path, subfolder="tokenizer")

        if category_classifier_ckpt:
            self.category_classifier = ClothCategoryClassfication(category_classifier_ckpt, device=device)
        else:
            self.category_classifier = None
        
        if tps_ckpt:
            tps_sd = torch.load(tps_ckpt, map_location="cpu")
            self.tps = ConvNet_TPS(256, 192, 21, 3)
            self.refinement = UNetVanilla(n_channels=24, n_classes=3, bilinear=True)
            self.tps.load_state_dict(tps_sd["tps"])
            self.refinement.load_state_dict(tps_sd["refinement"])
        else:
            self.tps, self.refinement = torch.hub.load(repo_or_dir='miccunifi/ladi-vton', source='github', 
                                                       model='warping_module', dataset="dresscode")
        
        if emasc_ckpt:
            emasc_sd = torch.load(emasc_ckpt, map_location="cpu")
            in_feature_channels = [128, 128, 128, 256, 512]
            out_feature_channels = [128, 256, 512, 512, 512]
            self.emasc = EMASC(in_feature_channels, out_feature_channels, 
                               kernel_size=3, padding=1, stride=1, type="nonlinear")
            self.emasc.load_state_dict(emasc_sd)
        else:
            self.emasc = torch.hub.load(repo_or_dir='miccunifi/ladi-vton', source='github', 
                                        model='emasc', dataset="dresscode")
        
        if inversion_adapter_ckpt:
            inversion_adapter_sd = torch.load(inversion_adapter_ckpt, map_location="cpu")
            self.inversion_adapter = InversionAdapter(input_dim=1280, hidden_dim=1280 * 4, output_dim=1024 * 16,
                                                      num_encoder_layers=1, config=self.vision_encoder.config)
            self.inversion_adapter.load_state_dict(inversion_adapter_sd)
        else:
            self.inversion_adapter = torch.hub.load(repo_or_dir='miccunifi/ladi-vton', source='github', 
                                                    model='inversion_adapter', dataset="dresscode")
        
        if unet_ckpt:
            unet_sd = torch.load(unet_ckpt, map_location="cpu")
            config = UNet2DConditionModel.load_config("stabilityai/stable-diffusion-2-inpainting", subfolder="unet")
            config['in_channels'] = 31
            self.unet = UNet2DConditionModel.from_config(config)
            self.unet.load_state_dict(unet_sd)
        else:
            self.unet = torch.hub.load(repo_or_dir='miccunifi/ladi-vton', source='github', 
                                       model='extended_unet', dataset="dresscode")

        self.text_encoder.to(device, dtype=weight_dtype)
        self.vae.to(device, dtype=weight_dtype)
        self.emasc.to(device, dtype=weight_dtype)
        self.inversion_adapter.to(device, dtype=weight_dtype)
        self.unet.to(device, dtype=weight_dtype)
        self.tps.to(device, dtype=torch.float32)
        self.refinement.to(device, dtype=torch.float32)
        self.vision_encoder.to(device, dtype=weight_dtype)

        self.text_encoder.eval()
        self.vae.eval()
        self.emasc.eval()
        self.inversion_adapter.eval()
        self.unet.eval()
        self.tps.eval()
        self.refinement.eval()
        self.vision_encoder.eval()

        self.generator = torch.Generator(device).manual_seed(0)
        self.vton_pipe = StableDiffusionTryOnePipeline(text_encoder=self.text_encoder,
                                                        vae=self.vae,
                                                        tokenizer=self.tokenizer,
                                                        unet=self.unet,
                                                        scheduler=self.val_scheduler,
                                                        emasc=self.emasc,
                                                        emasc_int_layers=[1,2,3,4,5]).to(device)
    
    @torch.no_grad()
    def cloth_tps_transform(self, cloth_img, masked_img, pose_map):
        low_cloth = torchvision.transforms.functional.resize(cloth_img, (256, 192),
                                                                torchvision.transforms.InterpolationMode.BILINEAR, antialias=True)
        low_im_mask = torchvision.transforms.functional.resize(masked_img, (256, 192),
                                                                torchvision.transforms.InterpolationMode.BILINEAR, antialias=True)
        low_pose_map = torchvision.transforms.functional.resize(pose_map, (256, 192),
                                                                torchvision.transforms.InterpolationMode.BILINEAR, antialias=True)
        
        agnostic = torch.cat([low_im_mask, low_pose_map], 1)
        low_grid, theta, rx, ry, cx, cy, rg, cg = self.tps(low_cloth.to(torch.float32), agnostic.to(torch.float32))

        # We upsample the grid to the original image size and warp the cloth using the predicted TPS parameters
        highres_grid = torchvision.transforms.functional.resize(low_grid.permute(0, 3, 1, 2),
                                                                size=(512, 384),
                                                                interpolation=torchvision.transforms.InterpolationMode.BILINEAR,
                                                                antialias=True).permute(0, 2, 3, 1)
        
        warped_cloth = F.grid_sample(cloth_img.to(torch.float32), highres_grid.to(torch.float32), padding_mode='border')
        
        # Refine the warped cloth using the refinement network
        warped_cloth = torch.cat([masked_img, pose_map, warped_cloth], 1)
        warped_cloth = self.refinement(warped_cloth.to(torch.float32))
        warped_cloth = warped_cloth.clamp(-1, 1)
        warped_cloth = warped_cloth.to(self.weight_dtype)

        return warped_cloth

    @torch.no_grad()
    def cloth_embedding(self, cloth_img, category):
        # Get the visual features of the in-shop cloths
        cloth_img = (cloth_img + 1) / 2
        input_image = torchvision.transforms.functional.resize(cloth_img, (224, 224), antialias=True).clamp(0,1)
        processed_images = self.processor(images=input_image, return_tensors="pt")
        clip_cloth_features = self.vision_encoder(processed_images.pixel_values.to(self.device, dtype=self.weight_dtype)).last_hidden_state

        # Compute the predicted PTEs
        word_embeddings = self.inversion_adapter(clip_cloth_features.to(self.device))
        word_embeddings = word_embeddings.reshape((word_embeddings.shape[0], 16, -1))
        
        if self.category_classifier:
            category_text = {
                'dresses': 'a suit of dress',
                'short_sleeved': 'a short sleeve tee',
                'long_sleeved': 'a long sleeve tee',
                'short_pants': 'a pair of shorts',
                'long_pants': 'a pair of pant',
                'skirts': 'a skirt'
            }
        else:
            category_text = {
                'dresses': 'a dress',
                'upper_body': 'an upper body garment',
                'lower_body': 'a lower body garment'
                }
        text = [f'a photo of a model wearing {category_text[c]} {" $ " * 16}' for c in category]
        
        # Tokenize text
        tokenized_text = self.tokenizer(text, max_length=self.tokenizer.model_max_length, padding="max_length", truncation=True, return_tensors="pt").input_ids
        tokenized_text = tokenized_text.to(self.device)
        
        # Encode the text using the PTEs extracted from the in-shop cloths
        encoder_hidden_states = encode_text_word_embedding(self.text_encoder, tokenized_text, word_embeddings, 16).last_hidden_state

        return encoder_hidden_states
    
    @torch.no_grad()
    def predict(self, kwargs):
        return self.vton_pipe(generator=self.generator, **kwargs).images[0]
            

