import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import numpy as np
from PIL import Image
from accelerate import Accelerator
from diffusers import DDIMScheduler
from diffusers.utils.import_utils import is_xformers_available
from transformers import CLIPTextModel, CLIPTokenizer, CLIPVisionModelWithProjection, AutoProcessor
from transformers import SegformerImageProcessor, AutoModelForSemanticSegmentation
from mmpose.apis import inference_topdown, init_model

from models.ladi_vton.AutoencoderKL import AutoencoderKL
from models.ladi_vton.tryon_pipe import StableDiffusionTryOnePipeline
from utils.encode_text_word_embedding import encode_text_word_embedding


class FashionPoseEstimation():
    def __init__(self, kind="short-sleeved-shirt", device="cpu"):
        cfgs_dict = {"short-sleeved-shirt" : "./models/mmpose/configs/fashion_2d_keypoint/topdown_heatmap/deepfashion2/td-hm_res50_6xb64-210e_deepfasion2-short-sleeved-shirt-256x192.py",
                     "long_sleeved_shirt" : "",
                     "short_sleeved_outwear": "",
                     "long_sleeved_outwear": "",
                     "vest": "",
                     "sling": "",
                     "shorts": "",
                     "trousers": "",
                     "skirt": "",
                     "short_sleeved_dress": "",
                     "long_sleeved_dress": "",
                     "vest_dress": "",
                     "sling_dress": ""}
        ckpts_dict = {"short-sleeved-shirt": "https://download.openmmlab.com/mmpose/fashion/resnet/res50_deepfashion2_short_sleeved_shirt_256x192-21e1c5da_20221208.pth",
                      "long_sleeved_shirt": "",
                      "short_sleeved_outwear": "",
                      "long_sleeved_outwear": "",
                      "vest": "",
                      "sling": "",
                      "shorts": "",
                      "trousers": "",
                      "skirt": "",
                      "short_sleeved_dress": "",
                      "long_sleeved_dress": "",
                      "vest_dress": "",
                      "sling_dress": ""}
        assert kind in cfgs_dict and kind in ckpts_dict 
        if device == "cuda":
            assert torch.cuda.is_available()
         
        self.device = device 
        self.model = init_model(cfgs_dict[kind], ckpts_dict[kind], device=device)

    def predict(self, img):
        return inference_topdown(self.model, img)
    
    
class BodyPoseEstimation():
    def __init__(self, device="cpu"):
        cfg = "./models/mmpose/configs/body_2d_keypoint/rtmo/coco/rtmo-s_8xb32-600e_coco-640x640.py"
        ckpt = "https://download.openmmlab.com/mmpose/v1/projects/rtmo/rtmo-s_8xb32-600e_coco-640x640-8db55a59_20231211.pth"
        
        if device == "cuda":
            assert torch.cuda.is_available()
        
        self.device = device
        self.model = init_model(cfg, ckpt, device=device)

    def predict(self, img):
        img = np.array(img)
        pred = inference_topdown(self.model, img)[0]
        key_pts = pred.pred_instances.key_points[0]
        return key_pts
        
    
class FashionSegmentation():
    def __init__(self, device="cpu"):
        if device == "cuda":
            assert torch.cuda.is_available()
        self.device = device

        self.processor = SegformerImageProcessor.from_pretrained("mattmdjaga/segformer_b2_clothes")
        self.model = AutoModelForSemanticSegmentation.from_pretrained("mattmdjaga/segformer_b2_clothes")

    def predict(self, img):
        input = self.extractor(img, return_tensors="pt")
        output = self.model(**input)
        logits = output.logits
        upsampled_logits = nn.functional.interpolate(
            logits,
            size=(512,384),
            mode="bilinear",
            align_corners=False
        )
        pred = upsampled_logits.argmax(dim=1)[0]
        return pred
    

class LadiVTON():
    def __init__(self, weight_dtype=torch.float16, device="cpu"):
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

        self.unet = torch.hub.load(repo_or_dir='miccunifi/ladi-vton', source='github', model='extended_unet', dataset="dress_code")
        self.emasc = torch.hub.load(repo_or_dir='miccunifi/ladi-vton', source='github', model='emasc', dataset="dress_code")
        self.inversion_adapter = torch.hub.load(repo_or_dir='miccunifi/ladi-vton', source='github', model='inversion_adapter', dataset="dress_code")
        self.tps, self.refinement = torch.hub.load(repo_or_dir='miccunifi/ladi-vton', source='github', model='warping_module', dataset="dress_code")

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
    
    def cloth_tps_transform(self, cloth_img, masked_img, pose_map):
        low_cloth = torchvision.transforms.functional.resize(cloth_img, (256, 192),
                                                                torchvision.transforms.InterpolationMode.BILINEAR, antialias=True)
        low_im_mask = torchvision.transforms.functional.resize(masked_img, (256, 192),
                                                                torchvision.transforms.InterpolationMode.BILINEAR, antialias=True)
        low_pose_map = torchvision.transforms.functional.resize(pose_map, (256, 192),
                                                                torchvision.transforms.InterpolationMode.BILINEAR, antialias=True)
        
        low_cloth, low_im_mask, low_pose_map = low_cloth.unsqueeze(0), low_im_mask.unsqueeze(0), low_pose_map.unsqueeze(0)
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

    def cloth_embedding(self, cloth_img, category):
        # Get the visual features of the in-shop cloths
        cloth_img = (cloth_img + 1) / 2
        input_image = torchvision.transforms.functional.resize(cloth_img, (224, 224), antialias=True)
        processed_images = self.processor(images=input_image, return_tensors="pt")
        clip_cloth_features = self.vision_encoder(processed_images.pixel_values.to(self.device, dtype=self.weight_dtype)).last_hidden_state

        # Compute the predicted PTEs
        word_embeddings = self.inversion_adapter(clip_cloth_features.to(self.device))
        word_embeddings = word_embeddings.reshape((word_embeddings.shape[0], 16, -1))
        
        category_text = {
            'dresses': 'a dress',
            'upper_body': 'an upper body garment',
            'lower_body': 'a lower body garment'
            }
        
        text = [f'a photo of a model wearing {category_text[category]} {" $ " * 16}']
        
        # Tokenize text
        tokenized_text = self.tokenizer(text, max_length=self.tokenizer.model_max_length, padding="max_length", truncation=True, return_tensors="pt").input_ids
        tokenized_text = tokenized_text.to(self.device)
        
        # Encode the text using the PTEs extracted from the in-shop cloths
        encoder_hidden_states = encode_text_word_embedding(self.text_encoder, tokenized_text, word_embeddings, 16).last_hidden_state

        return encoder_hidden_states

    def predict(self, kwargs):
        return self.vton_pipe(generator=self.generator, **kwargs).images[0]
            
            

