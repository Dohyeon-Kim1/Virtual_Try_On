import torch
import numpy as np
from PIL import Image
from torchvision import transforms

from models import BodyPoseEstimation, FashionSegmentation, LadiVTON
from utils import resize, create_mask, keypoint_to_heatmap, remove_background

class Inferencer():
    def __init__(self, device="cpu"):
        if device == "cuda":
            assert torch.cuda.is_available()
        self.device = device

        self.body_pose_model = BodyPoseEstimation(device=device)
        self.fashion_seg_model = FashionSegmentation(device=device)
        self.vton_model = LadiVTON(device=device)

        self.transform = transforms.Compose([transforms.ToTensor(),
                                             transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))])

    def inference(self, body_img, cloth_img, category, guidance_scale=5.0, num_inference_steps=50):
        assert category in ["dresses", "upper_body", "lower_body"]

        size = (512,384)

        ## input preprocessing
        body_img = resize(body_img, size=size, keep_ratio=True)
        cloth_img = resize(cloth_img, size=size, keep_ratio=True)
        
        body_img = self.transform(body_img).unsqueeze(0).to(self.device)
        cloth_img = self.transform(cloth_img).unsqueeze(0).to(self.device)

        key_pt = self.body_pose_model.predict(body_img)
        seg_map = self.fashion_seg_model.predict(body_img)
        mask_img, masked_img = create_mask(body_img, seg_map, key_pt, [f"{category}"])                                    
        pose_map = keypoint_to_heatmap(key_pt, size)

        body_img = remove_background(body_img, seg_map)
        cloth_img = remove_background(cloth_img, seg_map)

        mask_img = mask_img.to(self.device)
        masked_img = masked_img.to(self.device)
        pose_map = pose_map.to(self.device)

        warped_cloth = self.vton_model.cloth_tps_transform(cloth_img, masked_img, pose_map)
        prompt_embeds = self.vton_model.cloth_embedding(cloth_img, [f"{category}"])

        ## generate image
        kwargs = {
            "image": body_img,
            "mask_image": mask_img,
            "pose_map": pose_map,
            "warped_cloth": warped_cloth,
            "prompt_embeds": prompt_embeds,
            "height": size[0],
            "width": size[1],
            "guidance_scale": guidance_scale,
            "num_images_per_prompt": 1,
            "cloth_input_type": "warped",
            "num_inference_steps": num_inference_steps
        }
        vton_img = self.vton_model.predict(kwargs)
        
        return vton_img

