import torch
import numpy as np
from PIL import Image
from torchvision import transforms

from models import FashionPoseEstimation, BodyPoseEstimation, FashionSegmentation, LadiVTON
from utils import create_mask, resize, keypoint_to_heatmap, coco_keypoint_mapping

class Inferencer():
    def __init__(self, device="cpu"):
        if device == "cuda":
            assert torch.cuda.is_available()
        self.device = device

        self.fashion_pose_model = FashionPoseEstimation(device=device)
        self.body_pose_model = BodyPoseEstimation(device=device)
        self.fashion_seg_model = FashionSegmentation(device=device)
        self.vton_model = LadiVTON(device=device)

        self.transform = transforms.Compose([transforms.ToTensor(),
                                             transforms.Normalizae((0.5,0.5,0.5), (0.5,0.5,0.5))])

    def inference(self, body_img, cloth_img, category):
        assert isinstance(body_img, Image)
        assert isinstance(cloth_img, Image)
        assert category in ["dresses", "upper_body", "lower_body"]

        size = (512,384)
        guidance_scale = 5.0
        num_inference_steps = 50

        ## input preprocessing
        body_img = resize(body_img, keep_ratio=True)                                            # PIL.Image
        cloth_img = resize(cloth_img, keep_ratio=True)                                          # PIL.Image
        
        key_pts = self.body_pose_model.predict(body_img)                                        # np.ndarray (17,2)
        key_pts = coco_keypoint_mapping(key_pts)
        seg_map = self.fashion_seg_model.predict(body_img)                                      # torch.Tensor (512,384)

        body_img = self.transform(body_img).unsqieeze(0)
        cloth_img = self.transform(cloth_img).unsqueeze(0)

        mask_img, masked_img = create_mask(key_pts, seg_map)                                    # 
        pose_map = keypoint_to_heatmap(key_pts)                                                 # torch.Tensor (18,512,384)
        warped_cloth = self.vton_model.cloth_tps_transform(cloth_img, masked_img, pose_map)     # torch.Tensor (3,512,384)
        prompt_embeds = self.vton_model.cloth_embedding(cloth_img, category)


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
            "cloth_image_type": "warped_cloth",
            "num_inference_steps": num_inference_steps
        }
        vton_img = self.vton_model.predict(kwargs)
        
        return vton_img

