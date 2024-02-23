import torch
import numpy as np
from torchvision import transforms
from PIL import Image

from models import BodyPoseEstimation, FashionSegmentation, LadiVTON
from utils.data_utils import resize, create_mask, keypoint_to_heatmap, face_mask, tensor_to_arr


class Inferencer():
    def __init__(self, category_classifier_ckpt=None, tps_ckpt=None, emasc_ckpt=None,
                 inversion_adapter_ckpt=None, unet_ckpt=None, device="cuda"):
        if device == "cuda":
            assert torch.cuda.is_available()
        self.device = device

        self.body_pose_model = BodyPoseEstimation(device=device)
        self.fashion_seg_model = FashionSegmentation(device=device)
        self.vton_model = LadiVTON(category_classifier_ckpt, tps_ckpt, emasc_ckpt,
                                   inversion_adapter_ckpt, unet_ckpt, device=device)

        self.transform = transforms.Compose([transforms.ToTensor(),
                                             transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))])

    ## input: PIL.Image, PIL.Image, Str, Float, Int / Output: PIL.Image
    def inference(self, body_img, cloth_img, category, guidance_scale=5.0, num_inference_steps=50):
        size = (512,384)

        ## input preprocessing
        body_img, pad_size = resize(body_img, size=size, keep_ratio=True, return_pad_size=True)
        cloth_img = resize(cloth_img, size=size, keep_ratio=True)
        category = [category]
        
        body_img = self.transform(body_img).unsqueeze(0).to(self.device)
        cloth_img = self.transform(cloth_img).unsqueeze(0).to(self.device)

        if self.vton_model.category_classifier:
            subcategory = self.vton_model.category_classifier.predict(cloth_img, category)
        else:
            subcategory = category
        seg_map = self.fashion_seg_model.predict(body_img)
        key_pt = self.body_pose_model.predict(body_img)
        mask_img, masked_img = create_mask(body_img, seg_map, key_pt, category)                                    
        pose_map = keypoint_to_heatmap(key_pt, size)

        mask_img = mask_img.to(self.device)
        masked_img = masked_img.to(self.device)
        pose_map = pose_map.to(self.device)

        warped_cloth = self.vton_model.cloth_tps_transform(cloth_img, masked_img, pose_map)
        prompt_embeds = self.vton_model.cloth_embedding(cloth_img, subcategory)

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
        
        mask = np.array(face_mask(seg_map)[0].permute(1,2,0).cpu())
        vton_img = (tensor_to_arr(body_img, batch=False) * mask + np.array(vton_img) * (1-mask)).astype(np.uint8)
        if pad_size is not None:
            vton_img = vton_img[:,pad_size:-pad_size,:]
        vton_img = Image.fromarray(vton_img)
        return vton_img

