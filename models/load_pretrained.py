import torch

from mmpose.apis import inference_topdown, init_model


class FashionPoseEstimation:
    def __init__(self, kind, device="cpu"):
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
    
    
    class BodyPoseEstimation:
        def __init__(self, device="cpu"):
            cfg = ""
            ckpt = ""
            
            if device == "cuda":
                assert torch.cuda.is_available()
            
            self.device = device
            self.model = init_model(cfg, ckpt, device=device)

        def predict(self, img):
            return inference_topdown(self.model, img)
        
    
    class FashionSegmentation:
        def __init__(self):
            pass

        def predict(self, img):
            pass
    
    class LadiVTON:
        def __init__(self):
            pass    

        def predict(self, *input):
            pass

