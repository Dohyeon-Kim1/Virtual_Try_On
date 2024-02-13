import os
import torch
import torchvision.transforms as transforms
from PIL import Image

from utils import resize


DATA_ROOT = ["/DressCode", "/AI-Hub-Fashion"]


class BodyClothPairDataset(torch.utils.data.Dataset):
    def __init__(self):
        self.data = []
        for data_root in DATA_ROOT:
            for category in os.listdir(data_root):
                with open(f"{data_root}/{category}/pairs.txt", "r") as f:
                    for pair in f.readlines():
                        img_name1, img_name2 = pair.strip().split("\t")
                        body_img_name = f"{data_root}/{category}/images/{img_name1}"
                        cloth_img_name = f"{data_root}/{category}/images/{img_name2}"
                        self.data.append([body_img_name, cloth_img_name, category])
        
        self.transform = transforms.Compose([transforms.ToTensor(),
                                             transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))])
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, i):
        body_img = self.transform(resize(Image.open(self.data[i][0]).convert("RGB"), (512,384)))
        cloth_img = self.transform(resize(Image.open(self.data[i][1]).convert("RGB"), (512,384)))
        category = self.data[i][2]
        return (body_img, cloth_img, category)